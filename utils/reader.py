import json
import os
import random
import sys
from typing import List

import librosa
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.signal import fftconvolve
from utils.binary import DatasetReader


class CustomDataset(Dataset):
    def __init__(self,
                 data_list_path,
                 processor,
                 mono=True,
                 language=None,
                 timestamps=False,
                 sample_rate=16000,
                 min_duration=0.5,
                 max_duration=30,
                 augment_config_path=None):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            mono: 是否将音频转换成单通道，这个必须是True
            language: 微调数据的语言
            timestamps: 微调时是否使用时间戳
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            augment_config_path: 数据增强配置参数文件路径
        """
        super(CustomDataset, self).__init__()
        assert min_duration >= 0.5, f"min_duration不能小于0.5，当前为：{min_duration}"
        assert max_duration <= 30, f"max_duration不能大于30，当前为：{max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.data_list_path = data_list_path
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.timestamp_begin = self.vocab['<|notimestamps|>'] + 1
        self.startoftranscript = self.vocab['<|startoftranscript|>']
        self.endoftext = self.vocab['<|endoftext|>']
        if '<|nocaptions|>' in self.vocab:
            self.nocaptions = self.vocab['<|nocaptions|>']
        else:
            self.nocaptions = self.vocab['<|nospeech|>'] # support whisper_large_v3
        self.data_list: List[dict] = []
        # 加载数据列表
        self._load_data_list()
        # 数据增强配置参数
        self.augment_configs = None
        self.spec_aug = None

        self.noise_files = None
        self.rir_files = None
        self.speed_rates = None
        if augment_config_path:
            with open(augment_config_path, 'r', encoding='utf-8') as f:
                self.augment_configs = json.load(f)
            self.spec_aug = dict()
            for augment_config in self.augment_configs:
                if augment_config.get('type') == 'noise':
                    noise_dir = augment_config['params'].get('noise_dir', None)
                    if noise_dir:
                        self.noise_files = self._load_wav_files(noise_dir)
                    else:
                        print("Warning: 'noise_dir' is not specified in the configuration.")
                if augment_config.get('type') == 'reverb':
                    rir_dir = augment_config['params'].get('rir_dir', None)
                    if rir_dir:
                        self.rir_files = self._load_wav_files(rir_dir)  # 加载混响文件
                    else:
                        print("Warning: 'rir_dir' is not specified in the configuration.")
                if augment_config.get('type') == 'specaug':
                    params = augment_config.get('params', {})
                    prob = augment_config.get('prob', 0.5)
                    defaults = {
                        'num_t_mask': 2,
                        'num_f_mask': 2,
                        'max_t': 50,
                        'max_f': 10,
                        'prob': prob,
                    }
                    self.spec_aug.update({key: params.get(key, default_value) for key, default_value in defaults.items()})
                    
    # 从指定路径加载语音数据
    def _load_wav_files(self, data_dir):
        wave_files = list()
        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.wav'):
                        wave_files.append(os.path.join(root, file))
        return wave_files

    # 加载数据列表
    def _load_data_list(self):
        if self.data_list_path.endswith(".header"):
            # 获取二进制的数据列表
            self.dataset_reader = DatasetReader(data_header_path=self.data_list_path,
                                                min_duration=self.min_duration,
                                                max_duration=self.max_duration)
            self.data_list = self.dataset_reader.get_keys()
        else:
            # 获取数据列表
            with open(self.data_list_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.data_list = []
            for line in tqdm(lines, desc='读取数据列表'):
                if isinstance(line, str):
                    line = json.loads(line)
                if not isinstance(line, dict): continue
                # 跳过超出长度限制的音频
                if line["duration"] < self.min_duration:
                    continue
                if self.max_duration != -1 and line["duration"] > self.max_duration:
                    continue
                self.data_list.append(dict(line))

    # 从数据列表里面获取音频数据、采样率和文本
    def _get_list_data(self, idx):
        if self.data_list_path.endswith(".header"):
            data_list = self.dataset_reader.get_data(self.data_list[idx])
        else:
            data_list = self.data_list[idx]
        # 分割音频路径和标签
        audio_file = data_list["audio"]['path']
        transcript = data_list["sentences"] if self.timestamps else data_list["sentence"]
        language = data_list["language"] if 'language' in data_list.keys() else None
        if 'start_time' not in data_list["audio"].keys():
            sample, sample_rate = soundfile.read(audio_file, dtype='float32')
        else:
            start_time, end_time = data_list["audio"]["start_time"], data_list["audio"]["end_time"]
            # 分割读取音频
            sample, sample_rate = self.slice_from_file(audio_file, start=start_time, end=end_time)
        sample = sample.T
        # 转成单通道
        if self.mono:
            sample = librosa.to_mono(sample)
        # 数据增强
        if self.augment_configs:
            sample, sample_rate = self.augment(sample, sample_rate)
        # 重采样
        if self.sample_rate != sample_rate:
            sample = self.resample(sample, orig_sr=sample_rate, target_sr=self.sample_rate)
        if 'id' in data_list:
            return data_list['id'], sample, sample_rate, transcript, language
        else:
            return data_list['audio']['path'], sample, sample_rate, transcript, language

    def _load_timestamps_transcript(self, transcript: List[dict]):
        assert isinstance(transcript, list), f"transcript应该为list，当前为：{type(transcript)}"
        data = dict()
        labels = self.processor.tokenizer.prefix_tokens[:3]
        for t in transcript:
            # 将目标文本编码为标签ID
            start = t['start'] if round(t['start'] * 100) % 2 == 0 else t['start'] + 0.01
            start = self.timestamp_begin + round(start * 100) // 2
            end = t['end'] if round(t['end'] * 100) % 2 == 0 else t['end'] - 0.01
            end = self.timestamp_begin + round(end * 100) // 2
            label = self.processor(text=t['text']).input_ids[4:-1]
            labels.extend([start])
            labels.extend(label)
            labels.extend([end])
        data['labels'] = labels + [self.endoftext]
        return data

    def __getitem__(self, idx):
        try:
            # 从数据列表里面获取音频数据、采样率和文本
            audio_id, sample, sample_rate, transcript, language = self._get_list_data(idx=idx)
            # 可以为单独数据设置语言
            self.processor.tokenizer.set_prefix_tokens(language=language if language is not None else self.language)
            if len(transcript) > 0:
                # 加载带有时间戳的文本
                if self.timestamps:
                    data = self._load_timestamps_transcript(transcript=transcript)
                    # 从输入音频数组中计算log-Mel输入特征
                    data["input_features"] = self.processor(audio=sample, sampling_rate=self.sample_rate).input_features
                else:
                    # 获取log-Mel特征和标签ID
                    data = self.processor(audio=sample, sampling_rate=self.sample_rate, text=transcript)
            else:
                # 如果没有文本，则使用<|nocaptions|>标记
                data = self.processor(audio=sample, sampling_rate=self.sample_rate)
                data['labels'] = [self.startoftranscript, self.nocaptions, self.endoftext]
            data['id'] = audio_id
            if self.augment_configs and self.spec_aug:
                if random.random() < self.spec_aug.get('prob'):
                    data = self.spec_augmentation(data, 
                                                  num_t_mask=self.spec_aug['num_t_mask'], 
                                                  num_f_mask=self.spec_aug['num_f_mask'],
                                                  max_t=self.spec_aug['max_t'],
                                                  max_f=self.spec_aug['max_f'])
            return data
        except Exception as e:
            print(f'读取数据出错，序号：{idx}，错误信息：{e}', file=sys.stderr)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)

    # 分割读取音频
    @staticmethod
    def slice_from_file(file, start, end):
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = round(float(len(sndfile)) / sample_rate, 3)
        start = round(start, 3)
        end = round(end, 3)
        # 从末尾开始计
        if start < 0.0: start += duration
        if end < 0.0: end += duration
        # 保证数据不越界
        if start < 0.0: start = 0.0
        if end > duration: end = duration
        if end < 0.0:
            raise ValueError("切片结束位置(%f s)越界" % end)
        if start > end:
            raise ValueError("切片开始位置(%f s)晚于切片结束位置(%f s)" % (start, end))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        sample = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return sample, sample_rate

    # 数据增强
    def augment(self, sample, sample_rate):
        for config in self.augment_configs:
            if config['type'] == 'speed' and random.random() < config['prob']:
                if self.speed_rates is None:
                    min_speed_rate, max_speed_rate, num_rates = config['params']['min_speed_rate'], \
                        config['params']['max_speed_rate'], config['params']['num_rates']
                    self.speed_rates = np.linspace(min_speed_rate, max_speed_rate, num_rates, endpoint=True)
                rate = random.choice(self.speed_rates)
                sample = self.change_speed(sample, speed_rate=rate)
            if config['type'] == 'shift' and random.random() < config['prob']:
                min_shift_ms, max_shift_ms = config['params']['min_shift_ms'], config['params']['max_shift_ms']
                shift_ms = random.randint(min_shift_ms, max_shift_ms)
                sample = self.shift(sample, sample_rate, shift_ms=shift_ms)
            if config['type'] == 'volume' and random.random() < config['prob']:
                min_gain_dBFS, max_gain_dBFS = config['params']['min_gain_dBFS'], config['params']['max_gain_dBFS']
                gain = random.randint(min_gain_dBFS, max_gain_dBFS)
                sample = self.volume(sample, gain=gain)
            if config['type'] == 'resample' and random.random() < config['prob']:
                new_sample_rates = config['params']['new_sample_rates']
                new_sample_rate = np.random.choice(new_sample_rates)
                sample = self.resample(sample, orig_sr=sample_rate, target_sr=new_sample_rate)
                sample_rate = new_sample_rate
            if config['type'] == 'noise' and random.random() < config['prob']:
                min_snr_dB, max_snr_dB = config['params']['min_snr_dB'], config['params']['max_snr_dB']
                if self.noise_files and len(self.noise_files) > 0:
                    noise_file = random.choice(self.noise_files)
                    snr_dB = random.randint(min_snr_dB, max_snr_dB)
                    sample = self.add_noise(sample, sample_rate, noise_path=noise_file, snr_dB=snr_dB)
            if config['type'] == 'reverb' and random.random() < config['prob']:
                if self.rir_files and len(self.rir_files) > 0:
                    rir_file = random.choice(self.rir_files)
                    sample = self.add_reverb(sample, sample_rate, rir_file)
        return sample, sample_rate

    # 改变语速
    @staticmethod
    def change_speed(sample, speed_rate):
        if speed_rate == 1.0:
            return sample
        if speed_rate <= 0:
            raise ValueError("速度速率应大于零")
        old_length = sample.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        sample = np.interp(new_indices, old_indices, sample).astype(np.float32)
        return sample

    # 音频偏移
    @staticmethod
    def shift(sample, sample_rate, shift_ms):
        duration = sample.shape[0] / sample_rate
        if abs(shift_ms) / 1000.0 > duration:
            raise ValueError("shift_ms的绝对值应该小于音频持续时间")
        shift_samples = int(shift_ms * sample_rate / 1000)
        if shift_samples > 0:
            sample[:-shift_samples] = sample[shift_samples:]
            sample[-shift_samples:] = 0
        elif shift_samples < 0:
            sample[-shift_samples:] = sample[:shift_samples]
            sample[:-shift_samples] = 0
        return sample

    # 改变音量
    @staticmethod
    def volume(sample, gain):
        sample *= 10.**(gain / 20.)
        return sample

    # 声音重采样
    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    # 添加噪声
    def add_noise(self, sample, sample_rate, noise_path, snr_dB, max_gain_db=300.0):
        noise_sample, sr = librosa.load(noise_path, sr=sample_rate)
        # 标准化音频音量，保证噪声不会太大
        target_db = -20
        gain = min(max_gain_db, target_db - self.rms_db(sample))
        sample *= 10. ** (gain / 20.)
        # 指定噪声音量
        sample_rms_db, noise_rms_db = self.rms_db(sample), self.rms_db(noise_sample)
        noise_gain_db = min(sample_rms_db - noise_rms_db - snr_dB, max_gain_db)
        noise_sample *= 10. ** (noise_gain_db / 20.)
        # 固定噪声长度
        if noise_sample.shape[0] < sample.shape[0]:
            diff_duration = sample.shape[0] - noise_sample.shape[0]
            noise_sample = np.pad(noise_sample, (0, diff_duration), 'wrap')
        elif noise_sample.shape[0] > sample.shape[0]:
            start_frame = random.randint(0, noise_sample.shape[0] - sample.shape[0])
            noise_sample = noise_sample[start_frame:sample.shape[0] + start_frame]
        sample += noise_sample
        return sample
    
    def add_reverb(self, sample, sample_rate, rir_file):
        rir_data, _ = librosa.load(rir_file, sr=sample_rate)
        sample_convolved = fftconvolve(sample, rir_data, mode='same')
        # 能量归一化
        max_rate = np.max(np.abs(sample)) / np.max(np.abs(sample_convolved))
        sample_convolved = sample_convolved * max_rate
        return sample_convolved

    @staticmethod
    def rms_db(sample):
        mean_square = np.mean(sample ** 2)
        return 10 * np.log10(mean_square)
    
    def spec_augmentation(self, sample, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10):
        """ Do spec augmentation
            Inplace operation

            Args:
                sample: {id, input_features, ...}
                num_t_mask: number of time mask to apply
                num_f_mask: number of freq mask to apply
                max_t: max width of time mask
                max_f: max width of freq mask

            Returns
                {id, input_features, ....}
        """
        assert 'input_features' in sample
        input_features_array = np.array(sample['input_features'])
        x = torch.as_tensor(input_features_array)
        assert isinstance(x, torch.Tensor)
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for _ in range(num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for _ in range(num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        sample['input_features'] = y
        return sample