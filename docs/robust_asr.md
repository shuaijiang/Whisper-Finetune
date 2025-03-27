# Build Robust ASR

## Audio-augmentation
通过对音频的变速（0.9 1.0 1.1三种变速倍数），提升语音数据多样性。
参考工作： [AudioAugmentation](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.pdf)

```json
[
  {
    "type": "speed",
    "params": {
      "min_speed_rate": 0.9,
      "max_speed_rate": 1.1,
      "num_rates": 3
    },
    "prob": 0.0
  },
]
```

## Add-noise
通过叠加声学环境中的各种类型的噪声，模拟不同信噪比、不同噪声的声学场景。
数据来源：[RoomImpulseResponseandNoiseDatabase](http://openslr.org/28/)
```json
[
  {
    "type": "noise",
    "params": {
      "min_snr_dB": 10,
      "max_snr_dB": 50,
      "noise_dir": "path_for/RIRS_NOISES/pointsource_noises/"
    },
    "prob": 0.5
  },
]
```

## Add-reverb 
模拟房间（或者车内等空间的）混响，通过采集到的房间冲击响应，与干净语音数据做卷积实现加混响，从而提升语音识别的鲁棒性。
数据来源：[RoomImpulseResponseandNoiseDatabase](http://openslr.org/28/)
```json
[
  {
    "type": "reverb",
    "params": {
      "rir_dir": "path_for/RIRS_NOISES/simulated_rirs/"
    },
    "prob": 0.5
  },
]
```

## SpecAugment
在语音频谱上在时间、频率上分别随机做mask，从而提升语音识别的鲁棒性。
参考工作：[SpecAugment](https://arxiv.org/abs/1904.08779)

```json
[
  {
    "type": "specaug",
    "params": {
      "num_t_mask": 2,
      "num_f_mask": 2,
      "max_t": 50,
      "max_f": 10
    },
    "prob": 0.5
  },
]
```

Practice for building [Belle-Robust-whisper-v3-turbo-zh](https://huggingface.co/BELLE-2/Belle-Robust-whisper-large-v3-turbo-zh)
<details>
  <summary>
    <b>configs/augmentation.json</b>
  </summary>

```json
[
  {
    "type": "resample",
    "params": {
      "new_sample_rates": [8000, 32000, 44100]
    },
    "prob": 0.0
  },
  {
    "type": "noise",
    "params": {
      "min_snr_dB": 10,
      "max_snr_dB": 50,
      "noise_dir": "path_for/RIRS_NOISES/pointsource_noises/"
    },
    "prob": 0.5
  },
  {
    "type": "reverb",
    "params": {
      "rir_dir": "path_for/RIRS_NOISES/simulated_rirs/"
    },
    "prob": 0.5
  },
  {
    "type": "speed",
    "params": {
      "min_speed_rate": 0.9,
      "max_speed_rate": 1.1,
      "num_rates": 3
    },
    "prob": 0.0
  },
  {
    "type": "shift",
    "params": {
      "min_shift_ms": -5,
      "max_shift_ms": 5
    },
    "prob": 0.0
  },
  {
    "type": "volume",
    "params": {
      "min_gain_dBFS": -15,
      "max_gain_dBFS": 15
    },
    "prob": 0.0
  },
  {
    "type": "specaug",
    "params": {
      "num_t_mask": 2,
      "num_f_mask": 2,
      "max_t": 50,
      "max_f": 10
    },
    "prob": 0.5
  }
]
```
</details>