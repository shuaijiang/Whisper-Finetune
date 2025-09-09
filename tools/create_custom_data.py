import argparse
import json
import os
from tqdm import tqdm
import soundfile


def load_text(text_file):
    """加载文本文件，格式：utt_id sentence"""
    print(f"Loading text from {text_file}...")
    transcript_dict = {}
    with open(text_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)  # 分割第一个空格
            if len(parts) < 2:
                continue
            utt_id, text = parts
            text = ''.join(text.split())  # 去除所有空格
            transcript_dict[utt_id] = text
    print(f"Loaded {len(transcript_dict)} utterances from text.")
    return transcript_dict


def load_wav_scp(wav_scp_file):
    """加载 wav.scp 文件，格式：utt_id /path/to/audio.wav"""
    print(f"Loading wav.scp from {wav_scp_file}...")
    wav_dict = {}
    with open(wav_scp_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            utt_id, wav_path = parts
            wav_dict[utt_id] = wav_path.strip()
    print(f"Loaded {len(wav_dict)} utterances from wav.scp.")
    return wav_dict


def add_duration(data_list):
    """批量添加 duration 和 sentences 字段"""
    print("Adding duration info...")
    for item in tqdm(data_list):
        try:
            sample, sr = soundfile.read(item['audio']['path'])
            duration = round(sample.shape[-1] / float(sr), 2)
            item['duration'] = duration
            item['sentences'] = [{"start": 0, "end": duration, "text": item["sentence"]}]
        except Exception as e:
            print(f"Error reading {item['audio']['path']}: {e}")
            item['duration'] = 0.0
    return data_list


def create_custom_manifest(text_file, wav_scp_file, output_jsonl):
    """主函数：生成 custom jsonl 文件"""
    transcript_dict = load_text(text_file)
    wav_dict = load_wav_scp(wav_scp_file)

    # 取交集：只保留同时在 text 和 wav.scp 中的 utt_id
    common_utts = set(transcript_dict.keys()) & set(wav_dict.keys())
    if not common_utts:
        raise ValueError("No common utterances found between text and wav.scp!")

    print(f"Found {len(common_utts)} common utterances.")

    lines = []
    for utt_id in sorted(common_utts):
        text = transcript_dict[utt_id]
        wav_path = wav_dict[utt_id]

        # 检查音频文件是否存在
        if not os.path.exists(wav_path):
            print(f"Warning: Audio file not found: {wav_path}")
            continue

        line = {
            "utt_id": utt_id,
            "audio": {"path": wav_path},
            "sentence": text
        }
        lines.append(line)

    # 添加时长信息
    lines = add_duration(lines)

    # 写入 JSONL 文件
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"Saved manifest to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Generate custom JSONL manifest from text and wav.scp")
    parser.add_argument("--text", required=True, type=str, help="Path to text file, format: utt_id sentence")
    parser.add_argument("--wav_scp", required=True, type=str, help="Path to wav.scp file, format: utt_id /path/to/audio.wav")
    parser.add_argument("--output", required=True, type=str, help="Output JSONL file path")
    parser.add_argument("--add_pun", action='store_true', help="Whether to add punctuation (requires modelscope)")

    args = parser.parse_args()

    if args.add_pun:
        print("Punctuation restoration is not implemented here. You can extend it using ModelScope.")
        # 这里可以集成 modelscope 的标点恢复模型
        # from modelscope.pipelines import pipeline
        # pipe = pipeline('punctuation', model='damo/punc_ct-transformer...')
        # text = pipe(text_in=text)['text']

    create_custom_manifest(args.text, args.wav_scp, args.output)


if __name__ == '__main__':
    main()
