# convert_jsonl_to_parquet.py
import json
import pandas as pd
from tqdm import tqdm
import argparse

def valid_duration(duration, min_dur=0.5, max_dur=30.0):
    return min_dur <= duration <= max_dur

def jsonl_to_parquet(jsonl_path, parquet_path, min_duration=0.5, max_duration=30.0):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing JSONL"):
            try:
                item = json.loads(line.strip())
                dur = float(item.get("duration", 0))
                if not valid_duration(dur, min_duration, max_duration):
                    continue

                # 提取关键字段
                record = {
                    "utt_id": item.get("utt_id", ""),
                    "audio_path": item["audio"]["path"],
                    "sentence": item.get("sentence", ""),
                    "sentences": item.get("sentences", []),  # 用于 timestamps=True
                    "language": item.get("language", None),
                    "data_source": item.get("data_source", "default"),
                }
                records.append(record)
            except Exception as e:
                print(f"Skip invalid line: {e}")
                continue

    print(f"Total valid samples: {len(records)}")
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, engine='pyarrow', index=False)
    print(f"Saved to {parquet_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Path to input .jsonl file")
    parser.add_argument("--parquet", required=True, help="Output .parquet path")
    parser.add_argument("--min_duration", type=float, default=0.5)
    parser.add_argument("--max_duration", type=float, default=30.0)
    args = parser.parse_args()

    jsonl_to_parquet(
        args.jsonl,
        args.parquet,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )