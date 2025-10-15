import json, random, argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",       default="datasets/humaneval/humaneval-py.jsonl")
    p.add_argument("--out_dir",   default="datasets/humaneval/")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

def main():
    args = parse_args()
    src_path = Path(args.src)
    out_dir  = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 逐行读取，去掉行尾换行，保证之后手动加 '\n'
    with open(src_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n\r") for line in f if line.strip()]

    random.seed(args.seed)
    random.shuffle(lines)

    split_idx = int(len(lines) * args.train_ratio)
    train_lines, test_lines = lines[:split_idx], lines[split_idx:]

    def write_jsonl(path: Path, records):
        # 逐条写入并显式加 '\n'，最后一条也加
        with open(path, "w", encoding="utf-8") as w:
            for rec in records:
                w.write(rec + "\n")

    write_jsonl(out_dir / "train.jsonl", train_lines)
    write_jsonl(out_dir / "test.jsonl",  test_lines)

    print(f"Total {len(lines)}  →  train {len(train_lines)}  test {len(test_lines)}")

if __name__ == "__main__":
    main()