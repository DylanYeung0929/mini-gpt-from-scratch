"""Convert Alpaca-style JSON arrays to instruction jsonl (instruction/input/output)."""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in data:
            if not row.get("instruction") or not row.get("output"):
                continue
            f.write(json.dumps({
                "instruction": row["instruction"],
                "input": row.get("input", ""),
                "output": row["output"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} examples -> {args.output}")


if __name__ == "__main__":
    main()
