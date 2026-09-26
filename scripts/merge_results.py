"""
Merges per-chunk outputs of run_inference.py and reports accuracy, visual-token
usage, and latency.

Usage:
    python scripts/merge_results.py <output_dir>
"""
import sys
import os
import glob
import json


def main(output_dir):
    files = sorted(f for f in glob.glob(os.path.join(output_dir, "*_chunk*.json"))
                   if not f.endswith(".partial.json"))
    records = []
    for f in files:
        with open(f) as fd:
            records.extend(json.load(fd))

    out_path = os.path.join(output_dir, "merged.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    valid = [r for r in records if r.get("gt")]
    correct = sum(r.get("pred") == r["gt"] for r in valid)
    traces = [r["trace"] for r in records if r.get("trace")]

    print(f"Merged {len(records)} records from {len(files)} files -> {out_path}")
    print(f"Accuracy: {100.0 * correct / max(len(valid), 1):.2f}% ({correct}/{len(valid)})")
    if traces:
        tokens = [t["visual_tokens"] for t in traces if "visual_tokens" in t]
        online = [t["online_time"] for t in traces if "online_time" in t]
        offline = [t["offline_time"] for t in traces if t.get("offline_time")]
        if tokens:
            print(f"Visual tokens per query: {sum(tokens) / len(tokens):.1f}")
        if online:
            print(f"Online latency: {sum(online) / len(online):.2f} s/query")
        if offline:
            print(f"Offline indexing: {sum(offline) / len(offline):.2f} s/video ({len(offline)} videos)")
    errors = sum(1 for r in records if r.get("error"))
    if errors:
        print(f"Failed queries: {errors}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./results")
