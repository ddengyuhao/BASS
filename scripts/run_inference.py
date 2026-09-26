import sys
import os
import json
import argparse

from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bass.data import DATASET_REGISTRY
from bass.methods import METHOD_REGISTRY
from bass.utils import extract_answer_from_text


def parse_args():
    parser = argparse.ArgumentParser(description="BASS Inference")

    # --- Paths & Dataset ---
    parser.add_argument("--dataset", type=str, required=True, choices=["VideoMME", "CinePile", "VRBench"], help="Target dataset")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Root directory for datasets")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")

    # --- Model Config ---
    parser.add_argument("--method", type=str, default="BASS", choices=["BASS"])
    parser.add_argument("--backbone", type=str, default="Qwen2.5-VL-7B", choices=["Qwen2.5-VL-7B", "Qwen2-VL-72B"], help="LMM backbone")
    parser.add_argument("--model_path", type=str, default=None, help="Path to local LMM checkpoint")
    parser.add_argument("--clip_path", type=str, default=None, help="Path to local CLIP model")

    # --- Offline indexing (Sec. 3) ---
    parser.add_argument("--segmentation", type=str, default="transnetv2", choices=["transnetv2", "pyscenedetect", "uniform"], help="Temporal boundary detector used to partition videos into events")
    parser.add_argument("--edge_construction", type=str, default="index", choices=["index", "pairwise"], help="Semantic edges via inverted-index join or dense pairwise patch matching")
    parser.add_argument("--uniform_word_weight", action="store_true", help="Use uniform visual-word weights instead of inverse-event-frequency weights")
    parser.add_argument("--delta", type=float, default=0.65, help="Semantic similarity threshold for semantic edges")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Number of visual words for vector quantization")
    parser.add_argument("--rho", type=int, default=50, help="Posting-list cap; more frequent visual words are dropped as stop words")
    parser.add_argument("--frame_interval", type=float, default=4.0, help="Seconds per representative frame within an event")
    parser.add_argument("--max_frames_per_event", type=int, default=4, help="Maximum representative frames per event")
    parser.add_argument("--frame_size", type=int, default=336, help="Resolution of representative frames fed to the LMM")

    # --- Online planning and execution (Sec. 4) ---
    parser.add_argument("--planner", type=str, default="celf", choices=["celf", "random"], help="Cost-aware lazy greedy planner or cost-feasible random selection")
    parser.add_argument("--disable_relevance", action="store_true", help="Remove the query relevance term F_rel from the objective")
    parser.add_argument("--execution", type=str, default="graph", choices=["graph", "cot"], help="Graph-guided execution pipeline or standard Chain-of-Thought prompting")
    parser.add_argument("--token_budget", type=int, default=8192, help="Visual-token budget B")
    parser.add_argument("--lambda_param", type=float, default=1.0, help="Trade-off between query relevance and reachable information gain")
    parser.add_argument("--alpha", type=float, default=0.15, help="PPR restart probability")
    parser.add_argument("--max_paths", type=int, default=8, help="Maximum reasoning paths used in result synthesis")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generation length per LMM call")

    # --- Distributed ---
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None, help="Evaluate only the first N queries")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def load_backbone(args):
    if args.backbone == "Qwen2.5-VL-7B":
        from bass.models.qwen2_5_vl import Qwen2_5_VLWrapper
        path = args.model_path or os.environ.get("QWEN_PATH")
        return Qwen2_5_VLWrapper(model_path=path)
    if args.backbone == "Qwen2-VL-72B":
        from bass.models.qwen2_vl_72b import Qwen2_VL_72B_Wrapper
        return Qwen2_VL_72B_Wrapper(model_path=args.model_path)
    raise ValueError(f"Backbone {args.backbone} not implemented.")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Chunk {args.chunk_idx}/{args.num_chunks}] Inference: {args.dataset} | {args.backbone}")

    # 1. Load Dataset
    dataset_cls = DATASET_REGISTRY[args.dataset]
    dataset = dataset_cls(root_dir=args.data_root)

    if args.max_samples:
        dataset.samples = dataset.samples[:args.max_samples]

    # Group queries of the same video so that its event graph is built once and reused
    dataset.samples = sorted(
        dataset.samples, key=lambda s: (str(s.get('video_path') or ''), str(s.get('id'))))

    # 2. Chunking
    total = len(dataset)
    chunk_size = (total + args.num_chunks - 1) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total)
    dataset.samples = dataset.samples[start_idx:end_idx]

    print(f"   Processing indices {start_idx} -> {end_idx} (N={len(dataset)})")
    if len(dataset) == 0:
        return

    # 3. Load Model and Method
    model = load_backbone(args)
    processor = METHOD_REGISTRY[args.method](args, model)

    # 4. Loop
    results = []
    for sample in tqdm(dataset, desc=f"GPU {args.chunk_idx}"):
        options = sample.get('options', [])
        num_options = len(options) if isinstance(options, (list, tuple)) and options else 4
        try:
            if not sample.get('video_path'):
                raise FileNotFoundError("Video path is None")

            response = processor.process_and_inference(
                sample['video_path'], sample['question'], options)

            # Unparseable responses are kept as None and counted as incorrect
            pred = extract_answer_from_text(response, num_options=num_options)
            results.append({
                "id": sample['id'],
                "pred": pred,
                "raw_response": response,
                "gt": sample.get('answer', ''),
                "q": sample.get('question', ''),
                "trace": getattr(processor, 'last_trace', None),
            })

            # Periodic checkpoint
            if len(results) % 5 == 0:
                ckpt = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_chunk{args.chunk_idx}.partial.json")
                with open(ckpt, 'w') as f:
                    json.dump(results, f)

        except Exception as e:
            print(f"Error ID {sample.get('id')}: {e}")
            results.append({"id": sample.get('id'), "error": str(e), "pred": None,
                            "gt": sample.get('answer', '')})

    # Final Save
    out_file = os.path.join(args.output_dir, f"{args.dataset}_{args.method}_chunk{args.chunk_idx}.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if r.get('gt')]
    acc = sum(r.get('pred') == r['gt'] for r in valid) / max(len(valid), 1)
    print(f"Saved to {out_file} | accuracy on this chunk: {acc * 100:.2f}% ({len(valid)} queries)")


if __name__ == "__main__":
    main()
