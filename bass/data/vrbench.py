import os
import json

from .base_dataset import BaseDataset


class VRBenchDataset(BaseDataset):
    """
    VRBench (https://huggingface.co/datasets/OpenGVLab/VRBench): 1,010 long narrative
    videos with multi-step reasoning multiple-choice questions.

    Expected structure:
        root_dir/
            VRBench_eval.jsonl
            videos/{video_id}.mp4      (extracted from v001_360p_zips)

    Each line of VRBench_eval.jsonl describes one video; its "mcq" field maps
    question keys (qa1, qa2, ...) to {"question", "options": {"A": ...}, "answer"}.
    """

    def __init__(self, root_dir="./dataset/VRBench", split="test", **kwargs):
        super().__init__(root_dir)
        self.video_dir = os.path.join(root_dir, "videos")
        ann_file = os.path.join(root_dir, "VRBench_eval.jsonl")
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"VRBench annotation not found at {ann_file}")

        with open(ann_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                vid = item["video_id"]
                video_path = self.resolve_video([
                    os.path.join(self.video_dir, f"{vid}.mp4"),
                    os.path.join(self.video_dir, "v001", f"{vid}.mp4"),
                    os.path.join(root_dir, item.get("video_path", "")),
                ])
                mcq = item.get("mcq") or {}
                for key in sorted(mcq, key=self._qa_index):
                    qa = mcq[key]
                    opts = qa["options"]
                    if isinstance(opts, dict):
                        opts = [f"{k}. {opts[k]}" for k in sorted(opts)]
                    self.samples.append({
                        "id": f"{vid}_{key}",
                        "video_path": video_path,
                        "question": qa["question"],
                        "options": list(opts),
                        "answer": str(qa["answer"]).strip().upper()[:1],
                    })

    @staticmethod
    def _qa_index(key):
        digits = "".join(c for c in key if c.isdigit())
        return int(digits) if digits else 0
