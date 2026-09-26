import os

from .base_dataset import BaseDataset


class VideoMMEDataset(BaseDataset):
    """
    Video-MME (https://huggingface.co/datasets/lmms-lab/Video-MME): 900 videos, 2,700 QA pairs.

    Expected structure:
        root_dir/
            videos/{videoID}.mp4
            videomme/test-00000-of-00001.parquet   (optional local copy of the annotations)

    If no local annotation file is found, annotations are loaded from the
    HuggingFace hub.
    """

    HF_NAME = "lmms-lab/Video-MME"

    def __init__(self, root_dir="./dataset/VideoMME", split="test", **kwargs):
        super().__init__(root_dir)
        self.video_dir = os.path.join(root_dir, "videos")
        rows = self._load_annotations(split)

        for row in rows:
            vid = row["videoID"]
            self.samples.append({
                "id": row["question_id"],
                "video_path": self.resolve_video([
                    os.path.join(self.video_dir, f"{vid}.mp4"),
                    os.path.join(self.video_dir, f"{vid}.mkv"),
                ]),
                "question": row["question"],
                "options": list(row["options"]),
                "answer": row["answer"],
                "duration": row.get("duration"),
                "task_type": row.get("task_type"),
            })

    def _load_annotations(self, split):
        from datasets import load_dataset

        local = os.path.join(self.root_dir, "videomme", f"{split}-00000-of-00001.parquet")
        if os.path.exists(local):
            ds = load_dataset("parquet", data_files=local, split="train")
        else:
            ds = load_dataset(self.HF_NAME, split=split)
        return list(ds)
