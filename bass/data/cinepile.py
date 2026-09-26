import os
import subprocess

from .base_dataset import BaseDataset


class CinePileDataset(BaseDataset):
    """
    CinePile (https://huggingface.co/datasets/tomg-group-umd/cinepile).

    Annotations are loaded from the HuggingFace hub. Video clips are downloaded
    from YouTube with yt-dlp on first access and cached under root_dir/yt_videos.
    A cookies file at root_dir/cookies.txt is used if present.
    """

    HF_NAME = "tomg-group-umd/cinepile"
    ANSWER_LETTERS = "ABCDE"

    def __init__(self, root_dir="./dataset/CinePile", split="test", **kwargs):
        super().__init__(root_dir)
        from datasets import load_dataset

        self.video_dir = os.path.join(root_dir, "yt_videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.failed_urls = set()

        for idx, row in enumerate(load_dataset(self.HF_NAME, split=split)):
            safe_title = "".join(c if c.isalnum() else "_" for c in row["yt_clip_title"])
            self.samples.append({
                "id": f"cinepile_{split}_{idx}",
                "video_path": os.path.join(self.video_dir, f"{row['movie_name']}_{safe_title}.mp4"),
                "url": row["yt_clip_link"],
                "question": row["question"],
                "options": list(row["choices"]),
                "answer": self.ANSWER_LETTERS[int(row["answer_key_position"])],
            })

    def __getitem__(self, idx):
        sample = dict(self.samples[idx])
        path, url = sample["video_path"], sample["url"]
        if not os.path.exists(path):
            if url in self.failed_urls or not self._download_video(url, path):
                self.failed_urls.add(url)
                sample["video_path"] = None
        return sample

    def _download_video(self, url, output_path):
        cmd = [
            "yt-dlp",
            "-S", "height:224,ext:mp4:m4a",
            "--recode", "mp4",
            "-o", output_path,
            url,
        ]
        cookies_path = os.path.join(self.root_dir, "cookies.txt")
        if os.path.exists(cookies_path):
            cmd.extend(["--cookies", cookies_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except Exception as e:
            print(f"[CinePile] Download failed for {url}: {e}")
            return False
        if result.returncode != 0:
            print(f"[CinePile] Download failed for {url}: {result.stderr.strip()[:200]}")
            return False
        return True
