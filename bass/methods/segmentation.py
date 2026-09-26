"""
Temporal boundary detection used to partition a video into events (Sec. 3).

Supported segmenters (Sec. 5.4, Table 2):
    transnetv2     TransNetV2 shot transition detector (default)
    pyscenedetect  content-based scene cut detection
    uniform        fixed-duration windows
"""

import numpy as np
import torch


def _video_info(video_path):
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    return vr, vr.get_avg_fps(), len(vr)


def uniform_windows(video_path, window=5.0):
    _, fps, total = _video_info(video_path)
    duration = total / fps if fps > 0 else 0.0
    if duration <= 0:
        return [(0.0, 1.0)]
    return [(float(t), float(min(t + window, duration))) for t in np.arange(0, duration, window)]


class TransNetV2Segmenter:
    """TransNetV2 shot boundary detector (transnetv2-pytorch)."""

    def __init__(self, device="cuda", threshold=0.5, min_distance=15, batch_size=500):
        from transnetv2_pytorch import TransNetV2
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.min_distance = min_distance
        self.batch_size = batch_size
        self.model = TransNetV2().to(self.device).eval()

    def __call__(self, video_path):
        from PIL import Image

        vr, fps, total = _video_info(video_path)
        predictions = []
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            frames = vr.get_batch(list(range(start, end))).asnumpy()
            # TransNetV2 operates on 48x27 RGB frames
            small = np.stack([np.array(Image.fromarray(f).resize((48, 27), Image.BILINEAR))
                              for f in frames])
            x = torch.from_numpy(small[None]).to(torch.uint8).to(self.device)
            with torch.no_grad():
                out = self.model(x)[0]
            out = out.cpu().numpy() if torch.is_tensor(out) else np.asarray(out)
            predictions.append(np.squeeze(out).reshape(-1))
        del vr

        predictions = np.concatenate(predictions)
        boundaries = self._peak_boundaries(predictions)
        return self._boundaries_to_events(boundaries, total, fps)

    def _peak_boundaries(self, predictions):
        """Non-maximum suppression over frames whose transition score exceeds the threshold."""
        candidates = np.where(predictions > self.threshold)[0]
        boundaries = []
        i = 0
        while i < len(candidates):
            best = candidates[i]
            j = i
            while j < len(candidates) and candidates[j] - candidates[i] < self.min_distance:
                if predictions[candidates[j]] > predictions[best]:
                    best = candidates[j]
                j += 1
            boundaries.append(int(best))
            i = j
        return boundaries

    @staticmethod
    def _boundaries_to_events(boundaries, total_frames, fps):
        duration = total_frames / fps
        cuts = [0.0] + [b / fps for b in boundaries if 0 < b < total_frames - 1] + [duration]
        return [(cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]


class PySceneDetectSegmenter:
    """Content-based scene cut detection with PySceneDetect."""

    def __init__(self, threshold=27.0):
        self.threshold = threshold

    def __call__(self, video_path):
        from scenedetect import detect, ContentDetector
        scenes = detect(str(video_path), ContentDetector(threshold=self.threshold))
        if not scenes:
            return uniform_windows(video_path, window=1e9)
        return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def build_segmenter(name, device="cuda"):
    if name == "transnetv2":
        return TransNetV2Segmenter(device=device)
    if name == "pyscenedetect":
        return PySceneDetectSegmenter()
    if name == "uniform":
        return uniform_windows
    raise ValueError(f"Unknown segmentation method: {name}")
