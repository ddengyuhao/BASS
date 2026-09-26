"""
Temporal boundary detection used to partition a video into events (Sec. 3).

Supported segmenters (Sec. 5.4, Table 2):
    transnetv2     TransNetV2 shot transition detector (default)
    pyscenedetect  content-based scene cut detection
    uniform        fixed-duration windows
    precomputed    boundaries produced by an external segmenter (e.g., UBoCo),
                   read from <boundary_dir>/<video name>.json as [[start_sec, end_sec], ...]
"""

import os
import json

import numpy as np
import torch


def _video_duration(video_path):
    from decord import VideoReader, cpu
    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps, total = vr.get_avg_fps(), len(vr)
    del vr
    return total / fps if fps > 0 else 0.0


def cuts_to_events(cuts, duration):
    """Turns sorted cut points (seconds) into a partition [0, duration)."""
    points = [0.0] + [float(c) for c in cuts if 0.0 < c < duration] + [float(duration)]
    points = sorted(set(points))
    return [(points[k], points[k + 1]) for k in range(len(points) - 1)]


def uniform_windows(video_path, window=5.0):
    duration = _video_duration(video_path)
    if duration <= 0:
        return [(0.0, 1.0)]
    return cuts_to_events(np.arange(window, duration, window), duration)


class TransNetV2Segmenter:
    """TransNetV2 shot boundary detector (pretrained weights from transnetv2-pytorch)."""

    def __init__(self, device="cuda", threshold=0.5):
        from transnetv2_pytorch import TransNetV2
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.model = TransNetV2(device=self.device).eval()

    def __call__(self, video_path):
        from decord import VideoReader, cpu

        # TransNetV2 operates on 48x27 RGB frames
        vr = VideoReader(str(video_path), ctx=cpu(0), width=48, height=27)
        fps, total = vr.get_avg_fps(), len(vr)
        chunks = [vr.get_batch(list(range(s, min(s + 2000, total)))).asnumpy()
                  for s in range(0, total, 2000)]
        del vr
        frames = torch.from_numpy(np.concatenate(chunks)).to(self.device)

        with torch.no_grad():
            single_frame_pred, _ = self.model.predict_frames(frames, quiet=True)
        scenes = self.model.predictions_to_scenes(
            single_frame_pred.cpu().numpy(), threshold=self.threshold)

        # Each shot starts a new event; transition frames join the preceding event
        cuts = [start / fps for start, _ in scenes[1:]]
        return cuts_to_events(cuts, total / fps)


class PySceneDetectSegmenter:
    """Content-based scene cut detection with PySceneDetect."""

    def __init__(self, threshold=27.0):
        self.threshold = threshold

    def __call__(self, video_path):
        from scenedetect import detect, ContentDetector
        scenes = detect(str(video_path), ContentDetector(threshold=self.threshold))
        duration = _video_duration(video_path)
        return cuts_to_events([s.get_seconds() for s, _ in scenes[1:]], duration)


class PrecomputedSegmenter:
    """Loads event boundaries computed offline by an external segmenter."""

    def __init__(self, boundary_dir):
        if not boundary_dir:
            raise ValueError("--boundary_dir is required for precomputed segmentation")
        self.boundary_dir = boundary_dir

    def __call__(self, video_path):
        name = os.path.splitext(os.path.basename(video_path))[0]
        with open(os.path.join(self.boundary_dir, f"{name}.json")) as f:
            segments = json.load(f)
        duration = _video_duration(video_path)
        return cuts_to_events([s for s, _ in segments[1:]], duration)


def build_segmenter(name, device="cuda", boundary_dir=None):
    if name == "transnetv2":
        return TransNetV2Segmenter(device=device)
    if name == "pyscenedetect":
        return PySceneDetectSegmenter()
    if name == "uniform":
        return uniform_windows
    if name == "precomputed":
        return PrecomputedSegmenter(boundary_dir)
    raise ValueError(f"Unknown segmentation method: {name}")
