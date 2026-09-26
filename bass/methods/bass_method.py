import math
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

from .base_method import BaseMethod
from .graph_builder import (
    build_visual_vocabulary,
    quantize_patches,
    compute_semantic_edges,
    compute_semantic_edges_pairwise,
    build_adjacency,
    compute_pagerank_matrix,
)
from .celf_solver import CELFSelector, random_selection
from .segmentation import build_segmenter
from .execution import GraphGuidedExecutor
from ..utils import format_options

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None


class BASS(BaseMethod):
    """
    Budget-Aware Subgraph Selection.

    Offline:  event segmentation -> multi-granular CLIP features -> event graph
              (temporal + semantic edges) -> reachability matrix Pi.
    Online:   cost-aware lazy greedy event selection -> graph-guided execution.
    """

    def __init__(self, args, model):
        super().__init__(args, model)

        # Offline indexing
        self.delta = getattr(args, 'delta', 0.65)
        self.rho = getattr(args, 'rho', 50)
        self.vocab_size = getattr(args, 'vocab_size', 1024)
        self.frame_interval = getattr(args, 'frame_interval', 4.0)
        self.max_frames_per_event = getattr(args, 'max_frames_per_event', 4)
        self.min_event_duration = getattr(args, 'min_event_duration', 0.5)
        self.max_events = getattr(args, 'max_events', 800)
        self.segmentation = getattr(args, 'segmentation', 'transnetv2')
        self.edge_construction = getattr(args, 'edge_construction', 'index')
        self.uniform_word_weight = getattr(args, 'uniform_word_weight', False)

        # Online planning / execution
        self.alpha = getattr(args, 'alpha', 0.15)
        self.lambda_param = getattr(args, 'lambda_param', 1.0)
        self.token_budget = getattr(args, 'token_budget', 8192)
        self.frame_size = getattr(args, 'frame_size', 336)
        self.planner = getattr(args, 'planner', 'celf')
        self.use_relevance = not getattr(args, 'disable_relevance', False)
        self.execution = getattr(args, 'execution', 'graph')
        self.seed = getattr(args, 'seed', 0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Visual-token cost of one representative frame under the backbone preprocessing
        probe = Image.new("RGB", (self.frame_size, self.frame_size))
        self.tokens_per_frame = self.model.count_image_tokens(probe)
        print(f"[BASS] {self.tokens_per_frame} visual tokens per frame "
              f"({self.frame_size}x{self.frame_size})")

        self._load_clip(args)

        self.segmenter = build_segmenter(self.segmentation, device=str(self.device))

        self.executor = GraphGuidedExecutor(
            model,
            max_new_tokens=getattr(args, 'max_new_tokens', 512),
            max_paths=getattr(args, 'max_paths', 8),
        )
        self.index_cache_size = getattr(args, 'index_cache_size', 2)
        self._index_cache = OrderedDict()
        self.last_trace = None

    def _load_clip(self, args):
        path = getattr(args, 'clip_path', None) or "openai/clip-vit-large-patch14"
        print(f"[BASS] Loading CLIP from: {path}")
        self.clip_processor = CLIPProcessor.from_pretrained(path)
        self.clip_model = CLIPModel.from_pretrained(path).to(self.device).eval()

    # ------------------------------------------------------------------
    # Offline semantic indexing
    # ------------------------------------------------------------------

    def _detect_events(self, video_path):
        """Partitions the video into events with the configured segmenter."""
        return self._normalize_events(self.segmenter(video_path))

    def _normalize_events(self, events):
        """
        Keeps the events a partition of the video: very short shots are merged
        into the preceding event, and if there are too many events, adjacent
        events are merged pairwise.
        """
        events = sorted((float(s), float(e)) for s, e in events if e > s)
        merged = []
        for s, e in events:
            if merged and (e - s) < self.min_event_duration:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        if len(merged) > 1 and (merged[0][1] - merged[0][0]) < self.min_event_duration:
            merged[1] = (merged[0][0], merged[1][1])
            merged = merged[1:]

        while len(merged) > self.max_events:
            merged = [(merged[k][0], merged[min(k + 1, len(merged) - 1)][1])
                      for k in range(0, len(merged), 2)]
        return merged

    def _num_frames(self, start, end):
        """m_i grows with the event duration, capped at max_frames_per_event."""
        m = math.ceil((end - start) / self.frame_interval)
        return int(min(max(m, 1), self.max_frames_per_event))

    def _load_representative_frames(self, video_path, events):
        """Temporally ordered representative frames of each event."""
        if VideoReader is None:
            raise ImportError("decord is required for video reading.")
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        total = len(vr)

        frame_ids, owner = [], []
        for i, (s, e) in enumerate(events):
            m = self._num_frames(s, e)
            for k in range(m):
                t = s + (k + 0.5) * (e - s) / m
                frame_ids.append(min(int(t * fps), total - 1))
                owner.append(i)

        frames = {i: [] for i in range(len(events))}
        batch = 64
        for b in range(0, len(frame_ids), batch):
            arr = vr.get_batch(frame_ids[b:b + batch]).asnumpy()
            for img, i in zip(arr, owner[b:b + batch]):
                frames[i].append(Image.fromarray(img).resize(
                    (self.frame_size, self.frame_size), Image.BICUBIC))
        del vr
        return frames

    def _extract_event_features(self, event_frames):
        """
        Returns:
            global_feats: (N, D) mean-pooled frame-level CLIP embeddings.
            event_patches: list of (L_i, D') patch embeddings flattened over all
                           representative frames of event i (float16, CPU).
        """
        N = len(event_frames)
        flat, owner = [], []
        for i in range(N):
            flat.extend(event_frames[i])
            owner.extend([i] * len(event_frames[i]))

        g_list, p_list = [], []
        with torch.no_grad():
            for b in range(0, len(flat), 64):
                inputs = self.clip_processor(images=flat[b:b + 64], return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                vision_out = self.clip_model.vision_model(pixel_values=pixel_values)
                g = self.clip_model.visual_projection(vision_out.pooler_output)
                g_list.append(F.normalize(g, dim=-1).cpu())
                p_list.append(vision_out.last_hidden_state[:, 1:, :].half().cpu())

        g_frames = torch.cat(g_list)
        p_frames = torch.cat(p_list)
        owner = torch.tensor(owner)

        global_feats = torch.stack([g_frames[owner == i].mean(dim=0) for i in range(N)])
        event_patches = [p_frames[owner == i].reshape(-1, p_frames.shape[-1]) for i in range(N)]
        return global_feats, event_patches

    def build_index(self, video_path):
        """Builds the query-independent event graph of a video."""
        events = self._detect_events(video_path)
        event_frames = self._load_representative_frames(video_path, events)
        global_feats, event_patches = self._extract_event_features(event_frames)

        if self.edge_construction == "pairwise":
            semantic_edges, _ = compute_semantic_edges_pairwise(
                global_feats.to(self.device), event_patches, delta=self.delta)
        else:
            # Vector quantization of local patches into visual words
            all_patches = torch.cat(event_patches)
            codebook = build_visual_vocabulary(
                all_patches, vocab_size=self.vocab_size, device=self.device, seed=self.seed)
            words = quantize_patches(all_patches, codebook)
            event_words, offset = [], 0
            for p in event_patches:
                event_words.append(words[offset:offset + p.shape[0]])
                offset += p.shape[0]

            semantic_edges, _ = compute_semantic_edges(
                global_feats, event_words, delta=self.delta, rho=self.rho,
                uniform_weight=self.uniform_word_weight)
        adj = build_adjacency(len(events), semantic_edges).to(self.device)
        Pi = compute_pagerank_matrix(adj, alpha=self.alpha)

        costs = torch.tensor(
            [len(event_frames[i]) * self.tokens_per_frame for i in range(len(events))],
            dtype=torch.float32)

        return {
            "events": events,
            "frames": event_frames,
            "global_feats": global_feats,
            "semantic_edges": semantic_edges,
            "Pi": Pi.cpu(),
            "costs": costs,
        }

    # ------------------------------------------------------------------
    # Online query execution
    # ------------------------------------------------------------------

    def _query_relevance(self, query_text, global_feats):
        inputs = self.clip_processor(
            text=[query_text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            q = self.clip_model.get_text_features(**inputs)
        q = F.normalize(q, dim=-1).cpu()
        g = F.normalize(global_feats, dim=-1)
        return torch.clamp(torch.mm(g, q.t()).squeeze(1), min=0.0)

    def select_events(self, index, question, options):
        query_text = f"{question}\n{format_options(options)}"
        if self.planner == "random":
            return random_selection(index["costs"], self.token_budget, seed=f"{self.seed}-{query_text}")
        rel = self._query_relevance(query_text, index["global_feats"])
        selector = CELFSelector(index["Pi"], rel, index["costs"],
                                lambda_param=self.lambda_param,
                                use_relevance=self.use_relevance)
        return selector.select(budget=self.token_budget)

    def get_index(self, video_path):
        """Event graphs are built once per video and reused across its queries."""
        if video_path in self._index_cache:
            self._index_cache.move_to_end(video_path)
            return self._index_cache[video_path], 0.0
        t0 = time.time()
        index = self.build_index(video_path)
        offline_time = time.time() - t0
        self._index_cache[video_path] = index
        while len(self._index_cache) > self.index_cache_size:
            self._index_cache.popitem(last=False)
        return index, offline_time

    def process_and_inference(self, video_path, question, options):
        index, offline_time = self.get_index(video_path)

        t0 = time.time()
        selected = self.select_events(index, question, options)
        if not selected:
            # No event has positive utility: fall back to the most relevant affordable event
            affordable = (index["costs"] <= self.token_budget).nonzero().flatten().tolist()
            if affordable:
                rel = self._query_relevance(
                    f"{question}\n{format_options(options)}", index["global_feats"])
                selected = [max(affordable, key=lambda i: float(rel[i]))]
        if not selected:
            self.last_trace = {"selected": [], "offline_time": offline_time}
            return ""

        if self.execution == "cot":
            answer, trace = self.executor.run_cot(
                question, options, selected, index["events"], index["frames"])
        else:
            answer, trace = self.executor.run(
                question, options, selected,
                index["events"], index["frames"], index["semantic_edges"])
        trace["visual_tokens"] = float(index["costs"][selected].sum())
        trace["num_events"] = len(index["events"])
        trace["offline_time"] = offline_time
        trace["online_time"] = time.time() - t0
        self.last_trace = trace
        return f"The answer is {answer}." if answer else ""
