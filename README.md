# BASS: Budget-Aware Subgraph Selection for LMM-Based Long-Video Query Processing

This repository contains the official implementation of **BASS**, a training-free data access and query planning framework for answering open-ended queries over long videos with large multimodal models (LMMs) under a strict visual-token budget.

<p align="center">
  <img src="assets/framework.png" alt="Overall framework of BASS" width="100%">
</p>
<p align="center"><em>Overall framework of BASS (Figure 2 of the paper).</em></p>

## Overview

Feeding an entire long video to an LMM can produce hundreds of thousands of visual tokens, while keeping only the frames most similar to the query may discard indirectly relevant evidence needed for multi-hop reasoning. BASS treats a long video as a **queryable event graph** rather than a flat sequence of visual tokens:

1. **Offline semantic indexing.** Each video is partitioned into events and indexed once as a directed weighted graph. Temporal edges capture chronological continuity. Semantic edges capture candidate long-range visual associations, found through vector quantization of patch features, an inverted index, and a sparse similarity join.
2. **Budget-aware query planning.** Given a query and a visual-token budget, BASS maximizes a monotone submodular objective. The objective combines direct query relevance with reachable information gain computed from Personalized PageRank, and a cost-aware lazy greedy (CELF-style) algorithm selects a budget-feasible event set.
3. **Graph-guided execution.** The selected events are grounded into query-conditioned evidence records, candidate semantic relations are verified, and the answer is synthesized by a weighted vote over directed reasoning paths.

The index is query-independent and is reused by all queries on the same video.

## Paper-to-Code Correspondence

| Paper | Description | Code |
| :--- | :--- | :--- |
| Sec. 2.1, Eq. (1) | Budget-constrained event selection, event cost c_i | `BASS.build_index` in [`bass/methods/bass_method.py`](bass/methods/bass_method.py) |
| Sec. 3, Event Nodes | Event segmentation (TransNetV2) | [`bass/methods/segmentation.py`](bass/methods/segmentation.py) |
| Sec. 3, Event Nodes, Eq. (2) | Representative frames, global features x_i^g, local patches X_i^p | `BASS._load_representative_frames`, `BASS._extract_event_features` |
| Sec. 3, Eq. (3) | Dense pairwise patch matching (baseline "PM") | `compute_semantic_edges_pairwise` in [`bass/methods/graph_builder.py`](bass/methods/graph_builder.py) |
| Sec. 3, Eq. (4)-(5) | Vector quantization, inverted index, inverse-event-frequency weights, stop-word cap ρ, sparse similarity join, threshold δ | `build_visual_vocabulary`, `quantize_patches`, `build_inverted_index`, `compute_semantic_edges` |
| Sec. 3, Eq. (6) | Temporal edges | `build_adjacency` |
| Sec. 4.1, Eq. (7) | Reachability matrix Π (Personalized PageRank, power iteration) | `compute_pagerank_matrix` |
| Sec. 4.1, Eq. (8)-(9) | Query relevance, reachable information gain, unified objective | `BASS._query_relevance`, `CELFSelector` in [`bass/methods/celf_solver.py`](bass/methods/celf_solver.py) |
| Sec. 4.1, Algorithm 1 | Cost-aware lazy greedy query planning | `CELFSelector.select` |
| Sec. 4.2, Phase 1 | Event grounding | `GraphGuidedExecutor.run` in [`bass/methods/execution.py`](bass/methods/execution.py) |
| Sec. 4.2, Phase 2 | Relation verification of semantic edges in G[S] | `GraphGuidedExecutor.run` |
| Sec. 4.2, Phase 3, Eq. (10) | Reasoning paths, path score, weighted answer aggregation | `GraphGuidedExecutor._enumerate_paths`, `GraphGuidedExecutor.run` |
| Sec. 5.1 | Datasets, backbones, budget, decoding settings | [`bass/data/`](bass/data), [`bass/models/`](bass/models), [`scripts/run_inference.py`](scripts/run_inference.py) |

## Repository Structure

```text
BASS/
├── assets/
│   └── framework.png
├── bass/
│   ├── data/                 # VideoMME, VRBench, CinePile loaders
│   ├── methods/
│   │   ├── bass_method.py    # End-to-end BASS pipeline (offline index + online execution)
│   │   ├── segmentation.py   # TransNetV2 / PySceneDetect / uniform segmentation
│   │   ├── graph_builder.py  # Visual vocabulary, inverted index, semantic edges, PPR
│   │   ├── celf_solver.py    # Cost-aware lazy greedy planner (Algorithm 1)
│   │   └── execution.py      # Graph-guided execution pipeline
│   ├── models/               # Qwen2.5-VL-7B and Qwen2-VL-72B backbones
│   └── utils.py
├── scripts/
│   ├── run.sh                # Multi-GPU evaluation
│   ├── run_inference.py      # Evaluation entry point
│   └── merge_results.py      # Accuracy, visual tokens, and latency
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/ddengyuhao/BASS.git
cd BASS

conda create -n bass python=3.10 -y
conda activate bass
pip install -r requirements.txt

# Optional: FlashAttention-2 for faster LMM inference
pip install flash-attn --no-build-isolation
```

The experiments in the paper were run on 4 NVIDIA A100 (40GB) GPUs. The 7B backbone runs one data chunk per GPU. The 72B backbone is sharded across all GPUs.

Model weights (`Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2-VL-72B-Instruct`, `openai/clip-vit-large-patch14`) are downloaded from the HuggingFace Hub by default. Local checkpoints can be passed with `--model_path` and `--clip_path`.

## Data Preparation

Organize the datasets under `./dataset` (configurable with `--data_root`):

```text
dataset/
├── VideoMME/
│   └── videos/{videoID}.mp4
├── VRBench/
│   ├── VRBench_eval.jsonl
│   └── videos/{video_id}.mp4
└── CinePile/
    ├── yt_videos/            # created automatically
    └── cookies.txt           # optional, passed to yt-dlp
```

- **Video-MME** ([lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)): 900 videos and 2,700 QA pairs. Annotations are loaded from the HuggingFace Hub (or from `VideoMME/videomme/test-00000-of-00001.parquet` if present). Place the videos under `VideoMME/videos/`, named by `videoID`.
- **VRBench** ([OpenGVLab/VRBench](https://huggingface.co/datasets/OpenGVLab/VRBench)): 1,010 long narrative videos with multi-step reasoning questions. Download `VRBench_eval.jsonl` and extract the video archives into `VRBench/videos/`.
- **CinePile** ([tomg-group-umd/cinepile](https://huggingface.co/datasets/tomg-group-umd/cinepile)): annotations are loaded from the HuggingFace Hub. Video clips are downloaded with `yt-dlp` on first access and cached in `CinePile/yt_videos/`.

## Usage

### Quick Start

Evaluate BASS with Qwen2.5-VL-7B on Video-MME under the default budget of 8,192 visual tokens:

```bash
DATASET=VideoMME BACKBONE=Qwen2.5-VL-7B bash scripts/run.sh
```

`run.sh` splits the queries across the GPUs in `GPU_IDS` (default `0 1 2 3`), writes per-chunk results and logs to `./results/<dataset>_<backbone>_B<budget>/`, and prints the merged accuracy, average visual tokens per query, and latency. Any extra argument is forwarded to `run_inference.py`.

### Single-Process Evaluation

```bash
python scripts/run_inference.py \
    --dataset VRBench \
    --backbone Qwen2.5-VL-7B \
    --token_budget 8192 \
    --delta 0.65 \
    --lambda_param 1.0 \
    --output_dir ./results/VRBench_Qwen2.5-VL-7B

python scripts/merge_results.py ./results/VRBench_Qwen2.5-VL-7B
```

### Hyperparameters

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--token_budget` | 8192 | Visual-token budget B |
| `--delta` | 0.65 | Semantic similarity threshold δ for semantic edges |
| `--lambda_param` | 1.0 | Trade-off λ between F_rel and F_reach |
| `--alpha` | 0.15 | Restart probability α of Personalized PageRank |
| `--vocab_size` | 1024 | Number of visual words for vector quantization |
| `--rho` | 50 | Posting-list cap ρ; more frequent visual words are dropped as stop words |
| `--frame_interval` | 4.0 | One representative frame per `frame_interval` seconds of an event |
| `--max_frames_per_event` | 4 | Maximum number of representative frames per event |
| `--frame_size` | 336 | Resolution of representative frames fed to the LMM |
| `--max_paths` | 8 | Maximum number of reasoning paths used in result synthesis |
| `--max_new_tokens` | 512 | Maximum generation length per LMM call (greedy decoding) |
| `--segmentation` | `transnetv2` | Event segmentation: `transnetv2`, `pyscenedetect`, `uniform` |

The cost c_i of an event is the number of visual tokens of its representative frames under the backbone's preprocessing (144 tokens per 336×336 frame for the Qwen backbones).

## Reproducing the Experiments

All commands below can be combined with `DATASET=...` and `BACKBONE=...`. `{a,b,...}` denotes one run per value. Each setting is written to its own directory under `./results/`, named after the dataset, backbone, budget, and extra arguments.

| Paper | Setting | Command |
| :--- | :--- | :--- |
| Table 1 | Main results (7B and 72B) | `DATASET={VideoMME,VRBench,CinePile} BACKBONE={Qwen2.5-VL-7B,Qwen2-VL-72B} bash scripts/run.sh` |
| Figure 3 | w/o query relevance F_rel | `bash scripts/run.sh --disable_relevance` |
| Figure 4 | w/o reachable information gain F_reach | `bash scripts/run.sh --lambda_param 0` |
| Figure 5 | Cost-feasible random selection | `bash scripts/run.sh --planner random` |
| Figure 6 | Standard Chain-of-Thought instead of graph-guided execution | `bash scripts/run.sh --execution cot` |
| Table 2 | Segmentation strategies | `bash scripts/run.sh --segmentation {uniform,pyscenedetect,transnetv2}` |
| Table 3 | Pairwise matching (PM) | `bash scripts/run.sh --edge_construction pairwise` |
| Table 3 | BASS w/o weighting | `bash scripts/run.sh --uniform_word_weight` |
| Figure 7(a) | Trade-off parameter λ ∈ [0, 2.0] | `bash scripts/run.sh --lambda_param {0,0.5,1.0,1.5,2.0}` |
| Figure 7(b) | Similarity threshold δ ∈ [0.5, 0.8] | `bash scripts/run.sh --delta {0.5,0.6,0.65,0.7,0.8}` |
| Figure 8 | Visual-token budget from 1k to 12k | `TOKEN_BUDGET={1024,2048,4096,8192,12288} bash scripts/run.sh` |
| Table 5 | Offline / online latency | reported by `scripts/merge_results.py` |

Baselines are evaluated with their official implementations under the same visual-token budget (Sec. 5.1).

## Output Format

Each query produces one record:

```json
{
  "id": "...",
  "pred": "B",
  "gt": "B",
  "raw_response": "The answer is B.",
  "trace": {
    "selected": [3, 7, 12],
    "records": {"3": "...", "7": "...", "12": "..."},
    "verified_edges": [[3, 12, "The same painting appears in both events."]],
    "rejected_edges": [],
    "paths": [{"path": [3, 12], "score": 0.71, "answer": "B"}],
    "votes": {"B": 0.71},
    "visual_tokens": 1728.0,
    "num_events": 41,
    "offline_time": 38.2,
    "online_time": 6.1
  }
}
```

`offline_time` is recorded when a video is indexed for the first time and is 0 for later queries that reuse the index.

## Main Results

Accuracy (%) under a budget of 8,192 visual tokens (Table 1 of the paper).

| Method | Type | LMM backbone | VideoMME | VRBench | CinePile | Avg. |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| *Proprietary LMMs (full sequence)* | | | | | | |
| Gemini 1.5 Pro | API | -- | 75.0 | 70.7 | 60.1 | 68.6 |
| GPT-4o | API | -- | 71.9 | 68.7 | 56.1 | 65.6 |
| *Open-source LMMs (full sequence)* | | | | | | |
| InternVL2.5-78B | Dense | -- | 72.1 | 53.5 | 54.6 | 60.1 |
| Qwen2-VL-72B | Dense | -- | 71.2 | 59.1 | 54.2 | 61.5 |
| Qwen2.5-VL-7B | Dense | -- | 65.1 | 56.5 | 52.6 | 58.1 |
| LLaVA-NeXT-34B | Dense | -- | 70.6 | 48.5 | 41.5 | 53.5 |
| *Visual-token-efficient methods (Qwen2.5-VL-7B)* | | | | | | |
| LLaVA-Phi | Architecture | Phi-2 | 34.5 | 30.0 | 27.5 | 30.7 |
| FastV | Visual-token reduction | Qwen2.5-VL-7B | 52.3 | 43.1 | 38.4 | 44.6 |
| DyCoke | Visual-token reduction | Qwen2.5-VL-7B | 51.8 | 47.2 | 37.1 | 45.4 |
| VTR-VLM | Visual-token reduction | Qwen2.5-VL-7B | 54.1 | 53.3 | 44.5 | 50.6 |
| AdaReTaKe | Visual-token reduction | Qwen2.5-VL-7B | 50.8 | 47.1 | 46.4 | 48.1 |
| Q-Frame | Keyframe sampling | Qwen2.5-VL-7B | 53.5 | 48.5 | 40.7 | 47.6 |
| Nar-KFC | Keyframe sampling | Qwen2.5-VL-7B | 56.7 | 53.5 | 45.7 | 52.0 |
| MovieChat | Memory-based | Qwen2.5-VL-7B | 48.5 | 35.0 | 28.0 | 37.2 |
| SGVC | Caption-based | Qwen2.5-VL-7B | 45.0 | 32.0 | 30.2 | 35.7 |
| **BASS** | **Graph-based** | **Qwen2.5-VL-7B** | **61.5** | **54.8** | **48.1** | **54.8** |
| *Visual-token-efficient methods (Qwen2-VL-72B)* | | | | | | |
| FastV | Visual-token reduction | Qwen2-VL-72B | 56.5 | 45.5 | 41.6 | 47.9 |
| DyCoke | Visual-token reduction | Qwen2-VL-72B | 57.9 | 46.8 | 39.8 | 48.2 |
| VTR-VLM | Visual-token reduction | Qwen2-VL-72B | 58.2 | 55.4 | 45.5 | 53.0 |
| AdaReTaKe | Visual-token reduction | Qwen2-VL-72B | 55.5 | 48.2 | 48.0 | 50.6 |
| Q-Frame | Keyframe sampling | Qwen2-VL-72B | 62.0 | 52.1 | 41.9 | 52.0 |
| Nar-KFC | Keyframe sampling | Qwen2-VL-72B | 63.2 | 56.1 | 48.9 | 56.1 |
| MovieChat | Memory-based | Qwen2-VL-72B | 52.1 | 37.5 | 28.5 | 39.4 |
| SGVC | Caption-based | Qwen2-VL-72B | 49.5 | 34.8 | 31.1 | 38.5 |
| **BASS** | **Graph-based** | **Qwen2-VL-72B** | **69.2** | **58.5** | **51.3** | **59.7** |

## Citation

```bibtex
@misc{deng2026bass,
  title  = {BASS: Budget-Aware Subgraph Selection for LMM-Based Long-Video Query Processing},
  author = {Deng, Yuhao and Hu, Hanqing and He, Kun and Li, Feicheng and Deng, Qiyan and Qiao, Lianpeng and Wang, Yuping and Zhang, Aoqian and Yuan, Ye and Chai, Chengliang},
  year   = {2026},
}
```
