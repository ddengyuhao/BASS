# BASS: Budget-Aware Subgraph Selection for LMM-Based Long-Video Query Processing

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-green.svg)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)]()

**BASS** is a framework for budget-aware query processing over long videos. It uses pretrained vision and multimodal models and requires no task-specific training. BASS organizes a video as a reusable event graph and selects a query-specific set of events under a visual-token budget. The current evaluation focuses on multiple-choice long-video question answering.

[Overview](#overview) · [Method](#method) · [Reported Results](#reported-results) ·
[Installation](#installation) · [Data Preparation](#data-preparation) · [Usage](#usage) ·
[Citation](#citation) 

</div>

## Overview

Long videos can require hundreds of thousands of visual tokens when processed in full. Selecting only the frames with the highest direct query relevance can also remove intermediate evidence needed to answer questions involving several events.

BASS treats a long video as a queryable event graph rather than a flat sequence of visual tokens. It first constructs a reusable graph that represents temporal continuity and candidate long-range visual associations. At query time, BASS combines direct query relevance with graph-based reachable information gain in a monotone submodular objective and applies a cost-aware density-greedy heuristic with CELF-style lazy marginal evaluation. The selected events are passed to a graph-guided execution pipeline for candidate-relation filtering and answer synthesis.

In the experiments reported in the accompanying manuscript, BASS achieves the highest average accuracy among the evaluated visual-token-efficient methods under an 8,192-visual-token budget. It exceeds the strongest evaluated baseline by 2.8 and 3.6 percentage points with the 7B and 72B backbones, respectively.

## Method

1. **Offline event-graph construction.** The reported configuration segments videos with TransNetV2, extracts frozen visual features, and creates directed temporal edges and candidate semantic edges.
2. **Budget-aware event selection.** BASS optimizes a monotone submodular utility that combines query relevance and reachable information gain. A CELF-style lazy evaluation accelerates cost-aware density-greedy selection.
3. **Graph-guided execution.** The pipeline grounds selected events, checks candidate semantic relations, and organizes retained evidence for downstream LMM answer synthesis. These steps guide the model; they do not guarantee that every generated statement is correct.

## Reported Results

We evaluate BASS on VideoMME, VRBench, and CinePile. Under a strict budget of 8,192 visual tokens, our method significantly outperforms existing efficient inference baselines.

| Method | Type | LMM backbone | Video-MME | VRBench | CinePile | Avg. |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| *Proprietary LMMs (reference only)* | | | | | | |
| Gemini 1.5 Pro | API | -- | 75.0 | 70.7 | 60.1 | 68.6 |
| GPT-4o | API | -- | 71.9 | 68.7 | 56.1 | 65.6 |
| *Open-source LMMs with full-sequence inference (reference only)* | | | | | | |
| InternVL2.5-78B | Dense | -- | 72.1 | 53.5 | 54.6 | 60.1 |
| Qwen2-VL-72B | Dense | -- | 71.2 | 59.1 | 54.2 | 61.5 |
| Qwen2.5-VL-7B | Dense | -- | 65.1 | 56.5 | 52.6 | 58.1 |
| LLaVA-NeXT-34B | Dense | -- | 70.6 | 48.5 | 41.5 | 53.5 |
| *Visual-token-efficient methods evaluated with Qwen2.5-VL-7B* | | | | | | |
| LLaVA-Phi | Architecture | Phi-2 | 34.5 | 30.0 | 27.5 | 30.7 |
| FastV | Visual-token reduction | Qwen2.5-VL-7B | 52.3 | 43.1 | 38.4 | 44.6 |
| DyCoke | Visual-token reduction | Qwen2.5-VL-7B | 51.8 | 47.2 | 37.1 | 45.4 |
| VTR-VLM | Adaptive visual selection | Qwen2.5-VL-7B | 54.1 | 53.3 | 44.5 | 50.6 |
| AdaReTaKe | Visual-token reduction | Qwen2.5-VL-7B | 50.8 | 47.1 | 46.4 | 48.1 |
| Q-Frame | Keyframe sampling | Qwen2.5-VL-7B | 53.5 | 48.5 | 40.7 | 47.6 |
| Nar-KFC | Keyframe sampling | Qwen2.5-VL-7B | 56.7 | 53.5 | 45.7 | 52.0 |
| MovieChat | Memory-based | Qwen2.5-VL-7B | 48.5 | 35.0 | 28.0 | 37.2 |
| SGVC | Caption-based | Qwen2.5-VL-7B | 45.0 | 32.0 | 30.2 | 35.7 |
| **BASS** | **Graph-based** | **Qwen2.5-VL-7B** | **61.5** | **54.8** | **48.1** | **54.8** |
| *Visual-token-efficient methods evaluated with Qwen2-VL-72B* | | | | | | |
| FastV | Visual-token reduction | Qwen2-VL-72B | 56.5 | 45.5 | 41.6 | 47.9 |
| DyCoke | Visual-token reduction | Qwen2-VL-72B | 57.9 | 46.8 | 39.8 | 48.2 |
| VTR-VLM | Adaptive visual selection | Qwen2-VL-72B | 58.2 | 55.4 | 45.5 | 53.0 |
| AdaReTaKe | Visual-token reduction | Qwen2-VL-72B | 55.5 | 48.2 | 48.0 | 50.6 |
| Q-Frame | Keyframe sampling | Qwen2-VL-72B | 62.0 | 52.1 | 41.9 | 52.0 |
| Nar-KFC | Keyframe sampling | Qwen2-VL-72B | 63.2 | 56.1 | 48.9 | 56.1 |
| MovieChat | Memory-based | Qwen2-VL-72B | 52.1 | 37.5 | 28.5 | 39.4 |
| SGVC | Caption-based | Qwen2-VL-72B | 49.5 | 34.8 | 31.1 | 38.5 |
| **BASS** | **Graph-based** | **Qwen2-VL-72B** | **69.2** | **58.5** | **51.3** | **59.7** |


## 📦 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/ddengyuhao/BASS.git](https://github.com/ddengyuhao/BASS.git)
cd BASS

```

### 2. Environment Setup

We recommend using Conda to manage dependencies.

```bash
conda create -n bass python=3.10 -y
conda activate bass

# Install core dependencies
pip install -r requirements.txt

# Install TransNetV2 (for shot detection) and Decord (for video loading)
pip install transnetv2-pytorch decord

```

---

## 📂 Data Preparation

Please organize your datasets as follows. You can configure the `DATA_ROOT` in `scripts/run.sh`.

```text
dataset/
├── VideoMME/
│   ├── videos/       # Contains .mp4 files
│   └── test.json     # Annotation file (Hardcoded in code)
├── CinePile/
│   ├── yt_videos/    # Downloaded video files
│   └── cookies.txt   # (Optional) YouTube auth for downloading restricted videos
└── VRBench/
    ├── videos/       # Contains video files
    └── VRBench.json  # Annotation file
```

---

## 🏃 Usage

You can run inference using the provided shell script, which supports multi-GPU chunking.

### Quick Start

To evaluate on **VideoMME** using **Qwen2.5-VL-7B**:

```bash
bash scripts/run.sh

```

### Advanced Configuration

You can also run the Python script directly for specific configurations:

```bash
python scripts/run_inference.py \
    --dataset VideoMME \
    --data_root ./dataset \
    --backbone Qwen2.5-VL-7B \
    --method BASS \
    --token_budget 8192 \
    --tau 30.0 \
    --delta 0.65 \
    --output_dir ./results/debug

```

#### Key Arguments:

* `--method`: Selection strategy (Default: `BASS`).
* `--backbone`: Model backbone (e.g., `Qwen2.5-VL-7B`, `Qwen2-VL-72B`).
* `--token_budget`: Maximum number of visual tokens allowed (default: 8192).
* `--tau`: Temporal distance threshold for graph construction (default: 30.0).
* `--delta`: Semantic similarity threshold (default: 0.65).


## Citation

If you find this project useful for your research, please cite our paper:

```bibtex
@misc{deng2026bass,
  title  = {BASS: Budget-Aware Subgraph Selection for LMM-Based Long-Video Query Processing},
  author = {Deng, Yuhao and Hu, Hanqing and He, Kun and Li, Feicheng and Deng, Qiyan and Qiao, Lianpeng and Wang, Yuping and Zhang, Aoqian and Yuan, Ye and Chai, Chengliang},
  year   = {2026},
}
```
