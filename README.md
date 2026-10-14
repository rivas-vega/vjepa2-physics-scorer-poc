# V-JEPA 2 Physics Scorer: Violation-of-Expectation (VoE) Benchmark Adapter

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-10%20passing-brightgreen.svg)]()

A reproducible evaluation benchmark adapter and serverless inference pipeline for **Video Joint Embedding Predictive Architectures (V-JEPA 2)**, evaluating physical plausibility and temporal consistency via the **Violation-of-Expectation (VoE)** paradigm.

---

## 🔬 Motivation & Conceptual Formulation

In embodied AI and world models, evaluating physical commonsense understanding in video generation and prediction models remains an open challenge. Conventional metrics like **Fréchet Video Distance (FVD)** primarily evaluate texture statistics and distribution coverage rather than physical coherence or object permanence.

Following the **Violation-of-Expectation (VoE)** paradigm from developmental cognitive science and recent self-supervised video representations ([Garrido et al., 2025](https://arxiv.org/abs/2502.11831); [Luo et al., 2024](https://arxiv.org/abs/2410.05203)):
1. Physical systems exhibit continuous, predictable latent trajectories governed by conservation laws.
2. When presented with physical violations (e.g., disappearance, non-inertial teleports, shape mutability), an expressive world model in latent space exhibits an **expectation violation (SurPrise)**.

```
       Context Frames [1 : t]                 Target Frames [t+1 : t+k]
               │                                         │
        ┌──────▼──────┐                           ┌──────▼──────┐
        │   V-JEPA 2  │                           │   V-JEPA 2  │
        │   Encoder   │                           │   Encoder   │
        └──────┬──────┘                           └──────┬──────┘
               │ z_ctx                                   │ z_tgt (observed)
        ┌──────▼──────┐                                  │
        │   JEPA      │                                  │
        │  Predictor  │                                  │
        └──────┬──────┘                                  │
               │ ẑ_tgt (predicted)                       │
               └─────────────────►[ Δ(ẑ, z) ]◄───────────┘
                                        │
                                  SurPrise Score
```

### Mathematical Definitions

For a video sequence $V = (x_1, \dots, x_T)$:

#### 1. Predictor-Conditioned Latent VoE (Faithful Mode)
When the trained V-JEPA 2 predictor network $g_\phi$ is evaluated alongside the spatiotemporal encoder $f_\theta$:
$$\mathcal{S}_{\mathrm{pred}}(W) = \frac{1}{D} \sum_{d=1}^D \left| g_\phi(f_\theta(x_{t:t+C}))_d - f_\theta(x_{t+C:t+M})_d \right|$$
where $C$ is context length, $M$ is total window size, and $D$ is the embedding dimension.

#### 2. Temporal Representation Discontinuity Proxy (Encoder-Only Mode)
When evaluating frozen encoders without predictor checkpoints:
$$\mathcal{S}_{\mathrm{enc}}(W) = \frac{1}{D} \sum_{d=1}^D \left| \bar{z}_{\mathrm{ctx}} - \bar{z}_{\mathrm{tgt}} \right|$$
where $\bar{z}_{\mathrm{ctx}}$ and $\bar{z}_{\mathrm{tgt}}$ are spatially-pooled latent representations of context and target windows.

#### 3. Pairwise VoE Accuracy
Given balanced pairs of physically plausible ($v^+$) and physically impossible ($v^-$) events (e.g., IntPhys, Physion):
$$\text{VoE Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\mathcal{S}(v_i^-) > \mathcal{S}(v_i^+)\right)$$
An ideal intuitive physics world model achieves $\text{VoE Accuracy} \to 1.0$.

---

## 📁 Repository Structure

```text
├── surprise_scorer.py          # Core VoE scoring engine (native NumPy, zero private dependencies)
├── modal_app.py                # Cloud serverless API on Modal (A10G/A100 GPU tiers + FastAPI)
├── preprocess.py               # Video decoding, uniform/strided sampling & normalization
├── client.py                   # Python SDK client for remote API interaction
├── scripts/
│   └── verify_offline.py       # Instant offline verification demo (< 2 seconds)
├── tests/
│   └── test_surprise_scorer.py # Comprehensive unit test suite (L1/L2/Cosine, VoE, edge cases)
├── requirements.txt            # Minimal runtime dependencies
└── pyproject.toml              # Modern Python packaging specification
```

---

## ⚡ Quick Start

### 1. Offline Verification (< 2 seconds, CPU only)

No cloud credentials, GPUs, or external weights required:

```bash
git clone https://github.com/rivas-vega/vjepa2-physics-scorer-poc.git
cd vjepa2-physics-scorer-poc

# Run test suite
python3 -m unittest discover tests

# Run offline synthetic VoE benchmark
python3 scripts/verify_offline.py
```

Expected output:
```text
======================================================================
V-JEPA 2 Physics Scorer: Offline VoE Verification
======================================================================
Generating 25 synthetic physical trajectory pairs (T=32, D=128)...

[1] Evaluating Sliding-Window SurPrise (Encoder Temporal Coherence)...
  • Plausible mean surprise (L1):   0.0456
  • Implausible mean surprise (L1): 0.0645
  • Pairwise VoE Accuracy:          100.0% (25/25 pairs)
  ✓ Encoder-level VoE discrimination verified.

[2] Evaluating Predictor-Conditioned Latent Error...
  • Predictor L1 error (plausible):   0.0402
  • Predictor L1 error (implausible): 1.2215
  ✓ Predictor-level VoE error discrimination verified.
======================================================================
```

---

### 2. Python API Usage

```python
import numpy as np
from surprise_scorer import score_frame_sequence, evaluate_voe_dataset

# 1. Score single video frame representations (T, D)
frame_embeddings = np.random.randn(32, 1024).astype(np.float32)
result = score_frame_sequence(
    frame_embeddings,
    distance_metric="l1",
    window_size=8,
    context_length=4,
    stride=2,
)

print(f"SurPrise score: {result.surprise_score:.4f}")
print(f"Hallucination level: {result.hallucination_level}")
print(f"Windows evaluated: {result.n_windows}")

# 2. Evaluate VoE Dataset Accuracy
possible_scores = [0.04, 0.05, 0.06, 0.03]
impossible_scores = [0.12, 0.15, 0.09, 0.08]

voe_eval = evaluate_voe_dataset(possible_scores, impossible_scores)
print(f"VoE Pairwise Accuracy: {voe_eval.accuracy * 100:.1f}% ({voe_eval.n_correct}/{voe_eval.n_pairs})")
```

---

### 3. Serverless Cloud Deployment (Modal)

To deploy the GPU inference pipeline on [Modal](https://modal.com):

```bash
pip install -r requirements.txt
pip install modal

# Authenticate Modal CLI
modal setup

# Deploy FastAPI server on serverless GPUs
modal deploy modal_app.py
```

#### Endpoints:
- `POST /download-model`: Pulls official Meta V-JEPA 2 ViT-L/H/G weights into persistent storage.
- `POST /upload`: Chunked video ingestion with format validation.
- `POST /preprocess`: Spatial resize ($224 \times 224$), frame normalization, and temporal sampling.
- `POST /encode`: Spatiotemporal embedding extraction via V-JEPA 2.
- `POST /surprise/score`: Computes single-video surprise or pairwise VoE accuracy.
- `POST /surprise/pipeline`: End-to-end single-request pipeline (upload $\to$ preprocess $\to$ encode $\to$ score).

---

## 🔍 Scientific Discussion & Known Limitations

1. **Encoder-only vs. Predictor-conditioned metrics:**
   - In the absence of a trained predictor ($g_\phi$), measuring context-target distance $\|\bar{z}_{\mathrm{ctx}} - \bar{z}_{\mathrm{tgt}}\|_1$ functions as a proxy for velocity and acceleration shifts.
   - For rigorous physics benchmarking on benchmarks like IntPhys or Physion, the full predictor checkpoint must be loaded to measure deviations from predicted counterfactual futures.
2. **Texture and Camera Motion Confounders:**
   - Sharp camera ego-motion induces high feature distance even under rigid Newtonian physics. Spatial patch masking or camera-conditioned predictors are required to decouple camera motion from physical anomalies.
3. **Action Conditioning:**
   - For embodied robotics settings, world models must predict future representations conditioned on robot actions $a_t \in \mathcal{A}$. Extending this scorer to action-conditioned JEPAs (e.g. latent dynamics $z_{t+1} = T(z_t, a_t)$) is an active area of development.

---

## 📚 References

- **V-JEPA 2**: Bardes et al., *"Revisiting Joint Embedding Predictive Architectures for Video"* (Meta AI, 2024 / 2025).
- **SurPrise / Physical VoE**: Garrido et al., *"Intuitive physics understanding emerges from self-supervised pretraining on natural videos"*, arXiv:2502.11831 (2025).
- **JEDi Metric**: Luo et al., *"JEDi: Joint Embedding Distance for Video Evaluation"*, arXiv:2410.05203 (2024).

---

## 👤 Author

**Pedro Rivas Vega**  
*M2 MVA (Mathématiques, Vision, Apprentissage), École Normale Supérieure Paris-Saclay*  
*B.S. in Mathematics, Universidad de Sevilla*  
- GitHub: [@rivas-vega](https://github.com/rivas-vega)
- Contact: [pedrorivasvega@gmail.com](mailto:pedrorivasvega@gmail.com)
