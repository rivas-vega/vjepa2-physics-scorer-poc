# V-JEPA 2 Physics Consistency Scorer

This repository contains a Proof of Concept (PoC) for deploying Meta's **V-JEPA 2** (Video Joint Embedding Predictive Architecture) as a scalable API on [Modal.com](https://modal.com), along with an adapter for computing a **Physics Consistency Score** (temporal hallucination proxy).

---

## 📁 Project Structure

```text
├── modal_app.py         # Main Modal deployment (GPU inference + FastAPI)
├── surprise_scorer.py   # Physics consistency scoring (Violation of Expectation metric)
├── preprocess.py        # Video preprocessing utilities
├── client.py            # Python client SDK for the API
└── README.md            # This file
```

---

## 🔬 Physics Consistency Scoring (SurPrise)

The `surprise_scorer.py` implements the **SurPrise** metric (Violation of Expectation) to evaluate temporal coherence as a proxy for hallucination detection. 

The core idea is based on the VoE (Violation of Expectation) paradigm:
- **Temporally Coherent Videos** → Model exhibits **LOW** surprise.
- **Incoherent/Hallucinated Videos** → Model exhibits **HIGH** surprise.

It uses a sliding-window approach over the V-JEPA 2 frame embeddings to measure how well the contextual frames can predict the target frames.

---

## ⚡ Quick Start

### 1. Install Modal

```bash
pip install modal
modal setup          # authenticate with your Modal account
```

### 2. Deploy the API

```bash
modal deploy modal_app.py
```

Modal will print your app URL, e.g.:
`https://your-org--vjepa2-inference-api-server.modal.run`

### 3. Download Model Weights

Call the setup endpoint once to pull the V-JEPA 2 weights into the persistent volume:

```bash
curl -X POST https://<your-url>/download-model \
  -H "Content-Type: application/json" \
  -d '{"model_variant": "vjepa2-vitl-fpc64-256"}'
```

### 4. Run Inference & Scoring

You can run the full pipeline (upload → preprocess → encode) directly via the API:

```bash
curl -X POST https://<your-url>/pipeline \
  -F "file=@my_video.mp4"
```

---

## 🐍 Python Client

You can use the provided `client.py` SDK to interact with the deployed API:

```python
from client import VJepa2Client

client = VJepa2Client("https://<your-url>")

# Check health
print(client.health())

# Full pipeline
result = client.pipeline("video.mp4", num_frames=16, frame_size=224)
embeddings = result["encode"]["embeddings"]
print(f"Embedding dim: {len(embeddings[0])}")
```

---

## 📖 References

- [V-JEPA 2 Paper (Meta AI)](https://arxiv.org/abs/2506.09985)
- [V-JEPA 2 GitHub](https://github.com/facebookresearch/vjepa2)
- [SurPrise / VoE Metric](https://arxiv.org/abs/2502.11831)
- [Modal Documentation](https://modal.com/docs)
