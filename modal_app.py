"""
V-JEPA 2 + SurPrise Deployment on Modal.com
=====================================================
Full API server with video upload, preprocessing, V-JEPA 2 inference,
and the SurPrise hallucination scoring metric.

SurPrise — Violation of Expectation metric (Garrido et al., 2025 — arXiv:2502.11831):
  • Measures temporal coherence via sliding-window V-JEPA 2 prediction error
  • No reference set needed — scores individual videos
  • Two modes: single-video coherence score, or VoE pairwise accuracy
  • Implemented via hyle-surprice-metric-evaluator
"""

import modal
import os
from pathlib import Path

# ─── Modal App Definition ───────────────────────────────────────────────────

app = modal.App("vjepa2-inference")

# Persistent volumes
model_volume  = modal.Volume.from_name("vjepa2-weights",    create_if_missing=True)
video_volume  = modal.Volume.from_name("vjepa2-videos",     create_if_missing=True)

# Docker image
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "ffmpeg", "libsm6", "libxext6",
        "libgl1-mesa-glx", "libglib2.0-0", "wget", "curl",
    )
    .pip_install(
        # Core ML
        "torch>=2.3.1",
        "torchvision>=0.18.1",
        "torchaudio>=2.3.1",
        # V-JEPA 2
        "transformers>=4.40.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "accelerate>=0.29.0",
        # Video processing
        "opencv-python-headless>=4.9.0",
        "decord>=0.6.0",
        "av>=12.0.0",
        "imageio[ffmpeg]>=2.34.0",
        # API
        "fastapi>=0.111.0",
        "python-multipart>=0.0.9",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "aiofiles>=23.2.1",
        # Utilities
        "numpy>=1.26.0",
        "Pillow>=10.3.0",
        "scipy>=1.13.0",
        "tqdm>=4.66.0",
        "huggingface_hub>=0.23.0",
        # Admin / monetisation
        "sqlalchemy>=2.0.0",
        "pycryptodome>=3.20.0",
        # Hallucination scoring
        "scikit-learn>=1.4.0",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/vjepa2.git /opt/vjepa2 || true",
        "cd /opt/vjepa2 && pip install -e . --no-deps || true",
        "mkdir -p /opt/vjepa2_api",
    )
    .run_commands(
        # Install private hyle-surprice-metric-evaluator from GitHub.
        # Requires GITHUB_TOKEN secret (modal secret create hyle-github GITHUB_TOKEN=<pat>).
        # The token needs read access to EroPerez/hyle_surprise_metric_evaluator.
        'pip install "hyle-surprice-metric-evaluator[all] @ git+https://${GITHUB_TOKEN}@github.com/EroPerez/hyle_surprise_metric_evaluator.git"',
        secrets=[modal.Secret.from_name("hyle-github")],
    )
    .add_local_file("surprise_scorer.py", "/opt/vjepa2_api/surprise_scorer.py", copy=True)
    .env({"PYTHONPATH": "/opt/vjepa2_api"})
)

GPU_CONFIG_PREMIUM  = "A100-40GB"   # Professional / Enterprise
GPU_CONFIG_STANDARD = "A10G"        # Free / Pay-per-Minute / Starter
GPU_CONFIG          = GPU_CONFIG_PREMIUM  # kept for download_model compatibility


# ─── Model Download ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={"/weights": model_volume},
    timeout=3600,
    gpu=GPU_CONFIG,
)
def download_model(model_variant: str = "vjepa2-vitl-fpc64-256"):
    """Download V-JEPA 2 weights from HuggingFace into persistent Modal Volume."""
    from huggingface_hub import snapshot_download

    weights_dir = Path(f"/weights/{model_variant}")
    marker      = weights_dir / ".downloaded"

    if marker.exists():
        print(f"[cache hit] {model_variant} already downloaded.")
        return str(weights_dir)

    weights_dir.mkdir(parents=True, exist_ok=True)
    hf_repos = {
        "vjepa2-vitl-fpc64-256": "facebook/vjepa2-vitl-fpc64-256",
        "vjepa2-vith-fpc64-256": "facebook/vjepa2-vith-fpc64-256",
        "vjepa2-vitg-fpc64-256": "facebook/vjepa2-vitg-fpc64-256",
    }
    repo_id = hf_repos.get(model_variant, f"facebook/{model_variant}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(weights_dir),
        ignore_patterns=["*.msgpack", "flax_model*"],
    )
    marker.touch()
    model_volume.commit()
    return str(weights_dir)


# ─── Video Preprocessing ─────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={"/videos": video_volume},
    timeout=600,
    cpu=4,
    memory=8192,
)
def preprocess_video(video_bytes: bytes, video_id: str, config: dict) -> dict:
    """
    Decode → resize/crop → temporal sample → normalise → save tensor.
    Output tensor format: (C, T, H, W) float32 — ready for V-JEPA 2 and JEDi.
    """
    import cv2
    import numpy as np
    import torch

    target_fps    = config.get("target_fps",    8)
    num_frames    = config.get("num_frames",    16)
    frame_size    = config.get("frame_size",    224)
    sampling_mode = config.get("sampling_mode", "uniform")
    stride        = config.get("stride",        2)
    norm_mean     = config.get("normalize_mean", [0.485, 0.456, 0.406])
    norm_std      = config.get("normalize_std",  [0.229, 0.224, 0.225])

    raw_dir  = Path(f"/videos/{video_id}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "raw.mp4"
    raw_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video file.")

    orig_fps    = cap.get(cv2.CAP_PROP_FPS)
    orig_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = orig_frames / orig_fps if orig_fps > 0 else 0

    metadata = {
        "video_id": video_id,
        "original_fps": orig_fps,
        "original_frames": orig_frames,
        "original_width": orig_w,
        "original_height": orig_h,
        "duration_seconds": round(duration, 3),
    }

    step       = max(1, round(orig_fps / target_fps))
    frames_raw = []
    idx        = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames_raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames_raw:
        raise ValueError("No frames decoded.")

    def resize_crop(f, size):
        h, w  = f.shape[:2]
        scale = size / min(h, w)
        nw    = int(w * scale)
        nh    = int(h * scale)
        r     = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        t     = (nh - size) // 2
        l     = (nw - size) // 2
        return r[t:t + size, l:l + size]

    frames_cropped = [resize_crop(f, frame_size) for f in frames_raw]
    total          = len(frames_cropped)

    if sampling_mode == "uniform":
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    elif sampling_mode == "random":
        replace = total < num_frames
        indices = np.sort(np.random.choice(total, num_frames, replace=replace))
    elif sampling_mode == "strided":
        raw = np.arange(0, total, stride)[:num_frames]
        if len(raw) < num_frames:
            pad = np.full(num_frames - len(raw), raw[-1] if len(raw) else 0)
            raw = np.concatenate([raw, pad])
        indices = raw
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)

    frames_sampled = [frames_cropped[int(i)] for i in indices]

    mean = np.array(norm_mean, dtype=np.float32)
    std  = np.array(norm_std,  dtype=np.float32)
    arr  = np.stack(frames_sampled).astype(np.float32) / 255.0
    arr  = (arr - mean) / std

    tensor    = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    proc_path = raw_dir / "preprocessed.pt"
    torch.save(
        {"tensor": tensor, "metadata": metadata, "config": config, "indices": indices.tolist()},
        str(proc_path),
    )
    # Save lightweight metadata.json so billing/info code can read duration
    # without loading the full tensor.
    import json as _json
    video_duration_minutes = round(duration / 60.0, 6)
    (raw_dir / "metadata.json").write_text(_json.dumps({
        "video_id":              video_id,
        "duration_seconds":      round(duration, 3),
        "video_duration_minutes": video_duration_minutes,
        "original_fps":          orig_fps,
        "original_frames":       orig_frames,
        "original_width":        orig_w,
        "original_height":       orig_h,
    }))
    video_volume.commit()

    return {
        **metadata,
        "video_duration_minutes": video_duration_minutes,
        "preprocessed_frames":   len(frames_sampled),
        "tensor_shape":          list(tensor.shape),
        "sampling_mode":         sampling_mode,
        "frame_size":            frame_size,
        "status":                "ready",
    }


# ─── GPU Model Base (shared logic) ────────────────────────────────────────────

class _VJepa2Base:
    """
    Persistent GPU container.
    Handles V-JEPA 2 encoding AND JEDi hallucination scoring.
    """

    @modal.enter()
    def load_model(self):
        import torch, sys
        sys.path.insert(0, "/opt/vjepa2")
        sys.path.insert(0, "/opt/vjepa2_api")
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model         = None
        self.processor     = None
        self.model_variant = None
        self.predictor     = None
        self._try_load()

    def _try_load(self):
        import torch
        variant     = "vjepa2-vitl-fpc64-256"
        weights_dir = Path(f"/weights/{variant}")
        if not weights_dir.exists():
            print("[V-JEPA 2] Weights missing — call POST /download-model first.")
            return
        try:
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(str(weights_dir))
            self.model     = AutoModel.from_pretrained(
                str(weights_dir), torch_dtype=torch.float16, device_map="auto",
            ).eval()
            self.model_variant = variant
            # Always load predictor from native checkpoint (faithful VoE scoring)
            self._load_native_predictor(weights_dir)
        except Exception as e:
            print(f"[V-JEPA 2] transformers load failed ({e}); trying native...")
            self._native_load(weights_dir, variant)

    def _load_native_predictor(self, weights_dir):
        """Load V-JEPA 2 predictor from native checkpoint (key 'predictor' in .pth)."""
        import torch, sys
        sys.path.insert(0, "/opt/vjepa2")
        orig_dir = weights_dir / "original"
        search_dir = orig_dir if orig_dir.exists() else weights_dir
        ckpts = list(search_dir.glob("*.pth")) + list(search_dir.glob("*.pt"))
        if not ckpts:
            print("[V-JEPA 2] No checkpoint found — predictor not loaded.")
            return
        try:
            ckpt = torch.load(str(ckpts[0]), map_location="cpu")
            from src.models.predictor import vit_predictor  # type: ignore
            predictor = vit_predictor(
                num_frames=8,
                tubelet_size=2,
                img_size=256,
                patch_size=16,
                embed_dim=1024,           # vitl encoder output dim
                predictor_embed_dim=384,
                depth=12,
                num_heads=16,
                num_targets=4,
            )
            pred_state = ckpt.get("predictor")
            if pred_state:
                predictor.load_state_dict(pred_state, strict=False)
                print("[V-JEPA 2] Predictor weights loaded from checkpoint.")
            else:
                print("[V-JEPA 2] No 'predictor' key in checkpoint — using random-init predictor.")
            self.predictor = predictor.to(self.device).half().eval()
        except Exception as ep:
            print(f"[V-JEPA 2] Predictor load skipped: {ep}")

    def _native_load(self, weights_dir, variant):
        import torch, sys
        try:
            sys.path.insert(0, "/opt/vjepa2")
            from src.models.vision_transformer import vit_large  # type: ignore
            self.model = vit_large(img_size=256, patch_size=16, num_frames=64, tubelet_size=2)
            orig_dir   = weights_dir / "original"
            search_dir = orig_dir if orig_dir.exists() else weights_dir
            ckpts = list(search_dir.glob("*.pth")) + list(search_dir.glob("*.pt"))
            if ckpts:
                ckpt  = torch.load(str(ckpts[0]), map_location="cpu")
                state = ckpt.get("encoder", ckpt.get("model", ckpt))
                self.model.load_state_dict(state, strict=False)
            self.model = self.model.to(self.device).half().eval()
            self.model_variant = variant
            self._load_native_predictor(weights_dir)
        except Exception as e2:
            print(f"[V-JEPA 2] Native load also failed: {e2}")

    # ── Encode ────────────────────────────────────────────────────────────────

    @modal.method()
    def encode_video(self, video_id: str, task: str = "encode") -> dict:
        """Extract V-JEPA 2 spatiotemporal embeddings from a preprocessed video."""
        import torch

        if self.model is None:
            return {"error": "Model not loaded. Call POST /download-model first.", "video_id": video_id}

        proc_path = Path(f"/videos/{video_id}/preprocessed.pt")
        if not proc_path.exists():
            return {"error": f"Video {video_id} not preprocessed.", "video_id": video_id}

        data   = torch.load(str(proc_path), map_location="cpu")
        tensor = data["tensor"].unsqueeze(0).half().to(self.device)

        with torch.no_grad():
            if self.processor is not None:
                # HF expects (B, T, C, H, W); stored tensor is (C, T, H, W)
                pv = tensor.permute(0, 2, 1, 3, 4)
                outputs    = self.model(pixel_values_videos=pv)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                embeddings = self.model(tensor)
                if isinstance(embeddings, (list, tuple)):
                    embeddings = embeddings[-1]
                if embeddings.dim() > 2:
                    embeddings = embeddings.mean(dim=1)

        emb_list = embeddings.float().cpu().numpy().tolist()
        emb_dim  = len(emb_list[0]) if isinstance(emb_list[0], list) else len(emb_list)

        return {
            "video_id":      video_id,
            "task":          task,
            "model_variant": self.model_variant,
            "embedding_dim": emb_dim,
            "embeddings":    emb_list,
            "tensor_shape":  list(tensor.shape),
            "status":        "success",
        }



    # ── SurPrise: predictor-based window encoder ─────────────────────────────

    def _encode_frame_sequence_with_predictor(
        self,
        video_id:       str,
        window_size:    int = 8,
        context_length: int = 4,
        stride:         int = 2,
    ):
        """
        True predictor-based SurPrise encoding (arXiv:2502.11831).

        For each sliding window [t, t+window_size):
          1. Encode context frames [t, t+context_length)  → ctx_tokens
          2. Run predictor(ctx_tokens)                    → predicted_vec
          3. Encode target  frames [t+context_length, t+window_size) → observed_vec

        Surprise = distance(predicted_vec, observed_vec).

        Returns (predicted_windows, observed_windows, n_frames) where both arrays
        are (n_windows, D), or an error dict if loading fails.

        Falls back gracefully at each step if the predictor call signature
        doesn't match — using context tokens as the prediction proxy instead.
        """
        import torch, numpy as np

        if self.model is None:
            return {"error": "Model not loaded.", "video_id": video_id}

        proc_path = Path(f"/videos/{video_id}/preprocessed.pt")
        if not proc_path.exists():
            return {"error": f"Video {video_id} not preprocessed.", "video_id": video_id}

        data   = torch.load(str(proc_path), map_location="cpu")
        tensor = data["tensor"]   # (C, T, H, W)
        _, T, H, W = tensor.shape

        def _encode(frames_t):
            """Encode a (C, t, H, W) sub-tensor → (N, D) token matrix."""
            inp = frames_t.unsqueeze(0).half().to(self.device)
            with torch.no_grad():
                if self.processor is not None:
                    # Prefer encoder sub-module to avoid running the decoder head
                    # HF expects (B, T, C, H, W); inp is (B, C, T, H, W)
                    enc = getattr(self.model, 'encoder', self.model)
                    out = enc(pixel_values_videos=inp.permute(0, 2, 1, 3, 4))
                    h   = out.last_hidden_state       # (1, N, D)
                else:
                    h = self.model(inp)
                    if isinstance(h, (list, tuple)):
                        h = h[-1]
                    if h.dim() == 2:
                        h = h.unsqueeze(0)            # (1, N, D)
            return h.squeeze(0).float().cpu()         # (N, D)

        def _predict(ctx_tokens):
            """Run predictor on context tokens → predicted target tokens (N, D)."""
            ctx = ctx_tokens.unsqueeze(0).half().to(self.device)   # (1, N, D)
            with torch.no_grad():
                # ── HF decoder path ───────────────────────────────────────────
                if self.predictor is not None and self.processor is not None:
                    try:
                        out = self.predictor(
                            hidden_states=ctx,
                            encoder_hidden_states=ctx,
                        )
                        h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out
                        return h.squeeze(0).float().cpu()
                    except Exception:
                        pass  # fall through to native path

                # ── Native predictor path ─────────────────────────────────────
                if self.predictor is not None and self.processor is None:
                    try:
                        n         = ctx.shape[1]
                        ctx_masks = [torch.arange(n, device=self.device)]
                        tgt_masks = [torch.arange(n, device=self.device)]
                        out = self.predictor(ctx, ctx_masks, tgt_masks)
                        if isinstance(out, (list, tuple)):
                            out = out[0]
                        return out.squeeze(0).float().cpu()
                    except Exception:
                        pass

            # Fallback: context tokens as proxy (encoder-only, same as current default)
            return ctx.squeeze(0).float().cpu()

        predicted_windows: list = []
        observed_windows:  list = []

        t = 0
        while t + window_size <= T:
            ctx_frames = tensor[:, t : t + context_length, :, :]
            tgt_frames = tensor[:, t + context_length : t + window_size, :, :]

            ctx_tokens  = _encode(ctx_frames)
            pred_tokens = _predict(ctx_tokens)
            obs_tokens  = _encode(tgt_frames)

            predicted_windows.append(pred_tokens.mean(dim=0).numpy())
            observed_windows.append(obs_tokens.mean(dim=0).numpy())
            t += stride

        if not predicted_windows:
            return {"error": "Not enough frames for windowing.", "video_id": video_id}

        return (
            np.array(predicted_windows, dtype=np.float32),   # (n_windows, D)
            np.array(observed_windows,  dtype=np.float32),   # (n_windows, D)
            T,
        )

    # ── SurPrise: per-frame sequence encoder ────────────────────────────────

    def _encode_frame_sequence(self, video_id: str):
        """
        Internal: extract temporally-resolved (T, D) embeddings from V-JEPA 2.

        Unlike encode_video (which mean-pools all tokens into one vector), this
        returns one embedding per temporal token group by spatially mean-pooling
        each group — giving the frame-level sequence SurPrise needs.

        Returns np.ndarray of shape (num_temporal_tokens, dim) or an error dict.
        """
        import torch

        if self.model is None:
            return {"error": "Model not loaded.", "video_id": video_id}

        proc_path = Path(f"/videos/{video_id}/preprocessed.pt")
        if not proc_path.exists():
            return {"error": f"Video {video_id} not preprocessed.", "video_id": video_id}

        data   = torch.load(str(proc_path), map_location="cpu")
        tensor = data["tensor"]            # (C, T, H, W)
        _, T, H, W = tensor.shape
        t_in   = tensor.unsqueeze(0).half().to(self.device)

        with torch.no_grad():
            if self.processor is not None:
                # HF expects (B, T, C, H, W); t_in is (B, C, T, H, W)
                outputs = self.model(pixel_values_videos=t_in.permute(0, 2, 1, 3, 4))
                hidden  = outputs.last_hidden_state   # (1, num_tokens, dim)
            else:
                hidden = self.model(t_in)
                if isinstance(hidden, (list, tuple)):
                    hidden = hidden[-1]
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)      # (1, num_tokens, dim)

        hidden = hidden.squeeze(0).float().cpu()      # (num_tokens, dim)

        # Reshape to (num_temporal, num_spatial, dim) then mean-pool spatially
        tubelet_size = 2
        patch_size   = 16
        num_t        = max(1, T // tubelet_size)
        num_s        = max(1, (H // patch_size) * (W // patch_size))

        if hidden.shape[0] == num_t * num_s:
            hidden = hidden.reshape(num_t, num_s, -1).mean(dim=1)  # (num_t, dim)
        elif hidden.shape[0] == num_t:
            pass  # already temporal
        else:
            # Unknown layout: treat all tokens as a single frame
            hidden = hidden.mean(dim=0, keepdim=True)               # (1, dim)

        return hidden.numpy()

    # ── SurPrise: scoring helpers ─────────────────────────────────────────────

    def _score_one_video(self, vid_id, distance_metric, aggregation_mode, window_size, context_length, stride):
        """Score a single video; returns (SurpriseResult, None) or (None, error_dict)."""
        import sys
        sys.path.insert(0, "/opt/vjepa2_api")
        from surprise_scorer import score_frame_sequence, score_predictor_windows

        if self.predictor is not None:
            out = self._encode_frame_sequence_with_predictor(
                vid_id, window_size=window_size, context_length=context_length, stride=stride,
            )
            if isinstance(out, dict):
                return None, out
            pred_windows, obs_windows, n_frames = out
            return score_predictor_windows(
                pred_windows, obs_windows, n_frames=n_frames,
                distance_metric=distance_metric, aggregation_mode=aggregation_mode,
            ), None

        frames = self._encode_frame_sequence(vid_id)
        if isinstance(frames, dict):
            return None, frames
        return score_frame_sequence(
            frames, distance_metric=distance_metric, aggregation_mode=aggregation_mode,
            window_size=window_size, context_length=context_length, stride=stride,
        ), None

    def _surprise_voe_pair(self, possible_video_ids, impossible_video_ids, distance_metric, aggregation_mode, window_size, context_length, stride):
        from hyle_surprise_metric import SurpriseEvaluator

        def _collect(ids):
            scores, failed = [], []
            for vid_id in ids:
                r, err = self._score_one_video(vid_id, distance_metric, aggregation_mode, window_size, context_length, stride)
                if err:  failed.append(vid_id)
                else:    scores.append(r.surprise_score)
            return scores, failed

        poss_scores, poss_failed = _collect(possible_video_ids)
        imp_scores,  imp_failed  = _collect(impossible_video_ids)

        if not poss_scores or not imp_scores:
            return {
                "error":             "Could not encode videos for pair evaluation.",
                "possible_failed":   poss_failed,
                "impossible_failed": imp_failed,
            }

        dataset_result = SurpriseEvaluator(
            distance_metric=distance_metric, aggregation_mode=aggregation_mode
        ).evaluate_dataset(poss_scores, imp_scores)

        return {
            "mode":                "voe_pair",
            "pairwise_accuracy":   dataset_result.accuracy,
            "n_correct_pairs":     dataset_result.n_correct,
            "n_total_pairs":       dataset_result.n_pairs,
            "possible_scores":     poss_scores,
            "impossible_scores":   imp_scores,
            "possible_failed":     poss_failed,
            "impossible_failed":   imp_failed,
            "distance_metric":     distance_metric,
            "metric":              "SurPrise — Violation of Expectation (VoE) pairwise accuracy",
            "paper":               "https://arxiv.org/abs/2502.11831",
        }

    def _surprise_single_video(self, video_ids, distance_metric, aggregation_mode, window_size, context_length, stride):
        import numpy as np

        per_video, failed = [], []
        for vid_id in (video_ids or []):
            r, err = self._score_one_video(vid_id, distance_metric, aggregation_mode, window_size, context_length, stride)
            if err:
                failed.append({"video_id": vid_id, "error": err.get("error")})
            else:
                d = r.to_dict()
                d["video_id"] = vid_id
                per_video.append(d)

        if not per_video:
            return {"error": "No videos could be scored.", "failed": failed}

        return {
            "mode":                "single_video",
            "n_videos":            len(per_video),
            "mean_surprise_score": float(np.mean([v["surprise_score"] for v in per_video])),
            "videos":              per_video,
            "failed":              failed,
            "distance_metric":     distance_metric,
            "aggregation_mode":    aggregation_mode,
            "metric":              "SurPrise — temporal coherence proxy (V-JEPA 2 encoder)",
            "paper":               "https://arxiv.org/abs/2502.11831",
        }

    # ── SurPrise: public scoring method ──────────────────────────────────────

    @modal.method()
    def surprise_score(
        self,
        video_ids:            list,
        possible_video_ids:   list | None = None,
        impossible_video_ids: list | None = None,
        distance_metric:      str  = "l1",
        aggregation_mode:     str  = "average",
        window_size:          int  = 8,
        context_length:       int  = 4,
        stride:               int  = 2,
    ) -> dict:
        """
        Compute SurPrise hallucination proxy scores.

        Two modes:
        1. Single-video mode  — pass video_ids  → per-video temporal coherence scores.
        2. VoE pair mode      — pass possible_video_ids + impossible_video_ids
                                → pairwise accuracy (was the model more surprised
                                  by the physically impossible videos?).

        When a predictor is loaded, uses true predictor-based scoring
        (predicted vs observed target embeddings). Otherwise falls back to
        encoder-only temporal coherence (context vs target embeddings).
        """
        kwargs = {
            "distance_metric":  distance_metric,
            "aggregation_mode": aggregation_mode,
            "window_size":      window_size,
            "context_length":   context_length,
            "stride":           stride,
        }
        if possible_video_ids is not None and impossible_video_ids is not None:
            return self._surprise_voe_pair(possible_video_ids, impossible_video_ids, **kwargs)
        return self._surprise_single_video(video_ids, **kwargs)

    @modal.method()
    def health(self) -> dict:
        import torch
        return {
            "model_loaded":     self.model is not None,
            "predictor_loaded": self.predictor is not None,
            "model_variant":    self.model_variant,
            "device":           str(self.device),
            "cuda_available":   torch.cuda.is_available(),
            "gpu_name":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_gb":    round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
                              if torch.cuda.is_available() else None,
        }



# ─── GPU Model Classes (Standard = A10G, Premium = A100) ──────────────────────

_CLS_KWARGS = dict(
    image=image,
    volumes={"/weights": model_volume, "/videos": video_volume, "/jedi": jedi_volume},
    timeout=600,
    scaledown_window=120,
)


@app.cls(gpu=GPU_CONFIG_STANDARD, **_CLS_KWARGS)
class VJepa2ModelStandard(_VJepa2Base):
    """Standard GPU tier (A10G) — Free, Pay-per-Minute, Starter plans."""


@app.cls(gpu=GPU_CONFIG_PREMIUM, **_CLS_KWARGS)
class VJepa2ModelPremium(_VJepa2Base):
    """Premium GPU tier (A100-40GB) — Professional, Enterprise plans."""


# ─── FastAPI Server ────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={
        "/weights":   model_volume,
        "/videos":    video_volume,
        "/jedi":      jedi_volume,
        "/admin-db":  admin_volume,
    },
    timeout=900,
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("hyle-admin-secret")],
)
@modal.asgi_app()
def api_server():
    import uuid
    import aiofiles
    from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Query, Body
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    from typing import List, Literal, Optional

    from fastapi.openapi.utils import get_openapi
    from fastapi.security import APIKeyHeader

    api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

    api = FastAPI(
        title="V-JEPA 2 + JEDi + SurPrise API",
        description=(
            "## V-JEPA 2 Inference + JEDi & SurPrise Hallucination Scoring\n\n"
            "Upload videos, extract V-JEPA 2 embeddings, and measure hallucination quality "
            "using two complementary metrics:\n\n"
            "- **JEDi** ([Luo et al., 2024](https://arxiv.org/abs/2410.05203)) — "
            "MMD² between generated and reference video embeddings; 34% more aligned with "
            "human perception than FVD.\n"
            "- **SurPrise** ([Garrido et al., 2025](https://arxiv.org/abs/2502.11831)) — "
            "Violation-of-Expectation metric measuring temporal coherence via sliding-window "
            "V-JEPA 2 prediction error. No reference set needed.\n\n"
            "### Authentication\n"
            "All endpoints (except `/health` and `/docs`) require an API key:\n"
            "```\nX-API-Key: hyl_...\n```\n"
            "Obtain a key from your admin at `/admin/dashboard`.\n\n"
            "### JEDi Score Levels\n"
            "| MMD² | Level | Meaning |\n"
            "|------|-------|---------|\n"
            "| < 0.001 | 🟢 **None** | Indistinguishable from real |\n"
            "| 0.001–0.005 | 🟡 **Low** | Minor temporal/spatial artifacts |\n"
            "| 0.005–0.02 | 🟠 **Moderate** | Noticeable hallucinations |\n"
            "| 0.02–0.08 | 🔴 **High** | Significant hallucinations |\n"
            "| > 0.08 | 🚨 **Severe** | Far outside real distribution |\n\n"
            "### SurPrise Score Levels\n"
            "| surprise_score (L1) | Level | Meaning |\n"
            "|---------------------|-------|---------|\n"
            "| < 0.05 | 🟢 **Coherent** | Highly consistent temporal dynamics |\n"
            "| 0.05–0.15 | 🟡 **Minor** | Small temporal inconsistencies |\n"
            "| 0.15–0.35 | 🟠 **Moderate** | Noticeable prediction errors |\n"
            "| 0.35–0.60 | 🔴 **High** | Significant temporal incoherence |\n"
            "| > 0.60 | 🚨 **Severe** | Far from expected temporal dynamics |\n\n"
            "*VoE pairwise accuracy ≥ 0.5 means the model was more surprised by the "
            "implausible video than the plausible one (higher = better discrimination).*"
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[{"name": "Info"}, {"name": "Setup"}, {"name": "Video"},
                      {"name": "Inference"}, {"name": "JEDi — Hallucination Scoring"},
                      {"name": "SurPrise — Hallucination Scoring"}, {"name": "Admin"}],
    )

    def custom_openapi():
        if api.openapi_schema:
            return api.openapi_schema
        schema = get_openapi(
            title=api.title,
            version=api.version,
            description=api.description,
            routes=api.routes,
            tags=api.openapi_tags,
        )
        # Register both security schemes
        schemes = schema.setdefault("components", {}).setdefault("securitySchemes", {})
        schemes["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for all endpoints. Obtain from `/admin/dashboard`. Format: `hyl_...`",
        }
        schemes["AdminTokenAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-Admin-Token",
            "description": "Admin token for `/admin/*` endpoints. Set via Modal secret `hyle-admin-secret`.",
        }
        # Apply per-path: admin routes get AdminTokenAuth, everything else gets ApiKeyAuth
        _public = {"/", "/health", "/docs", "/redoc", "/openapi.json",
                   "/admin/dashboard", "/admin/"}
        for path, path_item in schema.get("paths", {}).items():
            for operation in path_item.values():
                if not isinstance(operation, dict):
                    continue
                if path in _public:
                    operation["security"] = []          # no auth badge
                elif path.startswith("/admin"):
                    operation["security"] = [{"AdminTokenAuth": []}]
                else:
                    operation.setdefault("security", [{"ApiKeyAuth": []}])
        api.openapi_schema = schema
        return schema

    api.openapi = custom_openapi

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Admin router ──────────────────────────────────────────────────────────
    get_engine()   # initialise DB + run migrations on cold start
    public_admin, protected_admin = build_admin_router(admin_volume)
    api.include_router(public_admin,    prefix="/admin")
    api.include_router(protected_admin, prefix="/admin")

    # ── Payment router ────────────────────────────────────────────────────────
    pub_pay, prot_pay = build_payment_router(admin_volume)
    api.include_router(pub_pay)                  # /payments/*
    api.include_router(prot_pay, prefix="/admin") # /admin/payments/*

    # ── Auth + billing middleware ─────────────────────────────────────────────
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import JSONResponse as StarletteJSONResponse
    from datetime import datetime, timezone as _tz

    # Paths that bypass API-key auth entirely
    _PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}

    # Endpoints that consume video processing time and are billed
    _BILLED_ENDPOINTS = {
        "/preprocess",
        "/encode",
        "/pipeline",
        "/jedi/score",
        "/jedi/corpus",
        "/jedi/score-corpus",
        "/jedi/pipeline",
        "/surprise/score",
        "/surprise/pipeline",
    }

    class AuthAndBillingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: StarletteRequest, call_next):
            path = request.url.path

            # Always allow: public info pages, admin routes (have own auth), CORS preflight
            if (
                request.method == "OPTIONS"
                or path in _PUBLIC_PATHS
                or path.startswith("/admin")
                or path.startswith("/payments")
            ):
                return await call_next(request)

            # ── 1. API key validation ─────────────────────────────────────────
            raw_key = request.headers.get("X-API-Key")
            if not raw_key:
                return StarletteJSONResponse(
                    {"detail": "Missing X-API-Key header."},
                    status_code=401,
                )

            db = get_session()
            try:
                api_key = lookup_api_key(db, raw_key)
                if not api_key or not api_key.is_active:
                    return StarletteJSONResponse(
                        {"detail": "Invalid or revoked API key."},
                        status_code=401,
                    )

                # ── 2. Quota enforcement for hard-cap plans (e.g. free tier) ─
                is_billed = any(path.endswith(ep) for ep in _BILLED_ENDPOINTS)
                if is_billed:
                    user = db.get(_User, api_key.user_id)
                    if user:
                        plan = _PLANS.get(user.plan_id, _PLANS["free"])
                        # Hard cap: plan has a minute limit but zero overage price
                        if (
                            plan.included_minutes is not None
                            and plan.overage_price_per_minute == 0.0
                            and plan.included_minutes > 0
                        ):
                            now         = datetime.now(_tz.utc).replace(tzinfo=None)
                            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                            used = _monthly_minutes(db, api_key.user_id, month_start)
                            if used >= plan.included_minutes:
                                return StarletteJSONResponse(
                                    {
                                        "detail": (
                                            f"Monthly limit reached: {plan.included_minutes} min "
                                            f"included in the {plan.name} plan. "
                                            "Upgrade your plan at /admin/dashboard."
                                        ),
                                        "plan_id":          plan.id,
                                        "included_minutes": plan.included_minutes,
                                        "used_minutes":     round(used, 3),
                                    },
                                    status_code=429,
                                )

                # ── 3. Stamp auth context onto request state ──────────────────
                request.state.user_id    = api_key.user_id
                request.state.api_key_id = api_key.id
                # Resolve GPU tier once per request so endpoints don't hit the DB
                if user is None:
                    user = db.get(_User, api_key.user_id)
                _plan = _PLANS.get(user.plan_id, _PLANS["free"]) if user else _PLANS["free"]
                request.state.priority_gpu = _plan.priority_gpu
                api_key.last_used_at = datetime.now(_tz.utc).replace(tzinfo=None)

                # ── 4. Process the request ────────────────────────────────────
                response = await call_next(request)

                # ── 5. Record usage from billing state set by the endpoint ────
                if is_billed:
                    minutes  = getattr(request.state, "billing_minutes",  0.0)
                    video_id = getattr(request.state, "billing_video_id", None)
                    if minutes and minutes > 0:
                        record_usage(
                            db=db,
                            user_id=api_key.user_id,
                            endpoint=path,
                            video_minutes=float(minutes),
                            video_id=video_id,
                            api_key_id=api_key.id,
                        )
                        admin_volume.commit()
                        return response

                db.commit()
                return response

            except Exception:
                pass  # never break the main API
            finally:
                db.close()

    api.add_middleware(AuthAndBillingMiddleware)

    # ── Pydantic Models ───────────────────────────────────────────────────────

    class PreprocessConfig(BaseModel):
        target_fps:     int   = Field(8,   ge=1,  le=60,  description="Decode the source video at this frame rate before sampling (1–60 fps).")
        num_frames:     int   = Field(16,  ge=1,  le=128, description="Number of frames to keep after sampling (1–128). Used as the temporal depth T of the (C,T,H,W) tensor.")
        frame_size:     int   = Field(224, ge=64, le=512, description="Spatial size in pixels to which each frame is resized (64–512). Both height and width are set to this value.")
        sampling_mode:  Literal["uniform", "random", "strided"] = Field("uniform", description="Frame-selection strategy: 'uniform' spreads frames evenly, 'random' picks randomly, 'strided' skips every `stride` frames.")
        stride:         int   = Field(2,   ge=1,  le=16,  description="Step between consecutive frames when sampling_mode='strided' (1–16). Ignored for other modes.")
        normalize_mean: list  = Field([0.485, 0.456, 0.406], description="Per-channel (R,G,B) mean subtracted during normalization. ImageNet defaults: [0.485, 0.456, 0.406].")
        normalize_std:  list  = Field([0.229, 0.224, 0.225], description="Per-channel (R,G,B) standard deviation used during normalization. ImageNet defaults: [0.229, 0.224, 0.225].")

    class EncodeRequest(BaseModel):
        video_id: str
        task:     str = "encode"

    class DownloadModelRequest(BaseModel):
        model_variant: Literal[
            "vjepa2-vitl-fpc64-256",
            "vjepa2-vith-fpc64-256",
            "vjepa2-vitg-fpc64-256",
        ] = "vjepa2-vitl-fpc64-256"

    class JEDiScoreRequest(BaseModel):
        generated_video_ids:  List[str] = Field(
            ..., description="IDs of generated/evaluated videos (already uploaded & preprocessed)"
        )
        reference_video_ids:  List[str] = Field(
            ..., min_length=2, description="IDs of real reference videos (min 2)"
        )
        kernel: Literal["polynomial", "rbf", "multiscale_rbf"] = Field(
            "polynomial",
            description=(
                "MMD kernel. **polynomial** (default, matches official JEDi paper), "
                "**rbf**, or **multiscale_rbf**."
            ),
        )
        use_official_package: bool = Field(
            True, description="Try videojedi PyPI package for official JEDi score"
        )

    class JEDiCorpusRequest(BaseModel):
        video_ids:   List[str] = Field(..., min_length=2, description="Reference video IDs")
        corpus_name: str       = Field("default")

    class JEDiCorpusScoreRequest(BaseModel):
        generated_video_ids: List[str] = Field(..., description="Generated video IDs")
        corpus_name:         str       = Field("default")
        kernel:              Literal["polynomial", "rbf", "multiscale_rbf"] = "polynomial"

    class SurpriseScoreRequest(BaseModel):
        video_ids: List[str] = Field(
            default=[],
            description="Video IDs to score individually (single-video mode).",
        )
        possible_video_ids: Optional[List[str]] = Field(
            default=None,
            description="Physically possible video IDs (VoE pair mode).",
        )
        impossible_video_ids: Optional[List[str]] = Field(
            default=None,
            description="Physically impossible video IDs (VoE pair mode).",
        )
        distance_metric: Literal["l1", "l2", "cosine"] = Field(
            "l1", description="Distance between context and target embeddings (paper default: l1)."
        )
        aggregation_mode: Literal["average", "maximum"] = Field(
            "average", description="'average' = mean window surprise; 'maximum' = worst-case window."
        )
        window_size:    int = Field(8, ge=2, le=64, description="Frames per sliding window.")
        context_length: int = Field(4, ge=1, le=32, description="Context frames per window (must be < window_size).")
        stride:         int = Field(2, ge=1, le=16, description="Step between consecutive windows.")

    # ── Upload helper ─────────────────────────────────────────────────────────

    async def _save_upload(file: UploadFile, request: Request):
        ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
        suffix  = Path(file.filename or "video.mp4").suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(400, f"Unsupported format: {suffix}")
        content = await file.read()

        # ── Plan-based limits ─────────────────────────────────────────────────
        import json as _json
        user_id = getattr(request.state, "user_id", None)
        plan    = _PLANS["free"]
        if user_id:
            db = get_session()
            try:
                user = db.get(_User, user_id)
                if user:
                    plan = _PLANS.get(user.plan_id, _PLANS["free"])
            finally:
                db.close()

        size_mb = len(content) / 1024 / 1024
        if size_mb > plan.max_video_size_mb:
            raise HTTPException(
                413,
                {
                    "detail":            f"File too large: {size_mb:.1f} MB exceeds the {plan.max_video_size_mb} MB limit for the {plan.name} plan.",
                    "plan_id":           plan.id,
                    "max_video_size_mb": plan.max_video_size_mb,
                    "file_size_mb":      round(size_mb, 2),
                },
            )

        # Count videos already owned by this user
        if user_id:
            videos_root = Path("/videos")
            owned = 0
            if videos_root.exists():
                for d in videos_root.iterdir():
                    owner_file = d / "owner.json"
                    if owner_file.exists():
                        try:
                            if _json.loads(owner_file.read_text()).get("user_id") == user_id:
                                owned += 1
                        except Exception:
                            pass
            if owned >= plan.max_videos_stored:
                raise HTTPException(
                    429,
                    {
                        "detail":             f"Storage limit reached: {plan.max_videos_stored} videos allowed on the {plan.name} plan. Delete unused videos or upgrade.",
                        "plan_id":            plan.id,
                        "max_videos_stored":  plan.max_videos_stored,
                        "current_stored":     owned,
                    },
                )

        video_id = str(uuid.uuid4())
        raw_dir  = Path(f"/videos/{video_id}")
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "owner.json").write_text(_json.dumps({"user_id": user_id}))
        async with aiofiles.open(str(raw_dir / f"raw{suffix}"), "wb") as f:
            await f.write(content)
        video_volume.commit()
        return video_id, content, suffix

    # ── Info ──────────────────────────────────────────────────────────────────

    @api.get("/", tags=["Info"])
    async def root():
        return {
            "service": "V-JEPA 2 + JEDi API",
            "version": "2.0.0",
            "jedi_paper": "https://arxiv.org/abs/2410.05203",
            "endpoints": {
                "Setup":        ["POST /download-model", "GET /health"],
                "Video":        ["POST /upload", "POST /preprocess", "GET /videos", "DELETE /videos/{id}"],
                "Inference":    ["POST /encode", "POST /pipeline"],
                "JEDi":         [
                    "POST /jedi/score          — ad-hoc: score generated vs reference videos",
                    "POST /jedi/corpus         — store reference corpus (one-time)",
                    "GET  /jedi/corpus         — list stored corpora",
                    "POST /jedi/score-corpus   — score against pre-stored corpus (fast)",
                    "POST /jedi/pipeline       — upload → preprocess → encode → JEDi in one call",
                ],
                "SurPrise":     [
                    "POST /surprise/score      — temporal coherence score (single-video or VoE pair mode)",
                    "POST /surprise/pipeline   — upload → preprocess → encode → SurPrise in one call",
                ],
                "Docs":         ["GET /docs", "GET /redoc"],
            },
        }

    @api.get("/health", tags=["Info"])
    async def health():
        model  = VJepa2ModelPremium()
        result = await model.health.remote.aio()
        return {"status": "ok", **result}

    # ── Setup ─────────────────────────────────────────────────────────────────

    @api.post("/download-model", tags=["Setup"])
    async def trigger_download(req: DownloadModelRequest):
        """Download V-JEPA 2 weights from HuggingFace into Modal Volume."""
        result = await download_model.remote.aio(req.model_variant)
        return {"status": "downloaded", "model_variant": req.model_variant, "path": result}

    # ── Video ─────────────────────────────────────────────────────────────────

    @api.post("/upload", tags=["Video"])
    async def upload_video(request: Request, file: UploadFile = File(...)):
        """Upload a video file. Returns video_id for subsequent calls."""
        video_id, content, _ = await _save_upload(file, request)
        return {
            "video_id": video_id,
            "filename": file.filename,
            "size_mb":  round(len(content) / 1024 / 1024, 2),
            "status":   "uploaded",
            "next":     "POST /preprocess or POST /jedi/pipeline",
        }

    @api.post("/preprocess", tags=["Video"])
    async def preprocess(
        request:     Request,
        video_id:    str = Query(..., description="video_id returned by /upload"),
        config_json: str = Body(
            default="{}",
            media_type="application/json",
            description=(
                "Optional JSON object with preprocessing parameters. "
                "All fields have sensible defaults — pass `{}` or omit the body to use them.\n\n"
                "| Field | Type | Default | Description |\n"
                "|-------|------|---------|-------------|\n"
                "| `target_fps` | int (1–60) | `8` | Decode source video at this frame rate before sampling. |\n"
                "| `num_frames` | int (1–128) | `16` | Frames to keep; becomes the temporal depth T of the (C,T,H,W) tensor. |\n"
                "| `frame_size` | int (64–512) | `224` | Resize each frame to this square size (px). |\n"
                "| `sampling_mode` | `uniform`\\|`random`\\|`strided` | `uniform` | Frame-selection strategy. |\n"
                "| `stride` | int (1–16) | `2` | Frame step when `sampling_mode=strided`; ignored otherwise. |\n"
                "| `normalize_mean` | [R,G,B] floats | `[0.485, 0.456, 0.406]` | Per-channel mean for normalization (ImageNet defaults). |\n"
                "| `normalize_std` | [R,G,B] floats | `[0.229, 0.224, 0.225]` | Per-channel std for normalization (ImageNet defaults). |"
            ),
        ),
    ):
        """Preprocess an uploaded video → save (C,T,H,W) tensor to Modal Volume."""
        import json
        try:
            config = PreprocessConfig(**json.loads(config_json))
        except (json.JSONDecodeError, ValueError) as exc:
           config = PreprocessConfig()

        raw_dir   = Path(f"/videos/{video_id}")
        raw_files = list(raw_dir.glob("raw.*")) if raw_dir.exists() else []
        if not raw_files:
            raise HTTPException(404, f"Video {video_id} not found.")
        result = await preprocess_video.remote.aio(
            video_bytes=raw_files[0].read_bytes(),
            video_id=video_id,
            config=config.model_dump(),
        )
        request.state.billing_minutes  = result.get("video_duration_minutes", 0.0)
        request.state.billing_video_id = video_id
        return result

    @api.get("/videos", tags=["Video"])
    async def list_videos():
        """List all uploaded and preprocessed videos."""
        root = Path("/videos")
        if not root.exists():
            return {"videos": [], "count": 0}
        videos = []
        for d in sorted(root.iterdir()):
            if d.is_dir():
                videos.append({
                    "video_id":     d.name,
                    "uploaded":     bool(list(d.glob("raw.*"))),
                    "preprocessed": (d / "preprocessed.pt").exists(),
                    "files":        [f.name for f in d.iterdir()],
                })
        return {"videos": videos, "count": len(videos)}

    @api.delete("/videos/{video_id}", tags=["Video"])
    async def delete_video(video_id: str):
        import shutil
        d = Path(f"/videos/{video_id}")
        if not d.exists():
            raise HTTPException(404, f"Video {video_id} not found.")
        shutil.rmtree(str(d))
        video_volume.commit()
        return {"deleted": video_id, "status": "ok"}

    # ── Inference ─────────────────────────────────────────────────────────────

    @api.post("/encode", tags=["Inference"])
    async def encode(request: Request, req: EncodeRequest):
        """Extract V-JEPA 2 spatiotemporal embeddings from a preprocessed video."""
        if not Path(f"/videos/{req.video_id}/preprocessed.pt").exists():
            raise HTTPException(404, f"Run POST /preprocess for {req.video_id} first.")
        model  = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        result = await model.encode_video.remote.aio(req.video_id, req.task)
        if "error" in result:
            raise HTTPException(503, result["error"])
        import json as _json
        meta_path = Path(f"/videos/{req.video_id}/metadata.json")
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text())
            request.state.billing_minutes  = meta.get("video_duration_minutes", 0.0)
            request.state.billing_video_id = req.video_id
        return result

    @api.post("/pipeline", tags=["Inference"])
    async def full_pipeline(
        request:     Request,
        file:        UploadFile = File(...),
        config_json: str = Body(
            default="{}",
            media_type="application/json",
            description=(
                "Optional JSON object with preprocessing parameters. "
                "All fields have sensible defaults — pass `{}` or omit the body to use them.\n\n"
                "| Field | Type | Default | Description |\n"
                "|-------|------|---------|-------------|\n"
                "| `target_fps` | int (1–60) | `8` | Decode source video at this frame rate before sampling. |\n"
                "| `num_frames` | int (1–128) | `16` | Frames to keep; becomes the temporal depth T of the (C,T,H,W) tensor. |\n"
                "| `frame_size` | int (64–512) | `224` | Resize each frame to this square size (px). |\n"
                "| `sampling_mode` | `uniform`\\|`random`\\|`strided` | `uniform` | Frame-selection strategy. |\n"
                "| `stride` | int (1–16) | `2` | Frame step when `sampling_mode=strided`; ignored otherwise. |\n"
                "| `normalize_mean` | [R,G,B] floats | `[0.485, 0.456, 0.406]` | Per-channel mean for normalization (ImageNet defaults). |\n"
                "| `normalize_std` | [R,G,B] floats | `[0.229, 0.224, 0.225]` | Per-channel std for normalization (ImageNet defaults). |"
            ),
        ),
        task:        str        = Query("encode"),
    ):
        """Full encode pipeline: upload → preprocess → V-JEPA 2 embeddings."""
        import json as _json
        try:
            config = PreprocessConfig(**_json.loads(config_json))
        except Exception:
            config = PreprocessConfig()

        video_id, _, _ = await _save_upload(file, request)
        raw_files      = list(Path(f"/videos/{video_id}").glob("raw.*"))
        preprocess_r   = await preprocess_video.remote.aio(
            video_bytes=raw_files[0].read_bytes(),
            video_id=video_id,
            config=config.model_dump(),
        )
        request.state.billing_minutes  = preprocess_r.get("video_duration_minutes", 0.0)
        request.state.billing_video_id = video_id
        model    = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        encode_r = await model.encode_video.remote.aio(video_id, task)
        if "error" in encode_r:
            raise HTTPException(503, encode_r["error"])
        return {
            "pipeline":   "upload → preprocess → encode",
            "video_id":   video_id,
            "preprocess": preprocess_r,
            "encode":     encode_r,
        }

    # ── JEDi Endpoints ───────────────────────────────────────────────────────

    @api.post(
        "/jedi/score",
        tags=["JEDi — Hallucination Scoring"],
        summary="Score generated videos vs reference videos (ad-hoc)",
        description=(
            "Compute the **JEDi** metric between two sets of already-uploaded & preprocessed videos.\n\n"
            "Internally: encodes all videos with V-JEPA 2 → MMD² in embedding space.\n\n"
            "**Returns:**\n"
            "- `jedi_score` — raw MMD² (lower = less hallucination)\n"
            "- `jedi_sqrt` — √MMD (same units as embedding distance)\n"
            "- `hallucination_level` — none / low / moderate / high / severe\n"
            "- `hallucination_score` — 0–100 normalised (0 = pristine)\n"
            "- `per_video_nn_distances` — cosine distance to nearest reference per video\n"
            "- `official_videojedi_score` — score from videojedi PyPI (if available)\n"
        ),
    )
    async def jedi_score(request: Request, req: JEDiScoreRequest):
        model  = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        result = await model.jedi_score.remote.aio(
            generated_video_ids=req.generated_video_ids,
            reference_video_ids=req.reference_video_ids,
            kernel=req.kernel,
            use_official_package=req.use_official_package,
        )
        if "error" in result:
            raise HTTPException(422, result["error"])
        import json as _json
        all_ids = req.generated_video_ids + req.reference_video_ids
        minutes = 0.0
        for vid in all_ids:
            meta_path = Path(f"/videos/{vid}/metadata.json")
            if meta_path.exists():
                minutes += _json.loads(meta_path.read_text()).get("video_duration_minutes", 0.0)
        request.state.billing_minutes = minutes
        return result

    @api.post(
        "/jedi/corpus",
        tags=["JEDi — Hallucination Scoring"],
        summary="Build & store a reference corpus (one-time setup)",
        description=(
            "Pre-encodes reference videos and persists their V-JEPA 2 embeddings "
            "in a named corpus on the Modal Volume.\n\n"
            "Call once. Then use `POST /jedi/score-corpus` for fast scoring "
            "without re-encoding references every time."
        ),
    )
    async def jedi_build_corpus(request: Request, req: JEDiCorpusRequest):
        model  = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        result = await model.store_reference_embeddings.remote.aio(
            video_ids=req.video_ids,
            corpus_name=req.corpus_name,
        )
        if "error" in result:
            raise HTTPException(422, result["error"])
        import json as _json
        minutes = 0.0
        for vid in req.video_ids:
            meta_path = Path(f"/videos/{vid}/metadata.json")
            if meta_path.exists():
                minutes += _json.loads(meta_path.read_text()).get("video_duration_minutes", 0.0)
        request.state.billing_minutes = minutes
        return result

    @api.get(
        "/jedi/corpus",
        tags=["JEDi — Hallucination Scoring"],
        summary="List stored reference corpora",
    )
    async def list_corpora():
        """List all stored reference corpora and their sizes."""
        import numpy as np
        corpus_dir = Path("/jedi/corpus")
        if not corpus_dir.exists():
            return {"corpora": [], "count": 0}
        corpora = []
        for p in sorted(corpus_dir.glob("*.npy")):
            arr = np.load(str(p))
            corpora.append({
                "corpus_name":   p.stem,
                "n_embeddings":  arr.shape[0],
                "embedding_dim": arr.shape[1],
                "size_mb":       round(p.stat().st_size / 1024 / 1024, 3),
            })
        return {"corpora": corpora, "count": len(corpora)}

    @api.post(
        "/jedi/score-corpus",
        tags=["JEDi — Hallucination Scoring"],
        summary="Score generated videos against a pre-stored corpus (fast path)",
        description=(
            "Like `POST /jedi/score` but uses a pre-built corpus instead of "
            "re-encoding reference videos on every call. Much faster for "
            "repeated evaluations against the same reference set."
        ),
    )
    async def jedi_score_corpus(request: Request, req: JEDiCorpusScoreRequest):
        model  = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        result = await model.score_against_corpus.remote.aio(
            generated_video_ids=req.generated_video_ids,
            corpus_name=req.corpus_name,
            kernel=req.kernel,
        )
        if "error" in result:
            raise HTTPException(422, result["error"])
        import json as _json
        minutes = 0.0
        for vid in req.generated_video_ids:
            meta_path = Path(f"/videos/{vid}/metadata.json")
            if meta_path.exists():
                minutes += _json.loads(meta_path.read_text()).get("video_duration_minutes", 0.0)
        request.state.billing_minutes = minutes
        return result

    @api.post(
        "/jedi/pipeline",
        tags=["JEDi — Hallucination Scoring"],
        summary="Full JEDi pipeline: upload → preprocess → encode → score",
        description=(
            "All-in-one endpoint. Upload a **generated** video and immediately get "
            "its JEDi hallucination score against already-uploaded reference videos.\n\n"
            "Pass `reference_video_ids` as a comma-separated query parameter.\n\n"
            "**Steps internally:**\n"
            "1. Upload & save generated video\n"
            "2. Preprocess (decode frames, crop, sample, normalise)\n"
            "3. Encode with V-JEPA 2\n"
            "4. Compute JEDi (MMD² vs reference embeddings)\n"
            "5. Return hallucination level, score, and full breakdown"
        ),
    )
    async def jedi_full_pipeline(
        request: Request,
        file: UploadFile = File(...),
        reference_video_ids: str = Query(
            ...,
            description="Comma-separated reference video IDs (already uploaded & preprocessed, min 2)",
        ),
        config_json: str = Form(default="{}"),
        kernel: Literal["polynomial", "rbf", "multiscale_rbf"] = Query("polynomial"),
    ):
        import json as _json
        try:
            config = PreprocessConfig(**_json.loads(config_json))
        except Exception:
            config = PreprocessConfig()
        ref_ids = [r.strip() for r in reference_video_ids.split(",") if r.strip()]
        if len(ref_ids) < 2:
            raise HTTPException(400, "Provide at least 2 reference_video_ids.")

        # 1. Upload
        video_id, _, _ = await _save_upload(file, request)
        raw_files      = list(Path(f"/videos/{video_id}").glob("raw.*"))

        # 2. Preprocess
        preprocess_r = await preprocess_video.remote.aio(
            video_bytes=raw_files[0].read_bytes(),
            video_id=video_id,
            config=config.model_dump(),
        )
        request.state.billing_minutes  = preprocess_r.get("video_duration_minutes", 0.0)
        request.state.billing_video_id = video_id

        # 3 & 4. Encode + JEDi
        model       = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        jedi_result = await model.jedi_score.remote.aio(
            generated_video_ids=[video_id],
            reference_video_ids=ref_ids,
            kernel=kernel,
            use_official_package=True,
        )

        if "error" in jedi_result:
            raise HTTPException(503, jedi_result["error"])

        return {
            "pipeline":         "upload → preprocess → encode → JEDi score",
            "generated_video":  {
                "video_id":  video_id,
                "filename":  file.filename,
                "preprocess": preprocess_r,
            },
            "jedi":             jedi_result,
        }

    # ── SurPrise Endpoints ────────────────────────────────────────────────────

    @api.post(
        "/surprise/score",
        tags=["SurPrise — Hallucination Scoring"],
        summary="SurPrise temporal coherence score (single-video or VoE pair mode)",
        description=(
            "Compute the **SurPrise** metric from *'Intuitive physics understanding emerges "
            "from self-supervised pretraining on natural videos'* (arXiv:2502.11831).\n\n"
            "### Single-video mode\n"
            "Pass `video_ids` → each video gets a per-window surprise score measuring "
            "temporal coherence via V-JEPA 2 encoder representations.\n\n"
            "### VoE pair mode\n"
            "Pass `possible_video_ids` + `impossible_video_ids` → computes pairwise "
            "**Violation-of-Expectation accuracy**: was the model more surprised by "
            "physically impossible videos than by possible ones?\n\n"
            "**Returned fields (single-video):**\n"
            "- `surprise_score` — mean L1 window distance (lower = more coherent)\n"
            "- `max_surprise` — worst-case window\n"
            "- `hallucination_level` — none / low / moderate / high / severe\n"
            "- `hallucination_score` — 0–100 normalised proxy\n"
            "- `window_scores` — per-window breakdown\n\n"
            "**Returned fields (VoE pair):**\n"
            "- `pairwise_accuracy` — fraction of (possible, impossible) pairs correctly ordered\n"
            "- `possible_scores` / `impossible_scores` — raw per-video scores\n\n"
            "All videos must be already uploaded and preprocessed."
        ),
    )
    async def surprise_score_endpoint(request: Request, req: SurpriseScoreRequest):
        if req.context_length >= req.window_size:
            raise HTTPException(400, "context_length must be less than window_size.")

        voe_mode = req.possible_video_ids is not None and req.impossible_video_ids is not None
        if not voe_mode and not req.video_ids:
            raise HTTPException(400, "Provide video_ids (single-video mode) or possible_video_ids + impossible_video_ids (VoE mode).")

        model  = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        result = await model.surprise_score.remote.aio(
            video_ids=req.video_ids,
            possible_video_ids=req.possible_video_ids,
            impossible_video_ids=req.impossible_video_ids,
            distance_metric=req.distance_metric,
            aggregation_mode=req.aggregation_mode,
            window_size=req.window_size,
            context_length=req.context_length,
            stride=req.stride,
        )
        if "error" in result:
            raise HTTPException(422, result["error"])

        import json as _json
        minutes = 0.0
        all_ids = list(req.video_ids) + list(req.possible_video_ids or []) + list(req.impossible_video_ids or [])
        for vid in all_ids:
            meta_path = Path(f"/videos/{vid}/metadata.json")
            if meta_path.exists():
                minutes += _json.loads(meta_path.read_text()).get("video_duration_minutes", 0.0)
        request.state.billing_minutes = minutes
        return result

    @api.post(
        "/surprise/pipeline",
        tags=["SurPrise — Hallucination Scoring"],
        summary="Full SurPrise pipeline: upload → preprocess → encode → score",
        description=(
            "All-in-one endpoint. Upload a video and immediately get its "
            "SurPrise temporal coherence score.\n\n"
            "**Steps internally:**\n"
            "1. Upload & save video\n"
            "2. Preprocess (decode frames, crop, sample, normalise)\n"
            "3. Extract frame-level V-JEPA 2 embeddings\n"
            "4. Compute SurPrise via sliding-window surprise\n"
            "5. Return hallucination level, score, and per-window breakdown"
        ),
    )
    async def surprise_full_pipeline(
        request:          Request,
        file:             UploadFile = File(...),
        config_json:      str        = Form(default="{}"),
        distance_metric:  Literal["l1", "l2", "cosine"]   = Query("l1"),
        aggregation_mode: Literal["average", "maximum"]    = Query("average"),
        window_size:      int = Query(8,  ge=2,  le=64),
        context_length:   int = Query(4,  ge=1,  le=32),
        stride:           int = Query(2,  ge=1,  le=16),
    ):
        if context_length >= window_size:
            raise HTTPException(400, "context_length must be less than window_size.")

        import json as _json
        try:
            config = PreprocessConfig(**_json.loads(config_json))
        except Exception:
            config = PreprocessConfig()

        video_id, _, _ = await _save_upload(file, request)
        raw_files      = list(Path(f"/videos/{video_id}").glob("raw.*"))

        preprocess_r = await preprocess_video.remote.aio(
            video_bytes=raw_files[0].read_bytes(),
            video_id=video_id,
            config=config.model_dump(),
        )
        request.state.billing_minutes  = preprocess_r.get("video_duration_minutes", 0.0)
        request.state.billing_video_id = video_id

        model  = (VJepa2ModelPremium if getattr(request.state, "priority_gpu", False) else VJepa2ModelStandard)()
        result = await model.surprise_score.remote.aio(
            video_ids=[video_id],
            distance_metric=distance_metric,
            aggregation_mode=aggregation_mode,
            window_size=window_size,
            context_length=context_length,
            stride=stride,
        )
        if "error" in result:
            raise HTTPException(503, result["error"])

        return {
            "pipeline":   "upload → preprocess → encode → SurPrise score",
            "video_id":   video_id,
            "filename":   file.filename,
            "preprocess": preprocess_r,
            "surprise":   result,
        }

    return api
