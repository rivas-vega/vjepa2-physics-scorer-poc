"""
V-JEPA 2 — Video Preprocessing Utilities
==========================================
Standalone utilities for video decoding, augmentation, and frame sampling.
Can be used independently for local testing before Modal deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal

import numpy as np


# ─── Config Dataclass ────────────────────────────────────────────────────────

@dataclass
class PreprocessConfig:
    """Full configuration for V-JEPA 2 video preprocessing."""

    # Temporal
    target_fps:    int   = 8
    num_frames:    int   = 16
    sampling_mode: Literal["uniform", "random", "strided"] = "uniform"
    stride:        int   = 2                        # used when sampling_mode="strided"
    clip_duration: Optional[float] = None           # seconds; None = full video

    # Spatial
    frame_size:    int   = 224                      # output H == W after square crop
    crop_mode:     Literal["center", "random"] = "center"
    scale_range:   Tuple[float, float] = (0.5, 1.0) # for random resize crop

    # Normalisation
    normalize:        bool  = True
    normalize_mean:   List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std:    List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Augmentation (training mode)
    augment:          bool  = False
    horizontal_flip_p: float = 0.5
    color_jitter:     bool  = False
    color_jitter_strength: float = 0.4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def for_vjepa2_large(cls) -> "PreprocessConfig":
        """Recommended settings for V-JEPA 2 ViT-L."""
        return cls(target_fps=8, num_frames=16, frame_size=224, sampling_mode="uniform")

    @classmethod
    def for_vjepa2_full(cls) -> "PreprocessConfig":
        """Full-resolution settings for V-JEPA 2 (ViT-H / ViT-G)."""
        return cls(target_fps=8, num_frames=64, frame_size=256, sampling_mode="uniform")


# ─── Video I/O ───────────────────────────────────────────────────────────────

def probe_video(path: Union[str, Path]) -> dict:
    """Return metadata about a video file using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps":        fps,
        "frame_count": frame_count,
        "width":      width,
        "height":     height,
        "duration":   round(duration, 3),
        "aspect":     round(width / height, 3) if height else 0,
    }


def decode_frames(
    path: Union[str, Path],
    target_fps: int = 8,
    max_frames: Optional[int] = None,
    start_sec:  float = 0.0,
    end_sec:    Optional[float] = None,
) -> List[np.ndarray]:
    """
    Decode video frames as RGB numpy arrays.

    Args:
        path:       Path to video file.
        target_fps: Desired output frame rate.
        max_frames: Hard cap on decoded frames.
        start_sec:  Start decoding from this second.
        end_sec:    Stop decoding at this second (None = end of file).

    Returns:
        List of RGB frames (H, W, 3) as np.uint8.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step     = max(1, round(orig_fps / target_fps))

    # Seek to start
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

    end_ms = (end_sec * 1000) if end_sec is not None else float("inf")

    frames = []
    global_idx = 0

    while True:
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms > end_ms:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if global_idx % step == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        global_idx += 1

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


# ─── Spatial Transforms ───────────────────────────────────────────────────────

def short_side_scale(frame: np.ndarray, size: int) -> np.ndarray:
    """Scale frame so shorter side == size, preserving aspect ratio."""
    import cv2
    h, w = frame.shape[:2]
    if min(h, w) == size:
        return frame
    scale = size / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to (size, size)."""
    h, w = frame.shape[:2]
    top  = (h - size) // 2
    left = (w - size) // 2
    return frame[top:top + size, left:left + size]


def random_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Randomly crop to (size, size)."""
    h, w = frame.shape[:2]
    top  = np.random.randint(0, max(1, h - size + 1))
    left = np.random.randint(0, max(1, w - size + 1))
    return frame[top:top + size, left:left + size]


def random_resize_crop(
    frame: np.ndarray,
    size: int,
    scale: Tuple[float, float] = (0.5, 1.0),
) -> np.ndarray:
    """Random-resize then crop — standard for self-supervised training."""
    import cv2
    h, w   = frame.shape[:2]
    area   = h * w
    s      = np.random.uniform(scale[0], scale[1])
    crop_a = int(area * s)
    # keep square
    crop_h = crop_w = int(np.sqrt(crop_a))
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    top  = np.random.randint(0, max(1, h - crop_h + 1))
    left = np.random.randint(0, max(1, w - crop_w + 1))
    cropped = frame[top:top + crop_h, left:left + crop_w]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LANCZOS4)


def apply_spatial(
    frames: List[np.ndarray],
    config: PreprocessConfig,
) -> List[np.ndarray]:
    """Apply resize + crop strategy to all frames."""
    out = []
    # For random crops we want the SAME crop per frame (temporal consistency)
    if config.crop_mode == "random":
        dummy = frames[0]
        dummy_scaled = short_side_scale(dummy, config.frame_size)
        h, w  = dummy_scaled.shape[:2]
        top   = np.random.randint(0, max(1, h - config.frame_size + 1))
        left  = np.random.randint(0, max(1, w - config.frame_size + 1))
        for f in frames:
            scaled = short_side_scale(f, config.frame_size)
            # pad if needed
            sh, sw = scaled.shape[:2]
            if sh < config.frame_size or sw < config.frame_size:
                scaled = np.pad(
                    scaled,
                    ((0, max(0, config.frame_size - sh)),
                     (0, max(0, config.frame_size - sw)),
                     (0, 0)),
                    mode="edge",
                )
            crop = scaled[top:top + config.frame_size, left:left + config.frame_size]
            out.append(crop)
    else:
        for f in frames:
            scaled = short_side_scale(f, config.frame_size)
            crop   = center_crop(scaled, config.frame_size)
            out.append(crop)
    return out


# ─── Temporal Sampling ────────────────────────────────────────────────────────

def sample_frames(
    frames: List[np.ndarray],
    num_frames: int,
    mode: Literal["uniform", "random", "strided"] = "uniform",
    stride: int = 2,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Sample exactly `num_frames` from a list of decoded frames.

    Returns:
        (sampled_frames, original_indices)
    """
    total = len(frames)
    if total == 0:
        raise ValueError("Empty frame list — cannot sample.")

    if mode == "uniform":
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    elif mode == "random":
        if total >= num_frames:
            indices = np.sort(np.random.choice(total, num_frames, replace=False))
        else:
            indices = np.sort(np.random.choice(total, num_frames, replace=True))
    elif mode == "strided":
        raw = np.arange(0, total, stride)[:num_frames]
        if len(raw) < num_frames:
            pad = np.full(num_frames - len(raw), raw[-1] if len(raw) else 0)
            raw = np.concatenate([raw, pad])
        indices = raw
    else:
        raise ValueError(f"Unknown sampling_mode: {mode}")

    sampled = [frames[int(i)] for i in indices]
    return sampled, indices.tolist()


# ─── Augmentation ─────────────────────────────────────────────────────────────

def augment_frames(
    frames: List[np.ndarray],
    config: PreprocessConfig,
) -> List[np.ndarray]:
    """Apply training-time augmentations consistently across all frames."""
    if not config.augment:
        return frames

    out = list(frames)

    # Horizontal flip (consistent across clip)
    if np.random.random() < config.horizontal_flip_p:
        out = [f[:, ::-1].copy() for f in out]

    # Color jitter (consistent per-clip for temporal consistency)
    if config.color_jitter:
        s = config.color_jitter_strength
        # brightness
        b = np.random.uniform(max(0, 1 - s), 1 + s)
        # contrast (simple)
        c = np.random.uniform(max(0, 1 - s), 1 + s)
        # saturation via HSV
        import cv2
        out_aug = []
        for f in out:
            f_float = f.astype(np.float32)
            # brightness + contrast
            f_float = f_float * b
            mean    = f_float.mean()
            f_float = mean + (f_float - mean) * c
            f_float = np.clip(f_float, 0, 255).astype(np.uint8)
            out_aug.append(f_float)
        out = out_aug

    return out


# ─── Normalisation ────────────────────────────────────────────────────────────

def normalize_frames(
    frames: List[np.ndarray],
    mean: List[float] = (0.485, 0.456, 0.406),
    std:  List[float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert list of (H, W, 3) uint8 frames to normalized float32 array.
    Returns shape (T, H, W, 3).
    """
    arr  = np.stack(frames).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr  = np.array(std,  dtype=np.float32)
    return (arr - mean_arr) / std_arr


def to_tensor(normalized: np.ndarray) -> "torch.Tensor":  # type: ignore
    """
    Convert (T, H, W, C) normalized numpy array to (C, T, H, W) PyTorch tensor.
    """
    import torch
    return torch.from_numpy(normalized).permute(3, 0, 1, 2).contiguous()


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def preprocess_video_file(
    path: Union[str, Path],
    config: Optional[PreprocessConfig] = None,
) -> dict:
    """
    Full preprocessing pipeline for a single video file.

    Returns:
        dict with keys:
            tensor     — torch.Tensor (C, T, H, W)
            metadata   — video probe info
            indices    — sampled frame indices
            config     — config used
    """
    if config is None:
        config = PreprocessConfig()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    # 1. Probe
    meta = probe_video(path)

    # 2. Decode
    frames = decode_frames(
        path,
        target_fps=config.target_fps,
        start_sec=0.0,
        end_sec=config.clip_duration,
    )
    if not frames:
        raise ValueError("No frames decoded from video.")

    # 3. Spatial transforms
    frames = apply_spatial(frames, config)

    # 4. Augmentation (training only)
    frames = augment_frames(frames, config)

    # 5. Temporal sampling
    frames, indices = sample_frames(
        frames,
        num_frames=config.num_frames,
        mode=config.sampling_mode,
        stride=config.stride,
    )

    # 6. Normalise
    if config.normalize:
        normalized = normalize_frames(frames, config.normalize_mean, config.normalize_std)
    else:
        normalized = np.stack(frames).astype(np.float32) / 255.0

    # 7. To tensor
    tensor = to_tensor(normalized)

    return {
        "tensor":   tensor,
        "metadata": meta,
        "indices":  indices,
        "config":   config.to_dict(),
        "shape":    list(tensor.shape),
    }


# ─── Checksum Utility ─────────────────────────────────────────────────────────

def video_checksum(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """SHA-256 checksum of a video file (first 64 KB for speed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(chunk_size))
    return h.hexdigest()[:16]


# ─── CLI Quick Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <video_path> [num_frames] [frame_size]")
        sys.exit(1)

    video_path  = sys.argv[1]
    num_frames  = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    frame_size  = int(sys.argv[3]) if len(sys.argv) > 3 else 224

    cfg = PreprocessConfig(num_frames=num_frames, frame_size=frame_size)
    result = preprocess_video_file(video_path, cfg)

    print(json.dumps({
        "shape":    result["shape"],
        "indices":  result["indices"],
        "metadata": result["metadata"],
        "config":   result["config"],
    }, indent=2))
