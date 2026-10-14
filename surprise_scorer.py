"""
SurPrise metric adapter for V-JEPA 2 frame-level embeddings.

Wraps hyle-surprice-metric-evaluator (arXiv:2502.11831) to evaluate
temporal coherence as a hallucination proxy using V-JEPA 2 encoder outputs.

The VoE (Violation of Expectation) paradigm:
  • possible videos   → model should have LOW surprise  (coherent temporal flow)
  • impossible videos → model should have HIGH surprise (incoherent temporal flow)

When used in single-video mode (no separate predictor), we measure temporal
coherence: context frames → target frames within each sliding window.

Score semantics:
  • Lower SurPrise → temporally coherent, low hallucination proxy
  • Higher SurPrise → temporally incoherent, high hallucination proxy
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np


# ── Thresholds (calibrated for L1 on V-JEPA 2 normalised embeddings) ─────────

_DEFAULT_THRESHOLDS = {
    "none":     0.05,   # temporally consistent
    "low":      0.15,
    "moderate": 0.35,
    "high":     0.70,
    # anything above → "severe"
}

_INTERPRETATIONS = {
    "none":     "Temporally coherent video. No hallucination proxy signal detected.",
    "low":      "Slightly elevated temporal surprise. Minor artifacts or inconsistencies possible.",
    "moderate": "Noticeable temporal inconsistency. Some hallucinated content or frame drift likely.",
    "high":     "Strong temporal surprise. Significant temporal hallucinations probable.",
    "severe":   "Extreme temporal incoherence. Video is far outside expected temporal patterns.",
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SurpriseResult:
    """Full result of a SurPrise hallucination proxy run."""

    surprise_score:      float        # average per-window surprise (lower = better)
    max_surprise:        float        # maximum per-window surprise
    n_windows:           int          # number of sliding windows evaluated
    n_frames:            int          # temporal frames in the video
    embedding_dim:       int          # feature dimensionality
    distance_metric:     str          # l1 / l2 / cosine
    aggregation_mode:    str          # average / maximum
    window_scores:       List[float]  # per-window surprise values
    hallucination_level: str          # none / low / moderate / high / severe
    hallucination_score: float        # 0–100 normalised (0 = pristine)
    interpretation:      str          # human-readable summary
    thresholds:          dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"SurPrise Score  : {self.surprise_score:.6f}  (max: {self.max_surprise:.6f})\n"
            f"Hallucination   : {self.hallucination_level.upper()} ({self.hallucination_score:.1f}/100)\n"
            f"Interpretation  : {self.interpretation}\n"
            f"Windows / Frames: {self.n_windows} / {self.n_frames}\n"
            f"Embedding dim   : {self.embedding_dim}\n"
            f"Distance metric : {self.distance_metric}"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify(score: float, thresholds: dict) -> tuple[str, float]:
    """Map raw surprise score to (level, 0-100 normalised)."""
    severe_th = thresholds["high"] * 4
    eps       = 1e-9
    log_val   = np.log1p(max(0.0, score) / (thresholds["none"] + eps))
    log_max   = np.log1p(severe_th       / (thresholds["none"] + eps))
    norm      = float(np.clip(100 * log_val / (log_max + eps), 0, 100))

    if   score <= thresholds["none"]:       level = "none"
    elif score <= thresholds["low"]:        level = "low"
    elif score <= thresholds["moderate"]:   level = "moderate"
    elif score <= thresholds["high"]:       level = "high"
    else:                                   level = "severe"

    return level, norm


# ── Main scoring function ─────────────────────────────────────────────────────

def score_frame_sequence(
    frame_embeddings: np.ndarray,
    distance_metric:  str            = "l1",
    aggregation_mode: str            = "average",
    window_size:      int            = 8,
    context_length:   int            = 4,
    stride:           int            = 2,
    thresholds:       Optional[dict] = None,
) -> SurpriseResult:
    """
    Compute SurPrise temporal coherence score from V-JEPA 2 frame embeddings.

    Uses the hyle-surprice-metric-evaluator library to measure how "surprised"
    the model is by consecutive frame representations — a proxy for temporal
    hallucination.

    Args:
        frame_embeddings: (T, D) array of per-temporal-token encoder embeddings.
        distance_metric:  "l1" (paper default), "l2", or "cosine".
        aggregation_mode: "average" (mean surprise) or "maximum" (worst window).
        window_size:      Total frames per sliding window (M = context + target).
        context_length:   Context frames per window; target = window_size - context_length.
        stride:           Step between consecutive windows.
        thresholds:       Override default level thresholds (keys: none/low/moderate/high).

    Returns:
        SurpriseResult with full scoring breakdown.
    """
    # Check if external library is available, otherwise use native NumPy implementation
    try:
        from hyle_surprise_metric import (
            SurpriseEvaluator,
            SlidingWindowExtractor,
            prepare_video_from_sequence,
        )
        use_external = True
    except ImportError:
        use_external = False

    th  = thresholds or _DEFAULT_THRESHOLDS
    arr = np.asarray(frame_embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    n_frames, emb_dim = arr.shape

    # Fallback: too few frames for sliding windows → frame-delta as proxy
    if n_frames < window_size:
        flat_score = (
            float(np.mean(np.abs(np.diff(arr, axis=0))))
            if n_frames > 1 else 0.0
        )
        level, norm = _classify(flat_score, th)
        return SurpriseResult(
            surprise_score=flat_score, max_surprise=flat_score,
            n_windows=0, n_frames=n_frames, embedding_dim=emb_dim,
            distance_metric=distance_metric, aggregation_mode=aggregation_mode,
            window_scores=[flat_score],
            hallucination_level=level, hallucination_score=norm,
            interpretation=_INTERPRETATIONS[level], thresholds=th,
        )

    if use_external:
        evaluator = SurpriseEvaluator(
            distance_metric=distance_metric,
            aggregation_mode=aggregation_mode,
        )
        extractor = SlidingWindowExtractor(
            window_size=window_size,
            context_length=context_length,
            stride=stride,
        )

        # Use the same sequence for predicted and observed.
        video = prepare_video_from_sequence(
            video_id="vjepa2_encode",
            predicted_sequence=arr,
            observed_sequence=arr,
            extractor=extractor,
        )

        avg_score, window_scores = evaluator.compute_video_surprise(
            video.predicted, video.observed
        )
    else:
        # Native NumPy implementation: extract sliding windows (context, target)
        window_scores = []
        t = 0
        while t + window_size <= n_frames:
            ctx = arr[t : t + context_length].mean(axis=0)
            tgt = arr[t + context_length : t + window_size].mean(axis=0)
            if distance_metric == "l2":
                dist = float(np.sqrt(np.sum((ctx - tgt) ** 2)))
            elif distance_metric == "cosine":
                denom = float(np.linalg.norm(ctx) * np.linalg.norm(tgt)) + 1e-9
                dist = float(1.0 - np.dot(ctx, tgt) / denom)
            else:  # l1
                dist = float(np.mean(np.abs(ctx - tgt)))
            window_scores.append(dist)
            t += stride

        avg_score = float(np.mean(window_scores)) if window_scores else 0.0
    max_score   = float(max(window_scores)) if window_scores else float(avg_score)
    main_score  = max_score if aggregation_mode == "maximum" else float(avg_score)
    level, norm = _classify(main_score, th)

    return SurpriseResult(
        surprise_score=float(avg_score),
        max_surprise=max_score,
        n_windows=len(window_scores),
        n_frames=n_frames,
        embedding_dim=emb_dim,
        distance_metric=distance_metric,
        aggregation_mode=aggregation_mode,
        window_scores=[float(s) for s in window_scores],
        hallucination_level=level,
        hallucination_score=norm,
        interpretation=_INTERPRETATIONS[level],
        thresholds=th,
    )


# ── Predictor-based scoring ───────────────────────────────────────────────────

def score_predictor_windows(
    predicted_windows: np.ndarray,
    observed_windows:  np.ndarray,
    n_frames:          int            = 0,
    distance_metric:   str            = "l1",
    aggregation_mode:  str            = "average",
    thresholds:        Optional[dict] = None,
) -> SurpriseResult:
    """
    Score using pre-computed per-window (predicted, observed) embedding pairs.

    Used when V-JEPA 2's predictor is available. For each sliding window:
      - predicted_windows[i]: what the predictor expected the target frames to be
                               (from running the predictor on context frame embeddings)
      - observed_windows[i]:  what the encoder actually saw in the target frames

    This is the faithful VoE implementation from arXiv:2502.11831 — surprise is
    the gap between the predictor's expectation and reality, not just frame-delta.

    Args:
        predicted_windows: (n_windows, D) — mean-pooled predictor output per window.
        observed_windows:  (n_windows, D) — mean-pooled encoder output for target frames.
        n_frames:          total temporal frames in the source video.
        distance_metric:   "l1" (paper default), "l2", or "cosine".
        aggregation_mode:  "average" or "maximum".
        thresholds:        override default level thresholds.

    Returns:
        SurpriseResult identical in structure to score_frame_sequence output.
    """
    th   = thresholds or _DEFAULT_THRESHOLDS
    pred = np.asarray(predicted_windows, dtype=np.float32)
    obs  = np.asarray(observed_windows,  dtype=np.float32)

    if pred.shape != obs.shape:
        raise ValueError(
            f"predicted_windows and observed_windows must have the same shape, "
            f"got {pred.shape} vs {obs.shape}"
        )

    n_windows, emb_dim = pred.shape

    if n_windows == 0:
        return SurpriseResult(
            surprise_score=0.0, max_surprise=0.0,
            n_windows=0, n_frames=n_frames, embedding_dim=emb_dim,
            distance_metric=distance_metric, aggregation_mode=aggregation_mode,
            window_scores=[],
            hallucination_level="none", hallucination_score=0.0,
            interpretation=_INTERPRETATIONS["none"], thresholds=th,
        )

    # Compute per-window distances directly — no re-windowing needed since
    # pred/obs are already at window granularity.
    if distance_metric == "l2":
        window_scores = [
            float(np.sqrt(np.sum((pred[i] - obs[i]) ** 2))) for i in range(n_windows)
        ]
    elif distance_metric == "cosine":
        def _cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            return float(1.0 - np.dot(a, b) / denom)
        window_scores = [_cosine(pred[i], obs[i]) for i in range(n_windows)]
    else:  # l1 default
        window_scores = [
            float(np.mean(np.abs(pred[i] - obs[i]))) for i in range(n_windows)
        ]

    avg_score  = float(np.mean(window_scores))
    max_score  = float(max(window_scores))
    main_score = max_score if aggregation_mode == "maximum" else avg_score
    level, norm = _classify(main_score, th)

    return SurpriseResult(
        surprise_score=avg_score,
        max_surprise=max_score,
        n_windows=n_windows,
        n_frames=n_frames,
        embedding_dim=emb_dim,
        distance_metric=distance_metric,
        aggregation_mode=aggregation_mode,
        window_scores=window_scores,
        hallucination_level=level,
        hallucination_score=norm,
        interpretation=_INTERPRETATIONS[level],
        thresholds=th,
    )


# ── Dataset-level VoE evaluation ──────────────────────────────────────────────

@dataclass
class VoEDatasetResult:
    """Result of Violation-of-Expectation pairwise dataset evaluation."""
    accuracy:        float
    n_correct:       int
    n_pairs:         int
    possible_mean:   float
    impossible_mean: float


def evaluate_voe_dataset(
    possible_scores:   List[float],
    impossible_scores: List[float],
) -> VoEDatasetResult:
    """
    Evaluate Violation-of-Expectation (VoE) pairwise accuracy.

    For paired physical evaluation (e.g. IntPhys / VoE benchmarks):
    An instance is correct if the physically impossible sequence incurs
    strictly higher surprise than its physically possible counterpart.

    Supports both 1:1 paired lists (equal lengths) and all-pairs evaluation.
    """
    poss = [float(s) for s in possible_scores]
    imp  = [float(s) for s in impossible_scores]

    if not poss or not imp:
        return VoEDatasetResult(
            accuracy=0.0, n_correct=0, n_pairs=0,
            possible_mean=float(np.mean(poss)) if poss else 0.0,
            impossible_mean=float(np.mean(imp)) if imp else 0.0,
        )

    # 1:1 paired comparison if equal length, else Cartesian product
    if len(poss) == len(imp):
        n_pairs = len(poss)
        n_correct = sum(1 for p, i in zip(poss, imp) if i > p)
    else:
        n_pairs = len(poss) * len(imp)
        n_correct = sum(1 for p in poss for i in imp if i > p)

    accuracy = float(n_correct / n_pairs) if n_pairs > 0 else 0.0

    return VoEDatasetResult(
        accuracy=accuracy,
        n_correct=n_correct,
        n_pairs=n_pairs,
        possible_mean=float(np.mean(poss)),
        impossible_mean=float(np.mean(imp)),
    )

