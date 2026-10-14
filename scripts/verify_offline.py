#!/usr/bin/env python3
"""
Offline Physical Violation-of-Expectation (VoE) Verification Script.

Demonstrates latent space temporal surprise scoring and VoE pairwise accuracy
without requiring GPU access, external download of V-JEPA 2 weights, or Modal cloud credentials.

Usage:
    python3 scripts/verify_offline.py
"""

import sys
import os
import numpy as np

# Ensure parent directory is in python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surprise_scorer import (
    score_frame_sequence,
    score_predictor_windows,
    evaluate_voe_dataset,
)


def generate_synthetic_physical_pairs(n_pairs=20, n_frames=32, dim=64, seed=42):
    """
    Generate synthetic plausible vs. implausible physical trajectories in latent space.

    - Plausible: Smooth harmonic / continuous motion in latent space with small Gaussian noise.
    - Implausible: Identical trajectory up to a critical frame, followed by a non-physical
                   discontinuity / teleportation (violating object permanence / inertia).
    """
    rng = np.random.RandomState(seed)
    time = np.linspace(0, 2 * np.pi, n_frames)

    plausible_videos = []
    implausible_videos = []

    for i in range(n_pairs):
        # Base physical signal: smooth latent trajectory
        freq = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0, np.pi)
        direction = rng.randn(dim)
        direction /= np.linalg.norm(direction)

        # Smooth latent trajectory
        base_signal = np.outer(np.sin(freq * time + phase), direction)
        noise = rng.randn(n_frames, dim) * 0.05
        plausible = base_signal + noise

        # Implausible: sudden teleportation / jump at mid-sequence
        implausible = plausible.copy()
        jump_idx = n_frames // 2
        jump_direction = rng.randn(dim)
        jump_direction /= np.linalg.norm(jump_direction)
        implausible[jump_idx:] += jump_direction * 2.5  # Large discontinuous displacement

        plausible_videos.append(plausible.astype(np.float32))
        implausible_videos.append(implausible.astype(np.float32))

    return plausible_videos, implausible_videos


def main():
    print("=" * 70)
    print("V-JEPA 2 Physics Scorer: Offline VoE Verification")
    print("=" * 70)

    n_pairs = 25
    n_frames = 32
    embedding_dim = 128
    print(f"Generating {n_pairs} synthetic physical trajectory pairs (T={n_frames}, D={embedding_dim})...")
    plausible_videos, implausible_videos = generate_synthetic_physical_pairs(
        n_pairs=n_pairs, n_frames=n_frames, dim=embedding_dim, seed=1337
    )

    print("\n[1] Evaluating Sliding-Window SurPrise (Encoder Temporal Coherence)...")
    poss_scores = []
    imp_scores = []

    for i, (poss, imp) in enumerate(zip(plausible_videos, implausible_videos)):
        r_poss = score_frame_sequence(poss, distance_metric="l1", window_size=8, context_length=4, stride=2)
        r_imp = score_frame_sequence(imp, distance_metric="l1", window_size=8, context_length=4, stride=2)
        poss_scores.append(r_poss.surprise_score)
        imp_scores.append(r_imp.surprise_score)

    voe_result = evaluate_voe_dataset(poss_scores, imp_scores)

    print(f"  • Plausible mean surprise (L1):   {np.mean(poss_scores):.4f}")
    print(f"  • Implausible mean surprise (L1): {np.mean(imp_scores):.4f}")
    print(f"  • Pairwise VoE Accuracy:          {voe_result.accuracy * 100:.1f}% ({voe_result.n_correct}/{voe_result.n_pairs} pairs)")

    assert voe_result.accuracy == 1.0, f"Expected 100% VoE accuracy on synthetic discontinuities, got {voe_result.accuracy}"
    print("  ✓ Encoder-level VoE discrimination verified.")

    print("\n[2] Evaluating Predictor-Conditioned Latent Error...")
    # Simulate predictor that predicts smooth continuation:
    # Target windows observed vs predicted
    n_windows = 10
    pred_windows = np.zeros((n_windows, embedding_dim), dtype=np.float32)
    obs_plausible = pred_windows + np.random.randn(n_windows, embedding_dim).astype(np.float32) * 0.05
    obs_implausible = pred_windows + np.random.randn(n_windows, embedding_dim).astype(np.float32) * 1.50

    pred_res_plausible = score_predictor_windows(pred_windows, obs_plausible)
    pred_res_implausible = score_predictor_windows(pred_windows, obs_implausible)

    print(f"  • Predictor L1 error (plausible):   {pred_res_plausible.surprise_score:.4f}")
    print(f"  • Predictor L1 error (implausible): {pred_res_implausible.surprise_score:.4f}")
    assert pred_res_implausible.surprise_score > pred_res_plausible.surprise_score * 5.0
    print("  ✓ Predictor-level VoE error discrimination verified.")

    print("\n" + "=" * 70)
    print("RESULT: ALL OFFLINE VERIFICATIONS PASSED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
