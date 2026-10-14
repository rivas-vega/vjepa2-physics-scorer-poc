"""
Unit tests for SurPrise temporal coherence and VoE physics consistency scoring.

These tests run without external GPU, Modal, or private packages.
Compatible with pytest and standard unittest runner:
    python3 -m unittest discover -s tests
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from surprise_scorer import (
    score_frame_sequence,
    score_predictor_windows,
    evaluate_voe_dataset,
    _classify,
    _DEFAULT_THRESHOLDS,
    SurpriseResult,
    VoEDatasetResult,
)


class TestSurpriseScorer(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.embedding_dim = 64
        self.num_frames = 16

    def test_constant_sequence_zero_surprise(self):
        """A video where representations never change must exhibit 0 surprise."""
        constant_emb = np.ones((self.num_frames, self.embedding_dim), dtype=np.float32)
        res = score_frame_sequence(constant_emb, window_size=8, context_length=4, stride=2)
        
        self.assertIsInstance(res, SurpriseResult)
        self.assertAlmostEqual(res.surprise_score, 0.0, places=6)
        self.assertAlmostEqual(res.max_surprise, 0.0, places=6)
        self.assertEqual(res.hallucination_level, "none")
        self.assertAlmostEqual(res.hallucination_score, 0.0, places=1)
        self.assertGreater(res.n_windows, 0)

    def test_temporal_discontinuity_increases_surprise(self):
        """A discontinuous temporal jump should produce higher surprise than a smooth sequence."""
        # Smooth sequence (slow linear drift)
        t = np.linspace(0, 0.1, self.num_frames)[:, np.newaxis]
        smooth_emb = np.repeat(t, self.embedding_dim, axis=1).astype(np.float32)
        res_smooth = score_frame_sequence(smooth_emb, window_size=8, context_length=4, stride=2)

        # Discontinuous sequence (sharp jump in middle)
        jump_emb = smooth_emb.copy()
        jump_emb[self.num_frames // 2 :] += 10.0
        res_jump = score_frame_sequence(jump_emb, window_size=8, context_length=4, stride=2)

        self.assertGreater(res_jump.surprise_score, res_smooth.surprise_score)
        self.assertGreater(res_jump.max_surprise, res_smooth.max_surprise)

    def test_short_sequence_fallback(self):
        """Sequences shorter than window_size should trigger frame-delta fallback without crashing."""
        short_emb = np.random.randn(4, self.embedding_dim).astype(np.float32)
        res = score_frame_sequence(short_emb, window_size=8)
        
        self.assertEqual(res.n_windows, 0)
        self.assertEqual(res.n_frames, 4)
        self.assertGreaterEqual(res.surprise_score, 0.0)

    def test_single_frame_sequence(self):
        """A 1-frame video should return 0 surprise without error."""
        single_frame = np.random.randn(1, self.embedding_dim).astype(np.float32)
        res = score_frame_sequence(single_frame, window_size=8)
        
        self.assertEqual(res.surprise_score, 0.0)
        self.assertEqual(res.n_windows, 0)
        self.assertEqual(res.hallucination_level, "none")

    def test_distance_metrics(self):
        """Compare L1, L2, and Cosine distance evaluations."""
        emb = np.random.randn(self.num_frames, self.embedding_dim).astype(np.float32)
        for metric in ["l1", "l2", "cosine"]:
            res = score_frame_sequence(emb, distance_metric=metric, window_size=8, context_length=4, stride=2)
            self.assertEqual(res.distance_metric, metric)
            self.assertGreaterEqual(res.surprise_score, 0.0)

    def test_score_predictor_windows_exact_math(self):
        """Test predictor-based window scoring with synthetic expectations and observations."""
        n_win = 3
        # Pred and Obs differ by exactly 2.0 along all dimensions
        pred = np.zeros((n_win, self.embedding_dim), dtype=np.float32)
        obs = np.full((n_win, self.embedding_dim), 2.0, dtype=np.float32)

        res_l1 = score_predictor_windows(pred, obs, n_frames=16, distance_metric="l1")
        self.assertAlmostEqual(res_l1.surprise_score, 2.0, places=5)

        res_l2 = score_predictor_windows(pred, obs, n_frames=16, distance_metric="l2")
        expected_l2 = float(np.sqrt(self.embedding_dim * (2.0 ** 2)))
        self.assertAlmostEqual(res_l2.surprise_score, expected_l2, places=4)

    def test_score_predictor_shape_mismatch(self):
        """Mismatched predicted and observed shapes must raise ValueError."""
        pred = np.zeros((3, self.embedding_dim), dtype=np.float32)
        obs = np.zeros((4, self.embedding_dim), dtype=np.float32)
        with self.assertRaises(ValueError):
            score_predictor_windows(pred, obs, n_frames=16)

    def test_evaluate_voe_dataset_paired(self):
        """Pairwise VoE accuracy: impossible surprise > possible surprise => 100% correct."""
        poss = [0.1, 0.15, 0.08]
        imp = [0.4, 0.50, 0.35]
        res = evaluate_voe_dataset(poss, imp)
        self.assertIsInstance(res, VoEDatasetResult)
        self.assertEqual(res.accuracy, 1.0)
        self.assertEqual(res.n_correct, 3)
        self.assertEqual(res.n_pairs, 3)

    def test_evaluate_voe_dataset_inversion(self):
        """When possible surprise > impossible surprise, VoE accuracy drops to 0."""
        poss = [0.8, 0.9]
        imp = [0.1, 0.2]
        res = evaluate_voe_dataset(poss, imp)
        self.assertEqual(res.accuracy, 0.0)
        self.assertEqual(res.n_correct, 0)

    def test_classification_thresholds(self):
        """Verify score classification across nominal ranges."""
        th = _DEFAULT_THRESHOLDS
        lvl_none, _ = _classify(th["none"] * 0.5, th)
        self.assertEqual(lvl_none, "none")

        lvl_low, _ = _classify((th["none"] + th["low"]) / 2, th)
        self.assertEqual(lvl_low, "low")

        lvl_sev, _ = _classify(th["high"] * 2.0, th)
        self.assertEqual(lvl_sev, "severe")


if __name__ == "__main__":
    unittest.main()
