"""
V-JEPA 2 + SurPrise API — Python Client SDK
====================================================
Convenience wrapper around the Modal-deployed REST API.

Quick example:
    from client import VJepa2Client

    client = VJepa2Client(
        "https://your-org--vjepa2-inference-api-server.modal.run",
        api_key="hyl_yourkey",
    )

    # Check GPU health
    print(client.health())

    # SurPrise — temporal coherence score (no reference set needed)
    vid_id = client.upload_and_preprocess("video.mp4")
    result = client.surprise_score([vid_id])
    video  = result["videos"][0]
    print(f"SurPrise      : {video['surprise_score']:.4f} ({video['hallucination_level']})")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional


class VJepa2Client:
    """
    Synchronous HTTP client for the V-JEPA 2 + SurPrise Modal API.
    Handles communication with the FastAPI endpoints deployed on Modal.
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600):
        """
        Args:
            base_url: Modal deployment URL, e.g. https://org--name.modal.run
            api_key:  X-API-Key header value (hyl_...). Required for all
                      non-admin endpoints. Omit only for /health or /docs.
            timeout:  Request timeout in seconds (default: 600).
        """
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.timeout  = timeout
        self._check_requests()

    def _check_requests(self):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise RuntimeError("Install 'requests': pip install requests")

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        import requests
        r = requests.get(self._url(path), params=params, headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post_json(self, path: str, body: dict, params: Optional[dict] = None) -> dict:
        import requests
        r = requests.post(
            self._url(path), json=body, params=params,
            headers=self._headers(), timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def _post_file(
        self,
        path: str,
        file_path: str,
        params: Optional[dict] = None,
        extra_data: Optional[dict] = None,
    ) -> dict:
        import requests
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "video/mp4")}
            r     = requests.post(
                self._url(path),
                files=files,
                params=params,
                data=extra_data,
                headers=self._headers(),
                timeout=self.timeout,
            )
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        import requests
        r = requests.delete(self._url(path), headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ── Setup ─────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """GPU and model health check."""
        return self._get("/health")

    def download_model(self, variant: str = "vjepa2-vitl-fpc64-256") -> dict:
        """Trigger model weight download to Modal Volume."""
        return self._post_json("/download-model", {"model_variant": variant})

    # ── Video ─────────────────────────────────────────────────────────────────

    def upload(self, file_path: str) -> dict:
        """Upload a video file. Returns dict with video_id."""
        return self._post_file("/upload", file_path)

    def preprocess(
        self,
        video_id:      str,
        target_fps:    int = 8,
        num_frames:    int = 16,
        frame_size:    int = 224,
        sampling_mode: str = "uniform",
        stride:        int = 2,
    ) -> dict:
        """Preprocess an uploaded video into a (C,T,H,W) tensor."""
        config = {
            "target_fps":    target_fps,
            "num_frames":    num_frames,
            "frame_size":    frame_size,
            "sampling_mode": sampling_mode,
            "stride":        stride,
        }
        return self._post_json("/preprocess", config, params={"video_id": video_id})

    def list_videos(self) -> dict:
        """List all uploaded videos."""
        return self._get("/videos")

    def delete_video(self, video_id: str) -> dict:
        """Delete a video and free storage."""
        return self._delete(f"/videos/{video_id}")

    # ── Inference ─────────────────────────────────────────────────────────────

    def encode(self, video_id: str, task: str = "encode") -> dict:
        """Extract V-JEPA 2 spatiotemporal embeddings from a preprocessed video."""
        return self._post_json("/encode", {"video_id": video_id, "task": task})

    def pipeline(
        self,
        file_path:     str,
        num_frames:    int = 16,
        frame_size:    int = 224,
        sampling_mode: str = "uniform",
    ) -> dict:
        """Full pipeline: upload → preprocess → encode in one call."""
        return self._post_file(
            "/pipeline",
            file_path,
            params={
                "task":          "encode",
                "num_frames":    num_frames,
                "frame_size":    frame_size,
                "sampling_mode": sampling_mode,
            },
        )


    # ── SurPrise ──────────────────────────────────────────────────────────────

    def surprise_score(
        self,
        video_ids:       List[str],
        distance_metric: str = "l1",
        aggregation_mode: str = "average",
        window_size:     int = 8,
        context_length:  int = 4,
        stride:          int = 2,
    ) -> dict:
        """
        Compute SurPrise temporal coherence score for individual videos.

        No reference set needed — measures how "surprised" the V-JEPA 2 encoder
        is by each video's own temporal progression via sliding-window prediction
        error. Low surprise = temporally coherent, low hallucination proxy.

        All videos must already be uploaded and preprocessed.

        Args:
            video_ids:        IDs of videos to score.
            distance_metric:  "l1" (paper default) | "l2" | "cosine"
            aggregation_mode: "average" (mean window surprise) | "maximum" (worst window)
            window_size:      Total frames per sliding window.
            context_length:   Context frames per window (must be < window_size).
            stride:           Step between consecutive windows.

        Returns:
            dict with mode="single_video", mean_surprise_score, and per-video
            breakdown (surprise_score, max_surprise, hallucination_level,
            hallucination_score, window_scores, ...).
        """
        return self._post_json("/surprise/score", {
            "video_ids":        video_ids,
            "distance_metric":  distance_metric,
            "aggregation_mode": aggregation_mode,
            "window_size":      window_size,
            "context_length":   context_length,
            "stride":           stride,
        })

    def surprise_score_voe(
        self,
        possible_video_ids:   List[str],
        impossible_video_ids: List[str],
        distance_metric:      str = "l1",
        aggregation_mode:     str = "average",
        window_size:          int = 8,
        context_length:       int = 4,
        stride:               int = 2,
    ) -> dict:
        """
        Compute SurPrise pairwise VoE (Violation of Expectation) accuracy.

        Evaluates whether the model assigns higher surprise to physically
        impossible videos than to possible ones — a measure of intuitive
        physics understanding.

        All videos must already be uploaded and preprocessed.

        Args:
            possible_video_ids:   IDs of physically possible videos.
            impossible_video_ids: IDs of physically impossible videos.
            distance_metric:      "l1" | "l2" | "cosine"
            aggregation_mode:     "average" | "maximum"
            window_size:          Total frames per sliding window.
            context_length:       Context frames per window.
            stride:               Step between consecutive windows.

        Returns:
            dict with mode="voe_pair", pairwise_accuracy, n_correct_pairs,
            n_total_pairs, possible_scores, impossible_scores.
        """
        return self._post_json("/surprise/score", {
            "possible_video_ids":   possible_video_ids,
            "impossible_video_ids": impossible_video_ids,
            "distance_metric":      distance_metric,
            "aggregation_mode":     aggregation_mode,
            "window_size":          window_size,
            "context_length":       context_length,
            "stride":               stride,
        })

    def surprise_pipeline(
        self,
        file_path:        str,
        distance_metric:  str = "l1",
        aggregation_mode: str = "average",
        window_size:      int = 8,
        context_length:   int = 4,
        stride:           int = 2,
        num_frames:       int = 16,
        frame_size:       int = 224,
    ) -> dict:
        """
        All-in-one SurPrise pipeline: upload → preprocess → encode → score.

        Args:
            file_path:        Path to video file.
            distance_metric:  "l1" | "l2" | "cosine"
            aggregation_mode: "average" | "maximum"
            window_size:      Total frames per sliding window.
            context_length:   Context frames per window.
            stride:           Step between consecutive windows.
            num_frames:       Frames to sample during preprocessing.
            frame_size:       Output square frame size (px).

        Returns:
            dict with pipeline steps + SurPrise result under key "surprise".
        """
        return self._post_file(
            "/surprise/pipeline",
            file_path,
            params={
                "distance_metric":  distance_metric,
                "aggregation_mode": aggregation_mode,
                "window_size":      window_size,
                "context_length":   context_length,
                "stride":           stride,
            },
            extra_data={
                "config_json": f'{{"num_frames":{num_frames},"frame_size":{frame_size}}}',
            },
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def upload_and_preprocess(
        self,
        file_path:  str,
        num_frames: int = 16,
        frame_size: int = 224,
    ) -> str:
        """Upload and preprocess a video. Returns video_id."""
        video_id = self.upload(file_path)["video_id"]
        self.preprocess(video_id, num_frames=num_frames, frame_size=frame_size)
        return video_id




# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pprint

    USAGE = """
V-JEPA 2 + SurPrise Client CLI

Usage:
  python client.py <base_url> <command> [args...]  [--key hyl_...]

Commands:
  health
  download-model [variant]
  upload <video.mp4>
  preprocess <video_id>
  encode <video_id>
  pipeline <video.mp4>
  videos
  delete <video_id>

  surprise-score <vid_id1,vid_id2,...>
  surprise-score-voe <possible_id1,...> <impossible_id1,...>
  surprise-pipeline <video.mp4>

Options:
  --key <api_key>   X-API-Key header value (or set HYLE_API_KEY env var)
"""

    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(0)

    import os

    # Parse --key flag anywhere in argv
    args    = sys.argv[1:]
    api_key = os.environ.get("HYLE_API_KEY", "")
    if "--key" in args:
        idx     = args.index("--key")
        api_key = args[idx + 1]
        args    = args[:idx] + args[idx + 2:]

    base_url = args[0]
    command  = args[1]
    rest     = args[2:]
    client   = VJepa2Client(base_url, api_key=api_key)

    if command == "health":
        pprint.pprint(client.health())

    elif command == "download-model":
        variant = rest[0] if rest else "vjepa2-vitl-fpc64-256"
        pprint.pprint(client.download_model(variant))

    elif command == "upload":
        pprint.pprint(client.upload(rest[0]))

    elif command == "preprocess":
        pprint.pprint(client.preprocess(rest[0]))

    elif command == "encode":
        r = client.encode(rest[0])
        emb = r.get("embeddings")
        if emb:
            r["embeddings"] = f"[{len(emb)} vectors — dim {len(emb[0]) if isinstance(emb[0], list) else len(emb)}]"
        pprint.pprint(r)

    elif command == "pipeline":
        pprint.pprint(client.pipeline(rest[0]))

    elif command == "videos":
        pprint.pprint(client.list_videos())

    elif command == "delete":
        pprint.pprint(client.delete_video(rest[0]))



    elif command == "surprise-score":
        vid_ids = rest[0].split(",")
        result  = client.surprise_score(vid_ids)
        print(f"\n{'='*55}")
        print(f"  SurPrise Temporal Coherence Report")
        print(f"{'='*55}")
        for v in result.get("videos", []):
            print(f"  Video      : {v['video_id'][:16]}…")
            print(f"  Score      : {v['surprise_score']:.6f}  (max: {v['max_surprise']:.6f})")
            print(f"  Level      : {v['hallucination_level'].upper()}")
            print(f"  Norm score : {v['hallucination_score']:.1f} / 100")
            print(f"  Windows    : {v['n_windows']}")
            print(f"  Summary    : {v['interpretation']}")
            print()
        print(f"{'='*55}\n")

    elif command == "surprise-score-voe":
        possible_ids   = rest[0].split(",")
        impossible_ids = rest[1].split(",")
        result = client.surprise_score_voe(possible_ids, impossible_ids)
        print(f"\n{'='*55}")
        print(f"  SurPrise VoE Pairwise Accuracy")
        print(f"{'='*55}")
        print(f"  Accuracy   : {result['pairwise_accuracy']:.2%}")
        print(f"  Correct    : {result['n_correct_pairs']} / {result['n_total_pairs']} pairs")
        print(f"  Possible   : {[f'{s:.4f}' for s in result['possible_scores']]}")
        print(f"  Impossible : {[f'{s:.4f}' for s in result['impossible_scores']]}")
        print(f"{'='*55}\n")

    elif command == "surprise-pipeline":
        result   = client.surprise_pipeline(rest[0])
        surprise = result.get("surprise", {})
        videos   = surprise.get("videos", [])
        v        = videos[0] if videos else {}
        print(f"\n{'='*55}")
        print(f"  SurPrise Pipeline Report")
        print(f"{'='*55}")
        print(f"  Video      : {result.get('filename', '')}")
        print(f"  Score      : {v.get('surprise_score', 'N/A'):.6f}")
        print(f"  Level      : {v.get('hallucination_level', 'N/A').upper()}")
        print(f"  Norm score : {v.get('hallucination_score', 'N/A'):.1f} / 100")
        print(f"  Windows    : {v.get('n_windows', 'N/A')}")
        print(f"  Summary    : {v.get('interpretation', '')}")
        print(f"{'='*55}\n")

    else:
        print(f"Unknown command: {command}")
        print(USAGE)
        sys.exit(1)
