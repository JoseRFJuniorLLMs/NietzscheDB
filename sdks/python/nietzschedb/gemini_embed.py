"""Gemini Embedding 2 → Poincaré projection for NietzscheDB.

Wraps Google's Gemini Embedding 2 API (text, image, audio, video, PDF)
and projects the resulting Euclidean vectors into the Poincaré ball
using the exponential map at origin.

Usage::

    from nietzschedb.gemini_embed import GeminiEmbedder

    embedder = GeminiEmbedder(api_key="...", dim=768)

    # Text
    poincare_vec = embedder.embed_text("hello world")

    # Image (file path or bytes)
    poincare_vec = embedder.embed_image("/path/to/image.png")

    # With NietzscheDB client
    from nietzschedb import NietzscheClient
    client = NietzscheClient("localhost:50051")
    node_id = client.insert_node(
        coords=poincare_vec,
        content={"text": "hello"},
        collection="my_collection",
    )

    # Or via insert_sensory with precomputed flag
    client.insert_sensory(
        node_id=node_id,
        modality="text",
        latent=poincare_vec,
        precomputed_poincare=True,  # skip exp_map_zero on server
        collection="my_collection",
    )

Requires: pip install google-genai
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


# ── Poincaré projection (pure Python, no Rust dependency) ─────────────────

def exp_map_zero(euclidean: List[float]) -> List[float]:
    """Exponential map at the origin of the Poincaré ball.

    Maps a Euclidean vector v to the Poincaré ball via:
        exp_0(v) = tanh(||v||) * v / ||v||

    This is the same formula used in nietzsche-hyp-ops (Rust),
    reimplemented here for client-side projection.
    """
    norm_sq = sum(x * x for x in euclidean)
    norm = math.sqrt(norm_sq)
    if norm < 1e-12:
        return [0.0] * len(euclidean)
    scale = math.tanh(norm) / norm
    return [x * scale for x in euclidean]


def normalize_l2(vec: List[float]) -> List[float]:
    """L2-normalize a vector (required for Gemini Embedding 2 sub-3072D outputs)."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


# ── Task types ────────────────────────────────────────────────────────────

TASK_TYPES = Literal[
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
]


# ── Embedder class ────────────────────────────────────────────────────────

class GeminiEmbedder:
    """Client for Gemini Embedding 2 with automatic Poincaré projection.

    Args:
        api_key: Google AI API key (or set GEMINI_API_KEY / GOOGLE_API_KEY env var).
        dim: Output embedding dimension (128–3072). Default 768.
            Gemini Embedding 2 uses Matryoshka, so smaller dims retain most quality.
            Recommended: 3072 (full), 1536 (balanced), 768 (compact), 128 (minimal).
        model: Model ID. Default "gemini-embedding-002".
        auto_normalize: L2-normalize embeddings below 3072D (recommended by Google).
        auto_project: Automatically apply exp_map_zero Poincaré projection.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        dim: int = 768,
        model: str = "gemini-embedding-002",
        auto_normalize: bool = True,
        auto_project: bool = True,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self.dim = dim
        self.model = model
        self.auto_normalize = auto_normalize and dim < 3072
        self.auto_project = auto_project

        # Lazy import — only fail when actually used
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package required. Install with: pip install google-genai"
                )
        return self._client

    def _post_process(self, embedding: List[float]) -> List[float]:
        """Apply normalization and Poincaré projection."""
        vec = list(embedding)
        if self.auto_normalize:
            vec = normalize_l2(vec)
        if self.auto_project:
            vec = exp_map_zero(vec)
        return vec

    # ── Text ──────────────────────────────────────────────────────────────

    def embed_text(
        self,
        text: str,
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
    ) -> List[float]:
        """Embed text content. Returns Poincaré ball vector.

        Args:
            text: Input text (up to 8192 tokens).
            task_type: One of RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT,
                SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING, etc.
            title: Optional document title for better retrieval quality.
        """
        client = self._get_client()
        config = {"output_dimensionality": self.dim}
        if title:
            config["title"] = title
        result = client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dim,
            },
        )
        return self._post_process(result.embeddings[0].values)

    def embed_texts(
        self,
        texts: List[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """Batch embed multiple texts. Returns list of Poincaré vectors."""
        client = self._get_client()
        result = client.models.embed_content(
            model=self.model,
            contents=texts,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dim,
            },
        )
        return [self._post_process(e.values) for e in result.embeddings]

    # ── Query (convenience for asymmetric retrieval) ──────────────────────

    def embed_query(self, query: str) -> List[float]:
        """Embed a search query (uses RETRIEVAL_QUERY task type)."""
        return self.embed_text(query, task_type="RETRIEVAL_QUERY")

    def embed_document(self, text: str, *, title: Optional[str] = None) -> List[float]:
        """Embed a document for indexing (uses RETRIEVAL_DOCUMENT task type)."""
        return self.embed_text(text, task_type="RETRIEVAL_DOCUMENT", title=title)

    # ── Image ─────────────────────────────────────────────────────────────

    def embed_image(
        self,
        image: Union[str, bytes, Path],
        *,
        mime_type: Optional[str] = None,
    ) -> List[float]:
        """Embed an image. Returns Poincaré ball vector.

        Args:
            image: File path (str/Path) or raw bytes (PNG/JPEG).
            mime_type: Override MIME type (auto-detected from extension).
        """
        client = self._get_client()
        from google.genai import types

        if isinstance(image, (str, Path)):
            path = Path(image)
            if mime_type is None:
                ext = path.suffix.lower()
                mime_type = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                }.get(ext, "image/png")
            image_bytes = path.read_bytes()
        else:
            image_bytes = image
            mime_type = mime_type or "image/png"

        part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        result = client.models.embed_content(
            model=self.model,
            contents=part,
            config={"output_dimensionality": self.dim},
        )
        return self._post_process(result.embeddings[0].values)

    # ── Audio ─────────────────────────────────────────────────────────────

    def embed_audio(
        self,
        audio: Union[str, bytes, Path],
        *,
        mime_type: Optional[str] = None,
    ) -> List[float]:
        """Embed audio. Returns Poincaré ball vector.

        Args:
            audio: File path or raw bytes (MP3/WAV, up to 80s).
            mime_type: Override MIME type.
        """
        client = self._get_client()
        from google.genai import types

        if isinstance(audio, (str, Path)):
            path = Path(audio)
            if mime_type is None:
                ext = path.suffix.lower()
                mime_type = {
                    ".mp3": "audio/mpeg",
                    ".wav": "audio/wav",
                    ".ogg": "audio/ogg",
                    ".flac": "audio/flac",
                }.get(ext, "audio/mpeg")
            audio_bytes = path.read_bytes()
        else:
            audio_bytes = audio
            mime_type = mime_type or "audio/mpeg"

        part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        result = client.models.embed_content(
            model=self.model,
            contents=part,
            config={"output_dimensionality": self.dim},
        )
        return self._post_process(result.embeddings[0].values)

    # ── Video ─────────────────────────────────────────────────────────────

    def embed_video(
        self,
        video: Union[str, bytes, Path],
        *,
        mime_type: Optional[str] = None,
    ) -> List[float]:
        """Embed video. Returns Poincaré ball vector.

        Args:
            video: File path or raw bytes (MP4/MOV, up to 120s).
            mime_type: Override MIME type.
        """
        client = self._get_client()
        from google.genai import types

        if isinstance(video, (str, Path)):
            path = Path(video)
            if mime_type is None:
                ext = path.suffix.lower()
                mime_type = {
                    ".mp4": "video/mp4",
                    ".mov": "video/quicktime",
                    ".avi": "video/x-msvideo",
                    ".webm": "video/webm",
                }.get(ext, "video/mp4")
            video_bytes = path.read_bytes()
        else:
            video_bytes = video
            mime_type = mime_type or "video/mp4"

        part = types.Part.from_bytes(data=video_bytes, mime_type=mime_type)
        result = client.models.embed_content(
            model=self.model,
            contents=part,
            config={"output_dimensionality": self.dim},
        )
        return self._post_process(result.embeddings[0].values)

    # ── Multimodal (interleaved) ──────────────────────────────────────────

    def embed_multimodal(
        self,
        parts: List[Dict[str, Any]],
    ) -> List[float]:
        """Embed interleaved multimodal content. Returns Poincaré ball vector.

        Args:
            parts: List of dicts, each with:
                - {"text": "..."} for text
                - {"image": "/path/to/img.png"} for image
                - {"audio": "/path/to/audio.mp3"} for audio
                - {"video": "/path/to/video.mp4"} for video

        Example::

            vec = embedder.embed_multimodal([
                {"text": "A cat sitting on a mat"},
                {"image": "cat.jpg"},
            ])
        """
        client = self._get_client()
        from google.genai import types

        content_parts = []
        for part in parts:
            if "text" in part:
                content_parts.append(part["text"])
            elif "image" in part:
                path = Path(part["image"])
                ext = path.suffix.lower()
                mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(ext, "image/png")
                content_parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
            elif "audio" in part:
                path = Path(part["audio"])
                ext = path.suffix.lower()
                mime = {".mp3": "audio/mpeg", ".wav": "audio/wav"}.get(ext, "audio/mpeg")
                content_parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
            elif "video" in part:
                path = Path(part["video"])
                ext = path.suffix.lower()
                mime = {".mp4": "video/mp4", ".mov": "video/quicktime"}.get(ext, "video/mp4")
                content_parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))

        result = client.models.embed_content(
            model=self.model,
            contents=content_parts,
            config={"output_dimensionality": self.dim},
        )
        return self._post_process(result.embeddings[0].values)

    # ── Raw Euclidean (no projection) ─────────────────────────────────────

    def embed_text_euclidean(self, text: str, *, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """Get raw Euclidean embedding without Poincaré projection.
        Useful when sending to InsertSensory with precomputed_poincare=False
        (server does exp_map_zero).
        """
        client = self._get_client()
        result = client.models.embed_content(
            model=self.model,
            contents=text,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dim,
            },
        )
        vec = list(result.embeddings[0].values)
        if self.auto_normalize:
            vec = normalize_l2(vec)
        return vec
