from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .io import item_key


def deterministic_mock_score(query: dict[str, Any], candidate: dict[str, Any], seed: int = 42) -> float:
    payload = f"{seed}\n{item_key(query)}\n{item_key(candidate)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return float(value)


def cosine_scores(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    return query_embeddings @ candidate_embeddings.T


def encode_with_nexus_embedder(
    model_name_or_path: str,
    queries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    batch_size: int,
    query_max_len: int,
    passage_max_len: int,
    device: str,
    devices: str | None,
    torch_dtype: str,
    backbone_load_strategy: str,
    trust_remote_code: bool,
    cache_dir: str | None,
    processor_name_or_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from Nexus.inference.embedder.multimodal_retrieval import MultimodalEmbedder

    target_devices = [x.strip() for x in devices.split(",") if x.strip()] if devices else [device]

    embedder = MultimodalEmbedder(
        model_name_or_path=model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        model_type="auto",
        normalize_embeddings=True,
        devices=target_devices,
        batch_size=batch_size,
        query_max_length=query_max_len,
        passage_max_length=passage_max_len,
        convert_to_numpy=True,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        backbone_load_strategy=backbone_load_strategy,
        use_chat_template=True,
    )
    query_embeddings = embedder.encode_queries(queries, batch_size=batch_size, convert_to_numpy=True)
    candidate_embeddings = embedder.encode_corpus(candidates, batch_size=batch_size, convert_to_numpy=True)
    return np.asarray(query_embeddings), np.asarray(candidate_embeddings)
