from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .arguments import MiningArguments
from .io import (
    append_global_candidate_items,
    append_own_negative_candidates,
    build_candidate_pool,
    item_key,
    item_media_paths,
    item_modality_signature,
    normalize_group,
    read_candidate_corpus_jsonl,
    read_jsonl,
    resolve_rows_media_paths,
    row_candidate_items,
    row_positive_keys,
    row_positive_signatures,
    row_query_item,
    validate_mined_rows,
    write_json,
    write_jsonl,
)
from .scoring import cosine_scores, deterministic_mock_score, encode_with_nexus_embedder


class _TopKScoreView:
    def __init__(self, indices: np.ndarray, scores: np.ndarray):
        self.indices = indices
        self.scores = scores

    def __getitem__(self, row_idx: int):
        return _TopKScoreRow(self.indices[row_idx], self.scores[row_idx])


class _TopKScoreRow:
    def __init__(self, indices: np.ndarray, scores: np.ndarray):
        self.indices = indices
        self.scores = scores
        self._scores_by_index: dict[int, float] | None = None

    def __getitem__(self, cand_idx: int) -> float:
        if self._scores_by_index is None:
            self._scores_by_index = {int(idx): float(score) for idx, score in zip(self.indices, self.scores) if idx >= 0}
        return self._scores_by_index.get(int(cand_idx), float("-inf"))

    @property
    def size(self) -> int:
        return int(np.count_nonzero(self.indices >= 0))


def _as_score_view(scores: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray | _TopKScoreView:
    if isinstance(scores, tuple):
        return _TopKScoreView(scores[0], scores[1])
    return scores


def _rank_candidates_from_scores(scores: Any, top_k: int) -> list[int]:
    if isinstance(scores, _TopKScoreRow):
        valid = scores.indices >= 0
        if not np.any(valid):
            return []
        indices = scores.indices[valid]
        values = scores.scores[valid]
        order = np.argsort(-values)[:top_k]
        return [int(indices[idx]) for idx in order]
    return _rank_candidates(scores, top_k)


def _default_score_device(args: MiningArguments) -> str:
    if args.score_device:
        return args.score_device
    if args.devices:
        return args.devices.split(",")[0].strip()
    return args.device


def _row_local_candidate_source_rows(candidates) -> np.ndarray:
    source_rows = np.full((len(candidates),), -1, dtype=np.int64)
    for cand_idx, cand in enumerate(candidates):
        if cand.source_field == "neg":
            source_rows[cand_idx] = int(cand.source_row)
    return source_rows


def _split_csv(value: str | None) -> list[str]:
    if value in [None, ""]:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _repo_root() -> Path:
    # .../Nexus/Nexus/data_mining/multimodal_hard_negative/mining.py -> .../Nexus
    return Path(__file__).resolve().parents[3]


def _default_mmeb_config_paths() -> list[Path]:
    config_dir = _repo_root() / "examples/multimodal_retrieval/evaluation/mmeb_v2/configs"
    return [
        config_dir / "image_retrieval.yaml",
        config_dir / "video_retrieval.yaml",
        config_dir / "visdoc_retrieval.yaml",
        config_dir / "image.yaml",
        config_dir / "video.yaml",
        config_dir / "visdoc.yaml",
    ]


def _default_mmeb_hf_cache_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "OFFLINE_HF_HUB_ROOT"]:
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(Path(hf_home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    return candidates


def _configure_mmeb_hf_cache(args: MiningArguments) -> dict[str, Any]:
    cache_dirs = [Path(path) for path in _split_csv(args.mmeb_hf_cache_dir)]
    cache_dirs.extend(_default_mmeb_hf_cache_dirs())
    selected = next((path for path in cache_dirs if path.exists()), None)
    if selected is None:
        return {"selected_hf_cache_dir": None, "offline_env_configured": False}

    selected_str = str(selected)
    for key in ["HF_HUB_CACHE", "OFFLINE_HF_HUB_ROOT"]:
        os.environ.setdefault(key, selected_str)
    # Prefer offline cache lookup for MMEB metadata when a local cache exists,
    # so mining does not stall on network access in restricted environments.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    return {
        "selected_hf_cache_dir": selected_str,
        "offline_env_configured": True,
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        "OFFLINE_HF_HUB_ROOT": os.environ.get("OFFLINE_HF_HUB_ROOT"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
    }


def _load_mmeb_task_configs(config_paths: list[str]) -> dict[str, dict[str, Any]]:
    import yaml

    paths = [Path(path) for path in config_paths] if config_paths else _default_mmeb_config_paths()
    configs: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            continue
        for dataset_name, task_config in data.items():
            if isinstance(task_config, dict):
                configs[str(dataset_name)] = dict(task_config)
    return configs


def _candidate_items_from_eval_dataset(dataset_name: str, task_config: dict[str, Any], args: MiningArguments):
    import Nexus.evaluation.mmeb_v2.data.datasets  # noqa: F401
    from Nexus.evaluation.mmeb_v2.data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
    from Nexus.evaluation.mmeb_v2.eval_embedding import prepare_task_paths

    task_config = deepcopy(task_config)
    if args.mmeb_num_sample_per_subset is not None:
        task_config["num_sample_per_subset"] = args.mmeb_num_sample_per_subset

    data_args = SimpleNamespace(
        data_basedir=args.mmeb_data_basedir,
        generated_media_basedir=(
            args.mmeb_generated_media_basedir
            or os.path.join(os.path.dirname(os.path.abspath(args.output_train_jsonl)), "generated_mmeb_media_cache")
        ),
        max_video_frames=args.mmeb_max_video_frames,
        video_fps=args.mmeb_video_fps,
    )
    model_args = SimpleNamespace()
    prepared_config = prepare_task_paths(task_config, data_args)
    eval_dataset, corpus = AutoEvalPairDataset.instantiate(
        model_args=model_args,
        data_args=data_args,
        **prepared_config,
    )
    cand_dataset = generate_cand_dataset(eval_dataset, corpus)
    items: list[dict[str, Any]] = []
    max_items = args.max_candidate_examples if args.max_candidate_examples is not None else 0
    for row_idx, row in enumerate(cand_dataset):
        if max_items > 0 and len(items) >= max_items:
            break
        row = dict(row)
        dataset_infos = row.get("dataset_infos") if isinstance(row.get("dataset_infos"), dict) else {}
        for item in row_candidate_items(row, include_fields=[]):
            metadata = deepcopy(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}
            metadata["mmeb_eval_dataset"] = dataset_name
            if dataset_infos.get("cand_name") not in [None, ""]:
                metadata["cand_name"] = dataset_infos["cand_name"]
            metadata["mmeb_eval_candidate_row"] = row_idx
            item["metadata"] = metadata
            items.append(item)
            if max_items > 0 and len(items) >= max_items:
                break
    return items


def _load_mmeb_eval_corpus_items(args: MiningArguments) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_names = _split_csv(args.mmeb_eval_corpus_datasets)
    if not dataset_names:
        return [], []
    cache_info = _configure_mmeb_hf_cache(args)
    configs = _load_mmeb_task_configs(_split_csv(args.mmeb_eval_dataset_config))
    items: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        if dataset_name not in configs:
            available = ", ".join(sorted(configs)[:30])
            raise ValueError(
                f"MMEB v2 dataset {dataset_name!r} was not found in the configured YAML files. "
                f"First available names: {available}"
            )
        start = len(items)
        dataset_items = _candidate_items_from_eval_dataset(dataset_name, configs[dataset_name], args)
        items.extend(dataset_items)
        stats.append(
            {
                "source": "mmeb_eval_corpus",
                "dataset": dataset_name,
                "raw_items": len(dataset_items),
                "items_before_dedup": len(items) - start,
                "items_start": start,
                "items_end": len(items),
                "cache_info": cache_info,
            }
        )
    return items, stats


def _topk_scores_blockwise(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    block_size: int,
    score_device: str,
    local_candidate_source_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if query_embeddings.shape[0] == 0 or candidate_embeddings.shape[0] == 0:
        return (
            np.full((query_embeddings.shape[0], 0), -1, dtype=np.int64),
            np.empty((query_embeddings.shape[0], 0), dtype=np.float32),
        )

    top_k = max(1, min(int(top_k), int(candidate_embeddings.shape[0])))
    block_size = max(1, int(block_size))
    try:
        import torch
    except ImportError:
        return _topk_scores_numpy_blockwise(
            query_embeddings,
            candidate_embeddings,
            top_k,
            block_size,
            local_candidate_source_rows=local_candidate_source_rows,
        )

    device = torch.device(score_device)
    query_tensor = torch.as_tensor(query_embeddings, dtype=torch.float32, device=device)
    best_scores = torch.full((query_tensor.shape[0], top_k), -float("inf"), dtype=torch.float32, device=device)
    best_indices = torch.full((query_tensor.shape[0], top_k), -1, dtype=torch.long, device=device)

    for start in range(0, candidate_embeddings.shape[0], block_size):
        end = min(start + block_size, candidate_embeddings.shape[0])
        cand_tensor = torch.as_tensor(candidate_embeddings[start:end], dtype=torch.float32, device=device)
        block_scores = query_tensor @ cand_tensor.T
        if local_candidate_source_rows is not None:
            local_rows = torch.as_tensor(local_candidate_source_rows[start:end], dtype=torch.long, device=device)
            query_rows = torch.arange(query_tensor.shape[0], dtype=torch.long, device=device)[:, None]
            disallowed = (local_rows[None, :] >= 0) & (local_rows[None, :] != query_rows)
            block_scores = block_scores.masked_fill(disallowed, -float("inf"))
        combined_scores = torch.cat([best_scores, block_scores], dim=1)
        block_indices = torch.arange(start, end, dtype=torch.long, device=device).expand(query_tensor.shape[0], end - start)
        combined_indices = torch.cat([best_indices, block_indices], dim=1)
        best_scores, top_positions = torch.topk(combined_scores, k=top_k, dim=1, largest=True, sorted=True)
        best_indices = torch.gather(combined_indices, 1, top_positions)
        del cand_tensor, block_scores, combined_scores, block_indices, combined_indices, top_positions
    return best_indices.cpu().numpy(), best_scores.cpu().numpy()


def _topk_scores_numpy_blockwise(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    block_size: int,
    local_candidate_source_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    best_scores = np.full((query_embeddings.shape[0], top_k), -np.inf, dtype=np.float32)
    best_indices = np.full((query_embeddings.shape[0], top_k), -1, dtype=np.int64)
    for start in range(0, candidate_embeddings.shape[0], block_size):
        end = min(start + block_size, candidate_embeddings.shape[0])
        block_scores = query_embeddings @ candidate_embeddings[start:end].T
        if local_candidate_source_rows is not None:
            local_rows = local_candidate_source_rows[start:end]
            row_ids = np.arange(query_embeddings.shape[0], dtype=np.int64)[:, None]
            disallowed = (local_rows[None, :] >= 0) & (local_rows[None, :] != row_ids)
            block_scores = block_scores.copy()
            block_scores[disallowed] = -np.inf
        block_indices = np.arange(start, end, dtype=np.int64)
        combined_scores = np.concatenate([best_scores, block_scores], axis=1)
        combined_indices = np.concatenate(
            [best_indices, np.broadcast_to(block_indices, (query_embeddings.shape[0], end - start))], axis=1
        )
        order = np.argpartition(-combined_scores, np.arange(top_k), axis=1)[:, :top_k]
        row_ids = np.arange(query_embeddings.shape[0])[:, None]
        selected_scores = combined_scores[row_ids, order]
        selected_indices = combined_indices[row_ids, order]
        sort_order = np.argsort(-selected_scores, axis=1)
        best_scores = selected_scores[row_ids, sort_order]
        best_indices = selected_indices[row_ids, sort_order]
    return best_indices, best_scores


def _append_required_candidate_scores(
    topk_scores: tuple[np.ndarray, np.ndarray],
    rows: list[dict[str, Any]],
    candidates,
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    top_indices, top_scores = topk_scores
    candidate_indices_by_key: dict[str, list[int]] = {}
    for cand_idx, cand in enumerate(candidates):
        candidate_indices_by_key.setdefault(cand.key, []).append(cand_idx)

    required_by_row: list[list[int]] = []
    max_extra = 0
    for row_idx, row in enumerate(rows):
        present = {int(idx) for idx in top_indices[row_idx] if idx >= 0}
        required: list[int] = []
        for key in row_positive_keys(row):
            for cand_idx in candidate_indices_by_key.get(key, []):
                if cand_idx not in present:
                    required.append(cand_idx)
                    present.add(cand_idx)
        required_by_row.append(required)
        max_extra = max(max_extra, len(required))
    if max_extra == 0:
        return top_indices, top_scores

    expanded_indices = np.full((top_indices.shape[0], top_indices.shape[1] + max_extra), -1, dtype=np.int64)
    expanded_scores = np.full((top_scores.shape[0], top_scores.shape[1] + max_extra), -np.inf, dtype=np.float32)
    expanded_indices[:, : top_indices.shape[1]] = top_indices
    expanded_scores[:, : top_scores.shape[1]] = top_scores
    for row_idx, required in enumerate(required_by_row):
        if not required:
            continue
        offset = top_indices.shape[1]
        for local_idx, cand_idx in enumerate(required):
            expanded_indices[row_idx, offset + local_idx] = cand_idx
            expanded_scores[row_idx, offset + local_idx] = float(
                query_embeddings[row_idx].astype(np.float32) @ candidate_embeddings[cand_idx].astype(np.float32)
            )
    return expanded_indices, expanded_scores


def _score_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
    }


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.casefold().split())


def _positive_texts(items: list[dict[str, Any]]) -> set[str]:
    return {text for text in (_normalize_text(item.get("text")) for item in items) if text}


def _media_dedupe_keys(item: dict[str, Any], hash_cache: dict[str, str]) -> set[str]:
    keys: set[str] = set()
    for media_path in item_media_paths(item):
        norm_path = os.path.normpath(media_path)
        keys.add(f"path:{norm_path}")
        basename = os.path.basename(norm_path).casefold()
        if basename:
            keys.add(f"basename:{basename}")
        digest = _media_md5(norm_path, hash_cache)
        if digest:
            keys.add(f"md5:{digest}")
    return keys


def _media_md5(media_path: str, hash_cache: dict[str, str]) -> str:
    if media_path in hash_cache:
        return hash_cache[media_path]
    if not os.path.isfile(media_path):
        hash_cache[media_path] = ""
        return ""
    try:
        digest = hashlib.md5()
        with open(media_path, "rb") as media_file:
            for chunk in iter(lambda: media_file.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
    except OSError:
        value = ""
    hash_cache[media_path] = value
    return value


def _items_media_dedupe_keys(items: list[dict[str, Any]], hash_cache: dict[str, str]) -> set[str]:
    keys: set[str] = set()
    for item in items:
        keys.update(_media_dedupe_keys(item, hash_cache))
    return keys


def _item_preview(item: dict[str, Any]) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    for field in ["text", "image_path", "image", "video_path", "video", "audio_path"]:
        value = item.get(field)
        if isinstance(value, str) and value:
            preview[field] = value if len(value) <= 240 else value[:237] + "..."
    for field in ["image_paths", "video_frames"]:
        value = item.get(field)
        if isinstance(value, list) and value:
            preview[field] = value[:3]
            preview[f"{field}_count"] = len(value)
    return preview


def _rank_candidates(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0:
        return []
    top_k = min(top_k, scores.shape[0])
    order = np.argpartition(-scores, np.arange(top_k))[:top_k]
    order = order[np.argsort(-scores[order])]
    return [int(x) for x in order]


def _mask_disallowed_row_local_negatives(scores: np.ndarray, candidates) -> np.ndarray:
    """Mask cross-row local negatives for full-matrix scoring.

    In `all_pos_plus_own_neg`, a candidate with `source_field == "neg"` is
    only legal for the row it came from. Full-matrix scoring computes every
    query against every candidate, so we explicitly set other rows' local
    negatives to -inf before top-k ranking.
    """

    masked = scores.copy()
    for cand_idx, cand in enumerate(candidates):
        if cand.source_field != "neg":
            continue
        masked[:, cand_idx] = -np.inf
        if 0 <= cand.source_row < masked.shape[0]:
            masked[cand.source_row, cand_idx] = scores[cand.source_row, cand_idx]
    return masked


def _mine_from_score_matrix(
    rows: list[dict[str, Any]],
    candidates,
    scores: np.ndarray | tuple[np.ndarray, np.ndarray],
    args: MiningArguments,
    teacher_scores: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
    student_scores: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
):
    filter_scores = _as_score_view(teacher_scores if teacher_scores is not None else scores)
    rank_scores = _as_score_view(student_scores if student_scores is not None else scores)
    student_score_view = _as_score_view(student_scores) if student_scores is not None else None
    output_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidate_indices_by_key: dict[str, list[int]] = {}
    for cand_idx, cand in enumerate(candidates):
        candidate_indices_by_key.setdefault(cand.key, []).append(cand_idx)
    kept = 0
    dropped_no_positive = 0
    dropped_low_positive = 0
    total_mined = 0
    skip_reasons: dict[str, int] = {
        "positive_key": 0,
        "modality_mismatch": 0,
        "above_false_negative_margin": 0,
        "above_max_negative_score_ratio": 0,
        "below_min_negative_score": 0,
        "duplicate_text": 0,
        "duplicate_media": 0,
        "duplicate_selected_text": 0,
        "duplicate_selected_media": 0,
    }
    if args.candidate_source == "all_pos_plus_own_neg":
        skip_reasons["not_own_negative"] = 0
    best_positive_scores: list[float] = []
    mined_negative_scores_all: list[float] = []
    hardness_margins: list[float] = []
    rows_with_zero_negatives = 0
    media_hash_cache: dict[str, str] = {}

    for row_idx, row in enumerate(rows):
        positive_keys = row_positive_keys(row)
        positive_signatures = row_positive_signatures(row)
        if not positive_keys:
            dropped_no_positive += 1
            continue

        row_scores = rank_scores[row_idx]
        row_filter_scores = filter_scores[row_idx]
        row_student_scores = student_score_view[row_idx] if student_score_view is not None else None
        positive_indices = []
        for key in positive_keys:
            positive_indices.extend(candidate_indices_by_key.get(key, []))
        best_positive_score = None
        if positive_indices:
            best_positive_score = float(max(row_filter_scores[idx] for idx in positive_indices))
        best_student_positive_score = None
        if row_student_scores is not None and positive_indices:
            best_student_positive_score = float(max(row_student_scores[idx] for idx in positive_indices))
        if best_positive_score is None:
            dropped_no_positive += 1
            continue
        if args.min_teacher_positive_score is not None and best_positive_score < args.min_teacher_positive_score:
            dropped_low_positive += 1
            continue

        pos_items = normalize_group(row, "pos")
        positive_texts = _positive_texts(pos_items)
        positive_media_keys = _items_media_dedupe_keys(pos_items, media_hash_cache) if args.filter_duplicate_media else set()
        original_negs = normalize_group(row, "neg") if args.keep_existing_negatives else []
        selected_texts = _positive_texts(original_negs) if args.filter_duplicate_text else set()
        selected_media_keys = (
            _items_media_dedupe_keys(original_negs, media_hash_cache) if args.filter_duplicate_media else set()
        )
        best_positive_scores.append(best_positive_score)

        selected = []
        selected_scores = []
        selected_teacher_scores = []
        selected_student_scores = []
        selected_audit = []
        skipped_for_row: dict[str, int] = {}
        for cand_idx in _rank_candidates_from_scores(row_scores, args.top_k):
            cand = candidates[cand_idx]
            if (
                args.candidate_source == "all_pos_plus_own_neg"
                and cand.source_field == "neg"
                and cand.source_row != row_idx
            ):
                skip_reasons["not_own_negative"] += 1
                skipped_for_row["not_own_negative"] = skipped_for_row.get("not_own_negative", 0) + 1
                continue
            if cand.key in positive_keys:
                skip_reasons["positive_key"] += 1
                skipped_for_row["positive_key"] = skipped_for_row.get("positive_key", 0) + 1
                continue
            if positive_signatures and item_modality_signature(cand.item) not in positive_signatures:
                skip_reasons["modality_mismatch"] += 1
                skipped_for_row["modality_mismatch"] = skipped_for_row.get("modality_mismatch", 0) + 1
                continue
            rank_score = float(row_scores[cand_idx])
            teacher_score = float(row_filter_scores[cand_idx])
            student_score = float(row_student_scores[cand_idx]) if row_student_scores is not None else None
            if teacher_score > best_positive_score + args.false_negative_margin:
                skip_reasons["above_false_negative_margin"] += 1
                skipped_for_row["above_false_negative_margin"] = skipped_for_row.get("above_false_negative_margin", 0) + 1
                continue
            if args.max_negative_score_ratio_to_positive is not None:
                max_allowed_negative_score = best_positive_score * args.max_negative_score_ratio_to_positive
                if teacher_score > max_allowed_negative_score:
                    skip_reasons["above_max_negative_score_ratio"] += 1
                    skipped_for_row["above_max_negative_score_ratio"] = (
                        skipped_for_row.get("above_max_negative_score_ratio", 0) + 1
                    )
                    continue
            if args.min_teacher_negative_score is not None and teacher_score < args.min_teacher_negative_score:
                skip_reasons["below_min_negative_score"] += 1
                skipped_for_row["below_min_negative_score"] = skipped_for_row.get("below_min_negative_score", 0) + 1
                continue
            candidate_text = _normalize_text(cand.item.get("text"))
            if args.filter_duplicate_text and candidate_text and candidate_text in positive_texts:
                skip_reasons["duplicate_text"] += 1
                skipped_for_row["duplicate_text"] = skipped_for_row.get("duplicate_text", 0) + 1
                continue
            if args.filter_duplicate_text and candidate_text and candidate_text in selected_texts:
                skip_reasons["duplicate_selected_text"] += 1
                skipped_for_row["duplicate_selected_text"] = skipped_for_row.get("duplicate_selected_text", 0) + 1
                continue
            candidate_media_keys: set[str] = set()
            if args.filter_duplicate_media:
                candidate_media_keys = _media_dedupe_keys(cand.item, media_hash_cache)
                if positive_media_keys.intersection(candidate_media_keys):
                    skip_reasons["duplicate_media"] += 1
                    skipped_for_row["duplicate_media"] = skipped_for_row.get("duplicate_media", 0) + 1
                    continue
                if selected_media_keys.intersection(candidate_media_keys):
                    skip_reasons["duplicate_selected_media"] += 1
                    skipped_for_row["duplicate_selected_media"] = skipped_for_row.get("duplicate_selected_media", 0) + 1
                    continue
            selected.append(deepcopy(cand.item))
            selected_scores.append(teacher_score)
            selected_teacher_scores.append(teacher_score)
            if args.filter_duplicate_text and candidate_text:
                selected_texts.add(candidate_text)
            if args.filter_duplicate_media:
                selected_media_keys.update(candidate_media_keys)
            if student_score is not None:
                selected_student_scores.append(student_score)
            mined_negative_scores_all.append(teacher_score)
            hardness_margins.append(best_positive_score - teacher_score)
            selected_audit.append(
                {
                    "score": teacher_score,
                    "teacher_score": teacher_score,
                    "student_score": student_score,
                    "rank_score": rank_score,
                    "margin_to_best_positive": best_positive_score - teacher_score,
                    "student_margin_to_best_positive": (
                        best_student_positive_score - student_score
                        if best_student_positive_score is not None and student_score is not None
                        else None
                    ),
                    "candidate_source_row": cand.source_row,
                    "candidate_source_field": cand.source_field,
                    "candidate_source_index": cand.source_index,
                    "candidate_key": cand.key,
                    "candidate_preview": _item_preview(cand.item),
                }
            )
            if len(selected) >= args.num_hard_negatives:
                break

        if not selected:
            rows_with_zero_negatives += 1

        new_row = deepcopy(row)
        new_row["neg"] = original_negs + selected
        if args.write_scores:
            pos_items = normalize_group(new_row, "pos")
            new_row["pos_scores"] = [best_positive_score for _ in pos_items]
            new_row["neg_scores"] = [None for _ in original_negs] + selected_scores
        metadata = deepcopy(new_row.get("metadata", {})) if isinstance(new_row.get("metadata"), dict) else {}
        metadata.setdefault("hard_negative_mining", {})
        metadata["hard_negative_mining"].update(
            {
                "mode": args.mode,
                "best_positive_score": best_positive_score,
                "best_student_positive_score": best_student_positive_score,
                "mined_negative_count": len(selected),
                "top_k": args.top_k,
                "teacher_model": args.teacher_model,
                "student_model": args.student_model,
                "dry_run": args.dry_run,
                "max_negative_score_ratio_to_positive": args.max_negative_score_ratio_to_positive,
            }
        )
        new_row["metadata"] = metadata
        output_rows.append(new_row)
        kept += 1
        total_mined += len(selected)
        report_rows.append(
            {
                "row_index": row_idx,
                "sample_id": row.get("sample_id"),
                "best_positive_score": best_positive_score,
                "best_student_positive_score": best_student_positive_score,
                "mined_negative_count": len(selected),
                "mined_negative_scores": selected_scores,
                "mined_negative_teacher_scores": selected_teacher_scores,
                "mined_negative_student_scores": selected_student_scores,
                "skipped_topk_counts": skipped_for_row,
            }
        )
        if args.audit_sample_path and len(audit_rows) < max(args.audit_sample_size, 0):
            audit_rows.append(
                {
                    "row_index": row_idx,
                    "sample_id": row.get("sample_id"),
                    "query_preview": _item_preview(row_query_item(row)),
                    "positive_preview": [_item_preview(item) for item in pos_items],
                    "best_positive_score": best_positive_score,
                    "best_student_positive_score": best_student_positive_score,
                    "mined_negative_count": len(selected),
                    "mined_negatives": selected_audit,
                    "skipped_topk_counts": skipped_for_row,
                }
            )

    report = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_train_jsonl": args.input_train_jsonl,
        "candidate_train_jsonl": args.candidate_train_jsonl or args.input_train_jsonl,
        "output_train_jsonl": args.output_train_jsonl,
        "mode": args.mode,
        "teacher_model": args.teacher_model,
        "media_root": args.media_root,
        "candidate_source": args.candidate_source,
        "max_examples": args.max_examples,
        "max_candidate_examples": args.max_candidate_examples,
        "input_rows": len(rows),
        "candidate_count": len(candidates),
        "output_rows": len(output_rows),
        "kept_rows": kept,
        "dropped_no_positive": dropped_no_positive,
        "dropped_low_positive": dropped_low_positive,
        "rows_with_zero_negatives": rows_with_zero_negatives,
        "total_mined_negatives": total_mined,
        "avg_mined_negatives_per_row": total_mined / kept if kept else 0.0,
        "score_summary": {
            "best_positive_score": _score_summary(best_positive_scores),
            "mined_negative_score": _score_summary(mined_negative_scores_all),
            "positive_minus_negative_margin": _score_summary(hardness_margins),
        },
        "skip_reasons_topk": skip_reasons,
        "args": vars(args),
        "rows": report_rows[: max(args.report_row_limit, 0)],
    }
    return output_rows, report, audit_rows


def run_mining(args: MiningArguments) -> dict[str, Any]:
    source_rows = read_jsonl(args.input_train_jsonl, max_examples=args.max_examples)
    rows = resolve_rows_media_paths(source_rows, fallback_media_root=args.media_root)
    if not rows:
        raise ValueError(f"No rows loaded from {args.input_train_jsonl}")

    candidate_path = args.candidate_train_jsonl or args.input_train_jsonl
    if args.max_candidate_examples is None and args.candidate_train_jsonl is None:
        candidate_source_rows = source_rows
    else:
        candidate_max_examples = args.max_candidate_examples if args.max_candidate_examples is not None else 0
        candidate_source_rows = read_jsonl(candidate_path, max_examples=candidate_max_examples)
    candidate_rows = resolve_rows_media_paths(candidate_source_rows, fallback_media_root=args.media_root)

    # Always include the output rows in the candidate pool. This guarantees
    # that each query's positive is scoreable even when mining from a larger or
    # separate candidate JSONL subset.
    if args.candidate_source == "all_pos_plus_own_neg":
        # Global retrieval candidates remain all positives. Existing negatives
        # are added only for their own row so they compete with global positives
        # without leaking into other queries as arbitrary cross-row negatives.
        candidates = build_candidate_pool(rows + candidate_rows, candidate_source="all_pos")
        append_own_negative_candidates(candidates, rows)
    else:
        candidates = build_candidate_pool(rows + candidate_rows, candidate_source=args.candidate_source)
    if not candidates:
        raise ValueError("No candidates were built from the input JSONL.")

    candidate_count_before_extra = len(candidates)
    extra_candidate_sources: list[dict[str, Any]] = []
    candidate_corpus_max_examples = args.max_candidate_examples if args.max_candidate_examples is not None else 0
    for corpus_path in _split_csv(args.candidate_corpus_jsonl):
        corpus_items = read_candidate_corpus_jsonl(
            corpus_path,
            max_examples=candidate_corpus_max_examples,
            fallback_media_root=args.media_root,
        )
        before = len(candidates)
        added = append_global_candidate_items(
            candidates,
            corpus_items,
            source_field=f"candidate_corpus_jsonl:{os.path.basename(corpus_path)}",
        )
        extra_candidate_sources.append(
            {
                "source": "candidate_corpus_jsonl",
                "path": corpus_path,
                "raw_items": len(corpus_items),
                "added_after_dedup": added,
                "candidate_count_before": before,
                "candidate_count_after": len(candidates),
            }
        )

    mmeb_eval_items, mmeb_eval_stats = _load_mmeb_eval_corpus_items(args)
    for stat in mmeb_eval_stats:
        start = int(stat.pop("items_start"))
        end = int(stat.pop("items_end"))
        before = len(candidates)
        added = append_global_candidate_items(
            candidates,
            mmeb_eval_items[start:end],
            source_field=f"mmeb_eval_corpus:{stat['dataset']}",
        )
        stat.update(
            {
                "added_after_dedup": added,
                "candidate_count_before": before,
                "candidate_count_after": len(candidates),
            }
        )
        extra_candidate_sources.append(stat)

    local_candidate_source_rows = (
        _row_local_candidate_source_rows(candidates) if args.candidate_source == "all_pos_plus_own_neg" else None
    )

    queries = [row_query_item(row) for row in rows]
    candidate_items = [candidate.item for candidate in candidates]

    if args.mode == "mock":
        score_matrix = np.zeros((len(rows), len(candidates)), dtype=np.float32)
        for row_idx, query in enumerate(queries):
            for cand_idx, candidate in enumerate(candidate_items):
                score_matrix[row_idx, cand_idx] = deterministic_mock_score(query, candidate, args.random_seed)
    elif args.mode == "teacher_only":
        if not args.teacher_model:
            raise ValueError("--teacher_model is required for teacher_only mode.")
        query_embeddings, candidate_embeddings = encode_with_nexus_embedder(
            model_name_or_path=args.teacher_model,
            queries=queries,
            candidates=candidate_items,
            batch_size=args.batch_size,
            query_max_len=args.query_max_len,
            passage_max_len=args.passage_max_len,
            device=args.device,
            devices=args.devices,
            torch_dtype=args.torch_dtype,
            backbone_load_strategy=args.backbone_load_strategy,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
        if args.score_block_size > 0:
            score_matrix = _topk_scores_blockwise(
                query_embeddings,
                candidate_embeddings,
                top_k=args.top_k,
                block_size=args.score_block_size,
                score_device=_default_score_device(args),
                local_candidate_source_rows=local_candidate_source_rows,
            )
            score_matrix = _append_required_candidate_scores(
                score_matrix,
                rows,
                candidates,
                query_embeddings,
                candidate_embeddings,
            )
        else:
            score_matrix = cosine_scores(query_embeddings, candidate_embeddings)
    elif args.mode == "student_aware":
        if not args.teacher_model:
            raise ValueError("--teacher_model is required for student_aware mode.")
        if not args.student_model:
            raise ValueError("--student_model is required for student_aware mode.")
        teacher_query_embeddings, teacher_candidate_embeddings = encode_with_nexus_embedder(
            model_name_or_path=args.teacher_model,
            queries=queries,
            candidates=candidate_items,
            batch_size=args.batch_size,
            query_max_len=args.query_max_len,
            passage_max_len=args.passage_max_len,
            device=args.device,
            devices=args.devices,
            torch_dtype=args.torch_dtype,
            backbone_load_strategy=args.backbone_load_strategy,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
        student_query_embeddings, student_candidate_embeddings = encode_with_nexus_embedder(
            model_name_or_path=args.student_model,
            processor_name_or_path=args.student_processor_name_or_path,
            queries=queries,
            candidates=candidate_items,
            batch_size=args.batch_size,
            query_max_len=args.student_query_max_len or args.query_max_len,
            passage_max_len=args.student_passage_max_len or args.passage_max_len,
            device=args.device,
            devices=args.devices,
            torch_dtype=args.student_torch_dtype or args.torch_dtype,
            backbone_load_strategy=args.student_backbone_load_strategy or args.backbone_load_strategy,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.student_cache_dir or args.cache_dir,
        )
        if args.score_block_size > 0:
            teacher_score_matrix = _topk_scores_blockwise(
                teacher_query_embeddings,
                teacher_candidate_embeddings,
                top_k=args.top_k,
                block_size=args.score_block_size,
                score_device=_default_score_device(args),
                local_candidate_source_rows=local_candidate_source_rows,
            )
            teacher_score_matrix = _append_required_candidate_scores(
                teacher_score_matrix,
                rows,
                candidates,
                teacher_query_embeddings,
                teacher_candidate_embeddings,
            )
            student_score_matrix = _topk_scores_blockwise(
                student_query_embeddings,
                student_candidate_embeddings,
                top_k=args.top_k,
                block_size=args.score_block_size,
                score_device=_default_score_device(args),
                local_candidate_source_rows=local_candidate_source_rows,
            )
            student_score_matrix = _append_required_candidate_scores(
                student_score_matrix,
                rows,
                candidates,
                student_query_embeddings,
                student_candidate_embeddings,
            )
        else:
            teacher_score_matrix = cosine_scores(teacher_query_embeddings, teacher_candidate_embeddings)
            student_score_matrix = cosine_scores(student_query_embeddings, student_candidate_embeddings)
        score_matrix = student_score_matrix
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    if args.mode == "student_aware":
        if args.candidate_source == "all_pos_plus_own_neg" and isinstance(score_matrix, np.ndarray):
            score_matrix = _mask_disallowed_row_local_negatives(score_matrix, candidates)
        if args.candidate_source == "all_pos_plus_own_neg" and isinstance(teacher_score_matrix, np.ndarray):
            teacher_score_matrix = _mask_disallowed_row_local_negatives(teacher_score_matrix, candidates)
        if args.candidate_source == "all_pos_plus_own_neg" and isinstance(student_score_matrix, np.ndarray):
            student_score_matrix = _mask_disallowed_row_local_negatives(student_score_matrix, candidates)
        output_rows, report, audit_rows = _mine_from_score_matrix(
            rows,
            candidates,
            score_matrix,
            args,
            teacher_scores=teacher_score_matrix,
            student_scores=student_score_matrix,
        )
    else:
        if args.candidate_source == "all_pos_plus_own_neg" and isinstance(score_matrix, np.ndarray):
            score_matrix = _mask_disallowed_row_local_negatives(score_matrix, candidates)
        output_rows, report, audit_rows = _mine_from_score_matrix(rows, candidates, score_matrix, args)
    report.update(
        {
            "candidate_count_before_extra": candidate_count_before_extra,
            "candidate_count_after_extra": len(candidates),
            "candidate_corpus_jsonl": args.candidate_corpus_jsonl,
            "mmeb_eval_corpus_datasets": args.mmeb_eval_corpus_datasets,
            "extra_candidate_sources": extra_candidate_sources,
        }
    )
    if args.validate_output:
        validation = validate_mined_rows(output_rows, require_existing_media=args.require_existing_media)
        report["validation"] = validation
    else:
        validation = None
    write_jsonl(args.output_train_jsonl, output_rows)

    report_path = args.report_path or os.path.splitext(args.output_train_jsonl)[0] + ".report.json"
    manifest_path = args.manifest_path or os.path.splitext(args.output_train_jsonl)[0] + ".manifest.json"
    if args.audit_sample_path:
        write_jsonl(args.audit_sample_path, audit_rows)
        report["audit_sample_path"] = args.audit_sample_path
        report["audit_sample_rows"] = len(audit_rows)
    write_json(report_path, report)
    manifest = {k: v for k, v in report.items() if k != "rows"}
    write_json(manifest_path, manifest)
    if validation is not None and not validation["valid"]:
        raise ValueError(f"Mined output validation failed; see {report_path}")
    return report
