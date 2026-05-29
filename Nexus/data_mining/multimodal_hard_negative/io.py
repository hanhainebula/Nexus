from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Candidate:
    key: str
    item: dict[str, Any]
    source_row: int
    source_field: str
    source_index: int


def read_jsonl(path: str, max_examples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_examples > 0 and len(rows) >= max_examples:
                break
    return rows


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _is_remote_or_special_path(path: str) -> bool:
    return path.startswith(("http://", "https://", "s3://", "oss://", "file://", "data:"))


def _resolve_path(path: str, roots: list[str]) -> str:
    if not path or os.path.isabs(path) or _is_remote_or_special_path(path):
        return path
    for root in roots:
        if root:
            return os.path.abspath(os.path.join(root, path))
    return path


def _row_roots(row: dict[str, Any], fallback_media_root: str | None = None) -> dict[str, list[str]]:
    generic_roots = [str(row.get("media_root"))] if row.get("media_root") else []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if metadata.get("media_root"):
        generic_roots.append(str(metadata["media_root"]))
    if fallback_media_root:
        generic_roots.append(fallback_media_root)

    roots: dict[str, list[str]] = {}
    for modality in ["image", "video", "audio"]:
        modality_roots: list[str] = []
        field_name = f"{modality}_root"
        if row.get(field_name):
            modality_roots.append(str(row[field_name]))
        if metadata.get(field_name):
            modality_roots.append(str(metadata[field_name]))
        modality_roots.extend(generic_roots)
        roots[modality] = modality_roots
    roots["generic"] = generic_roots
    return roots


def resolve_item_media_paths(item: dict[str, Any], roots: dict[str, list[str]]) -> dict[str, Any]:
    resolved = deepcopy(item)
    scalar_fields = {
        "image_path": "image",
        "image": "image",
        "video_path": "video",
        "video": "video",
        "audio_path": "audio",
    }
    list_fields = {
        "image_paths": "image",
        "video_frames": "image",
    }
    for field, modality in scalar_fields.items():
        value = resolved.get(field)
        if isinstance(value, str) and value:
            resolved[field] = _resolve_path(value, roots.get(modality, []) + roots.get("generic", []))
    for field, modality in list_fields.items():
        value = resolved.get(field)
        if isinstance(value, list):
            resolved[field] = [
                _resolve_path(x, roots.get(modality, []) + roots.get("generic", [])) if isinstance(x, str) else x
                for x in value
            ]
    return resolved


def resolve_row_media_paths(row: dict[str, Any], fallback_media_root: str | None = None) -> dict[str, Any]:
    resolved = deepcopy(row)
    roots = _row_roots(row, fallback_media_root=fallback_media_root)
    if isinstance(resolved.get("query"), dict):
        resolved["query"] = resolve_item_media_paths(resolved["query"], roots)
    for field in ["pos", "neg"]:
        value = resolved.get(field)
        if isinstance(value, dict):
            resolved[field] = resolve_item_media_paths(value, roots)
        elif isinstance(value, list):
            resolved[field] = [resolve_item_media_paths(x, roots) if isinstance(x, dict) else x for x in value]
    metadata = deepcopy(resolved.get("metadata", {})) if isinstance(resolved.get("metadata"), dict) else {}
    metadata.setdefault("path_resolution", {})
    metadata["path_resolution"].update(
        {
            "resolved_for_hard_negative_mining": True,
            "fallback_media_root": fallback_media_root,
            "row_image_root": row.get("image_root"),
            "row_video_root": row.get("video_root"),
            "row_audio_root": row.get("audio_root"),
        }
    )
    resolved["metadata"] = metadata
    return resolved


def resolve_rows_media_paths(rows: list[dict[str, Any]], fallback_media_root: str | None = None) -> list[dict[str, Any]]:
    return [resolve_row_media_paths(row, fallback_media_root=fallback_media_root) for row in rows]


def normalize_candidate_item(item: dict[str, Any]) -> dict[str, Any]:
    """Convert common corpus/eval row formats to a Nexus training candidate item.

    MMEB training JSONL usually stores candidates as `text` + `image_path` or
    `video_path`, while MMEB evaluation parsers commonly emit `image`/`video`
    and M-BEIR candidate pools use `txt`/`img_path`.  Hard-negative output is a
    training JSONL, so this function canonicalizes those aliases before they
    enter the candidate pool.
    """

    normalized: dict[str, Any] = {}
    metadata = deepcopy(item.get("metadata", {})) if isinstance(item.get("metadata"), dict) else {}

    text_parts = []
    title = item.get("title")
    if isinstance(title, str) and title.strip():
        text_parts.append(title.strip())
    for field in ["text", "txt", "caption", "body"]:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            text_parts.append(value.strip())
            break
    if text_parts:
        normalized["text"] = " ".join(text_parts).strip()

    for source, target in [
        ("image_path", "image_path"),
        ("img_path", "image_path"),
        ("image", "image_path"),
        ("video_path", "video_path"),
        ("video", "video_path"),
        ("audio_path", "audio_path"),
        ("audio", "audio_path"),
    ]:
        value = item.get(source)
        if isinstance(value, str) and value:
            normalized[target] = value

    for source, target in [
        ("image_paths", "image_paths"),
        ("images", "image_paths"),
        ("pages", "image_paths"),
        ("video", "video_frames"),
        ("video_frames", "video_frames"),
    ]:
        value = item.get(source)
        if isinstance(value, list) and value:
            normalized[target] = value

    # Preserve useful lightweight provenance without putting parser-specific
    # fields directly into the training item.
    for field in ["did", "corpus_id", "corpus-id", "cand_name", "modality", "src_content"]:
        if item.get(field) not in [None, ""]:
            metadata[field.replace("-", "_")] = item[field]
    if metadata:
        normalized["metadata"] = metadata
    return normalized or deepcopy(item)


def _candidate_items_from_cand_input(value: Any) -> list[dict[str, Any]]:
    if value in [None, "", []]:
        return []
    if isinstance(value, dict):
        return [normalize_candidate_item(value)]
    if isinstance(value, list):
        items: list[dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, dict):
                items.append(normalize_candidate_item(entry))
        return items
    return []


def row_candidate_items(row: dict[str, Any], include_fields: list[str] | None = None) -> list[dict[str, Any]]:
    """Extract candidate documents from train-style, eval-style, or corpus rows."""

    items: list[dict[str, Any]] = []
    if "cand_input" in row:
        items.extend(_candidate_items_from_cand_input(row.get("cand_input")))
    fields = include_fields if include_fields is not None else ["pos", "neg"]
    for field in fields:
        for item in normalize_group(row, field):
            items.append(normalize_candidate_item(item))
    if not items:
        # Single corpus item row, e.g. M-BEIR cand_pool with txt/img_path.
        candidate = normalize_candidate_item(row)
        if item_key(candidate):
            items.append(candidate)
    return items


def read_candidate_corpus_jsonl(
    path: str,
    max_examples: int = 0,
    fallback_media_root: str | None = None,
) -> list[dict[str, Any]]:
    """Read extra candidate corpus JSONL and return canonical candidate items."""

    items: list[dict[str, Any]] = []
    rows = read_jsonl(path, max_examples=max_examples)
    for row in rows:
        roots = _row_roots(row, fallback_media_root=fallback_media_root)
        for item in row_candidate_items(row):
            items.append(resolve_item_media_paths(item, roots))
    return items


def item_key(item: dict[str, Any]) -> str:
    parts = []
    for field in ["image_path", "image", "video_path", "video", "audio_path", "audio", "text"]:
        value = item.get(field)
        if value:
            parts.append(f"{field}={value}")
    frames = item.get("video_frames") or item.get("image_paths")
    if frames:
        if isinstance(frames, list):
            parts.append("frames=" + "|".join(map(str, frames[:4])) + f"|n={len(frames)}")
        else:
            parts.append(f"frames={frames}")
    if not parts:
        parts.append(json.dumps(item, sort_keys=True, ensure_ascii=False))
    return "\u241f".join(parts)


def item_media_paths(item: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for field in ["image_path", "image", "video_path", "video", "audio_path"]:
        value = item.get(field)
        if isinstance(value, str) and value and not _is_remote_or_special_path(value):
            paths.append(value)
    for field in ["image_paths", "video_frames"]:
        value = item.get(field)
        if isinstance(value, list):
            paths.extend(x for x in value if isinstance(x, str) and x and not _is_remote_or_special_path(x))
    return paths


def item_modality_signature(item: dict[str, Any]) -> tuple[str, ...]:
    """Return a coarse modality signature used to avoid cross-task negatives.

    A mixed dataset such as Fashion200K may contain text-to-image and
    image-to-text rows in the same JSONL. Mining negatives across those rows can
    produce invalid candidates, e.g. a text-only negative for an image target.
    The signature is intentionally coarse and only checks which media/text
    fields are populated.
    """
    signature = []
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        signature.append("text")
    if item.get("image_path") or item.get("image"):
        signature.append("image")
    if item.get("image_paths"):
        signature.append("images")
    if item.get("video_path") or item.get("video"):
        signature.append("video")
    if item.get("video_frames"):
        signature.append("video_frames")
    if item.get("audio_path"):
        signature.append("audio")
    return tuple(signature) or ("empty",)


def normalize_group(row: dict[str, Any], field: str) -> list[dict[str, Any]]:
    value = row.get(field, [])
    if value is None:
        return []
    if isinstance(value, dict):
        return [deepcopy(value)]
    if isinstance(value, list):
        return [deepcopy(x) for x in value if isinstance(x, dict)]
    return []


def append_own_negative_candidates(candidates: list[Candidate], rows: list[dict[str, Any]]) -> list[list[int]]:
    """Append row-local original negatives and return their candidate indices.

    The `all_pos_plus_own_neg` mode is intentionally row-local: a negative
    originally attached to row A must not become a candidate for row B. We keep
    the real item key so duplicate/positive filtering still works, but we allow
    the same item to appear as separate row-local candidate instances when it is
    provided by multiple rows.
    """

    global_positive_keys = {cand.key for cand in candidates}
    row_local_candidate_indices: list[list[int]] = []
    for row_idx, row in enumerate(rows):
        local_indices: list[int] = []
        local_seen: set[str] = set()
        for item_idx, item in enumerate(normalize_group(row, "neg")):
            item = normalize_candidate_item(item)
            key = item_key(item)
            if key in global_positive_keys or key in local_seen:
                continue
            local_seen.add(key)
            local_indices.append(len(candidates))
            candidates.append(
                Candidate(
                    key=key,
                    item=item,
                    source_row=row_idx,
                    source_field="neg",
                    source_index=item_idx,
                )
            )
        row_local_candidate_indices.append(local_indices)
    return row_local_candidate_indices


def build_candidate_pool(rows: list[dict[str, Any]], candidate_source: str = "all_pos") -> list[Candidate]:
    seen: set[str] = set()
    candidates: list[Candidate] = []
    if candidate_source == "all_pos_plus_own_neg":
        candidates = build_candidate_pool(rows, candidate_source="all_pos")
        append_own_negative_candidates(candidates, rows)
        return candidates
    fields = ["pos"]
    if candidate_source in {"all_pos_and_existing_neg", "train_all_docs"}:
        fields.append("neg")
    for row_idx, row in enumerate(rows):
        for field in fields:
            for item_idx, item in enumerate(normalize_group(row, field)):
                item = normalize_candidate_item(item)
                key = item_key(item)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    Candidate(
                        key=key,
                        item=item,
                        source_row=row_idx,
                        source_field=field,
                        source_index=item_idx,
                    )
                )
    return candidates


def append_global_candidate_items(
    candidates: list[Candidate],
    items: list[dict[str, Any]],
    source_field: str = "corpus",
) -> int:
    """Append deduplicated global candidate documents to an existing pool."""

    seen = {cand.key for cand in candidates}
    added = 0
    for item_idx, item in enumerate(items):
        item = normalize_candidate_item(item)
        key = item_key(item)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            Candidate(
                key=key,
                item=item,
                source_row=-1,
                source_field=source_field,
                source_index=item_idx,
            )
        )
        added += 1
    return added


def row_positive_keys(row: dict[str, Any]) -> set[str]:
    return {item_key(normalize_candidate_item(item)) for item in normalize_group(row, "pos")}


def row_positive_signatures(row: dict[str, Any]) -> set[tuple[str, ...]]:
    return {item_modality_signature(normalize_candidate_item(item)) for item in normalize_group(row, "pos")}


def row_query_item(row: dict[str, Any]) -> dict[str, Any]:
    query = row.get("query")
    if not isinstance(query, dict):
        raise ValueError("Each row must contain a dict `query` field.")
    return deepcopy(query)


def validate_mined_rows(rows: list[dict[str, Any]], require_existing_media: bool = True) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    total_negatives = 0
    rows_with_metadata = 0
    rows_with_zero_negatives = 0
    missing_media_paths: list[str] = []

    for row_idx, row in enumerate(rows):
        if not isinstance(row.get("query"), dict):
            errors.append({"row_index": row_idx, "error": "missing_query_dict"})
            continue
        pos_items = normalize_group(row, "pos")
        neg_items = normalize_group(row, "neg")
        if not pos_items:
            errors.append({"row_index": row_idx, "error": "missing_positive"})
            continue
        positive_keys = {item_key(normalize_candidate_item(item)) for item in pos_items}
        positive_signatures = {item_modality_signature(normalize_candidate_item(item)) for item in pos_items}
        if not neg_items:
            rows_with_zero_negatives += 1
        for neg in neg_items:
            total_negatives += 1
            neg_key = item_key(normalize_candidate_item(neg))
            neg_signature = item_modality_signature(normalize_candidate_item(neg))
            if neg_key in positive_keys:
                errors.append({"row_index": row_idx, "error": "negative_equals_positive", "key": neg_key})
            if neg_signature not in positive_signatures:
                errors.append(
                    {
                        "row_index": row_idx,
                        "error": "negative_modality_mismatch",
                        "negative_signature": list(neg_signature),
                        "positive_signatures": [list(x) for x in sorted(positive_signatures)],
                    }
                )
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        if metadata.get("hard_negative_mining"):
            rows_with_metadata += 1
        if require_existing_media:
            items_to_check = [row["query"], *pos_items, *neg_items]
            for item in items_to_check:
                for path in item_media_paths(item):
                    if not os.path.isabs(path):
                        errors.append({"row_index": row_idx, "error": "relative_media_path", "path": path})
                    elif not os.path.exists(path):
                        missing_media_paths.append(path)
                        if len(missing_media_paths) <= 20:
                            errors.append({"row_index": row_idx, "error": "missing_media_path", "path": path})

    return {
        "valid": not errors,
        "row_count": len(rows),
        "total_negatives": total_negatives,
        "rows_with_zero_negatives": rows_with_zero_negatives,
        "rows_with_hard_negative_metadata": rows_with_metadata,
        "require_existing_media": require_existing_media,
        "missing_media_count": len(missing_media_paths),
        "error_count": len(errors),
        "errors": errors[:100],
    }
