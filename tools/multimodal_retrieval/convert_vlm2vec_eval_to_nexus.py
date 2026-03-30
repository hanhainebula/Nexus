#!/usr/bin/env python3
"""Convert VLM2Vec pair-style evaluation data into a Nexus local retrieval dataset."""

import argparse
import base64
import io
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional


def require_datasets():
    try:
        from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise ImportError(
            "The conversion script requires `datasets`. Create an isolated environment and install Nexus dependencies there."
        ) from exc

    return Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk


def default_hf_datasets_cache() -> str:
    candidates = []
    for env_key in ["HF_DATASETS_CACHE", "HF_HOME", "XDG_CACHE_HOME"]:
        env_value = os.getenv(env_key)
        if env_value not in [None, ""]:
            expanded = os.path.expanduser(env_value)
            if env_key != "HF_DATASETS_CACHE":
                expanded = os.path.join(expanded, "huggingface", "datasets")
            candidates.append(expanded)
    candidates.append("/tmp/huggingface/datasets")

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=candidate):
                pass
            return candidate
        except OSError:
            continue
    return "/tmp/huggingface/datasets"


def find_local_data_files(directory: str) -> List[str]:
    data_files = []
    for current_root, _, file_names in os.walk(directory):
        for file_name in sorted(file_names):
            file_path = os.path.join(current_root, file_name)
            if file_name.endswith((".json", ".jsonl", ".parquet")):
                data_files.append(file_path)
    return data_files


def is_git_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as input_file:
            header = input_file.read(128)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def filter_lfs_pointer_files(data_files: List[str], source_description: str) -> List[str]:
    usable_files = [file_path for file_path in data_files if not is_git_lfs_pointer(file_path)]
    if len(usable_files) == 0:
        raise ValueError(
            f"All local data files under {source_description} are Git LFS pointers. "
            "Download the actual files first with `git lfs pull` or `prepare_public_data.py`."
        )
    return usable_files


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="HF dataset name, local dataset path, or a json/jsonl/parquet file.")
    parser.add_argument("--output-dir", required=True, help="Output directory containing corpus/queries/qrels jsonl files.")
    parser.add_argument("--subset", default=None, help="Optional HF subset name.")
    parser.add_argument("--split", default="test", help="Dataset split.")
    parser.add_argument("--cache-dir", default=None, help="Datasets cache directory. Defaults to a writable local path.")
    parser.add_argument("--dataset-name", default=None, help="Optional dataset name stored in metadata.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap.")
    parser.add_argument("--media-root", default=None, help="Default media root written to `dataset_meta.json`.")
    parser.add_argument("--image-root", default=None, help="Image root written to `dataset_meta.json`.")
    parser.add_argument("--video-root", default=None, help="Video root written to `dataset_meta.json`.")
    parser.add_argument(
        "--sequence-mode",
        default="auto",
        choices=["auto", "image", "video"],
        help="How to interpret multi-frame candidate/query media.",
    )
    parser.add_argument("--query-prefix", default="q", help="Prefix used for generated query ids.")
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "pair_style", "beir_configs"],
        help="How to interpret the input. `beir_configs` expects separate HF configs such as queries/corpus/qrels.",
    )
    parser.add_argument("--queries-config", default="queries", help="Queries config used by `beir_configs` mode.")
    parser.add_argument("--corpus-config", default="corpus", help="Corpus config used by `beir_configs` mode.")
    parser.add_argument("--qrels-config", default="qrels", help="Qrels config used by `beir_configs` mode.")
    return parser.parse_args()


def load_any_dataset(path_or_name: str, subset: Optional[str], split: str, cache_dir: Optional[str] = None):
    Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk = require_datasets()
    if cache_dir is None:
        cache_dir = default_hf_datasets_cache()

    if os.path.isdir(path_or_name) and os.path.exists(os.path.join(path_or_name, "dataset_info.json")):
        dataset = load_from_disk(path_or_name)
    elif os.path.isdir(path_or_name):
        data_files = filter_lfs_pointer_files(find_local_data_files(path_or_name), path_or_name)
        if len(data_files) == 0:
            raise ValueError(f"No supported local dataset files were found in directory: {path_or_name}")
        extensions = {os.path.splitext(file_path)[1].lower() for file_path in data_files}
        if extensions.issubset({".json", ".jsonl"}):
            dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
        elif extensions == {".parquet"}:
            dataset = load_dataset("parquet", data_files=data_files, cache_dir=cache_dir)
        else:
            raise ValueError(
                f"Mixed local dataset file types are not supported in {path_or_name}. Found: {sorted(extensions)}"
            )
    elif os.path.isfile(path_or_name):
        if is_git_lfs_pointer(path_or_name):
            raise ValueError(
                f"Local file {path_or_name} is a Git LFS pointer. Download the actual file first."
            )
        extension = os.path.splitext(path_or_name)[1].lower()
        if extension in [".json", ".jsonl"]:
            dataset = load_dataset("json", data_files=path_or_name, cache_dir=cache_dir)
        elif extension == ".parquet":
            dataset = load_dataset("parquet", data_files=path_or_name, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported local file type: {extension}")
    else:
        dataset = load_dataset(path_or_name, subset, cache_dir=cache_dir)

    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        if split not in dataset:
            available_splits = list(dataset.keys())
            if len(available_splits) == 1:
                split = available_splits[0]
            else:
                raise ValueError(f"Split `{split}` is not available. Found: {available_splits}")
        dataset = dataset[split]
    elif subset is not None and not isinstance(dataset, (Dataset, IterableDataset)):
        raise ValueError(f"Unsupported dataset object returned from {path_or_name}: {type(dataset)}")

    return dataset


def load_beir_eval_components(
    path_or_name: str,
    queries_config: str,
    corpus_config: str,
    qrels_config: str,
    split: str,
    cache_dir: Optional[str] = None,
):
    Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset, load_from_disk = require_datasets()
    del Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_from_disk
    if cache_dir is None:
        cache_dir = default_hf_datasets_cache()

    return (
        load_dataset(path_or_name, queries_config, split=split, cache_dir=cache_dir),
        load_dataset(path_or_name, corpus_config, split=split, cache_dir=cache_dir),
        load_dataset(path_or_name, qrels_config, split=split, cache_dir=cache_dir),
    )


def iter_rows(dataset, max_rows: Optional[int] = None):
    for index, row in enumerate(dataset):
        if max_rows is not None and index >= max_rows:
            break
        yield index, row


def maybe_b64_encode(value: Any) -> Optional[str]:
    if value in [None, b"", ""]:
        return None
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("utf-8")
    return None


def serialize_entry_from_path_or_bytes(path: Any, image_bytes: Any) -> Optional[Any]:
    if path not in [None, ""] and image_bytes in [None, b""]:
        return path
    if path in [None, ""] and image_bytes in [None, b""]:
        return None

    payload = {}
    if path not in [None, ""]:
        payload["path"] = path
    b64_value = maybe_b64_encode(image_bytes)
    if b64_value is not None:
        payload["b64"] = b64_value
    return payload or None


def extract_media_entries(media_struct: Any) -> List[Any]:
    if media_struct in [None, "", []]:
        return []

    if isinstance(media_struct, list):
        merged_entries = []
        for element in media_struct:
            merged_entries.extend(extract_media_entries(element))
        return merged_entries

    if isinstance(media_struct, str):
        return [media_struct]

    if isinstance(media_struct, dict):
        if "frames" in media_struct and media_struct["frames"] not in [None, "", []]:
            merged_entries = []
            for frame in media_struct["frames"]:
                if isinstance(frame, str):
                    merged_entries.append(frame)
                elif isinstance(frame, dict):
                    payload = {}
                    if frame.get("path") not in [None, ""]:
                        payload["path"] = frame["path"]
                    b64_value = maybe_b64_encode(frame.get("bytes"))
                    if b64_value is not None:
                        payload["b64"] = b64_value
                    if payload:
                        merged_entries.append(payload)
            return merged_entries

        paths = media_struct.get("paths") or []
        bytes_values = media_struct.get("bytes") or []
        if not isinstance(paths, list):
            paths = [paths]
        if not isinstance(bytes_values, list):
            bytes_values = [bytes_values]

        count = max(len(paths), len(bytes_values))
        entries = []
        for index in range(count):
            entry = serialize_entry_from_path_or_bytes(
                paths[index] if index < len(paths) else None,
                bytes_values[index] if index < len(bytes_values) else None,
            )
            if entry is not None:
                entries.append(entry)
        return entries

    pil_image_path = getattr(media_struct, "filename", None)
    if pil_image_path not in [None, ""]:
        return [pil_image_path]
    return []


def resolve_sequence_mode(args, row: Optional[Dict[str, Any]] = None) -> str:
    if args.sequence_mode != "auto":
        return args.sequence_mode

    hints = [args.dataset_name or ""]
    if isinstance(row, dict):
        dataset_infos = row.get("dataset_infos") or {}
        hints.extend(
            [
                dataset_infos.get("dataset_name", ""),
                dataset_infos.get("global_dataset_name", ""),
            ]
        )

    lowered = " ".join(hints).lower()
    if any(token in lowered for token in ["video", "msrvtt", "msvd", "youcook", "vatex", "charades", "moment"]):
        return "video"
    return "image"


def attach_media(item: Dict[str, Any], media_entries: List[Any], sequence_mode: str):
    valid_entries = [entry for entry in media_entries if entry not in [None, "", [], {}]]
    if len(valid_entries) == 0:
        return

    if sequence_mode == "video":
        if all(isinstance(entry, str) for entry in valid_entries):
            item["video_frames"] = valid_entries
        else:
            item["videos"] = [{"frames": valid_entries}]
        return

    if len(valid_entries) == 1:
        entry = valid_entries[0]
        if isinstance(entry, str):
            item["image_path"] = entry
        else:
            item["image"] = entry
        return

    item["images"] = valid_entries


def first_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(text for text in value if isinstance(text, str) and text.strip())
    return value if isinstance(value, str) else ""


def clean_instruction_text(value: Any) -> str:
    text = first_text_value(value)
    if text == "":
        return ""
    return text.replace("<|image_1|>", "").replace(" \n", "\n").strip()


def combine_instruction_and_text(instruction: Any, text: Any) -> str:
    parts = [clean_instruction_text(instruction), first_text_value(text).strip()]
    return " ".join(part for part in parts if part).strip()


def serialize_image_like(value: Any) -> Optional[Any]:
    if value in [None, "", []]:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return {"b64": base64.b64encode(value).decode("utf-8")}

    image_save = getattr(value, "save", None)
    if callable(image_save):
        buffer = io.BytesIO()
        image_format = getattr(value, "format", None) or "PNG"
        value.save(buffer, format=image_format)
        return {"b64": base64.b64encode(buffer.getvalue()).decode("utf-8")}

    return None


def build_eval_item(text_value: Any, media_value: Any, sequence_mode: str) -> Dict[str, Any]:
    item = {}
    text = first_text_value(text_value)
    if text not in [None, ""]:
        item["text"] = text
    attach_media(item, extract_media_entries(media_value), sequence_mode=sequence_mode)
    return item


def row_uses_instruction_style_schema(row: Dict[str, Any]) -> bool:
    return any(
        key in row
        for key in [
            "qry_inst",
            "qry_img_path",
            "tgt_inst",
            "tgt_img_path",
        ]
    )


def row_uses_legacy_pair_style_schema(row: Dict[str, Any]) -> bool:
    return any(
        key in row
        for key in [
            "qry",
            "qry_image_path",
            "qry_video_path",
            "pos_text",
            "pos_image_path",
            "neg_text",
            "neg_image_path",
        ]
    )


def build_instruction_style_query(row: Dict[str, Any], sequence_mode: str) -> Dict[str, Any]:
    item = {}
    query_text = combine_instruction_and_text(row.get("qry_inst"), row.get("qry_text"))
    if query_text not in [None, ""]:
        item["text"] = query_text

    if row.get("qry_img_path") not in [None, "", []]:
        attach_media(item, [row.get("qry_img_path")], sequence_mode=sequence_mode)
    elif row.get("qry_video_path") not in [None, "", []]:
        attach_media(item, [row.get("qry_video_path")], sequence_mode="video")
    return item


def build_instruction_style_candidates(row: Dict[str, Any], sequence_mode: str) -> List[Dict[str, Any]]:
    target_texts = row.get("tgt_text") or []
    target_image_paths = row.get("tgt_img_path") or []
    if not isinstance(target_texts, list):
        target_texts = [target_texts]
    if not isinstance(target_image_paths, list):
        target_image_paths = [target_image_paths]

    candidate_count = max(len(target_texts), len(target_image_paths))
    if candidate_count == 0:
        return []

    candidates = []
    for index in range(candidate_count):
        caption = target_texts[index] if index < len(target_texts) else None
        image_path = target_image_paths[index] if index < len(target_image_paths) else None
        candidate = {}

        candidate_text = combine_instruction_and_text(row.get("tgt_inst"), caption)
        if candidate_text not in [None, ""]:
            candidate["text"] = candidate_text
        if image_path not in [None, "", []]:
            attach_media(candidate, [image_path], sequence_mode=sequence_mode)

        name_parts = []
        if image_path not in [None, "", []]:
            name_parts.append(str(image_path))
        clean_caption = first_text_value(caption).strip()
        if clean_caption not in [None, ""]:
            name_parts.append(clean_caption)
        candidate["_name"] = ":".join(name_parts) if len(name_parts) > 1 else (name_parts[0] if name_parts else f"cand_{index}")
        candidates.append(candidate)
    return candidates


def build_legacy_pair_style_query(row: Dict[str, Any], sequence_mode: str) -> Dict[str, Any]:
    item = {}
    query_text = clean_instruction_text(row.get("qry"))
    if query_text not in [None, ""]:
        item["text"] = query_text
    if row.get("qry_image_path") not in [None, "", []]:
        attach_media(item, [row.get("qry_image_path")], sequence_mode=sequence_mode)
    elif row.get("qry_video_path") not in [None, "", []]:
        attach_media(item, [row.get("qry_video_path")], sequence_mode="video")
    return item


def build_legacy_pair_style_candidates(row: Dict[str, Any], sequence_mode: str) -> List[Dict[str, Any]]:
    candidates = []
    for prefix in ["pos", "neg"]:
        text_value = row.get(f"{prefix}_text")
        image_path = row.get(f"{prefix}_image_path")
        if text_value in [None, ""] and image_path in [None, "", []]:
            continue

        candidate = {}
        candidate_text = first_text_value(text_value).strip()
        if candidate_text not in [None, ""]:
            candidate["text"] = candidate_text
        if image_path not in [None, "", []]:
            attach_media(candidate, [image_path], sequence_mode=sequence_mode)

        name_parts = []
        if image_path not in [None, "", []]:
            name_parts.append(str(image_path))
        if candidate_text not in [None, ""]:
            name_parts.append(candidate_text)
        candidate["_name"] = ":".join(name_parts) if len(name_parts) > 1 else (name_parts[0] if name_parts else prefix)
        candidates.append(candidate)
    return candidates


def convert_legacy_pair_style_row(
    row_index: int,
    row: Dict[str, Any],
    args,
):
    sequence_mode = resolve_sequence_mode(args, row=row)
    query_record = {"_id": f"{args.query_prefix}{row_index}"}
    query_record.update(build_legacy_pair_style_query(row, sequence_mode=sequence_mode))

    candidates = build_legacy_pair_style_candidates(row, sequence_mode=sequence_mode)
    corpus_records = []
    qrels = []
    for candidate_index, candidate in enumerate(candidates):
        corpus_id = candidate.pop("_name")
        corpus_record = {"_id": corpus_id}
        corpus_record.update(candidate)
        corpus_records.append(corpus_record)
        if candidate_index == 0:
            qrels.append({"query-id": query_record["_id"], "corpus-id": corpus_id, "score": 1})
    return query_record, corpus_records, qrels


def convert_instruction_style_row(
    row_index: int,
    row: Dict[str, Any],
    args,
):
    sequence_mode = resolve_sequence_mode(args, row=row)
    query_record = {"_id": f"{args.query_prefix}{row_index}"}
    query_record.update(build_instruction_style_query(row, sequence_mode=sequence_mode))

    candidates = build_instruction_style_candidates(row, sequence_mode=sequence_mode)
    corpus_records = []
    qrels = []
    for candidate_index, candidate in enumerate(candidates):
        corpus_id = candidate.pop("_name")
        corpus_record = {"_id": corpus_id}
        corpus_record.update(candidate)
        corpus_records.append(corpus_record)
        if candidate_index == 0:
            qrels.append({"query-id": query_record["_id"], "corpus-id": corpus_id, "score": 1})
    return query_record, corpus_records, qrels


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_beir_eval_configs(args):
    queries_dataset, corpus_dataset, qrels_dataset = load_beir_eval_components(
        path_or_name=args.input,
        queries_config=args.queries_config,
        corpus_config=args.corpus_config,
        qrels_config=args.qrels_config,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    selected_query_ids = None
    if args.max_rows is not None:
        queries_dataset = queries_dataset.select(range(min(args.max_rows, len(queries_dataset))))
        selected_query_ids = {str(row["query-id"]) for row in queries_dataset}

    filtered_qrels = []
    relevant_corpus_ids = set()
    for row in qrels_dataset:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        if selected_query_ids is not None and query_id not in selected_query_ids:
            continue
        filtered_qrels.append({"query-id": query_id, "corpus-id": corpus_id, "score": int(row["score"])})
        relevant_corpus_ids.add(corpus_id)

    queries = []
    for row in queries_dataset:
        queries.append({"_id": str(row["query-id"]), "text": first_text_value(row.get("query"))})

    corpus_rows = []
    for row in corpus_dataset:
        corpus_id = str(row["corpus-id"])
        if relevant_corpus_ids and corpus_id not in relevant_corpus_ids:
            continue
        corpus_record = {"_id": corpus_id}
        if first_text_value(row.get("text")) not in [None, ""]:
            corpus_record["text"] = first_text_value(row.get("text"))
        image_payload = serialize_image_like(row.get("image"))
        if image_payload is not None:
            if isinstance(image_payload, str):
                corpus_record["image_path"] = image_payload
            else:
                corpus_record["image"] = image_payload
        corpus_rows.append(corpus_record)

    split_name = args.split
    os.makedirs(args.output_dir, exist_ok=True)
    write_jsonl(os.path.join(args.output_dir, "corpus.jsonl"), corpus_rows)
    write_jsonl(os.path.join(args.output_dir, f"{split_name}_queries.jsonl"), queries)
    write_jsonl(os.path.join(args.output_dir, f"{split_name}_qrels.jsonl"), filtered_qrels)

    metadata = {
        "dataset_name": args.dataset_name,
        "source_input": args.input,
        "subset": args.subset,
        "split": args.split,
        "media_root": args.media_root,
        "image_root": args.image_root,
        "video_root": args.video_root,
        "num_queries": len(queries),
        "num_corpus": len(corpus_rows),
        "num_qrels": len(filtered_qrels),
        "input_format": "beir_configs",
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(queries)} queries, {len(corpus_rows)} corpus rows, and {len(filtered_qrels)} qrels "
        f"to {args.output_dir}"
    )


def main():
    args = parse_args()
    if args.input_format == "beir_configs":
        convert_beir_eval_configs(args)
        return

    dataset = load_any_dataset(args.input, subset=args.subset, split=args.split, cache_dir=args.cache_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    corpus: Dict[str, Dict[str, Any]] = {}
    queries: List[Dict[str, Any]] = []
    qrels: List[Dict[str, Any]] = []

    for row_index, row in iter_rows(dataset, max_rows=args.max_rows):
        if row_uses_instruction_style_schema(row):
            query_record, corpus_records, row_qrels = convert_instruction_style_row(row_index, row, args)
            queries.append(query_record)
            for corpus_record in corpus_records:
                corpus[corpus_record["_id"]] = corpus_record
            qrels.extend(row_qrels)
            continue
        if row_uses_legacy_pair_style_schema(row):
            query_record, corpus_records, row_qrels = convert_legacy_pair_style_row(row_index, row, args)
            queries.append(query_record)
            for corpus_record in corpus_records:
                corpus[corpus_record["_id"]] = corpus_record
            qrels.extend(row_qrels)
            continue

        sequence_mode = resolve_sequence_mode(args, row=row)
        query_item = build_eval_item(row.get("query_text"), row.get("query_image"), sequence_mode=sequence_mode)
        query_id = f"{args.query_prefix}{row_index}"
        query_record = {"_id": query_id}
        query_record.update(query_item)
        queries.append(query_record)

        dataset_infos = row.get("dataset_infos") or {}
        cand_names = dataset_infos.get("cand_names") or []
        cand_texts = row.get("cand_text") or []
        cand_images = row.get("cand_image") or []

        if not isinstance(cand_names, list):
            cand_names = [cand_names]
        if not isinstance(cand_texts, list):
            cand_texts = [cand_texts]
        if not isinstance(cand_images, list):
            cand_images = [cand_images]

        for index, cand_name in enumerate(cand_names):
            if cand_name in [None, ""]:
                continue
            if cand_name not in corpus:
                corpus_item = build_eval_item(
                    cand_texts[index] if index < len(cand_texts) else None,
                    cand_images[index] if index < len(cand_images) else None,
                    sequence_mode=sequence_mode,
                )
                corpus_record = {"_id": cand_name}
                corpus_record.update(corpus_item)
                corpus[cand_name] = corpus_record

        label_names = dataset_infos.get("label_name") or []
        if isinstance(label_names, str):
            label_names = [label_names]
        for label_name in label_names:
            if label_name in [None, ""]:
                continue
            qrels.append({"query-id": query_id, "corpus-id": label_name, "score": 1})

    split_name = args.split
    write_jsonl(os.path.join(args.output_dir, "corpus.jsonl"), corpus.values())
    write_jsonl(os.path.join(args.output_dir, f"{split_name}_queries.jsonl"), queries)
    write_jsonl(os.path.join(args.output_dir, f"{split_name}_qrels.jsonl"), qrels)

    metadata = {
        "dataset_name": args.dataset_name,
        "source_input": args.input,
        "subset": args.subset,
        "split": args.split,
        "media_root": args.media_root,
        "image_root": args.image_root,
        "video_root": args.video_root,
        "num_queries": len(queries),
        "num_corpus": len(corpus),
        "num_qrels": len(qrels),
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(queries)} queries, {len(corpus)} corpus rows, and {len(qrels)} qrels to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
