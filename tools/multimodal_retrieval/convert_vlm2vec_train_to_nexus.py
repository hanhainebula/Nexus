#!/usr/bin/env python3
"""Convert MMEB/VLM2Vec-style training data into Nexus multimodal retrieval JSONL."""

import argparse
import base64
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional


VRET_QRY_PROMPT = "Find a video that contains the following visual content: "
VRET_TGT_PROMPT = "Understand the content of the provided video."


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
    parser.add_argument("--output", required=True, help="Output Nexus training jsonl path.")
    parser.add_argument(
        "--source-format",
        default="auto",
        choices=[
            "auto",
            "vlm2vec_pairs",
            "mmeb_train",
            "vidore",
            "visrag",
            "llavahound_caption",
            "llavahound_qa",
        ],
        help="Input schema to convert.",
    )
    parser.add_argument("--subset", default=None, help="Optional HF subset name.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--cache-dir", default=None, help="Datasets cache directory. Defaults to a writable local path.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional maximum number of rows to export.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name stored in output metadata.")
    parser.add_argument("--global-dataset-name", default=None, help="Optional global dataset name stored in metadata.")
    parser.add_argument("--media-root", default=None, help="Default media root stored into each Nexus row.")
    parser.add_argument("--image-root", default=None, help="Image root stored into each Nexus row.")
    parser.add_argument("--video-root", default=None, help="Video root stored into each Nexus row.")
    parser.add_argument(
        "--sequence-mode",
        default="auto",
        choices=["auto", "image", "video"],
        help="How to interpret multi-frame media structures from pair-style datasets.",
    )
    parser.add_argument(
        "--llavahound-mode",
        default="caption_retrieval",
        choices=["caption_retrieval", "video_retrieval", "qa"],
        help="How to interpret LLaVA-Hound style conversations.",
    )
    parser.add_argument(
        "--raw-video",
        action="store_true",
        help="Prefer `video_path` over `video_frames` for LLaVA-Hound style data.",
    )
    parser.add_argument(
        "--frame-basedir",
        default=None,
        help="Directory containing pre-extracted video frames under `<frame_basedir>/<video_id>/...`.",
    )
    return parser.parse_args()


def detect_source_format(args) -> str:
    if args.source_format != "auto":
        return args.source_format

    if args.input.endswith(".json") or args.input.endswith(".jsonl") or args.input.endswith(".parquet"):
        lower_name = os.path.basename(args.input).lower()
        if "video" in lower_name or "hound" in lower_name:
            return "llavahound_caption"

    if args.dataset_name:
        lowered = args.dataset_name.lower()
        if "visrag" in lowered:
            return "visrag"
        if "vidore" in lowered or "colpali" in lowered:
            return "vidore"
        if "mmeb" in lowered:
            return "mmeb_train"

    return "vlm2vec_pairs"


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


def serialize_image_like(value: Any) -> Optional[Any]:
    if value in [None, ""]:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        path = value.get("path")
        image_bytes = value.get("bytes")
        if path not in [None, ""] and image_bytes in [None, b""]:
            return path
        payload = {}
        if path not in [None, ""]:
            payload["path"] = path
        b64_value = maybe_b64_encode(image_bytes)
        if b64_value is not None:
            payload["b64"] = b64_value
        return payload or None

    pil_image_path = getattr(value, "filename", None)
    if pil_image_path not in [None, ""]:
        return pil_image_path

    try:
        import io

        buffer = io.BytesIO()
        value.save(buffer, format=getattr(value, "format", None) or "PNG")
        return {"b64": base64.b64encode(buffer.getvalue()).decode("utf-8")}
    except Exception as exc:
        raise ValueError(f"Unsupported image value: {type(value)}") from exc


def extract_pair_media_entries(media_struct: Any) -> List[Any]:
    if media_struct in [None, "", []]:
        return []

    if isinstance(media_struct, list):
        merged_entries = []
        for element in media_struct:
            merged_entries.extend(extract_pair_media_entries(element))
        return merged_entries

    if isinstance(media_struct, str):
        return [media_struct]

    if isinstance(media_struct, dict):
        if "frames" in media_struct and media_struct["frames"] not in [None, "", []]:
            serialized_frames = []
            for frame in media_struct["frames"]:
                serialized = serialize_image_like(frame)
                if serialized is not None:
                    serialized_frames.append(serialized)
            return serialized_frames

        paths = media_struct.get("paths") or []
        bytes_values = media_struct.get("bytes") or []
        if not isinstance(paths, list):
            paths = [paths]
        if not isinstance(bytes_values, list):
            bytes_values = [bytes_values]

        count = max(len(paths), len(bytes_values))
        if count == 0:
            return []

        entries = []
        for index in range(count):
            path = paths[index] if index < len(paths) else None
            image_bytes = bytes_values[index] if index < len(bytes_values) else None
            if path in [None, ""] and image_bytes in [None, b""]:
                continue
            if path not in [None, ""] and image_bytes in [None, b""]:
                entries.append(path)
            else:
                payload = {}
                if path not in [None, ""]:
                    payload["path"] = path
                b64_value = maybe_b64_encode(image_bytes)
                if b64_value is not None:
                    payload["b64"] = b64_value
                if payload:
                    entries.append(payload)
        return entries

    serialized = serialize_image_like(media_struct)
    return [serialized] if serialized is not None else []


def resolve_sequence_mode(args, row: Optional[Dict[str, Any]] = None) -> str:
    if args.sequence_mode != "auto":
        return args.sequence_mode

    dataset_hints = [
        args.dataset_name or "",
        args.global_dataset_name or "",
        row.get("global_dataset_name", "") if isinstance(row, dict) else "",
    ]
    lowered = " ".join(dataset_hints).lower()
    if any(token in lowered for token in ["video", "llavahound", "msrvtt", "msvd", "youcook", "vatex"]):
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


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(text for text in value if isinstance(text, str) and text.strip())
    return value if isinstance(value, str) else ""


def build_pair_item(text: Optional[str], media_struct: Any, sequence_mode: str) -> Dict[str, Any]:
    item = {}
    normalized_text = normalize_text_value(text)
    if normalized_text not in [None, ""]:
        item["text"] = normalized_text
    attach_media(item, extract_pair_media_entries(media_struct), sequence_mode=sequence_mode)
    return item


def add_common_roots(output_row: Dict[str, Any], args):
    if args.media_root is not None:
        output_row["media_root"] = args.media_root
    if args.image_root is not None:
        output_row["image_root"] = args.image_root
    if args.video_root is not None:
        output_row["video_root"] = args.video_root


def convert_pair_style_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    sequence_mode = resolve_sequence_mode(args, row=row)
    query = build_pair_item(row.get("query_text"), row.get("query_image"), sequence_mode=sequence_mode)
    positive = build_pair_item(row.get("pos_text"), row.get("pos_image"), sequence_mode=sequence_mode)

    neg_texts = row.get("neg_text") or []
    neg_images = row.get("neg_image") or []
    if not isinstance(neg_texts, list):
        neg_texts = [neg_texts]
    if not isinstance(neg_images, list):
        neg_images = [neg_images]

    max_negatives = max(len(neg_texts), len(neg_images))
    negatives = []
    for index in range(max_negatives):
        neg_item = build_pair_item(
            neg_texts[index] if index < len(neg_texts) else None,
            neg_images[index] if index < len(neg_images) else None,
            sequence_mode=sequence_mode,
        )
        if neg_item:
            negatives.append(neg_item)

    output_row = {
        "query": query,
        "pos": [positive],
        "neg": negatives,
        "metadata": {
            "source_format": "vlm2vec_pairs",
            "dataset_name": args.dataset_name,
            "global_dataset_name": args.global_dataset_name or row.get("global_dataset_name"),
        },
    }
    add_common_roots(output_row, args)
    return output_row


def convert_mmeb_train_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    query = {}
    if row.get("qry") not in [None, ""]:
        query["text"] = row["qry"]
    if row.get("qry_image_path") not in [None, ""]:
        query["image_path"] = row["qry_image_path"]

    positive = {}
    if row.get("pos_text") not in [None, ""]:
        positive["text"] = row["pos_text"]
    if row.get("pos_image_path") not in [None, ""]:
        positive["image_path"] = row["pos_image_path"]

    negatives = []
    if row.get("neg_text") not in [None, ""] or row.get("neg_image_path") not in [None, ""]:
        negative = {}
        if row.get("neg_text") not in [None, ""]:
            negative["text"] = row["neg_text"]
        if row.get("neg_image_path") not in [None, ""]:
            negative["image_path"] = row["neg_image_path"]
        if negative:
            negatives.append(negative)

    output_row = {
        "query": query,
        "pos": [positive],
        "neg": negatives,
        "metadata": {
            "source_format": "mmeb_train",
            "dataset_name": args.dataset_name,
            "global_dataset_name": args.global_dataset_name,
        },
    }
    add_common_roots(output_row, args)
    return output_row


def convert_vidore_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    query = build_pair_item(row.get("query"), row.get("image"), sequence_mode="image")
    positive = {}
    if row.get("answer") not in [None, ""]:
        positive["text"] = row["answer"]

    output_row = {
        "query": query,
        "pos": [positive],
        "neg": [],
        "metadata": {
            "source_format": "vidore",
            "dataset_name": args.dataset_name,
            "global_dataset_name": args.global_dataset_name,
            "source": row.get("source"),
            "answer_type": row.get("answer_type"),
            "prompt": row.get("prompt"),
        },
    }
    add_common_roots(output_row, args)
    return output_row


def convert_visrag_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    query = {}
    if row.get("query") not in [None, ""]:
        query["text"] = row["query"]

    positive = build_pair_item(None, row.get("image"), sequence_mode="image")
    output_row = {
        "query": query,
        "pos": [positive],
        "neg": [],
        "metadata": {
            "source_format": "visrag",
            "dataset_name": args.dataset_name,
            "global_dataset_name": args.global_dataset_name,
            "source": row.get("source"),
        },
    }
    add_common_roots(output_row, args)
    return output_row


def resolve_video_reference(row: Dict[str, Any], args) -> Dict[str, Any]:
    video_ref = row.get("video")
    if video_ref in [None, ""]:
        return {}

    if args.raw_video or args.frame_basedir is None:
        return {"video_path": video_ref}

    frame_dir = os.path.join(args.frame_basedir, str(video_ref))
    if not os.path.isdir(frame_dir):
        return {"video_path": video_ref}

    frame_names = sorted(
        file_name
        for file_name in os.listdir(frame_dir)
        if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )
    if len(frame_names) == 0:
        return {"video_path": video_ref}
    return {"video_frames": [os.path.join(frame_dir, file_name) for file_name in frame_names]}


def get_conversation_value(conversations: Any, index: int) -> str:
    if not isinstance(conversations, list) or index >= len(conversations):
        return ""
    value = conversations[index].get("value", "")
    return value if isinstance(value, str) else ""


def convert_llavahound_row(row: Dict[str, Any], args) -> Dict[str, Any]:
    conversations = row.get("conversations") or []
    user_turn = get_conversation_value(conversations, 0)
    assistant_turn = get_conversation_value(conversations, 1)
    video_payload = resolve_video_reference(row, args)

    if args.llavahound_mode == "video_retrieval":
        query = {"text": f"{VRET_QRY_PROMPT}{assistant_turn}".strip()}
        positive = {"text": VRET_TGT_PROMPT.strip()}
        positive.update(video_payload)
    else:
        cleaned_user_turn = user_turn.replace("<video>", " ").strip()
        query = {"text": cleaned_user_turn or "Describe the video content."}
        query.update(video_payload)
        positive = {"text": assistant_turn}

    output_row = {
        "query": query,
        "pos": [positive],
        "neg": [],
        "metadata": {
            "source_format": args.source_format,
            "dataset_name": args.dataset_name,
            "global_dataset_name": args.global_dataset_name,
            "sample_id": row.get("id"),
            "video": row.get("video"),
            "llavahound_mode": args.llavahound_mode,
        },
    }
    add_common_roots(output_row, args)
    return output_row


def row_converter_for(source_format: str):
    if source_format == "vlm2vec_pairs":
        return convert_pair_style_row
    if source_format == "mmeb_train":
        return convert_mmeb_train_row
    if source_format == "vidore":
        return convert_vidore_row
    if source_format == "visrag":
        return convert_visrag_row
    if source_format in ["llavahound_caption", "llavahound_qa"]:
        return convert_llavahound_row
    raise ValueError(f"Unsupported source format: {source_format}")


def write_jsonl(rows: Iterable[Dict[str, Any]], output_path: str):
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    args = parse_args()
    args.source_format = detect_source_format(args)
    dataset = load_any_dataset(args.input, subset=args.subset, split=args.split, cache_dir=args.cache_dir)
    converter = row_converter_for(args.source_format)

    def converted_rows():
        for _, row in iter_rows(dataset, max_rows=args.max_rows):
            converted = converter(row, args)
            if converted.get("query") and converted.get("pos"):
                yield converted

    count = write_jsonl(converted_rows(), args.output)
    print(f"Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
