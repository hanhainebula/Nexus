#!/usr/bin/env python3
"""Convert locally collected MMEB v2 public train sources into Nexus JSONL files."""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

STAGE_TO_MODALITIES = {
    "all": {"image", "visdoc", "video"},
    "stage_a": {"image", "visdoc"},
    "stage_b": {"video"},
    "stage_c": {"image", "visdoc", "video"},
}

DEFAULT_DATA_CONFIG = {
    "train_data": [],
    "cache_path": None,
    "media_root": None,
    "train_group_size": 4,
    "query_max_len": 512,
    "passage_max_len": 1024,
    "query_instruction_for_retrieval": "Represent this multimodal query for retrieving matching targets: ",
    "query_instruction_format": "{}{}",
    "passage_instruction_for_retrieval": None,
    "passage_instruction_format": "{}{}",
    "knowledge_distillation": False,
    "shuffle_ratio": 0.0,
}

RELATIVE_PATH_ALIASES = (
    ("vlm2vec_train/train_video_and_instruction", "vlm2vec_train/ShareGPTVideo"),
    ("vlm2vec_train/VisRAG", "vlm2vec_train/VisRAG-Ret-Train-In-domain-data"),
)


def load_train_converter():
    module_path = SCRIPT_DIR / "convert_vlm2vec_train_to_nexus.py"
    spec = importlib.util.spec_from_file_location("convert_vlm2vec_train_to_nexus", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "docs" / "multimodal_retrieval" / "MMEB_v2_manifest.json"),
        help="Path to the MMEB v2 manifest JSON.",
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Root directory containing raw public data with `vlm2vec_train/` underneath.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where converted Nexus train jsonl files will be written.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=sorted(STAGE_TO_MODALITIES.keys()),
        help="Subset of modalities to convert.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["image", "visdoc", "video"],
        default=None,
        help="Optional modality override.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Optional list of source_name values to convert.",
    )
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Optional cap applied to each converted source.",
    )
    parser.add_argument("--cache-dir", default=None, help="Datasets cache directory.")
    parser.add_argument(
        "--write-stage-configs-dir",
        default=None,
        help="Optional directory where generated Nexus data_config json files will be written.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary JSON path. Defaults to `<output-root>/conversion_summary.json`.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip sources whose local raw data is missing instead of failing.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Do not fall back to remote Hugging Face loading when local raw inputs are absent.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted jsonl files.",
    )
    parser.add_argument(
        "--raw-video",
        action="store_true",
        help="Prefer raw video paths for video training sources instead of frame folders.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._") or "dataset"


def iter_train_entries(manifest: Dict, modalities: Iterable[str], selected_sources: Optional[set]) -> Iterable[Dict]:
    train_manifest = manifest.get("train", {})
    for modality in ["image", "visdoc", "video"]:
        if modality not in modalities:
            continue
        for entry in train_manifest.get(modality, []):
            source_name = entry.get("source_name", "")
            if selected_sources is not None and source_name not in selected_sources:
                continue
            yield {"modality": modality, **entry}


def find_first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_relative_candidates(raw_root: Path, relative_path: str) -> List[Path]:
    candidates = [raw_root / relative_path]
    for before, after in RELATIVE_PATH_ALIASES:
        if relative_path.startswith(before):
            candidates.append(raw_root / relative_path.replace(before, after, 1))
    return candidates


def resolve_repo_dir(raw_root: Path, candidate_names: List[str]) -> Optional[Path]:
    candidates = []
    for candidate_name in candidate_names:
        if candidate_name in [None, ""]:
            continue
        candidates.extend(
            [
                raw_root / "vlm2vec_train" / candidate_name,
                raw_root / candidate_name,
            ]
        )
    return find_first_existing_path(candidates)


def resolve_train_source(entry: Dict, raw_root: Path) -> Dict:
    parser_name = entry["dataset_parser"]
    dataset_name = entry.get("dataset_name", "")

    resolved = {
        "input": None,
        "subset": entry.get("subset_name"),
        "split": entry.get("dataset_split", "train"),
        "image_root": None,
        "video_root": None,
        "frame_basedir": None,
        "is_local": False,
    }

    if parser_name == "mmeb":
        repo_dir = resolve_repo_dir(raw_root, ["MMEB-train", dataset_name.split("/")[-1]])
        if repo_dir is not None:
            subset_dir = repo_dir / entry["subset_name"]
            if subset_dir.exists():
                resolved["input"] = str(subset_dir)
                resolved["subset"] = None
                resolved["split"] = entry.get("dataset_split", "original")
                resolved["is_local"] = True
        if "image_dir" in entry:
            resolved["image_root"] = str(raw_root / entry["image_dir"])
        if resolved["input"] is None:
            resolved["input"] = dataset_name
        return resolved

    if parser_name in {"llavahound_caption", "llavahound_qa"}:
        input_path = find_first_existing_path(resolve_relative_candidates(raw_root, entry["dataset_path"]))
        frame_dir = find_first_existing_path(resolve_relative_candidates(raw_root, entry["video_frame_basedir"]))
        if input_path is not None:
            resolved["input"] = str(input_path)
            resolved["is_local"] = True
        if frame_dir is not None:
            resolved["frame_basedir"] = str(frame_dir)
            resolved["video_root"] = str(frame_dir)
        return resolved

    if parser_name == "vidore":
        repo_dir = resolve_repo_dir(raw_root, ["colpali_train_set", dataset_name.split("/")[-1]])
        if repo_dir is not None:
            resolved["input"] = str(repo_dir)
            resolved["is_local"] = True
        else:
            resolved["input"] = dataset_name
        return resolved

    if parser_name == "visrag":
        repo_dir = resolve_repo_dir(
            raw_root,
            [
                "VisRAG-Ret-Train-In-domain-data",
                "VisRAG",
                dataset_name.split("/")[-1],
            ],
        )
        if repo_dir is not None:
            resolved["input"] = str(repo_dir)
            resolved["is_local"] = True
        else:
            resolved["input"] = dataset_name
        return resolved

    raise ValueError(f"Unsupported dataset parser in manifest: {parser_name}")


def build_converter_args(entry: Dict, resolved: Dict, output_path: Path, cli_args) -> SimpleNamespace:
    parser_name = entry["dataset_parser"]
    source_format = {
        "mmeb": "mmeb_train",
        "vidore": "vidore",
        "visrag": "visrag",
        "llavahound_caption": "llavahound_caption",
        "llavahound_qa": "llavahound_qa",
    }[parser_name]

    max_rows = cli_args.max_rows_per_source
    if max_rows is None:
        max_rows = entry.get("num_sample_per_subset", entry.get("num_rows"))

    return SimpleNamespace(
        input=resolved["input"],
        output=str(output_path),
        source_format=source_format,
        subset=resolved["subset"],
        split=resolved["split"],
        cache_dir=cli_args.cache_dir,
        max_rows=max_rows,
        dataset_name=entry["source_name"],
        global_dataset_name=entry.get("global_dataset_name"),
        media_root=None,
        image_root=resolved["image_root"],
        video_root=resolved["video_root"],
        sequence_mode="auto",
        llavahound_mode=entry.get("data_mode", "caption_retrieval"),
        raw_video=cli_args.raw_video,
        frame_basedir=resolved["frame_basedir"],
    )


def convert_source(converter, entry: Dict, raw_root: Path, output_root: Path, cli_args) -> Dict:
    resolved = resolve_train_source(entry, raw_root)
    if resolved["input"] is None:
        if cli_args.allow_missing:
            return {"source_name": entry["source_name"], "modality": entry["modality"], "status": "missing"}
        raise FileNotFoundError(f"Local raw data for source `{entry['source_name']}` was not found under {raw_root}.")
    if cli_args.local_only and not resolved["is_local"]:
        if cli_args.allow_missing:
            return {"source_name": entry["source_name"], "modality": entry["modality"], "status": "missing"}
        raise FileNotFoundError(
            f"Source `{entry['source_name']}` is not available locally under {raw_root} and --local-only was set."
        )

    output_path = output_root / entry["modality"] / f"{sanitize_name(entry['source_name'])}.jsonl"
    if output_path.exists() and not cli_args.overwrite:
        return {
            "source_name": entry["source_name"],
            "modality": entry["modality"],
            "status": "skipped_existing",
            "output": str(output_path),
        }

    converter_args = build_converter_args(entry, resolved, output_path, cli_args)
    dataset = converter.load_any_dataset(
        converter_args.input,
        subset=converter_args.subset,
        split=converter_args.split,
        cache_dir=converter_args.cache_dir,
    )
    row_converter = converter.row_converter_for(converter_args.source_format)

    def converted_rows():
        for _, row in converter.iter_rows(dataset, max_rows=converter_args.max_rows):
            converted = row_converter(row, converter_args)
            if converted.get("query") and converted.get("pos"):
                yield converted

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = converter.write_jsonl(converted_rows(), str(output_path))
    return {
        "source_name": entry["source_name"],
        "modality": entry["modality"],
        "status": "converted",
        "count": count,
        "output": str(output_path),
        "input": converter_args.input,
        "weight": entry.get("weight"),
    }


def write_stage_configs(write_dir: Path, summary_rows: List[Dict]):
    converted_rows = [row for row in summary_rows if row.get("status") == "converted"]
    outputs_by_modality = {"image": [], "visdoc": [], "video": []}
    for row in converted_rows:
        outputs_by_modality[row["modality"]].append(row["output"])

    stage_to_train_data = {
        "stage_a": outputs_by_modality["image"] + outputs_by_modality["visdoc"],
        "stage_b": outputs_by_modality["video"],
        "stage_c": outputs_by_modality["image"] + outputs_by_modality["visdoc"] + outputs_by_modality["video"],
    }

    write_dir.mkdir(parents=True, exist_ok=True)
    for stage_name, train_data in stage_to_train_data.items():
        config = dict(DEFAULT_DATA_CONFIG)
        config["train_data"] = train_data
        config["cache_path"] = str(write_dir.parent / "_cache" / stage_name)
        with (write_dir / f"{stage_name}_data_config.json").open("w", encoding="utf-8") as output_file:
            json.dump(config, output_file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_modalities = set(args.modalities or STAGE_TO_MODALITIES[args.stage])
    selected_sources = set(args.sources) if args.sources is not None else None
    converter = load_train_converter()

    summary_rows = []
    for entry in iter_train_entries(manifest, selected_modalities, selected_sources):
        summary_rows.append(convert_source(converter, entry, raw_root, output_root, args))

    summary_path = Path(args.summary_path) if args.summary_path is not None else output_root / "conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary_rows, output_file, indent=2, ensure_ascii=False)

    if args.write_stage_configs_dir is not None:
        write_stage_configs(Path(args.write_stage_configs_dir), summary_rows)

    converted = [row for row in summary_rows if row.get("status") == "converted"]
    skipped = [row for row in summary_rows if row.get("status") != "converted"]
    print(f"Converted {len(converted)} sources into {output_root}")
    if skipped:
        print(f"Skipped {len(skipped)} sources; see {summary_path}")


if __name__ == "__main__":
    main()
