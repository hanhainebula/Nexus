#!/usr/bin/env python3
"""Plan, download, and convert public MMEB/VLM2Vec training sources."""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, Iterable, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from hf_dataset_manager import build_plan, download_plan, fetch_dataset_tree, select_entries
from vlm2vec_manifest_lib import augment_train_source


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="docs/multimodal_retrieval/MMEB_v2_manifest.json",
        help="Path to the machine-readable MMEB manifest.",
    )
    parser.add_argument("--raw-root", required=True, help="Root directory for raw downloads.")
    parser.add_argument("--nexus-root", required=True, help="Root directory for converted Nexus data.")
    parser.add_argument(
        "--train-modality",
        action="append",
        choices=["image", "video", "visdoc", "alltasks"],
        default=[],
        help="Training modality section to include. Repeat for multiple sections. Defaults to `image`.",
    )
    parser.add_argument("--source-name", action="append", default=[], help="Restrict to specific train sources.")
    parser.add_argument("--download", action="store_true", help="Download selected raw files.")
    parser.add_argument("--extract-archives", action="store_true", help="Extract downloaded zip archives when relevant.")
    parser.add_argument("--convert-train", action="store_true", help="Convert selected train sources to Nexus JSONL.")
    parser.add_argument("--skip-media", action="store_true", help="Download only metadata files when possible.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing downloads/conversions.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for the conversion CLI. Defaults to the current interpreter.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for conversion smoke runs.")
    parser.add_argument("--json-output", default=None, help="Optional path to save the computed plan JSON.")
    return parser.parse_args()


def load_manifest(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as input_file:
        return json.load(input_file)


def default_train_repo_dir(raw_root: str, repo: str) -> str:
    base_root = os.path.join(raw_root, "vlm2vec_train")
    mapping = {
        "TIGER-Lab/MMEB-train": "MMEB-train",
        "ShareGPTVideo/train_video_and_instruction": "ShareGPTVideo",
        "vidore/colpali_train_set": "colpali_train_set",
        "openbmb/VisRAG-Ret-Train-In-domain-data": "VisRAG-Ret-Train-In-domain-data",
    }
    return os.path.join(base_root, mapping.get(repo, repo.split("/")[-1]))


def select_train_sources(manifest: Dict[str, object], modalities: Iterable[str], source_names: Iterable[str]):
    modalities = list(modalities) or ["image"]
    source_names = set(source_names)
    selected = []
    for modality in modalities:
        for source in manifest["train"][modality]:
            source_name = source["source_name"]
            if source_names and source_name not in source_names:
                continue
            if "metadata_hf_repo" not in source or "download_patterns" not in source:
                source = augment_train_source(modality=modality, source_name=source_name, source_config=source)
            selected.append(source)
    return selected


def plan_download_for_source(raw_root: str, source: Dict[str, object], skip_media: bool = False) -> Optional[Dict[str, object]]:
    repo = source.get("metadata_hf_repo")
    if not repo:
        return None

    patterns = list(source.get("download_patterns") or [])
    if skip_media:
        if source.get("dataset_parser") == "mmeb":
            patterns = [pattern for pattern in patterns if not pattern.startswith("images_zip/")]
        if source.get("dataset_parser") in {"llavahound_caption", "llavahound_qa"}:
            patterns = [pattern for pattern in patterns if not pattern.endswith(".tar.gz")]
    repo_dir = default_train_repo_dir(raw_root, repo)
    selected_entries = select_entries(
        fetch_dataset_tree(repo),
        include_patterns=patterns,
        exclude_patterns=[],
        max_files=None,
        max_bytes=None,
    )
    plan = build_plan(repo=repo, output_root=repo_dir, entries=selected_entries)
    plan["source_name"] = source["source_name"]
    plan["dataset_parser"] = source.get("dataset_parser")
    plan["train_modality"] = source.get("train_modality")
    return plan


def resolve_existing_dir(base_dir: str, candidates: Iterable[str]) -> str:
    candidates = list(candidates)
    for candidate in candidates:
        path = os.path.join(base_dir, candidate)
        if os.path.isdir(path):
            return path
    return os.path.join(base_dir, candidates[0]) if candidates else base_dir


def build_convert_command(args, raw_root: str, nexus_root: str, source: Dict[str, object]) -> List[str]:
    converter = os.path.join(os.path.dirname(__file__), "convert_vlm2vec_train_to_nexus.py")
    repo_dir = default_train_repo_dir(raw_root, source["metadata_hf_repo"])
    output_path = os.path.join(nexus_root, "train", source["train_modality"], f"{source['source_name']}.jsonl")
    command = [args.python_bin, converter]

    dataset_parser = source["dataset_parser"]
    if dataset_parser == "mmeb":
        command.extend(["--input", os.path.join(repo_dir, source["local_rel_input"])])
        command.extend(["--source-format", "mmeb_train"])
        command.extend(["--dataset-name", source["source_name"]])
        command.extend(["--global-dataset-name", f"{source['train_modality']}/{source['source_name']}"])
        image_root = resolve_existing_dir(repo_dir, source.get("image_root_candidates") or [])
        command.extend(["--image-root", image_root])
    elif dataset_parser in {"llavahound_caption", "llavahound_qa"}:
        command.extend(["--input", os.path.join(repo_dir, source["metadata_rel_path"])])
        command.extend(["--source-format", dataset_parser])
        command.extend(["--dataset-name", source["source_name"]])
        command.extend(["--global-dataset-name", f"{source['train_modality']}/{source['source_name']}"])
        if source.get("data_mode"):
            command.extend(["--llavahound-mode", source["data_mode"]])
        command.append("--raw-video")
        video_root = resolve_existing_dir(repo_dir, source.get("video_root_candidates") or [])
        command.extend(["--video-root", video_root])
    elif dataset_parser in {"vidore", "visrag"}:
        command.extend(["--input", os.path.join(repo_dir, source["local_rel_input"])])
        command.extend(["--source-format", dataset_parser])
        command.extend(["--dataset-name", source["source_name"]])
        command.extend(["--global-dataset-name", f"{source['train_modality']}/{source['source_name']}"])
    else:
        raise ValueError(f"Unsupported dataset_parser in prepare_public_data.py: {dataset_parser}")

    if args.max_rows is not None:
        command.extend(["--max-rows", str(args.max_rows)])
    command.extend(["--output", output_path])
    return command


def maybe_extract_archives(plan: Dict[str, object], dry_run: bool = False):
    for item in plan["files"]:
        if not str(item["path"]).endswith(".zip"):
            continue
        archive_path = item["destination"]
        extract_dir = os.path.join(os.path.dirname(os.path.dirname(archive_path)), "images")
        os.makedirs(extract_dir, exist_ok=True)
        print(f"extract {archive_path} -> {extract_dir}")
        if dry_run:
            continue
        subprocess.run(["unzip", "-n", archive_path, "-d", extract_dir], check=True)


def main():
    args = parse_args()
    manifest = load_manifest(os.path.abspath(args.manifest))
    raw_root = os.path.abspath(args.raw_root)
    nexus_root = os.path.abspath(args.nexus_root)
    selected_sources = select_train_sources(
        manifest,
        modalities=args.train_modality,
        source_names=args.source_name,
    )

    plan = {"sources": [], "conversions": []}

    for source in selected_sources:
        download_plan_item = plan_download_for_source(raw_root, source, skip_media=args.skip_media)
        conversion_command = build_convert_command(args, raw_root=raw_root, nexus_root=nexus_root, source=source)
        plan["sources"].append(
            {
                "source_name": source["source_name"],
                "train_modality": source["train_modality"],
                "dataset_parser": source["dataset_parser"],
                "download_plan": download_plan_item,
                "convert_command": conversion_command,
            }
        )
        plan["conversions"].append(
            {
                "source_name": source["source_name"],
                "command": conversion_command,
            }
        )

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.json_output:
        json_output = os.path.abspath(args.json_output)
        output_dir = os.path.dirname(json_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as output_file:
            json.dump(plan, output_file, indent=2, ensure_ascii=False)

    if args.download:
        for item in plan["sources"]:
            if item["download_plan"] is None:
                continue
            download_plan(item["download_plan"], dry_run=args.dry_run)
            if args.extract_archives:
                maybe_extract_archives(item["download_plan"], dry_run=args.dry_run)

    if args.convert_train:
        for conversion in plan["conversions"]:
            print("convert " + " ".join(conversion["command"]))
            if args.dry_run:
                continue
            subprocess.run(conversion["command"], check=True)


if __name__ == "__main__":
    main()
