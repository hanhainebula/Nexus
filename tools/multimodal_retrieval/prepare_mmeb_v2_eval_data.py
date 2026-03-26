#!/usr/bin/env python3
"""Prepare MMEB v2 evaluation subsets in Nexus local-eval format."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

STAGE_TO_MODALITIES = {
    "all": {"image", "video", "visdoc"},
    "image_only": {"image"},
    "video_only": {"video"},
    "visdoc_only": {"visdoc"},
}

DEFAULT_EVAL_CONFIG = {
    "eval_name": "multimodal_local",
    "dataset_dir": None,
    "media_root": None,
    "image_root": None,
    "video_root": None,
    "dataset_names": None,
    "splits": ["test"],
    "corpus_embd_save_dir": "./outputs/multimodal_eval_cache",
    "search_top_k": 100,
    "cache_path": None,
    "ignore_identical_ids": False,
    "force_redownload": False,
    "overwrite": True,
    "eval_output_dir": "./outputs/multimodal_eval_results",
    "eval_output_method": "markdown",
    "eval_output_path": "./outputs/multimodal_eval_results/summary.md",
    "eval_metrics": ["ndcg_at_10", "recall_at_10"],
    "k_values": [1, 3, 5, 10, 100],
}


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
        help="Root directory containing raw eval data. Supports both VLM2Vec-style mirrors and repo-cache layouts.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where converted Nexus local-eval datasets will be written.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=sorted(STAGE_TO_MODALITIES.keys()),
        help="Subset of modalities to prepare.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["image", "video", "visdoc"],
        default=None,
        help="Optional modality override.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of MMEB dataset_key values to convert.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional cap applied to each converted eval dataset.",
    )
    parser.add_argument("--cache-dir", default=None, help="Datasets cache directory.")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for the conversion CLI. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--write-eval-configs-dir",
        default=None,
        help="Optional directory where per-dataset eval_config json files will be written.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary JSON path. Defaults to `<output-root>/conversion_summary.json`.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip datasets whose local metadata input is missing instead of failing.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Fail or skip when a local metadata input is not found instead of falling back to HF dataset loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted dataset directories and config files.",
    )
    parser.add_argument(
        "--write-configs-only",
        action="store_true",
        help="Only write eval_config files and summary rows without re-running dataset conversion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversion commands without executing them.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def repo_fs_name(repo: str) -> str:
    return repo.replace("/", "--")


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    deduped = []
    seen = set()
    for value in values:
        if value in [None, ""]:
            continue
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def iter_eval_entries(manifest: Dict, modalities: Iterable[str], selected_datasets: Optional[set]) -> Iterable[Dict]:
    for entry in manifest.get("eval", []):
        if entry.get("modality") not in modalities:
            continue
        dataset_key = entry.get("dataset_key")
        if selected_datasets is not None and dataset_key not in selected_datasets:
            continue
        yield entry


def find_first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def repo_root_candidates(raw_root: Path, repo: Optional[str], purpose: str) -> List[Path]:
    if repo in [None, ""]:
        return []
    repo_name = repo_fs_name(repo)
    repo_basename = repo.split("/")[-1]
    return [
        raw_root / "vlm2vec_eval" / purpose / repo_name,
        raw_root / "vlm2vec_eval" / "repo" / repo_name,
        raw_root / "vlm2vec_eval" / repo_basename,
        raw_root / repo_basename,
    ]


def relative_path_candidates(raw_root: Path, relative_path: Optional[str], repo: Optional[str], purpose: str) -> List[Path]:
    if relative_path in [None, ""]:
        return []
    relative_path = relative_path.rstrip("/")
    candidates = []
    for repo_root in repo_root_candidates(raw_root, repo, purpose=purpose):
        candidates.append(repo_root / relative_path)
    candidates.append(raw_root / "vlm2vec_eval" / relative_path)
    candidates.append(raw_root / relative_path)
    return candidates


def resolve_existing_or_default(
    raw_root: Path,
    relative_path: Optional[str],
    repo: Optional[str],
    purpose: str,
) -> Optional[str]:
    if relative_path in [None, ""]:
        return None
    candidates = relative_path_candidates(raw_root, relative_path, repo=repo, purpose=purpose)
    resolved = find_first_existing_path(candidates)
    if resolved is not None:
        return str(resolved)
    return str((raw_root / relative_path).resolve())


def resolve_metadata_input(entry: Dict, raw_root: Path) -> Tuple[Optional[str], bool]:
    metadata_repo = entry.get("metadata_hf_repo")
    local_repo = entry.get("local_repo")
    local_subset = entry.get("local_subset")
    metadata_subset = entry.get("metadata_hf_subset")

    path_candidates: List[Path] = []
    metadata_relative_candidates: List[str] = []
    if metadata_subset not in [None, ""]:
        metadata_relative_candidates.append(str(metadata_subset))
    for relative_path in metadata_relative_candidates:
        path_candidates.extend(relative_path_candidates(raw_root, relative_path, repo=metadata_repo, purpose="metadata"))
    path_candidates.extend(repo_root_candidates(raw_root, metadata_repo, purpose="metadata"))

    local_relative_candidates: List[str] = []
    if local_repo not in [None, ""]:
        if local_subset not in [None, ""]:
            local_relative_candidates.append(os.path.join(local_repo, local_subset))
        local_relative_candidates.append(local_repo)
    for relative_path in unique_preserve_order(local_relative_candidates):
        path_candidates.extend(relative_path_candidates(raw_root, relative_path, repo=metadata_repo, purpose="metadata"))

    resolved = find_first_existing_path(path_candidates)
    if resolved is not None:
        return str(resolved), True
    return None, False


def resolve_eval_source(entry: Dict, raw_root: Path) -> Dict[str, Optional[str]]:
    config = entry.get("config", {})
    input_path, is_local = resolve_metadata_input(entry, raw_root)

    frame_root = resolve_existing_or_default(
        raw_root=raw_root,
        relative_path=config.get("frame_root"),
        repo=entry.get("media_hf_repo"),
        purpose="media",
    )
    image_root = resolve_existing_or_default(
        raw_root=raw_root,
        relative_path=config.get("image_root"),
        repo=entry.get("media_hf_repo"),
        purpose="media",
    )
    video_root = resolve_existing_or_default(
        raw_root=raw_root,
        relative_path=config.get("video_root"),
        repo=entry.get("media_hf_repo"),
        purpose="media",
    )
    media_root = resolve_existing_or_default(
        raw_root=raw_root,
        relative_path=entry.get("media_rel_root"),
        repo=entry.get("media_hf_repo"),
        purpose="media",
    )

    # VLM2Vec video eval datasets are typically frame-based, so Nexus should use the
    # resolved frame root as the image root when normalizing `video_frames`.
    if entry.get("modality") == "video" and frame_root not in [None, ""]:
        image_root = frame_root
        if media_root in [None, ""]:
            media_root = frame_root

    return {
        "input": input_path,
        "subset": entry.get("metadata_hf_subset"),
        "split": entry.get("metadata_hf_split") or entry.get("local_split") or "test",
        "is_local": is_local,
        "media_root": media_root,
        "image_root": image_root,
        "video_root": video_root,
    }


def build_convert_command(entry: Dict, resolved: Dict[str, Optional[str]], output_dir: Path, args) -> List[str]:
    converter = SCRIPT_DIR / "convert_vlm2vec_eval_to_nexus.py"
    input_value = resolved["input"]
    if input_value in [None, ""]:
        if args.local_only:
            raise FileNotFoundError(
                f"Local metadata input was not found for `{entry['dataset_key']}` under `{args.raw_root}`."
            )
        input_value = entry["metadata_hf_repo"]

    command = [
        args.python_bin,
        str(converter),
        "--input",
        input_value,
        "--output-dir",
        str(output_dir),
        "--dataset-name",
        entry["dataset_key"],
        "--split",
        resolved["split"],
        "--sequence-mode",
        "video" if entry.get("modality") == "video" else "image",
    ]
    metadata_repo = entry.get("metadata_hf_repo") or ""
    if entry.get("dataset_parser") in {"vidore", "visrag"} or metadata_repo.startswith("vidore/"):
        command.extend(["--input-format", "beir_configs"])
    if resolved.get("subset") not in [None, ""]:
        command.extend(["--subset", str(resolved["subset"])])
    if args.cache_dir not in [None, ""]:
        command.extend(["--cache-dir", args.cache_dir])
    if args.max_rows_per_dataset is not None:
        command.extend(["--max-rows", str(args.max_rows_per_dataset)])
    if resolved.get("media_root") not in [None, ""]:
        command.extend(["--media-root", resolved["media_root"]])
    if resolved.get("image_root") not in [None, ""]:
        command.extend(["--image-root", resolved["image_root"]])
    if resolved.get("video_root") not in [None, ""]:
        command.extend(["--video-root", resolved["video_root"]])
    return command


def build_eval_config(
    entry: Dict,
    output_root: Path,
    resolved: Dict[str, Optional[str]],
) -> Dict[str, object]:
    dataset_key = entry["dataset_key"]
    split = resolved["split"]
    eval_config = dict(DEFAULT_EVAL_CONFIG)
    eval_config.update(
        {
            "eval_name": dataset_key,
            "dataset_dir": str(output_root),
            "dataset_names": [dataset_key],
            "splits": [split],
            "media_root": resolved.get("media_root"),
            "image_root": resolved.get("image_root"),
            "video_root": resolved.get("video_root"),
            "corpus_embd_save_dir": str(output_root / "_cache" / dataset_key),
            "cache_path": str(output_root / "_cache" / "hf_datasets"),
            "eval_output_dir": str(output_root / "_results" / dataset_key),
            "eval_output_path": str(output_root / "_results" / dataset_key / "summary.md"),
        }
    )
    return eval_config


def plan_conversion_output_dir(dataset_output_dir: Path, overwrite: bool) -> Path:
    if not dataset_output_dir.exists():
        return dataset_output_dir
    if not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing dataset directory without `--overwrite`: {dataset_output_dir}"
        )
    staging_output_dir = dataset_output_dir.parent / f".{dataset_output_dir.name}.tmp_convert"
    if staging_output_dir.exists():
        shutil.rmtree(staging_output_dir)
    return staging_output_dir


def finalize_conversion_output_dir(staging_output_dir: Path, dataset_output_dir: Path):
    if staging_output_dir == dataset_output_dir:
        return
    if dataset_output_dir.exists():
        shutil.rmtree(dataset_output_dir)
    staging_output_dir.rename(dataset_output_dir)


def write_json(path: Path, payload: Dict[str, object], overwrite: bool = False):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without `--overwrite`: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    raw_root = Path(args.raw_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    modalities = set(args.modalities or STAGE_TO_MODALITIES[args.stage])
    selected_datasets = set(args.datasets) if args.datasets else None
    summary_path = Path(args.summary_path) if args.summary_path else output_root / "conversion_summary.json"

    summary = {
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "modalities": sorted(modalities),
        "datasets": [],
    }

    for entry in iter_eval_entries(manifest, modalities=modalities, selected_datasets=selected_datasets):
        dataset_key = entry["dataset_key"]
        resolved = resolve_eval_source(entry, raw_root=raw_root)
        dataset_output_dir = output_root / dataset_key
        staging_output_dir = plan_conversion_output_dir(dataset_output_dir, overwrite=args.overwrite)

        dataset_summary = {
            "dataset_key": dataset_key,
            "modality": entry.get("modality"),
            "input": resolved.get("input") or entry.get("metadata_hf_repo"),
            "subset": resolved.get("subset"),
            "split": resolved.get("split"),
            "is_local_input": resolved.get("is_local", False),
            "media_root": resolved.get("media_root"),
            "image_root": resolved.get("image_root"),
            "video_root": resolved.get("video_root"),
            "output_dir": str(dataset_output_dir),
            "status": "planned",
        }

        if resolved.get("input") in [None, ""] and args.local_only:
            if args.allow_missing:
                dataset_summary["status"] = "missing"
                summary["datasets"].append(dataset_summary)
                continue
            raise FileNotFoundError(
                f"Local metadata input was not found for `{dataset_key}` under `{raw_root}` and `--local-only` is set."
            )

        command = build_convert_command(entry=entry, resolved=resolved, output_dir=staging_output_dir, args=args)
        dataset_summary["convert_command"] = command
        if staging_output_dir != dataset_output_dir:
            dataset_summary["staging_output_dir"] = str(staging_output_dir)

        if args.write_eval_configs_dir not in [None, ""]:
            eval_config = build_eval_config(entry=entry, output_root=output_root, resolved=resolved)
            config_path = Path(args.write_eval_configs_dir) / f"{dataset_key}.eval_config.json"
            write_json(config_path, eval_config, overwrite=args.overwrite)
            dataset_summary["eval_config_path"] = str(config_path)

        if args.write_configs_only:
            dataset_summary["status"] = "config_only"
            summary["datasets"].append(dataset_summary)
            continue

        if args.dry_run:
            dataset_summary["status"] = "dry_run"
            summary["datasets"].append(dataset_summary)
            continue

        subprocess.run(command, check=True)
        finalize_conversion_output_dir(staging_output_dir=staging_output_dir, dataset_output_dir=dataset_output_dir)
        dataset_summary["status"] = "converted"
        summary["datasets"].append(dataset_summary)

    write_json(summary_path, summary, overwrite=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
