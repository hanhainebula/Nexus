#!/usr/bin/env python3
"""Export an MMEB v2 inventory markdown report from local VLM2Vec references."""

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from vlm2vec_manifest_lib import (
    build_eval_manifest_entries,
    build_train_manifest,
    discover_vlm2vec_root,
    load_vlm2vec_context,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vlm2vec-root",
        default=None,
        help="Path to the local VLM2Vec repository. Defaults to auto-discovery via VLM2VEC_ROOT or sibling repos.",
    )
    parser.add_argument("--output", required=True, help="Output markdown path.")
    return parser.parse_args()


def format_mapping_table(rows: Iterable[Dict[str, Any]], columns: List[str]) -> List[str]:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        formatted_values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (list, tuple)):
                value = ", ".join("" if item is None else str(item) for item in value)
            elif isinstance(value, dict):
                value = ", ".join(f"{key}={value[key]}" for key in sorted(value))
            formatted_values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(formatted_values) + " |")
    return lines


def main():
    args = parse_args()
    vlm2vec_root = discover_vlm2vec_root(args.vlm2vec_root)
    if vlm2vec_root is None:
        raise FileNotFoundError(
            "Unable to locate a local VLM2Vec repository. Pass --vlm2vec-root explicitly or set VLM2VEC_ROOT."
        )

    context = load_vlm2vec_context(vlm2vec_root)
    eval_entries = build_eval_manifest_entries(vlm2vec_root)
    train_manifest = build_train_manifest(context["train_configs"])

    lines = []
    lines.append("# MMEB v2 Inventory")
    lines.append("")
    lines.append("Generated from local VLM2Vec references.")
    lines.append("")
    lines.append("## Evaluation Coverage")
    lines.append("")

    for modality in ["image", "video", "visdoc"]:
        modality_entries = [row for row in eval_entries if row["modality"] == modality]
        lines.append(f"### {modality}")
        lines.append("")
        lines.append(f"- Task count: {len(modality_entries)}")
        lines.append("")
        lines.extend(
            format_mapping_table(
                modality_entries,
                [
                    "dataset_key",
                    "dataset_parser",
                    "eval_type",
                    "metadata_hf_repo",
                    "metadata_hf_subset",
                    "metadata_hf_split",
                    "media_hf_repo",
                    "media_rel_root",
                ],
            )
        )
        lines.append("")

    lines.append("## Public Training Sources")
    lines.append("")
    for modality in ["image", "video", "visdoc", "alltasks"]:
        lines.append(f"### {modality}")
        lines.append("")
        lines.extend(
            format_mapping_table(
                train_manifest[modality],
                [
                    "source_name",
                    "dataset_parser",
                    "dataset_name",
                    "subset_name",
                    "dataset_split",
                    "metadata_hf_repo",
                    "media_hf_repo",
                    "download_patterns",
                ],
            )
        )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Image eval metadata comes from `ziyjiang/MMEB_Test_Instruct`, while image media comes from `TIGER-Lab/MMEB-V2`.")
    lines.append("- Many video and visdoc eval datasets combine metadata from one HF repo with media extracted under `TIGER-Lab/MMEB-V2` layouts.")
    lines.append("- MMEB-train image media is downloaded as `images_zip/*.zip` and extracted under `images/`, not `image/`.")
    lines.append("- ShareGPTVideo raw media layout differs across repos and scripts; use the manifest path candidates instead of hardcoding one directory name.")
    lines.append("")

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print(f"Wrote inventory markdown to {args.output}")


if __name__ == "__main__":
    main()
