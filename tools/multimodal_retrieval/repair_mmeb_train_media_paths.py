#!/usr/bin/env python3
"""Rewrite converted MMEB train JSONL rows to match a local media layout."""

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IMAGE_SUFFIX_PATTERN = re.compile(r"(image_\d+\.[A-Za-z0-9]+)$")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Converted Nexus JSONL file to rewrite.")
    parser.add_argument("--output", required=True, help="Destination JSONL path.")
    parser.add_argument(
        "--image-root",
        required=True,
        help="Local directory that should be written into each row's `image_root`.",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Fail if any image path cannot be resolved under the local `image_root`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def iter_image_path_candidates(relative_path: str) -> Iterable[str]:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    yield normalized

    basename = Path(normalized).name
    if basename != normalized:
        yield basename

    match = IMAGE_SUFFIX_PATTERN.search(basename)
    if match is not None:
        candidate = match.group(1)
        if candidate != basename:
            yield candidate


def resolve_local_image_path(relative_path: str, image_root: Path) -> Optional[str]:
    for candidate in iter_image_path_candidates(relative_path):
        if (image_root / candidate).exists():
            return candidate
    return None


def rewrite_row_image_paths(row: Dict, image_root: Path) -> Tuple[Dict, List[str]]:
    rewritten = deepcopy(row)
    unresolved = []

    def maybe_rewrite(item: Dict):
        image_path = item.get("image_path")
        if image_path in [None, ""]:
            return
        resolved = resolve_local_image_path(image_path, image_root)
        if resolved is None:
            unresolved.append(image_path)
            return
        item["image_path"] = resolved

    maybe_rewrite(rewritten.get("query", {}))
    for positive in rewritten.get("pos", []):
        maybe_rewrite(positive)
    for negative in rewritten.get("neg", []):
        maybe_rewrite(negative)

    rewritten["image_root"] = str(image_root)
    return rewritten, unresolved


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_root = Path(args.image_root).resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")

    rewritten_rows = []
    unresolved_rows = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_no, line in enumerate(input_file, start=1):
            row = json.loads(line)
            rewritten, unresolved = rewrite_row_image_paths(row, image_root)
            rewritten_rows.append(rewritten)
            if unresolved:
                unresolved_rows.append({"line": line_no, "paths": unresolved})

    if unresolved_rows and args.fail_on_unresolved:
        sample = unresolved_rows[:5]
        raise FileNotFoundError(
            f"Failed to resolve {len(unresolved_rows)} rows under {image_root}. Sample: {sample}"
        )

    write_jsonl(output_path, rewritten_rows)
    print(f"Rewrote {len(rewritten_rows)} rows into {output_path}")
    if unresolved_rows:
        print(f"Unresolved rows: {len(unresolved_rows)}")


if __name__ == "__main__":
    main()
