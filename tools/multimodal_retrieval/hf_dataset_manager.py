#!/usr/bin/env python3
"""Plan or download public Hugging Face dataset files via the HTTP API."""

import argparse
import fnmatch
import json
import os
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote


HF_DATASET_API = "https://huggingface.co/api/datasets/{repo}/tree/main"
HF_DATASET_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Dataset repo id, for example `TIGER-Lab/MMEB-train`.")
    parser.add_argument("--output-root", required=True, help="Local directory used for planned or downloaded files.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Glob pattern to include. Repeat for multiple patterns. Defaults to every file.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude. Repeat for multiple patterns.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional file-count cap after filtering.")
    parser.add_argument("--max-bytes", type=int, default=None, help="Optional total byte cap after filtering.")
    parser.add_argument("--download", action="store_true", help="Download the selected files instead of only planning.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without downloading.")
    parser.add_argument("--json-output", default=None, help="Optional path to write the resolved plan JSON.")
    return parser.parse_args()


def split_headers_and_body(text: str) -> Tuple[str, str]:
    sections = re.split(r"\r?\n\r?\n", text)
    for index, section in enumerate(sections):
        if section.lstrip().startswith(("[", "{")):
            headers = sections[index - 1] if index > 0 else ""
            body = "\n\n".join(sections[index:])
            return headers, body
    return "", text


def parse_next_link(headers_text: str) -> Optional[str]:
    for line in headers_text.splitlines():
        if not line.lower().startswith("link:"):
            continue
        match = re.search(r"<([^>]+)>;\s*rel=\"next\"", line)
        if match:
            return match.group(1)
    return None


def run_curl_json_with_headers(url: str):
    result = subprocess.run(["curl", "-L", "-s", "-D", "-", url], check=True, capture_output=True, text=True)
    headers_text, body_text = split_headers_and_body(result.stdout)
    return headers_text, json.loads(body_text)


def fetch_dataset_tree(repo: str) -> List[Dict[str, object]]:
    url = HF_DATASET_API.format(repo=quote(repo, safe="/")) + "?recursive=1&expand=1"
    files: List[Dict[str, object]] = []
    seen_paths = set()

    while url:
        headers_text, payload = run_curl_json_with_headers(url)
        for entry in payload:
            if entry.get("type") != "file":
                continue
            path = str(entry.get("path"))
            if path in seen_paths:
                continue
            files.append(entry)
            seen_paths.add(path)
        url = parse_next_link(headers_text)

    return files


def match_any(path: str, patterns: Iterable[str]) -> bool:
    patterns = list(patterns)
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def select_entries(
    entries: Iterable[Dict[str, object]],
    include_patterns: Iterable[str],
    exclude_patterns: Iterable[str],
    max_files: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    total_bytes = 0

    for entry in sorted(entries, key=lambda item: str(item["path"])):
        path = str(entry["path"])
        if not match_any(path, include_patterns):
            continue
        if exclude_patterns and match_any(path, exclude_patterns):
            continue

        entry_size = int(entry.get("size", 0) or 0)
        if max_files is not None and len(selected) >= max_files:
            break
        if max_bytes is not None and total_bytes + entry_size > max_bytes:
            break

        selected.append(entry)
        total_bytes += entry_size

    return selected


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def build_plan(repo: str, output_root: str, entries: List[Dict[str, object]]) -> Dict[str, object]:
    planned_files = []
    total_bytes = 0
    for entry in entries:
        path = str(entry["path"])
        size = int(entry.get("size", 0) or 0)
        planned_files.append(
            {
                "repo": repo,
                "path": path,
                "size": size,
                "size_human": format_bytes(size),
                "destination": os.path.join(output_root, path),
            }
        )
        total_bytes += size

    return {
        "repo": repo,
        "output_root": output_root,
        "num_files": len(planned_files),
        "total_bytes": total_bytes,
        "total_size_human": format_bytes(total_bytes),
        "files": planned_files,
    }


def ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def download_plan(plan: Dict[str, object], dry_run: bool = False):
    for item in plan["files"]:
        destination = item["destination"]
        expected_size = int(item["size"])
        if os.path.exists(destination) and os.path.getsize(destination) == expected_size:
            print(f"skip {destination}")
            continue

        ensure_parent_dir(destination)
        url = HF_DATASET_RESOLVE.format(repo=quote(item["repo"], safe="/"), path=quote(item["path"], safe="/"))
        print(f"download {item['path']} -> {destination}")
        if dry_run:
            continue
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "-C",
                "-",
                "-o",
                destination,
                url,
            ],
            check=True,
        )


def main():
    args = parse_args()

    all_entries = fetch_dataset_tree(args.repo)
    selected_entries = select_entries(
        all_entries,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        max_files=args.max_files,
        max_bytes=args.max_bytes,
    )
    plan = build_plan(repo=args.repo, output_root=os.path.abspath(args.output_root), entries=selected_entries)

    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if args.json_output:
        json_output = os.path.abspath(args.json_output)
        ensure_parent_dir(json_output)
        with open(json_output, "w", encoding="utf-8") as output_file:
            json.dump(plan, output_file, indent=2, ensure_ascii=False)

    if args.download:
        download_plan(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
