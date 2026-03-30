#!/usr/bin/env python3
import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an exploration-friendly multimodal train jsonl by sampling or repeating "
            "existing JSON/JSONL records to a target size."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more source files or directories containing training JSON/JSONL records.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1000,
        help="Desired number of output records.",
    )
    parser.add_argument(
        "--fill-mode",
        choices=["repeat_to_target", "at_most_target"],
        default="repeat_to_target",
        help=(
            "`repeat_to_target` oversamples with replacement when the source is smaller than the target. "
            "`at_most_target` never exceeds the available record count."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and shuffling.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep the sampled records in deterministic selection order.",
    )
    return parser.parse_args()


def iter_input_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix in {".json", ".jsonl"}:
                    files.append(candidate)
            continue
        if path.suffix not in {".json", ".jsonl"}:
            raise ValueError(f"Unsupported input file type: {path}")
        files.append(path)
    if len(files) == 0:
        raise ValueError("No JSON/JSONL input files were found.")
    return files


def load_records_from_file(path: Path) -> List[dict]:
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line == "":
                continue
            records.append(json.loads(line))
        return records

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON payload in {path}: expected an object or a list of objects.")


def load_records(inputs: Iterable[str]) -> List[dict]:
    records: List[dict] = []
    for file_path in iter_input_files(inputs):
        records.extend(load_records_from_file(file_path))
    if len(records) == 0:
        raise ValueError("No training records were loaded from the provided inputs.")
    return records


def materialize_target_records(records: List[dict], target_size: int, seed: int, fill_mode: str, shuffle: bool) -> List[dict]:
    if target_size <= 0:
        raise ValueError("`target_size` must be positive.")

    rng = random.Random(seed)

    if len(records) >= target_size:
        output_records = rng.sample(records, target_size)
    elif fill_mode == "repeat_to_target":
        output_records = list(records)
        while len(output_records) < target_size:
            output_records.append(deepcopy(rng.choice(records)))
    else:
        output_records = list(records)

    if shuffle:
        rng.shuffle(output_records)
    return output_records


def write_jsonl(records: List[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    source_records = load_records(args.input)
    output_records = materialize_target_records(
        records=source_records,
        target_size=args.target_size,
        seed=args.seed,
        fill_mode=args.fill_mode,
        shuffle=not args.no_shuffle,
    )
    output_path = Path(args.output).expanduser().resolve()
    write_jsonl(output_records, output_path)
    print(
        json.dumps(
            {
                "input_records": len(source_records),
                "output_records": len(output_records),
                "output_path": str(output_path),
                "fill_mode": args.fill_mode,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
