#!/usr/bin/env python3
"""Report likely idle GPUs based on `nvidia-smi` memory and utilization."""

import argparse
import subprocess
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-memory-used", type=int, default=1024, help="Maximum memory usage in MiB for an idle GPU.")
    parser.add_argument("--max-utilization", type=int, default=10, help="Maximum utilization percentage for an idle GPU.")
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to a CSV dump from `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits`.",
    )
    return parser.parse_args()


def parse_gpu_lines(lines: List[str]):
    gpus = []
    for raw_line in lines:
        line = raw_line.strip()
        if line in ["", None] or line.lower().startswith("index"):
            continue
        normalized_fields = [field.strip().replace(" MiB", "").replace(" %", "") for field in line.split(",")]
        if len(normalized_fields) != 3:
            raise ValueError(f"Unexpected GPU row: {raw_line}")
        index, memory_used, utilization = normalized_fields
        gpus.append(
            {
                "index": int(index),
                "memory_used": int(memory_used),
                "utilization": int(utilization),
            }
        )
    return gpus


def query_gpus():
    commands = [
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
    ]
    last_error = None
    for command in commands:
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            return parse_gpu_lines(completed.stdout.strip().splitlines())
        except subprocess.CalledProcessError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def main():
    args = parse_args()
    if args.input is not None:
        with open(args.input, "r", encoding="utf-8") as input_file:
            gpus = parse_gpu_lines(input_file.readlines())
    else:
        try:
            gpus = query_gpus()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Failed to query `nvidia-smi` directly. If you are inside a constrained environment, "
                "dump the command output to a file and rerun with `--input`."
            ) from exc

    idle_gpus = [
        gpu
        for gpu in gpus
        if gpu["memory_used"] <= args.max_memory_used and gpu["utilization"] <= args.max_utilization
    ]

    for gpu in gpus:
        print(
            f"GPU {gpu['index']}: memory_used={gpu['memory_used']} MiB, utilization={gpu['utilization']}%"
        )

    if idle_gpus:
        selected = ",".join(str(gpu["index"]) for gpu in idle_gpus)
        print("")
        print(f"Idle GPUs: {selected}")
        print(f"CUDA_VISIBLE_DEVICES={selected}")
    else:
        print("")
        print("No GPUs matched the current idle thresholds.")


if __name__ == "__main__":
    main()
