#!/usr/bin/env python3
"""Create a mutable DiskANN3 Bf-tree run shadow from a pristine snapshot."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
from pathlib import Path


REQUIRED_SUFFIXES = ("_params.json", "_vectors.bftree", "_neighbors.bftree")
OPTIONAL_SUFFIXES = ("_quant.bftree", "_delete.bin", "_quant_data.bin", "_manifest.json")


def files_for(prefix: Path) -> list[tuple[str, Path]]:
    required = [(suffix, Path(f"{prefix}{suffix}")) for suffix in REQUIRED_SUFFIXES]
    missing = [str(path) for _, path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing pristine snapshot files: {missing}")
    return required + [
        (suffix, Path(f"{prefix}{suffix}"))
        for suffix in OPTIONAL_SUFFIXES
        if Path(f"{prefix}{suffix}").is_file()
    ]


def copy_reflink(source: Path, destination: Path) -> None:
    subprocess.run(
        ["cp", "--reflink=auto", "--sparse=always", "--preserve=mode,timestamps", source, destination],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-prefix", type=Path, required=True)
    parser.add_argument("--dest-prefix", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--copy-workers", type=int, default=3)
    args = parser.parse_args()
    if args.copy_workers < 1:
        parser.error("--copy-workers must be positive")

    sources = files_for(args.source_prefix)
    args.dest_prefix.parent.mkdir(parents=True, exist_ok=True)
    destinations = [(suffix, Path(f"{args.dest_prefix}{suffix}")) for suffix, _ in sources]
    existing = [path for _, path in destinations if path.exists()]
    if existing and not args.force:
        raise SystemExit(f"refusing to overwrite {existing[0]}; pass --force")
    for path in existing:
        path.unlink()

    logical_bytes = sum(source.stat().st_size for _, source in sources)
    free_bytes = shutil.disk_usage(args.dest_prefix.parent).free
    if free_bytes < logical_bytes + (1 << 30):
        raise SystemExit(
            f"insufficient free space: need at least {logical_bytes + (1 << 30)} bytes "
            f"for a non-reflink fallback but have {free_bytes}"
        )

    copy_jobs = [
        (source, destination)
        for (suffix, source), (_, destination) in zip(sources, destinations, strict=True)
        if suffix != "_params.json"
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.copy_workers, len(copy_jobs))
    ) as executor:
        future_to_paths = {
            executor.submit(copy_reflink, source, destination): (source, destination)
            for source, destination in copy_jobs
        }
        for future in concurrent.futures.as_completed(future_to_paths):
            source, destination = future_to_paths[future]
            future.result()
            print(f"copied {source} -> {destination}")

    source_params = Path(f"{args.source_prefix}_params.json")
    destination_params = Path(f"{args.dest_prefix}_params.json")
    params = json.loads(source_params.read_text())
    params["prefix"] = str(args.dest_prefix.resolve())
    destination_params.write_text(json.dumps(params, separators=(",", ":")) + "\n")
    print(f"rewrote snapshot prefix in {destination_params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
