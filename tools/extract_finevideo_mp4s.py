#!/usr/bin/env python3
"""Extract base64 MP4s from one FineVideo parquet shard."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


def _mp4_bytes(value) -> bytes:
    if isinstance(value, bytes):
        if b"ftyp" in value[:32]:
            return value
        return base64.b64decode(value)
    if isinstance(value, str):
        return base64.b64decode(value)
    raise TypeError(f"Unsupported mp4 column value type: {type(value).__name__}")


def _local_snapshot_parquets(repo_id: str) -> list[Path]:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_cache = cache_root / f"datasets--{repo_id.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"
    if not snapshots.exists():
        return []
    return sorted(snapshots.glob("*/data/*.parquet"))


def _parquet_paths(args) -> list[Path]:
    if args.include:
        return [
            Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    filename=include,
                )
            )
            for include in args.include
        ]

    if args.download_all:
        snapshot = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                allow_patterns=["data/*.parquet"],
            )
        )
        return sorted(snapshot.glob("data/*.parquet"))

    paths = _local_snapshot_parquets(args.repo_id)
    if not paths:
        raise FileNotFoundError(
            "No local parquet shards found. Run once with --include data/train-00009-of-01357.parquet "
            "or pass --download-all to fetch every remote parquet shard."
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="HuggingFaceFV/finevideo")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--include", action="append", default=None,
                        help="Specific parquet filename to download/extract. May be repeated. Default: all local cached shards.")
    parser.add_argument("--download-all", action="store_true",
                        help="Download and extract every remote data/*.parquet shard instead of only local cached shards.")
    parser.add_argument("--out-dir", default="/data/datasets/multimodal/video")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prompt", default="Describe what happens in this video.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    parquet_paths = _parquet_paths(args)

    manifest_path = out_dir / "finevideo_calib.jsonl"
    written = 0
    with manifest_path.open("w") as manifest:
        for shard_idx, parquet_path in enumerate(parquet_paths):
            df = pd.read_parquet(parquet_path, columns=["mp4"])
            for row_idx, encoded in enumerate(df["mp4"]):
                if args.limit is not None and written >= args.limit:
                    break
                if encoded is None:
                    continue
                video_bytes = _mp4_bytes(encoded)
                video_path = video_dir / f"{written:06d}.mp4"
                video_path.write_bytes(video_bytes)

                record = {
                    "source_parquet": str(parquet_path),
                    "source_row": row_idx,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "video", "video": str(video_path)},
                                {"type": "text", "text": args.prompt},
                            ],
                        }
                    ],
                }
                manifest.write(json.dumps(record) + "\n")
                written += 1
            print(f"[{shard_idx + 1}/{len(parquet_paths)}] {parquet_path}: total videos={written}", flush=True)
            if args.limit is not None and written >= args.limit:
                break

    print(f"Processed {len(parquet_paths)} parquet shard(s)")
    print(f"Wrote {written} videos to {video_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
