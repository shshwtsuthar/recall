"""
SWE-Gym OpenHands parquet shards -> ADP converter.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from recall.wrappers.chat_to_adp import (
    convert_chat_trajectory,
    deterministic_trajectory_id,
    write_adp_trajectory,
)

logger = logging.getLogger("recall.swe_gym_to_adp")

SOURCE_FORMAT = "swe_gym_openhands_sampled"
SOURCE_DATASET = "SWE-Gym/OpenHands-Sampled-Trajectories"


def _resolve_inputs(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(sorted(path.glob("train.raw-*.parquet")))
        else:
            resolved.append(path)
    return resolved


def convert_row(row: dict) -> dict:
    source_example_id = f"{row['instance_id']}::{row['run_id']}"
    trajectory_id = deterministic_trajectory_id(
        SOURCE_FORMAT,
        str(row["instance_id"]),
        str(row["run_id"]),
    )

    test_result = row.get("test_result") or {}
    report = test_result.get("report") or {}

    extra_details = {
        "source_dataset": SOURCE_DATASET,
        "source_example_id": source_example_id,
        "instance_id": row.get("instance_id"),
        "run_id": row.get("run_id"),
        "resolved": row.get("resolved"),
        "has_git_patch": bool(test_result.get("git_patch")),
        "has_apply_patch_output": bool(test_result.get("apply_patch_output")),
        "test_result_summary": report,
    }

    return convert_chat_trajectory(
        messages=row["messages"],
        trajectory_id=trajectory_id,
        source_format=SOURCE_FORMAT,
        tools=row.get("tools"),
        extra_details=extra_details,
    )


def convert_parquet_files(
    inputs: list[str],
    output_dir: str | Path,
    *,
    limit: int | None = None,
    pretty: bool = False,
    overwrite: bool = False,
) -> tuple[int, int]:
    import pyarrow.parquet as pq

    written = 0
    skipped = 0
    paths = _resolve_inputs(inputs)
    columns = [
        "instance_id",
        "run_id",
        "resolved",
        "messages",
        "tools",
        "test_result",
    ]

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                trajectory = convert_row(row)
                output = write_adp_trajectory(
                    trajectory,
                    output_dir,
                    pretty=pretty,
                    overwrite=overwrite,
                )
                if output is None:
                    skipped += 1
                else:
                    written += 1

                if limit is not None and (written + skipped) >= limit:
                    return written, skipped

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWE-Gym OpenHands parquet trajectories to ADP JSON files",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Parquet shard paths, or directories containing train.raw-*.parquet",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to write ADP JSON files into",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert at most N trajectories",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    written, skipped = convert_parquet_files(
        args.inputs,
        args.output_dir,
        limit=args.limit,
        pretty=args.pretty,
        overwrite=args.overwrite,
    )
    logger.info("Converted %d trajectories (%d skipped)", written, skipped)


if __name__ == "__main__":
    main()
