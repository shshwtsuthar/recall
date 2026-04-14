"""
Nebius SWE-rebench OpenHands trajectories -> ADP converter.
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

logger = logging.getLogger("recall.nebius_to_adp")

SOURCE_FORMAT = "nebius_swe_rebench_openhands"
SOURCE_DATASET = "nebius/SWE-rebench-openhands-trajectories"


def convert_row(row: dict) -> dict:
    source_trajectory_id = str(row["trajectory_id"])
    trajectory_id = deterministic_trajectory_id(SOURCE_FORMAT, source_trajectory_id)

    extra_details = {
        "source_dataset": SOURCE_DATASET,
        "source_trajectory_id": source_trajectory_id,
        "instance_id": row.get("instance_id"),
        "repo": row.get("repo"),
        "resolved": row.get("resolved"),
        "exit_status": row.get("exit_status"),
        "gen_tests_correct": row.get("gen_tests_correct"),
        "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
        "has_model_patch": bool(row.get("model_patch")),
    }

    return convert_chat_trajectory(
        messages=row["trajectory"],
        trajectory_id=trajectory_id,
        source_format=SOURCE_FORMAT,
        tools=row.get("tools"),
        extra_details=extra_details,
    )


def convert_parquet(
    parquet_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int | None = None,
    pretty: bool = False,
    overwrite: bool = False,
) -> tuple[int, int]:
    import pyarrow.parquet as pq

    parquet_path = Path(parquet_path)
    written = 0
    skipped = 0

    parquet = pq.ParquetFile(parquet_path)
    columns = [
        "trajectory_id",
        "instance_id",
        "repo",
        "trajectory",
        "tools",
        "model_patch",
        "exit_status",
        "resolved",
        "gen_tests_correct",
        "pred_passes_gen_tests",
    ]

    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        for row in batch.to_pylist():
            trajectory = convert_row(row)
            path = write_adp_trajectory(
                trajectory,
                output_dir,
                pretty=pretty,
                overwrite=overwrite,
            )
            if path is None:
                skipped += 1
            else:
                written += 1

            if limit is not None and (written + skipped) >= limit:
                return written, skipped

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Nebius SWE-rebench OpenHands trajectories to ADP JSON files",
    )
    parser.add_argument("parquet_path", help="Path to trajectories.parquet")
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

    written, skipped = convert_parquet(
        args.parquet_path,
        args.output_dir,
        limit=args.limit,
        pretty=args.pretty,
        overwrite=args.overwrite,
    )
    logger.info("Converted %d trajectories (%d skipped)", written, skipped)


if __name__ == "__main__":
    main()
