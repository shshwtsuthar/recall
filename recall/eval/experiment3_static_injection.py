"""
Experiment 3 scaffold: static memory injection for OpenHands runs.

This module deliberately separates three artifacts:

1. Task/run manifests: what should be evaluated.
2. Retrieval/candidate files: which memories are attached to each task.
3. Rendered prompts: what is passed to OpenHands for each condition.

The local candidate mode is only for smoke tests and pilots. Main paper runs
should replace it with candidates produced by representation-specific retrievers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONDITIONS = ("no_memory", "flat", "summary", "adp_lite", "adp_full")
MEMORY_CONDITIONS = {"flat", "summary", "adp_lite", "adp_full"}
ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|failures|passed|pytest|assert)\b",
    re.IGNORECASE,
)
PATH_RE = re.compile(r"(?:(?:[\w.-]+/)+[\w.-]+\.\w+|[\w.-]+\.\w+)")


@dataclass(frozen=True)
class TrajectoryMeta:
    trajectory_id: str
    path: str
    source_format: str
    source_dataset: str
    instance_id: str
    repo: str
    resolved: bool | None
    task_goal: str
    total_events: int
    adp_item_count: int


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    target_trajectory_id: str
    target_path: str
    source_format: str
    instance_id: str
    repo: str
    original_resolved: bool | None
    task_goal: str
    eligible_same_repo_memory: int


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    task_id: str
    condition: str
    seed: int
    prompt_path: str | None = None


def stable_hash(value: str, seed: int = 0) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_run_id(task_id: str, condition: str, seed: int) -> str:
    digest = hashlib.md5(f"{task_id}\x1f{condition}\x1f{seed}".encode()).hexdigest()
    return digest[:16]


def normalize_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "resolved"}:
        return True
    if text in {"0", "false", "no", "unresolved"}:
        return False
    return None


def infer_repo(instance_id: str, working_dir: str = "") -> str:
    if "__" in instance_id:
        slug = instance_id.rsplit("-", 1)[0]
        owner, repo = slug.split("__", 1)
        if owner and repo:
            return f"{owner}/{repo}"

    if working_dir:
        name = Path(working_dir).name
        if "__" in name:
            slug = name.rsplit("__", 1)[0]
            owner, repo = slug.split("__", 1)
            if owner and repo:
                return f"{owner}/{repo}"

    return ""


def details_repo(details: dict[str, Any]) -> str:
    return str(details.get("repo", "")) or infer_repo(
        str(details.get("instance_id", "")),
        str(details.get("working_dir", "")),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_adp_files(adp_dirs: Iterable[Path]) -> Iterable[Path]:
    for adp_dir in adp_dirs:
        if adp_dir.is_file() and adp_dir.suffix == ".json":
            yield adp_dir
            continue
        if adp_dir.is_dir():
            yield from sorted(adp_dir.rglob("*.json"))


def load_trajectory(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(path: Path) -> TrajectoryMeta:
    trajectory = load_trajectory(path)
    details = trajectory.get("details", {})
    content = trajectory.get("content", [])
    instance_id = str(details.get("instance_id", ""))
    repo = details_repo(details)
    return TrajectoryMeta(
        trajectory_id=str(trajectory.get("id", path.stem)),
        path=str(path),
        source_format=str(details.get("source_format", "")),
        source_dataset=str(details.get("source_dataset", "")),
        instance_id=instance_id,
        repo=repo,
        resolved=normalize_bool(details.get("resolved")),
        task_goal=str(details.get("task_goal", "")),
        total_events=int(details.get("total_events") or 0),
        adp_item_count=len(content),
    )


def load_all_meta(adp_dirs: Iterable[Path]) -> list[TrajectoryMeta]:
    metas: list[TrajectoryMeta] = []
    for path in iter_adp_files(adp_dirs):
        try:
            meta = load_meta(path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Skipping malformed ADP file {path}: {exc}")
            continue
        if meta.instance_id and meta.repo and meta.task_goal:
            metas.append(meta)
    return metas


def instance_key(meta: TrajectoryMeta) -> str:
    return meta.instance_id or meta.trajectory_id


def choose_one_target_per_instance(
    metas: Iterable[TrajectoryMeta],
    seed: int,
) -> list[TrajectoryMeta]:
    best: dict[str, TrajectoryMeta] = {}
    for meta in metas:
        key = instance_key(meta)
        current = best.get(key)
        if current is None:
            best[key] = meta
            continue

        # Prefer a shorter target prompt for live runs; break ties stably.
        current_rank = (len(current.task_goal), stable_hash(current.trajectory_id, seed))
        candidate_rank = (len(meta.task_goal), stable_hash(meta.trajectory_id, seed))
        if candidate_rank < current_rank:
            best[key] = meta
    return list(best.values())


def build_manifest(
    *,
    adp_dirs: list[Path],
    out_dir: Path,
    max_tasks: int,
    eval_fraction: float,
    seed: int,
    min_repo_memory: int,
    run_seeds: list[int],
    conditions: list[str],
) -> None:
    metas = load_all_meta(adp_dirs)
    if not metas:
        raise SystemExit("No eligible ADP trajectories found.")

    memory: list[TrajectoryMeta] = []
    eval_pool: list[TrajectoryMeta] = []
    for meta in metas:
        bucket = stable_hash(instance_key(meta), seed) % 10_000
        if bucket < int(eval_fraction * 10_000):
            eval_pool.append(meta)
        else:
            memory.append(meta)

    memory_by_repo: dict[str, list[TrajectoryMeta]] = {}
    for meta in memory:
        memory_by_repo.setdefault(meta.repo, []).append(meta)

    target_pool = choose_one_target_per_instance(eval_pool, seed)
    eligible: list[EvalTask] = []
    for meta in target_pool:
        same_repo = [
            m
            for m in memory_by_repo.get(meta.repo, [])
            if instance_key(m) != instance_key(meta)
        ]
        if len(same_repo) < min_repo_memory:
            continue
        task_id = f"{meta.source_format}:{meta.instance_id}"
        eligible.append(
            EvalTask(
                task_id=task_id,
                target_trajectory_id=meta.trajectory_id,
                target_path=meta.path,
                source_format=meta.source_format,
                instance_id=meta.instance_id,
                repo=meta.repo,
                original_resolved=meta.resolved,
                task_goal=meta.task_goal,
                eligible_same_repo_memory=len(same_repo),
            )
        )

    eligible.sort(key=lambda t: stable_hash(t.task_id, seed))
    tasks = eligible[:max_tasks]

    runs: list[ExperimentRun] = []
    for task in tasks:
        for condition in conditions:
            if condition not in DEFAULT_CONDITIONS:
                raise SystemExit(f"Unsupported condition: {condition}")
            for run_seed in run_seeds:
                runs.append(
                    ExperimentRun(
                        run_id=stable_run_id(task.task_id, condition, run_seed),
                        task_id=task.task_id,
                        condition=condition,
                        seed=run_seed,
                    )
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "tasks.jsonl", (asdict(t) for t in tasks))
    write_jsonl(out_dir / "runs.jsonl", (asdict(r) for r in runs))
    write_jsonl(out_dir / "memory_bank.jsonl", (asdict(m) for m in memory))

    summary = {
        "adp_dirs": [str(p) for p in adp_dirs],
        "seed": seed,
        "eval_fraction": eval_fraction,
        "max_tasks": max_tasks,
        "min_repo_memory": min_repo_memory,
        "conditions": conditions,
        "run_seeds": run_seeds,
        "counts": {
            "all_eligible_trajectories": len(metas),
            "memory_trajectories": len(memory),
            "eval_pool_trajectories": len(eval_pool),
            "eligible_eval_tasks": len(eligible),
            "selected_eval_tasks": len(tasks),
            "runs": len(runs),
        },
    }
    (out_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def attach_local_candidates(
    *,
    tasks_path: Path,
    memory_bank_path: Path,
    out_path: Path,
    k: int,
    seed: int,
    mode: str,
) -> None:
    tasks = read_jsonl(tasks_path)
    memory_rows = read_jsonl(memory_bank_path)

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in memory_rows:
        by_repo.setdefault(row.get("repo", ""), []).append(row)

    retrievals: list[dict[str, Any]] = []
    for task in tasks:
        candidates = [
            row
            for row in by_repo.get(task["repo"], [])
            if row.get("instance_id") != task["instance_id"]
        ]
        if mode == "same_repo_success":
            successful = [row for row in candidates if row.get("resolved") is True]
            candidates = successful or candidates
        elif mode == "same_repo_any":
            pass
        elif mode == "random_success":
            successful = [row for row in memory_rows if row.get("resolved") is True]
            candidates = successful or memory_rows
        else:
            raise SystemExit(f"Unsupported local candidate mode: {mode}")

        rng = random.Random(stable_hash(task["task_id"], seed))
        ranked = sorted(
            candidates,
            key=lambda row: (
                stable_hash(str(row.get("trajectory_id", "")), seed),
                str(row.get("trajectory_id", "")),
            ),
        )
        rng.shuffle(ranked)

        retrievals.append(
            {
                "task_id": task["task_id"],
                "condition": None,
                "candidate_source": mode,
                "candidates": [
                    {
                        "rank": idx + 1,
                        "trajectory_id": row["trajectory_id"],
                        "path": row["path"],
                        "score": None,
                        "repo": row.get("repo", ""),
                        "instance_id": row.get("instance_id", ""),
                        "resolved": row.get("resolved"),
                    }
                    for idx, row in enumerate(ranked[:k])
                ],
            }
        )

    count = write_jsonl(out_path, retrievals)
    print(f"Wrote {count} retrieval rows to {out_path}")


def attach_qdrant_candidates(
    *,
    tasks_path: Path,
    memory_bank_path: Path,
    out_path: Path,
    qdrant_url: str,
    collection: str,
    embedding_model: str,
    embedding_dim: int,
    api_key: str | None,
    k: int,
    fetch_limit: int,
    condition: str | None,
    score_threshold: float | None,
) -> None:
    from recall.embeddings.embedder import Embedder
    from recall.storage.qdrant_store import QdrantStore

    tasks = read_jsonl(tasks_path)
    memory_rows = read_jsonl(memory_bank_path)
    memory_by_id = {row["trajectory_id"]: row for row in memory_rows}

    embedder = Embedder(embedding_model)
    store = QdrantStore(
        url=qdrant_url,
        collection_name=collection,
        embedding_dim=embedding_dim,
        api_key=api_key,
    )

    retrievals: list[dict[str, Any]] = []
    for task in tasks:
        query_vector = embedder.embed(task["task_goal"])
        hits = store.search(
            query_vector,
            limit=fetch_limit,
            score_threshold=score_threshold,
        )
        candidates: list[dict[str, Any]] = []
        for hit in hits:
            trajectory_id = str(hit["id"])
            row = memory_by_id.get(trajectory_id)
            payload = hit.get("payload") or {}
            instance_id = (row or {}).get("instance_id") or payload.get("instance_id", "")
            if instance_id == task["instance_id"]:
                continue
            if row is None or not row.get("path"):
                continue
            resolved = row.get("resolved")
            if resolved is None:
                resolved = normalize_bool(payload.get("resolved"))
            candidates.append(
                {
                    "rank": len(candidates) + 1,
                    "trajectory_id": trajectory_id,
                    "path": row.get("path", ""),
                    "score": hit.get("score"),
                    "repo": row.get("repo") or payload.get("repo", ""),
                    "instance_id": instance_id,
                    "resolved": resolved,
                }
            )
            if len(candidates) >= k:
                break

        retrievals.append(
            {
                "task_id": task["task_id"],
                "condition": condition,
                "candidate_source": f"qdrant:{collection}",
                "candidates": candidates,
            }
        )

    count = write_jsonl(out_path, retrievals)
    print(f"Wrote {count} Qdrant retrieval rows to {out_path}")


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 20)].rstrip() + "\n...[truncated]"


def clean_one_line(text: Any, max_chars: int = 240) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    value = re.sub(r"\s+", " ", value)
    return truncate(value, max_chars)


def compact_json(value: Any, max_chars: int = 260) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return truncate(text, max_chars)


def extract_paths_from_text(text: str) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for match in PATH_RE.finditer(text):
        path = match.group(0)
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def extract_memory_features(trajectory: dict[str, Any]) -> dict[str, Any]:
    files: list[str] = []
    commands: list[str] = []
    observations: list[str] = []
    actions: list[str] = []
    seen_files: set[str] = set()

    for item in trajectory.get("content", []):
        cls = item.get("class_", "")
        if cls == "api_action":
            function = str(item.get("function", ""))
            kwargs = item.get("kwargs") or {}
            path = kwargs.get("path")
            command = kwargs.get("command")
            if path and str(path) not in seen_files:
                seen_files.add(str(path))
                files.append(str(path))
            if command:
                actions.append(f"{function} command={clean_one_line(command, 160)}")
                for extracted in extract_paths_from_text(str(command)):
                    if extracted not in seen_files:
                        seen_files.add(extracted)
                        files.append(extracted)
            else:
                actions.append(f"{function} kwargs={compact_json(kwargs, 180)}")
        elif cls == "code_action":
            command = str(item.get("content", ""))
            if item.get("language") == "bash" and command:
                commands.append(command)
            actions.append(f"bash: {clean_one_line(command, 180)}")
            for extracted in extract_paths_from_text(command):
                if extracted not in seen_files:
                    seen_files.add(extracted)
                    files.append(extracted)
        elif cls == "text_observation":
            content = str(item.get("content", ""))
            if ERROR_RE.search(content):
                observations.append(clean_one_line(content, 300))

    return {
        "files": files[:20],
        "commands": commands[:20],
        "observations": observations[:12],
        "actions": actions[:80],
    }


def render_summary(trajectory: dict[str, Any], max_chars: int) -> str:
    details = trajectory.get("details", {})
    features = extract_memory_features(trajectory)
    lines = [
        f"Memory id: {trajectory.get('id', '')}",
        f"Repo: {details_repo(details)}",
        f"Instance: {details.get('instance_id', '')}",
        f"Outcome: resolved={details.get('resolved', None)}",
        f"Prior task: {clean_one_line(details.get('task_goal', ''), 800)}",
    ]
    if features["files"]:
        lines.append("Files mentioned or touched: " + ", ".join(features["files"][:12]))
    if features["commands"]:
        lines.append("Representative bash commands:")
        lines.extend(f"- {clean_one_line(cmd, 220)}" for cmd in features["commands"][:8])
    if features["observations"]:
        lines.append("Notable observations:")
        lines.extend(f"- {obs}" for obs in features["observations"][:6])
    return truncate("\n".join(lines), max_chars)


def render_adp_lite(trajectory: dict[str, Any], max_chars: int) -> str:
    details = trajectory.get("details", {})
    features = extract_memory_features(trajectory)
    lite = {
        "id": trajectory.get("id"),
        "repo": details_repo(details),
        "instance_id": details.get("instance_id"),
        "resolved": details.get("resolved"),
        "source_format": details.get("source_format"),
        "task_goal": clean_one_line(details.get("task_goal", ""), 900),
        "files": features["files"][:16],
        "ordered_actions": features["actions"][:50],
    }
    return truncate(json.dumps(lite, indent=2, ensure_ascii=False), max_chars)


def render_flat(trajectory: dict[str, Any], max_chars: int) -> str:
    details = trajectory.get("details", {})
    lines = [
        f"Trajectory {trajectory.get('id', '')}",
        f"Repo: {details_repo(details)}",
        f"Instance: {details.get('instance_id', '')}",
        f"Resolved: {details.get('resolved', None)}",
        f"Task: {details.get('task_goal', '')}",
        "",
        "Trace:",
    ]
    for item in trajectory.get("content", []):
        cls = item.get("class_", "")
        if cls == "text_observation":
            source = item.get("source", "")
            name = item.get("name") or ""
            lines.append(
                f"[observation source={source} name={name}] "
                f"{clean_one_line(item.get('content', ''), 320)}"
            )
        elif cls == "message_action":
            lines.append(f"[assistant] {clean_one_line(item.get('content', ''), 320)}")
        elif cls == "api_action":
            lines.append(
                f"[api {item.get('function', '')}] "
                f"{compact_json(item.get('kwargs') or {}, 320)}"
            )
        elif cls == "code_action":
            lines.append(
                f"[code {item.get('language', '')}] "
                f"{clean_one_line(item.get('content', ''), 360)}"
            )
        if sum(len(line) + 1 for line in lines) > max_chars:
            break
    return truncate("\n".join(lines), max_chars)


def render_adp_full(trajectory: dict[str, Any], max_chars: int) -> str:
    # Full ADP can exceed practical prompt budgets. We preserve schema fields and
    # truncate only at the final text boundary so every condition has a budget.
    return truncate(json.dumps(trajectory, indent=2, ensure_ascii=False), max_chars)


def render_memory_artifact(
    trajectory: dict[str, Any],
    representation: str,
    max_chars: int,
) -> str:
    if representation == "flat":
        return render_flat(trajectory, max_chars)
    if representation == "summary":
        return render_summary(trajectory, max_chars)
    if representation == "adp_lite":
        return render_adp_lite(trajectory, max_chars)
    if representation == "adp_full":
        return render_adp_full(trajectory, max_chars)
    raise ValueError(f"Unsupported memory representation: {representation}")


def retrieval_key(task_id: str, condition: str | None) -> str:
    return f"{task_id}\x1f{condition or ''}"


def load_retrievals(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["task_id"]
        condition = row.get("condition")
        by_key[retrieval_key(task_id, condition)] = row
    return by_key


def candidates_for_run(
    retrievals: dict[str, dict[str, Any]],
    task_id: str,
    condition: str,
) -> list[dict[str, Any]]:
    row = retrievals.get(retrieval_key(task_id, condition))
    if row is None:
        row = retrievals.get(retrieval_key(task_id, None))
    if row is None:
        return []
    return list(row.get("candidates") or [])


def render_openhands_prompt(
    *,
    task_goal: str,
    condition: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    max_memory_chars: int,
) -> str:
    if condition == "no_memory":
        return task_goal
    if condition not in MEMORY_CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")

    per_memory_budget = max(1_000, max_memory_chars // max(1, min(top_k, len(candidates) or 1)))
    blocks: list[str] = []
    for candidate in candidates[:top_k]:
        trajectory = load_trajectory(candidate["path"])
        artifact = render_memory_artifact(
            trajectory,
            representation=condition,
            max_chars=per_memory_budget,
        )
        blocks.append(
            "\n".join(
                [
                    f"<memory rank=\"{candidate.get('rank')}\" "
                    f"id=\"{candidate.get('trajectory_id')}\">",
                    artifact,
                    "</memory>",
                ]
            )
        )

    memory_text = "\n\n".join(blocks) if blocks else "No retrieved memories available."
    return "\n\n".join(
        [
            "<retrieved_prior_experience>",
            "The following prior trajectories are optional background. They may be irrelevant, stale, or wrong.",
            "Use them only when they provide concrete evidence for this repository task. Do not copy code blindly.",
            memory_text,
            "</retrieved_prior_experience>",
            "",
            "<current_task>",
            task_goal,
            "</current_task>",
        ]
    )


def render_prompts(
    *,
    tasks_path: Path,
    runs_path: Path,
    retrievals_path: Path,
    out_dir: Path,
    top_k: int,
    max_memory_chars: int,
) -> None:
    tasks = {row["task_id"]: row for row in read_jsonl(tasks_path)}
    runs = read_jsonl(runs_path)
    retrievals = load_retrievals(retrievals_path)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    rendered_runs: list[dict[str, Any]] = []
    for run in runs:
        task = tasks[run["task_id"]]
        candidates = candidates_for_run(
            retrievals,
            task_id=run["task_id"],
            condition=run["condition"],
        )
        prompt = render_openhands_prompt(
            task_goal=task["task_goal"],
            condition=run["condition"],
            candidates=candidates,
            top_k=top_k,
            max_memory_chars=max_memory_chars,
        )
        prompt_path = prompt_dir / f"{run['run_id']}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        updated = dict(run)
        updated["prompt_path"] = str(prompt_path)
        updated["candidate_count"] = len(candidates)
        rendered_runs.append(updated)

    write_jsonl(out_dir / "runs_with_prompts.jsonl", rendered_runs)
    print(f"Wrote {len(rendered_runs)} prompts to {prompt_dir}")


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 3 static-injection manifests and prompts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-manifest")
    build.add_argument(
        "--adp-dir",
        action="append",
        type=Path,
        default=None,
        help="ADP directory or JSON file. Repeat for multiple sources.",
    )
    build.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/exp3_static_injection"),
    )
    build.add_argument("--max-tasks", type=int, default=50)
    build.add_argument("--eval-fraction", type=float, default=0.15)
    build.add_argument("--seed", type=int, default=13)
    build.add_argument("--min-repo-memory", type=int, default=25)
    build.add_argument("--run-seeds", default="0")
    build.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))

    attach = subparsers.add_parser("attach-local-candidates")
    attach.add_argument("--tasks", type=Path, required=True)
    attach.add_argument("--memory-bank", type=Path, required=True)
    attach.add_argument("--out", type=Path, required=True)
    attach.add_argument("--k", type=int, default=3)
    attach.add_argument("--seed", type=int, default=13)
    attach.add_argument(
        "--mode",
        choices=("same_repo_success", "same_repo_any", "random_success"),
        default="same_repo_success",
    )

    qdrant = subparsers.add_parser("attach-qdrant-candidates")
    qdrant.add_argument("--tasks", type=Path, required=True)
    qdrant.add_argument("--memory-bank", type=Path, required=True)
    qdrant.add_argument("--out", type=Path, required=True)
    qdrant.add_argument("--qdrant-url", default="http://localhost:6333")
    qdrant.add_argument("--collection", default="trajectories")
    qdrant.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    qdrant.add_argument("--embedding-dim", type=int, default=384)
    qdrant.add_argument("--api-key", default=None)
    qdrant.add_argument("--k", type=int, default=3)
    qdrant.add_argument("--fetch-limit", type=int, default=50)
    qdrant.add_argument("--condition", default=None)
    qdrant.add_argument("--score-threshold", type=float, default=None)

    render = subparsers.add_parser("render-prompts")
    render.add_argument("--tasks", type=Path, required=True)
    render.add_argument("--runs", type=Path, required=True)
    render.add_argument("--retrievals", type=Path, required=True)
    render.add_argument("--out-dir", type=Path, required=True)
    render.add_argument("--top-k", type=int, default=3)
    render.add_argument("--max-memory-chars", type=int, default=12_000)

    args = parser.parse_args()

    if args.command == "build-manifest":
        adp_dirs = args.adp_dir or [
            Path("data/adp/nebius"),
            Path("data/adp/swe-gym"),
        ]
        build_manifest(
            adp_dirs=adp_dirs,
            out_dir=args.out_dir,
            max_tasks=args.max_tasks,
            eval_fraction=args.eval_fraction,
            seed=args.seed,
            min_repo_memory=args.min_repo_memory,
            run_seeds=parse_csv_ints(args.run_seeds),
            conditions=parse_csv_strings(args.conditions),
        )
    elif args.command == "attach-local-candidates":
        attach_local_candidates(
            tasks_path=args.tasks,
            memory_bank_path=args.memory_bank,
            out_path=args.out,
            k=args.k,
            seed=args.seed,
            mode=args.mode,
        )
    elif args.command == "attach-qdrant-candidates":
        attach_qdrant_candidates(
            tasks_path=args.tasks,
            memory_bank_path=args.memory_bank,
            out_path=args.out,
            qdrant_url=args.qdrant_url,
            collection=args.collection,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            api_key=args.api_key,
            k=args.k,
            fetch_limit=args.fetch_limit,
            condition=args.condition,
            score_threshold=args.score_threshold,
        )
    elif args.command == "render-prompts":
        render_prompts(
            tasks_path=args.tasks,
            runs_path=args.runs,
            retrievals_path=args.retrievals,
            out_dir=args.out_dir,
            top_k=args.top_k,
            max_memory_chars=args.max_memory_chars,
        )


if __name__ == "__main__":
    main()
