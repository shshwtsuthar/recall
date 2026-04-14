# Experiment 3: Static OpenHands Memory Injection

This experiment tests whether retrieved prior trajectories help a live
OpenHands run when injected once at task start.

The experiment is intentionally static first. Dynamic retrieval should come
later, after this experiment establishes whether memory is useful at all.

## Scientific Question

Does injecting retrieved trajectory memory improve live OpenHands software
engineering performance compared with no memory and weaker memory artifacts?

Primary comparison:

- `no_memory`
- `flat`
- `summary`
- `adp_lite`
- `adp_full`

The first live pilot can use the same retrieved candidates for all memory
conditions to isolate the injection artifact. The paper-level result should use
representation-specific retrieval candidates when testing retrieval
representation quality.

## Leakage Rules

Use instance-level splits.

- Evaluation tasks are selected by `instance_id`.
- The memory bank excludes trajectories with the same `instance_id`.
- The default task filter requires many same-repo memory trajectories from
  different instances so the run is not dominated by no-match cases.
- Exact same-instance retrieval is only an oracle diagnostic, not a main result.

## Step 1: Build The Manifest

Small smoke test:

```bash
python -m recall.eval.experiment3_static_injection build-manifest \
  --adp-dir data/adp/nebius \
  --adp-dir data/adp/swe-gym \
  --out-dir experiments/exp3_static_injection/smoke \
  --max-tasks 5 \
  --min-repo-memory 25 \
  --run-seeds 0 \
  --conditions no_memory,flat,summary,adp_lite,adp_full
```

Larger pilot:

```bash
python -m recall.eval.experiment3_static_injection build-manifest \
  --adp-dir data/adp/nebius \
  --adp-dir data/adp/swe-gym \
  --out-dir experiments/exp3_static_injection/pilot_50 \
  --max-tasks 50 \
  --min-repo-memory 50 \
  --run-seeds 0,1,2 \
  --conditions no_memory,flat,summary,adp_lite,adp_full
```

Outputs:

- `tasks.jsonl`: one row per held-out evaluation task.
- `runs.jsonl`: one row per task, condition, and seed.
- `memory_bank.jsonl`: allowed memory trajectories after the instance split.
- `manifest_summary.json`: counts and configuration.

## Step 2: Attach Candidate Memories

For a local smoke test, attach same-repo successful memories:

```bash
python -m recall.eval.experiment3_static_injection attach-local-candidates \
  --tasks experiments/exp3_static_injection/smoke/tasks.jsonl \
  --memory-bank experiments/exp3_static_injection/smoke/memory_bank.jsonl \
  --out experiments/exp3_static_injection/smoke/retrievals.local_same_repo_success.jsonl \
  --k 3 \
  --mode same_repo_success
```

This is not the main retrieval experiment. It is a deterministic way to verify
that prompt rendering and OpenHands execution work before Qdrant retrieval is
wired in.

For paper runs, replace this file with representation-specific Qdrant retrievals
using this schema:

```json
{
  "task_id": "nebius_swe_rebench_openhands:repo__project-123",
  "condition": "adp_lite",
  "candidate_source": "qdrant_adp_lite",
  "candidates": [
    {
      "rank": 1,
      "trajectory_id": "abc123",
      "path": "data/adp/nebius/abc123.json",
      "score": 0.82,
      "repo": "org/project",
      "instance_id": "org__project-999",
      "resolved": true
    }
  ]
}
```

If `condition` is `null`, the same candidates are reused for all memory
conditions of that task.

Example for the current single Qdrant collection:

```bash
python -m recall.eval.experiment3_static_injection attach-qdrant-candidates \
  --tasks experiments/exp3_static_injection/smoke/tasks.jsonl \
  --memory-bank experiments/exp3_static_injection/smoke/memory_bank.jsonl \
  --out experiments/exp3_static_injection/smoke/retrievals.qdrant_adp_lite.jsonl \
  --qdrant-url http://localhost:6333 \
  --collection trajectories \
  --condition adp_lite \
  --k 3 \
  --fetch-limit 50
```

For the main representation comparison, create separate collections or named
vectors for `flat`, `summary`, `adp_lite`, and `adp_full`, then run this command
once per condition and concatenate the retrieval JSONL files before rendering
prompts.

## Step 3: Render OpenHands Prompts

```bash
python -m recall.eval.experiment3_static_injection render-prompts \
  --tasks experiments/exp3_static_injection/smoke/tasks.jsonl \
  --runs experiments/exp3_static_injection/smoke/runs.jsonl \
  --retrievals experiments/exp3_static_injection/smoke/retrievals.local_same_repo_success.jsonl \
  --out-dir experiments/exp3_static_injection/smoke \
  --top-k 3 \
  --max-memory-chars 12000
```

Outputs:

- `prompts/<run_id>.txt`: the exact user prompt for OpenHands.
- `runs_with_prompts.jsonl`: run manifest with prompt paths.

The memory conditions are char-budgeted so `flat`, `summary`, `adp_lite`, and
`adp_full` are easier to compare. For the main paper, report the actual token
count sent to the model as a cost metric.

## Step 4: Run OpenHands

Run each row in `runs_with_prompts.jsonl` through the same OpenHands evaluator
configuration.

Record at minimum:

- `run_id`
- `task_id`
- `condition`
- `seed`
- model and OpenHands version
- prompt path
- trajectory output path
- final patch path
- resolved/pass status
- wall time
- total input/output tokens
- number of bash actions
- number of editor actions

The exact OpenHands command depends on how you launch the SWE-Gym or
SWE-rebench environment. The important part is that every condition receives
the same task environment and differs only by the prompt file.

## Step 5: Score The Runs

Primary metrics:

- success or resolved rate
- patch generated rate
- wall time
- input/output tokens
- bash/editor action counts
- harm rate: no-memory succeeds but memory condition fails

Use paired statistics by `task_id`.

- Binary success: paired bootstrap confidence intervals and McNemar or paired
  permutation tests.
- Cost/action counts: paired bootstrap confidence intervals over task-level
  deltas.

## Pilot Progression

Run in this order:

1. 5-task smoke test with local same-repo candidates.
2. 20-task pilot with no-memory vs `adp_lite` only.
3. 50-task pilot with all five conditions and one seed.
4. Full paired run with all conditions and three seeds.
5. Representation-specific Qdrant retrievals once the live harness is stable.
