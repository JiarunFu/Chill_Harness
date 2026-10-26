# CHILL-Harness

CHILL-Harness is an experimental agent harness for containerized software
engineering and terminal benchmarks. It includes a base agent, a counterfactual
planner, and learned CHILL policy hooks for adaptive intervention, efficiency
control, and completion decisions.

The repository is intended to make the paper experiments reproducible without
including private job logs, external upload tooling, or full training artifacts.
Internal job identifiers and machine-specific metadata are omitted from this release.

## Components

| Component | Purpose |
|---|---|
| Base agent | Native tool-calling agent for Harbor benchmark runs. |
| CHILL / CF agent | Agent wrapper that invokes the counterfactual planner. |
| Counterfactual planner | Generates, scores, and selects planner interventions. |
| Learned policy support | Loads lightweight CHILL router/change/stop/intervention policies. |
| Reasoning controls | Normalizes reasoning-related model options. |
| Prompt template | System prompt used by the agent. |
| Run scripts | Convenience launchers for base, CF, and terminal profiles. |
| Analysis scripts | Job comparison, task filtering, event summarization, and offline policy training utilities. |

## Installation

This project uses Python 3.12 or newer.

```bash
uv sync
```

Alternatively, install the package into an existing environment:

```bash
pip install -e .
```

The main runtime dependencies are `harbor`, `litellm`, `anthropic`, and
`tenacity`. Benchmark execution requires Docker when using Harbor's Docker
environment.

## Configuration

Start from the example environment files:

```bash
cp .env.example .env
cp .env.chill .env.chill.local
```

Fill in `.env` with the model endpoint, API key, dataset, and Harbor runtime
settings. Keep secrets out of git. `.env.chill` contains non-secret CHILL
thresholds and profile defaults loaded by the CF launcher.

The experiments call remote LLM APIs through LiteLLM. Local hardware is used for
Docker environments, Harbor orchestration, planner control logic, task command
execution, and evaluation; model inference is not performed locally.

## Running Experiments

The public release keeps the checked-in launcher filenames unchanged for
reproducibility. This README refers to them by role:

| Role | Description |
|---|---|
| Base launcher | Runs the base agent profile. |
| CF launcher | Runs the CHILL / CF profile. |
| Terminal-profile launcher | Runs the Terminal-Bench CHILL profile. |

Example environment overrides:

```bash
CF_MODE=adaptive
CHILL_MODE=full
CHILL_POLICY_DIR=chill-policy/example-policy
```

## Learned Policy

CHILL learned policies are lightweight JSON models. A policy directory may
contain:

```text
router_model.json
change_model.json
stop_model.json
intervention_model.json
```

Training datasets and full historical experiment logs are intentionally not part
of the clean release. To train from local job events, use:

```bash
python scripts/train_chill_offline.py \
  --event-source jobs/<cf_job> \
  --output-dir artifacts/chill_policy
```

## Analysis

Compare paired base and CHILL runs:

```bash
python scripts/compare_runs.py jobs/<base_job> jobs/<cf_job> --verbose
```

Filter tasks from a completed job:

```bash
python scripts/filter_tasks.py jobs/<job> --status errored --format names
```

Summarize planner events:

```bash
python scripts/summarize_cf_events.py jobs/<cf_job>/cf_planner_events.jsonl
```

## Repository Hygiene

The public repository should not include:

- `.env` or any API keys
- raw `jobs/` outputs
- full training datasets
- machine-local paths
- private upload, notification, or deployment tooling

Use `.env.example` and `.env.chill` for reproducible configuration examples.
