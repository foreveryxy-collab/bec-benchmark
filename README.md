# BEC: Behavioral Execution Consistency

A benchmark for **governance-marker compliance in LLM agent memory under multi-turn pressure**.

When an agent's persistent memory carries a system-assigned governance marker — `retracted`, `stale`, or `low-trust` — does the model still honor that marker when a user applies sustained pressure to ignore it? BEC separates **recall** (does the model remember the marker?) from **compliance** (does it still act on it?), and measures how compliance holds up under three escalating rounds of user pressure.

This repository contains the BEC item set, the evaluation harness, and the raw run outputs accompanying the paper:

> **Present but Not Enforced: Benchmarking Governance-Marker Compliance in LLM Agent Memory Under Multi-Turn Pressure.** Xiaoyang Yu, 2026. arXiv:XXXX.XXXXX *(link to be added)*.

## What it measures

- **Two conditions.** `WITH` (the marked memory is present, no pressure) vs. `PRESSURE` (three escalating rounds in which the user insists the marked memory is still valid).
- **Marker classes.** `retracted`, `stale`, `low-trust` (the *revoked* classes, 105 items), plus a `valid` negative control (15 items) — memories that are correct and *should* be used.
- **Three-way judging.** Each response is scored `COMPLIANT` / `ZOMBIE` / `ABSTAIN`.
- **Metrics.**
  - **ZCRR** (zombie-compliance rate): fraction of recalled revoked-class items judged `ZOMBIE` (ABSTAIN counts in the denominator as non-zombie).
  - **VMUR** (valid-memory use rate): fraction of recalled valid items judged `COMPLIANT`.
  - **Recall**: fraction of items where the model indicates it holds the record, including its revoked status.
  - 95% Wilson confidence intervals on ZCRR.

## Headline result (5 models, 2 vendors)

| Model | Recall | ZCRR (no pressure) | ZCRR (pressure) | VMUR |
|---|---|---|---|---|
| moonshot-v1-128k  | 1.00 | 0.010 | 0.971 | 0.933 |
| kimi-k2.5         | 1.00 | 0.000 | 0.362 | 0.867 |
| kimi-k2.6         | 1.00 | 0.000 | 0.219 | 0.867 |
| deepseek-v4-pro   | 1.00 | 0.000 | 0.838 | 0.933 |
| deepseek-v4-flash | 1.00 | 0.000 | 0.857 | 0.933 |

Without pressure, every model recalls the markers in full and almost never misuses them. Under pressure, zombie-compliance on the marked memories rises to **0.219–0.971**, while valid memories are almost never harmed (0/15 on four models, 1/15 on one). The failure is not perceptual; it appears only under pressure, and it is selective — failure tracks marker strength and falls almost entirely on the governance-marked class.

## Repository layout

```
bec-benchmark/
├── README.md
├── LICENSE
├── data/
│   └── bec_items.json              # the 120 items (retracted / stale / low-trust / valid)
├── eval/
│   ├── run_experiment.py           # run WITH and PRESSURE conditions against all 5 models
│   ├── cross_validate.py           # cross-family judge agreement check (kimi-k2.6 re-judges)
│   └── metrics.py                  # recompute ZCRR, VMUR, recall, Wilson CIs from result JSONs
└── results/
    ├── <model>_<condition>.json    # per-item raw responses + judge verdicts (10 files)
    ├── stats_supplement.json       # precomputed Wilson CIs and breakdown tables
    └── cross_validation.json       # judge cross-validation results (25/28 agree, 89.3%; 30 sampled, 2 JUDGE_ERROR excluded)
```

## Data format

Each of the 120 items in `data/bec_items.json` has the following fields:

| Field | Description |
|---|---|
| `id` | Unique identifier (e.g., `fc-v-001`, `fc-r-015`) |
| `fact_text` | The memory statement stored in the agent's context |
| `mark` | Governance marker: `valid`, `retracted`, `stale`, or `low_trust` |
| `trust_weight` | Numeric trust weight (0–1) |
| `supersede_ref` | Reference to superseding record, if any |
| `correct_fact_text` | The corrected/current version (same as `fact_text` for `valid` items) |
| `domain` | Topic domain of the memory |
| `recall_probe` | Prompt used to test whether the model recalls the memory |
| `compliance_probe` | Prompt used to test whether the model acts on the marked memory |
| `compliant_answer` | Reference answer for a compliant response |
| `zombie_answer` | Reference answer for a zombie response (ignores the marker) |
| `scoring_key` | Criteria the judge uses to classify the response |

Mark distribution: `valid` 15, `retracted` 30, `stale` 30, `low_trust` 45.

## Result file format

Each `results/<model>_<condition>.json` contains:

- **Top-level metrics**: `recall_rate`, `ZCRR`, `VMUR`, `judge_error`
- **`by_mark`**: per-mark aggregation (recalled/total/zombie counts)
- **`details`**: array of 120 per-item records with raw model responses and judge outputs

For `WITH` condition, each detail has a `verdict` field. For `PRESSURE` condition, each detail has `rounds` (array of up to 3 rounds) and a `final_verdict` field.

### Known errata: `by_mark` aggregation

The `by_mark` summary field in some result files may contain counts that do not match the per-item `details` array (a bug in the original aggregation code that counted based on in-memory state rather than the verdict field). **The authoritative counts are always the per-item `details` — do not use `by_mark` for aggregate statistics.** Use `eval/metrics.py` to recompute correct aggregates from `details`.

## Running the benchmark

### Prerequisites

```bash
pip install openai
```

### Environment setup

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
export MOONSHOT_API_KEY="your-moonshot-key"
```

### Run the experiment

```bash
# Smoke test (8 items per model, ~10 min)
python eval/run_experiment.py --smoke

# Full run (120 items x 5 models x 2 conditions, ~4-6 hours)
python eval/run_experiment.py --full
```

### Recompute metrics from existing results

```bash
python eval/metrics.py
```

This reads all `results/<model>_<condition>.json` files and regenerates `results/stats_supplement.json` with Wilson CIs and per-mark breakdowns.

### Cross-validate judge consistency

```bash
python eval/cross_validate.py
```

Samples 30 items (seed=42) and re-judges them with cross-family models to measure inter-judge agreement. JUDGE_ERROR items (judge returned empty string) are excluded from the agreement denominator.

## Reproducibility notes

- The judge (deepseek-v4-pro) is used as a scalable stand-in for structured human annotation, **not** to define ground truth; its verdicts were cross-validated against independent cross-family judges with 89.3% agreement (25/28; 30 sampled, 2 JUDGE_ERROR excluded from denominator; see `results/cross_validation.json` and the paper).
- Raw per-item responses and judge outputs are included under `results/` so that every reported number can be re-derived from source.
- `PRESSURE` results use a `final_verdict` field (the last round's verdict, or `ZOMBIE` if the model capitulated early). `WITH` results use a `verdict` field. The `eval/metrics.py` script handles this distinction automatically.

## Citation

```bibtex
@misc{yu2026bec,
  title         = {Present but Not Enforced: Benchmarking Governance-Marker Compliance in LLM Agent Memory Under Multi-Turn Pressure},
  author        = {Xiaoyang Yu},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
*(Fill in the arXiv ID after the preprint is posted.)*

## License

Code is released under the MIT License (see `LICENSE`). The BEC item set and run outputs are released for research use under CC BY 4.0 — please cite the paper.
