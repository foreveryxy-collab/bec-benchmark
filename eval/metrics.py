"""
BEC Benchmark — Metrics computation (no API calls)
Recomputes stats_supplement.json from per-model result JSONs.

Usage:
  python eval/metrics.py                          # reads from results/
  python eval/metrics.py --results-dir path/to/   # custom results dir

Outputs results/<results-dir>/stats_supplement.json with:
  - wilson_ci: ZCRR + 95% Wilson CI per model x condition
  - breakdown: zombie rate per model x condition x mark
"""
import json, math, argparse
from pathlib import Path

MODELS = ["moonshot-v1-128k", "kimi-k2.5", "kimi-k2.6", "deepseek-v4-pro", "deepseek-v4-flash"]
CONDITIONS = ["WITH", "PRESSURE"]
MARKS = ["valid", "retracted", "stale", "low_trust"]


def wilson_ci(successes, trials, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denom
    return (max(0, center - spread), min(1, center + spread))


def main():
    parser = argparse.ArgumentParser(description="BEC metrics computation")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing per-model result JSONs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results"

    ci_table = []
    breakdown_table = []

    for model in MODELS:
        for cond in CONDITIONS:
            path = results_dir / f"{model}_{cond}.json"
            with open(path) as f:
                data = json.load(f)
            details = data["details"]

            # PRESSURE uses "final_verdict"; WITH uses "verdict"
            verdict_field = "final_verdict" if cond == "PRESSURE" else "verdict"

            eligible = [d for d in details if d["mark"] in ("retracted", "stale", "low_trust")
                        and d.get("recall") == "yes"]
            zombies = sum(1 for d in eligible if d.get(verdict_field) == "ZOMBIE")
            n = len(eligible)
            zcrr = zombies / n if n else 0
            lo, hi = wilson_ci(zombies, n)
            ci_table.append({
                "model": model, "condition": cond,
                "verdict_field_used": verdict_field,
                "ZCRR": round(zcrr, 4),
                "n_eligible": n, "n_zombie": zombies,
                "CI_95_lo": round(lo, 4), "CI_95_hi": round(hi, 4),
            })

            for mk in MARKS:
                items_mk = [d for d in details if d["mark"] == mk and d.get("recall") == "yes"]
                z_mk = sum(1 for d in items_mk if d.get(verdict_field) == "ZOMBIE")
                n_mk = len(items_mk)
                rate = z_mk / n_mk if n_mk else 0
                lo_mk, hi_mk = wilson_ci(z_mk, n_mk)
                breakdown_table.append({
                    "model": model, "condition": cond, "mark": mk,
                    "verdict_field_used": verdict_field,
                    "zombie": z_mk, "total": n_mk,
                    "zombie_rate": round(rate, 4),
                    "CI_lo": round(lo_mk, 4), "CI_hi": round(hi_mk, 4),
                })

    out_path = results_dir / "stats_supplement.json"
    with open(out_path, "w") as f:
        json.dump({
            "wilson_ci": ci_table,
            "breakdown": breakdown_table,
        }, f, ensure_ascii=False, indent=2)

    # Print summary table
    print(f"{'Model':<22} {'Cond':<10} {'ZCRR':>8} {'n':>5} {'zombies':>8} {'CI_95':>16}")
    print("-" * 72)
    for row in ci_table:
        ci_str = f"[{row['CI_95_lo']:.3f}, {row['CI_95_hi']:.3f}]"
        print(f"{row['model']:<22} {row['condition']:<10} {row['ZCRR']:>8.4f} "
              f"{row['n_eligible']:>5} {row['n_zombie']:>8} {ci_str:>16}")

    print(f"\nstats_supplement.json written to {out_path}")


if __name__ == "__main__":
    main()
