"""Summarize paired ODE evaluations without changing the scoring rule."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path


METRICS = (
    "total", "clap_condition", "content_enjoyment",
    "production_quality", "signal_quality", "hybrid_reward",
)


def metric(row, key):
    if key == "hybrid_reward":
        learned = (
            0.5 * row["clap_condition"]
            + 0.2 * row["content_enjoyment"]
            + 0.3 * row["production_quality"]
        )
        return 0.7 * learned + 0.3 * row["signal_quality"]
    return row[key]


def summarize(paths, baseline, candidates):
    records = [json.loads(Path(path).read_text()) for path in paths]
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("Duplicate seed in evaluation inputs")
    result = {"seeds": [record["seed"] for record in records], "comparisons": {}}
    for candidate in candidates:
        comparisons = {}
        for key in METRICS:
            by_seed = []
            baseline_by_seed = []
            candidate_by_seed = []
            for record in records:
                base_rows = record["models"][baseline]["rows"]
                cand_rows = record["models"][candidate]["rows"]
                base_index = {(r["condition_id"], r["candidate"]): r for r in base_rows}
                cand_index = {(r["condition_id"], r["candidate"]): r for r in cand_rows}
                if base_index.keys() != cand_index.keys():
                    raise ValueError(f"Unpaired rows for seed {record['seed']}")
                order = sorted(base_index)
                base_values = [metric(base_index[i], key) for i in order]
                cand_values = [metric(cand_index[i], key) for i in order]
                baseline_by_seed.append(statistics.mean(base_values))
                candidate_by_seed.append(statistics.mean(cand_values))
                by_seed.append(statistics.mean(c - b for b, c in zip(base_values, cand_values)))
            n = len(by_seed)
            comparisons[key] = {
                "baseline_mean": statistics.mean(baseline_by_seed),
                "candidate_mean": statistics.mean(candidate_by_seed),
                "paired_delta": statistics.mean(by_seed),
                "seed_level_se": statistics.stdev(by_seed) / math.sqrt(n) if n > 1 else None,
                "positive_seeds": sum(delta > 0 for delta in by_seed),
                "n_seeds": n,
                "n_audio": sum(len(r["models"][baseline]["rows"]) for r in records),
                "per_seed_delta": by_seed,
            }
        result["comparisons"][candidate] = comparisons
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="JSON path or glob")
    parser.add_argument("--baseline", default="awm800")
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted({p for pattern in args.input for p in glob.glob(pattern)})
    if not paths:
        raise FileNotFoundError(args.input)
    result = summarize(paths, args.baseline, args.candidate)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
