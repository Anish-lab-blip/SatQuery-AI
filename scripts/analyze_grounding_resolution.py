"""Paired analysis of the grounding resolution experiment.

Reads the per-sample JSONL files written by `exp_grounding_resolution.py` and
computes the statistics that two independent means cannot give:

  * the mean PAIRED difference in best IoU, with a 95% CI
  * the win/loss/tie split, so a mean cannot hide a bimodal distribution
  * the paired difference at every threshold on the recall ladder

This is ANALYSIS, not the pre-registered verdict. The verdict rule lives in
`exp_grounding_resolution.py` and is reported alongside; if the two disagree,
both are shown and the disagreement is the finding.

Paired beats independent here because both resolutions see the SAME samples.
The variance that matters is the variance of the per-sample difference, not the
variance of either resolution's score -- and objects that are hard for 224 tend
to be hard for 448 too, which makes the paired standard error much smaller.

    python scripts/analyze_grounding_resolution.py \
        --dir artifacts/grounding --tag _full

Exit codes:
    0  analysis produced
    1  one or both per-sample files missing
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESOLUTIONS: tuple[int, ...] = (224, 448)
THRESHOLD_LADDER: tuple[float, ...] = (0.10, 0.25, 0.50)
IOU_BUCKETS: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.01)


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def load_records(path: Path) -> dict[str, dict]:
    """sample_id -> record. Keyed so the pairing is by identity, not by order."""
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records[row["sample_id"]] = row
    return records


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Mean difference (b - a) with a 95% CI and a paired t statistic."""
    diff = b - a
    n = diff.size
    if n == 0:
        return {"n": 0}
    mean = float(diff.mean())
    sd = float(diff.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    # Normal approximation. At n in the thousands the t and z quantiles are
    # indistinguishable; at small n this is reported as approximate.
    half = 1.96 * se
    t_stat = mean / se if se > 0 else 0.0
    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "se_diff": se,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "t_stat": t_stat,
        "approx": n < 100,
    }


def histogram(values: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for lo, hi in zip(IOU_BUCKETS[:-1], IOU_BUCKETS[1:]):
        out[f"{lo:.2f}-{hi:.2f}"] = int(np.sum((values >= lo) & (values < hi)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired resolution analysis")
    ap.add_argument("--dir", default=str(REPO_ROOT / "artifacts" / "grounding"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--min-iou-for-example", type=float, default=None,
                    help="print up to 10 samples whose 448 best-IoU exceeds this")
    args = ap.parse_args()

    base = Path(args.dir)
    tag = args.tag
    paths = {r: base / f"per_sample_{r}{tag}.jsonl" for r in RESOLUTIONS}

    print("=" * 70)
    print("GROUNDING RESOLUTION -- PAIRED ANALYSIS")
    print("=" * 70)
    for r, p in paths.items():
        print(f"  {r:>3}px : {p}  {'FOUND' if p.exists() else 'MISSING'}")
    print()

    missing = [r for r, p in paths.items() if not p.exists()]
    if missing:
        print(f"missing per-sample file(s) for {missing}; run the experiment first")
        return 1

    data = {r: load_records(paths[r]) for r in RESOLUTIONS}
    shared = sorted(set(data[224]) & set(data[448]))
    only_coarse = len(set(data[224]) - set(data[448]))
    only_fine = len(set(data[448]) - set(data[224]))

    hr("PAIRING")
    print(f"  224 only : {only_coarse}")
    print(f"  448 only : {only_fine}")
    print(f"  paired   : {len(shared)}")
    if not shared:
        print("  no shared sample ids -- the two files are not from the same run")
        return 1
    if only_coarse or only_fine:
        print("  [!] unpaired samples exist and are EXCLUDED from the paired test")
    print()

    a_best = np.asarray([data[224][s]["best_iou"] for s in shared], dtype=np.float64)
    b_best = np.asarray([data[448][s]["best_iou"] for s in shared], dtype=np.float64)

    # -- headline ---------------------------------------------------------
    hr("BEST IoU  (paired)")
    stats = paired_stats(a_best, b_best)
    print(f"  mean 224            : {a_best.mean():.4f}")
    print(f"  mean 448            : {b_best.mean():.4f}")
    print(f"  mean paired diff    : {stats['mean_diff']:+.4f}"
          f"  (95% CI {stats['ci95_low']:+.4f} .. {stats['ci95_high']:+.4f})")
    print(f"  paired SD           : {stats['sd_diff']:.4f}")
    print(f"  paired SE           : {stats['se_diff']:.4f}")
    print(f"  t statistic         : {stats['t_stat']:+.2f}")
    if stats["approx"]:
        print("  [!] n < 100: the normal-approximation CI is approximate")
    ci_excludes_zero = (stats["ci95_low"] > 0) or (stats["ci95_high"] < 0)
    print(f"  CI excludes zero    : {ci_excludes_zero}")
    print()

    wins = int(np.sum(b_best > a_best))
    losses = int(np.sum(b_best < a_best))
    ties = int(np.sum(b_best == a_best))
    total = len(shared)
    print(f"  448 better on       : {wins}/{total} ({100.0 * wins / total:.1f}%)")
    print(f"  448 worse on        : {losses}/{total} ({100.0 * losses / total:.1f}%)")
    print(f"  identical           : {ties}/{total} ({100.0 * ties / total:.1f}%)")
    print()

    # -- distribution -----------------------------------------------------
    hr("BEST IoU DISTRIBUTION")
    ha, hb = histogram(a_best), histogram(b_best)
    print(f"  {'bucket':<12}{'224':>8}{'448':>8}")
    for key in ha:
        print(f"  {key:<12}{ha[key]:>8}{hb[key]:>8}")
    print()

    # -- recall ladder ----------------------------------------------------
    hr("RECALL LADDER  (paired, over samples with targets)")
    with_targets = [s for s in shared if data[224][s]["n_targets"] > 0]
    print(f"  samples with targets: {len(with_targets)}")
    print()
    print(f"  {'threshold':<12}{'224':>10}{'448':>10}{'diff':>10}{'95% CI':>24}")
    ladder_rows = []
    for thr in THRESHOLD_LADDER:
        key = f"{thr:.2f}"
        if with_targets:
            ra = float(np.mean([data[224][s]["recall"][key] for s in with_targets]))
            rb = float(np.mean([data[448][s]["recall"][key] for s in with_targets]))
            da = np.asarray([data[224][s]["recall"][key] for s in with_targets])
            db = np.asarray([data[448][s]["recall"][key] for s in with_targets])
            st = paired_stats(da, db)
            ci = f"{st['ci95_low']:+.4f} .. {st['ci95_high']:+.4f}"
        else:
            ra = rb = 0.0
            ci = "n/a"
        ladder_rows.append({"threshold": thr, "recall_224": ra, "recall_448": rb})
        print(f"  {thr:<12.2f}{ra:>10.4f}{rb:>10.4f}{rb - ra:>+10.4f}{ci:>24}")
    print()

    # -- decision ---------------------------------------------------------
    hr("DECISION")
    pre_reg = None
    summary_path = base / f"resolution_experiment{tag}.json"
    if summary_path.exists():
        pre_reg = json.loads(summary_path.read_text(encoding="utf-8")).get("verdict")

    if pre_reg:
        print(f"  pre-registered verdict : {pre_reg['verdict']}")
        print(f"  reason                 : {pre_reg['reason']}")
    else:
        print("  pre-registered verdict : not found (summary json missing)")

    print()
    paired_positive = stats["ci95_low"] > 0
    paired_negative = stats["ci95_high"] < 0
    if paired_positive:
        print("  paired test           : 448 is significantly better on best IoU")
    elif paired_negative:
        print("  paired test           : 224 is significantly better on best IoU")
    else:
        print("  paired test           : no significant difference in best IoU")

    ladder_gain = max((r["recall_448"] - r["recall_224"]) for r in ladder_rows)
    ladder_best = max(ladder_rows, key=lambda r: r["recall_448"] - r["recall_224"])
    print(f"  best ladder gain      : {ladder_gain:+.4f} at IoU {ladder_best['threshold']:.2f}")

    print()
    if pre_reg and paired_positive and pre_reg["verdict"] == "224 WINS":
        print("  [DISAGREEMENT] The pre-registered rule says 224 WINS while the")
        print("  paired test finds a significant 448 advantage. Do NOT freeze on")
        print("  this run: report the disagreement and decide explicitly.")
    elif pre_reg and pre_reg["verdict"] == "INCONCLUSIVE":
        print("  INCONCLUSIVE under the pre-registered rule. Do not freeze.")
    hr("=")

    if args.min_iou_for_example is not None:
        hr("EXAMPLES WHERE 448 WINS BIGGEST")
        order = np.argsort(b_best - a_best)[::-1]
        shown = 0
        for i in order:
            if b_best[i] < args.min_iou_for_example:
                break
            print(f"  {shared[i]:<28} 224={a_best[i]:.3f}  448={b_best[i]:.3f}"
                  f"  (+{b_best[i] - a_best[i]:.3f})")
            shown += 1
            if shown >= 10:
                break
        if shown == 0:
            print(f"  none above {args.min_iou_for_example}")
        hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())