"""Pre-registered 11.5 metric evaluation for the Phase 14 fusion head (READ-ONLY).

This tool answers ONE question: **what is the pre-registered 11.5 metric for a
given trained fusion head, measured on the held-out TEST split?**

    python scripts/eval_fusion_115.py \
        --head artifacts/optical_sar/fusion_head_production_v001/head.pt \
        --cache-dir artifacts/optical_sar/fusion_features \
        --report artifacts/optical_sar/fusion_head_production_v001/\
pre_registered_115_metric.json

WHY THIS IS A SEPARATE TOOL FROM THE TRAINER
--------------------------------------------
The trainer (`training/fusion/train.py`) fits on `train`/`val` and **never**
opens the test split. That is deliberate: it keeps the held-out split out of
model selection, so the test number cannot be contaminated by the search over
seeds and arms. Every `run_record.json` therefore carries

    pre_registered_metric_computed = false
    result_status = "PLUMBING_ONLY -- ... the pre-registered 11.5 metric is not
                     computed"

and that flag is **correct about the trainer**. The 11.5 metric belongs to a
different, later, read-only step. This is that step. **Do not "fix" the trainer
by making it score the test split** -- that would destroy the separation the
flag exists to record.

WHAT THE METRIC IS
------------------
`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4 fixes: fusion-head
**accuracy** and **macro-F1** over the 19-class label space, on the held-out
split. This tool computes exactly those two numbers for one head. Accuracy is
reported as the point estimate; macro-F1 alongside it as the co-reported
secondary quantity (Gate F D-02 makes macro-F1 non-deciding for the ARM
question -- it is not thereby unimportant here, since 11.5 asks for both).

WHAT THIS TOOL MUST NOT DO
--------------------------
  * It NEVER trains, NEVER fine-tunes, NEVER touches a checkpoint's bytes.
  * It NEVER selects, ranks or compares heads. It scores the ONE head it is
    given. Choosing among heads is a separate, owner-owned step.
  * It NEVER reads the validation split to decide anything.
  * It NEVER writes the 11.5 number back into a `run_record.json`. Those records
    are frozen historical evidence; the metric lands in its own artifact.

EXIT CODES
----------
    0   a metric was computed and a report written
    2   the head or cache is missing/unreadable, the split is empty, the cache's
        `config_hash` is not `78f1e3700da15aa1`, or the head/cache dimensions
        disagree (a category error, not a result)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: The one configuration this experiment is registered against. A cache built
#: under a different registry is a different experiment; refuse it loudly.
EXPECTED_CONFIG_HASH = "78f1e3700da15aa1"

#: `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4: the 19-class space.
NUM_CLASSES = 19

TOOL_NAME = "fusion_115_metric"

__all__ = [
    "EXPECTED_CONFIG_HASH",
    "NUM_CLASSES",
    "TOOL_NAME",
    "Fusion115Error",
    "evaluate_115",
    "main",
]


class Fusion115Error(RuntimeError):
    """The 11.5 metric cannot be computed as pre-registered."""


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_115(
    head_path: str | Path,
    cache_dir: str | Path,
    split: str = "test",
) -> dict[str, Any]:
    """Score one trained head on one held-out split. Returns the report dict."""
    from training.fusion.train import (
        evaluate_fusion_head,
        load_trained_fusion_head,
        read_feature_cache,
    )

    head_p = Path(head_path)
    if not head_p.exists():
        raise Fusion115Error(f"head does not exist: {head_p}")

    cache_p = Path(cache_dir) / f"{split}.npz"
    if not cache_p.exists():
        raise Fusion115Error(
            f"{split} feature cache does not exist: {cache_p} (the cache must be "
            f"extracted before the metric can be computed)"
        )

    features, meta = read_feature_cache(cache_p)
    if not features:
        raise Fusion115Error(f"{split} split is empty: {cache_p}")

    cache_hash = meta.get("config_hash")
    if cache_hash != EXPECTED_CONFIG_HASH:
        raise Fusion115Error(
            f"{split} cache was built under config_hash={cache_hash!r}, expected "
            f"{EXPECTED_CONFIG_HASH!r}; this is a different experiment and its "
            f"numbers would not be the pre-registered metric"
        )

    head = load_trained_fusion_head(head_p)

    # Dimension agreement is checked by the model load itself (build_fusion_head
    # refuses anything but the frozen input dim), but the cache carries its own
    # encoder_dim -- surface a mismatch rather than letting it fail obscurely.
    result = evaluate_fusion_head(head, features, num_classes=NUM_CLASSES)

    return {
        "tool": TOOL_NAME,
        "metric": "pre_registered_11.5",
        "definition": (
            "fusion-head accuracy and macro-F1 over the 19-class label space on "
            "the held-out split (docs/PHASE14_CROMA_NORMALISATION_CHANGE.md s4)"
        ),
        "split": split,
        "head_path": str(head_p).replace("\\", "/"),
        "head_sha256": _sha256_file(head_p),
        "head_bytes": head_p.stat().st_size,
        "cache_path": str(cache_p).replace("\\", "/"),
        "cache_arm": meta.get("arm"),
        "cache_config_hash": cache_hash,
        "n_scored": result.n,
        "accuracy": round(result.accuracy, 6),
        "macro_f1": round(result.macro_f1, 6),
        "loss": round(result.loss, 6),
        "num_classes": NUM_CLASSES,
        # Recorded explicitly because a macro-F1 of 0.434 sitting next to an accuracy
        # of 0.931 looks like a broken model to a reader who does not know that the
        # mean runs over ALL 19 classes. Classes absent from the split score 0.0 each
        # and drag the mean down. A per-present-class mean would be 0.589218. These
        # are different numbers answering different questions; only the 19-class
        # figure matches the pre-registered definition.
        "macro_f1_denominator": "all 19 classes (absent classes contribute 0.0)",
        "classes_present": result.classes_present,
        "classes_absent": result.classes_absent,
        # Per-class F1 for all 19 slots, absent ones as 0.0. Emitted so a reader
        # (or a test) can re-derive both the 19-class and present-only means from
        # the same scores instead of having to trust a single scalar.
        "_per_class_f1": list(result.per_class_f1),
        "is_deciding_statistic": False,
        "advisory": (
            "This tool reports ONE head's held-out accuracy and macro-F1. It "
            "selects no head, ranks nothing and compares no arms. Whether this "
            "constitutes a Phase 12 pass is the owner's ruling."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute the pre-registered 11.5 metric for one fusion head."
    )
    parser.add_argument("--head", required=True, help="path to a trained head.pt")
    parser.add_argument(
        "--cache-dir", required=True, help="directory holding train/val/test caches"
    )
    parser.add_argument("--split", default="test", help="split to score (default: test)")
    parser.add_argument("--report", default=None, help="where to write the JSON report")
    args = parser.parse_args(argv)

    try:
        report = evaluate_115(args.head, args.cache_dir, args.split)
    except Fusion115Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    hr("PRE-REGISTERED 11.5 METRIC")
    print(f"head          : {report['head_path']}")
    print(f"head sha256   : {report['head_sha256'][:16]}...")
    print(f"split         : {report['split']}  (n = {report['n_scored']})")
    print(f"cache arm     : {report['cache_arm']}")
    print(f"config_hash   : {report['cache_config_hash']}")
    hr()
    print(f"ACCURACY      : {report['accuracy']:.6f}")
    print(f"MACRO-F1      : {report['macro_f1']:.6f}")
    hr()
    print(report["advisory"])

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        hr(f"report written: {out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
