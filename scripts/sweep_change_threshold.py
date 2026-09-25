"""Phase 9 -- select a change-detection threshold on the VALIDATION split.

    python scripts/sweep_change_threshold.py --data-root data/levir \
        --checkpoint artifacts/change/levir_change_v001/head.pt

WHAT IS REPORTED, AND WHY A SWEEP AT ALL
----------------------------------------
`evaluation/metrics/change.py` binarizes a probability map with a `>= threshold`
rule and documents its `DEFAULT_THRESHOLD = 0.5` as "a config value, not a tuned
one". The shipped head was scored at exactly 0.50 and reported precision 0.9195
against recall 0.8745 on a corpus where only ~5% of pixels change -- the shape of
a threshold that is too high. This script forwards the split ONCE (see
`training.change.train.collect_change_predictions`) and scores the same maps at
every threshold in a grid, so the whole precision/recall/IoU trade-off is visible
before one point is chosen.

Both aggregation conventions are carried for every threshold, exactly as the
single-shot evaluation carries them: POOLED (sum the confusion counts across
tiles) and MACRO (per-tile, then averaged, with empty-change tiles excluded and
counted separately). They can disagree by several points, so neither is silently
preferred.

Every threshold is scored by `evaluation.metrics.change.score_dataset` -- the
same function the shipped evaluation calls. The mask that is scored is the RAW
binarized one; no morphology is applied, for the reason spelled out in
`training/change/train.py`.

THE THRESHOLD IS SELECTED ON VALIDATION, AND ONLY ON VALIDATION
---------------------------------------------------------------
This is the point of the whole script, so it is enforced, not documented.

    * `val`   -- allowed. A held-out split the model did not fit on, so a
                 threshold chosen here is a legitimate selection.
    * `test`  -- REFUSED. Test is the one-shot benchmark. Selecting on it spends
                 it: the number stops being held-out and can never be reported as
                 such again.
    * `train` -- REFUSED. The fitting set. A threshold tuned there measures
                 memorisation, not generalisation.

Any `--split` other than `val` exits 4 and writes nothing. There is no override
flag, on purpose: an override is just a way to do the forbidden thing later.

SELECTION AND THE TIE-BREAK
---------------------------
`--select-by` chooses the scalar: `macro_iou` (default), `pooled_iou` or `f1`.
On an exact tie the HIGHER threshold wins -- fewer asserted pixels, fewer false
positives, the conservative direction on a 5%-change corpus. See
`training.change.train.sweep_thresholds`.

CONFIG DRIFT IS REFUSED, NOT WARNED ABOUT
-----------------------------------------
The detector's architecture travels inside the checkpoint, but the run's
configuration does not: `specialists.change.stanet.save_change_model` writes the
`config_hash` to the `model_metadata.json` sidecar beside the `.pt`. If that
recorded hash differs from the hash of the configuration in force now, the
comparison is not reproducible, so this script exits 3 and scores nothing unless
`--allow-config-drift` is passed -- in which case the drift is recorded in the
artifact. A missing sidecar is reported as "not checked", never as agreement.

BASELINE, AND WHY IT IS NOT ALWAYS 0.50
---------------------------------------
`delta_vs_baseline` compares the selected row against the grid point NEAREST the
shipped 0.50, and `baseline_threshold` records which point that actually was. On
the default grid it is 0.50; on an explicit grid that excludes 0.50 it is the
nearest point, so the artifact never claims a 0.50 baseline it did not use.

OVERWRITING IS DELIBERATE, NOT ACCIDENTAL
-----------------------------------------
An existing `threshold_sweep_val.json` is preserved by default (exit 2), because
a sweep is evidence. `--force` overwrites it and stamps `forced_overwrite=true`
into the payload, so the replacement is recorded rather than silent.

EXIT CODES
----------
    0   swept; artifact written
    2   the dataset or the checkpoint is missing / empty; the thresholds are
        malformed or outside [0,1] (rejected up front, before any compute); or
        the artifact already exists and --force was not given
    3   config drift, refused without --allow-config-drift
    4   a forbidden split was requested (anything other than `val`)

A ValueError from the scoring path is NOT an exit-2 case: it would mean a bug in
the model or metric code, so it surfaces as a traceback rather than being
disguised as bad input. See the note on the scoring `try`.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "artifacts" / "change"

#: The split a threshold may be selected on. Anything else exits 4.
ALLOWED_SPLIT = "val"

#: The shipped, untuned threshold. The sweep reports the row nearest this so the
#: delta against what was actually shipped is always visible.
SHIPPED_THRESHOLD = 0.50


def hr(title: str = "") -> None:
    print("-" * 70)
    if title:
        print(title)
        print("-" * 70)


def _environment_report(device: str) -> dict:
    facts: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": device,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        facts["torch"] = f"unavailable ({type(exc).__name__})"
    return facts


def _build_thresholds(args: argparse.Namespace) -> list[float]:
    """The thresholds to score: an explicit list, or start/stop/step inclusive.

    Rounded to 6 dp so `0.05 * 19` does not end up as 0.9500000000000001 and get
    printed as 0.95 in the curve but compared as something else by a test.

    Every failure mode raises ValueError HERE, before the model is built or a
    tile is read, so a typo costs nothing. `binarize` raises the same class for
    an out-of-range threshold, but only deep inside the scoring pass -- after the
    full forward -- so validating the range up front is what turns `--thresholds
    1.5` into a clean exit 2 instead of a 15-minute CPU run that dies with a
    traceback.
    """
    if args.thresholds is not None:
        # `is not None`, not truthiness: `--thresholds ""` is a request for zero
        # thresholds, and letting it fall through to the default grid would
        # silently answer a different question than the one asked.
        values = [part.strip() for part in args.thresholds.split(",")]
        values = [part for part in values if part]
        if not values:
            raise ValueError("--thresholds was given but parsed to nothing")
        out: list[float] = []
        for raw in values:
            try:
                t = round(float(raw), 6)
            except ValueError:
                raise ValueError(
                    f"--thresholds entry is not a number: {raw!r}"
                ) from None
            if not 0.0 <= t <= 1.0:
                raise ValueError(
                    f"--thresholds entry outside [0,1]: {t} (from {raw!r})"
                )
            out.append(t)
        return out

    if args.step <= 0:
        raise ValueError(f"--step must be > 0, got {args.step}")

    out = []
    t = args.start
    # `+ step/2` tolerates float accumulation without admitting an extra point
    # when stop is not exactly on the grid.
    while t <= args.stop + args.step / 2:
        out.append(round(t, 6))
        t += args.step
    if not out:
        raise ValueError(
            f"no thresholds between start={args.start} and stop={args.stop} "
            f"at step={args.step}"
        )
    # A generated grid can leave [0,1] too (e.g. `--stop 1.2`), and `binarize`
    # would reject it only after the forward pass.
    for t in out:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"generated threshold outside [0,1]: {t}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep change-detection thresholds on the validation split"
    )
    ap.add_argument("--data-root", required=True,
                    help="LEVIR-CD root holding train/val/test A,B,label trees")
    ap.add_argument("--checkpoint", required=True,
                    help="a head.pt written by scripts/train_change.py")
    ap.add_argument("--output-dir", default=None,
                    help="defaults to artifacts/change")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap scored tiles; for a smoke run, not a result")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split", default=ALLOWED_SPLIT,
                    help="the split to select on; ONLY 'val' is accepted")
    ap.add_argument("--thresholds", default=None,
                    help="explicit comma-separated list, e.g. '0.3,0.4,0.5'")
    ap.add_argument("--start", type=float, default=0.05)
    ap.add_argument("--stop", type=float, default=0.95)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--select-by", default="macro_iou",
                    choices=("macro_iou", "pooled_iou", "f1"))
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing threshold_sweep_val.json; without "
                         "it the artifact is preserved (exit 2) and the payload "
                         "records forced_overwrite=true when it is replaced")
    ap.add_argument("--allow-config-drift", action="store_true",
                    help="sweep anyway when the checkpoint's config hash differs; "
                         "the artifact then records that the run is not a "
                         "frozen-config selection")
    args = ap.parse_args()

    # -- HARD SPLIT FIREWALL, before anything is loaded or written -----------
    if args.split != ALLOWED_SPLIT:
        print("=" * 70)
        print("REFUSING -- a threshold may only be selected on validation")
        print("=" * 70)
        print(f"  requested split : {args.split!r}")
        print(f"  allowed split   : {ALLOWED_SPLIT!r}")
        print()
        print("  The threshold has to be chosen on the VALIDATION split:")
        print("    * val   -- held out from fitting, so selecting here is fair")
        print("    * test  -- the ONE-SHOT benchmark. Selecting on it spends it:")
        print("               the number stops being held-out and can never be")
        print("               reported as such again.")
        print("    * train -- the fitting set. Tuning here measures memorisation,")
        print("               not generalisation.")
        print()
        print("  There is no override for this. Pass --split val.")
        return 4

    from core.config import load_config
    from core.errors import SatQueryError
    from training.change.dataset import load_levir_dataset
    from training.change.train import (
        checkpoint_metadata,
        collect_change_predictions,
        load_trained_change_model,
        sweep_thresholds,
    )

    cfg = load_config()
    device = args.device or cfg.device_preference
    seed = args.seed if args.seed is not None else int(cfg.get("project.seed", 42))
    tile_size = int(cfg.get("change.tile_size", 256))
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR

    try:
        thresholds = _build_thresholds(args)
    except ValueError as exc:
        print(f"bad --thresholds / --start/--stop/--step: {exc}")
        return 2

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"checkpoint does not exist: {ckpt_path}")
        return 2

    # Never overwrite an existing artifact by default: a sweep is evidence, and
    # silently replacing it would make the recorded number un-auditable. --force
    # is the deliberate escape hatch (it also stamps forced_overwrite into the
    # payload), not a silent default.
    out_path = out_dir / "threshold_sweep_val.json"
    if out_path.exists() and not args.force:
        print(f"refusing to overwrite an existing artifact: {out_path}")
        print("  Move or remove it deliberately, pass a different --output-dir,")
        print("  or pass --force to overwrite it (the new artifact will record")
        print("  forced_overwrite=true).")
        return 2

    print("=" * 70)
    print("PHASE 9 -- CHANGE-DETECTION THRESHOLD SWEEP (VALIDATION ONLY)")
    print("=" * 70)
    print(f"data root   : {args.data_root}")
    print(f"checkpoint  : {ckpt_path}")
    print(f"split       : {args.split}")
    print(f"tile size   : {tile_size}px")
    print(f"device      : {device}")
    print(f"config hash : {cfg.hash}")
    print(f"select by   : {args.select_by}")
    print(f"thresholds  : {len(thresholds)} points "
          f"[{thresholds[0]:.2f} .. {thresholds[-1]:.2f}]")
    print(f"limit       : {args.limit or 'ALL tiles'}")
    print()

    hr("ENVIRONMENT")
    env = _environment_report(device)
    for key in sorted(env):
        print(f"  {key:<22} {env[key]}")
    print()

    # -- checkpoint identity, before any data is loaded --------------------
    hr("CHECKPOINT")
    import torch

    try:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"could not read checkpoint: {type(exc).__name__}: {exc}")
        return 2

    embedded_config = raw.get("config")
    if not embedded_config:
        print("checkpoint carries no embedded model config; refusing to guess")
        return 2

    sidecar = checkpoint_metadata(ckpt_path)
    ckpt_config_hash = sidecar.get("config_hash")
    drift_checked = "config_hash" in sidecar
    drift = bool(drift_checked and ckpt_config_hash not in (None, cfg.hash))

    print(f"  architecture      : {embedded_config}")
    print(f"  sidecar           : {ckpt_path.parent / 'model_metadata.json'}"
          f"{'' if drift_checked else '  (absent)'}")
    print(f"  checkpoint hash   : {ckpt_config_hash if drift_checked else 'NOT CHECKED'}")
    print(f"  current hash      : {cfg.hash}")
    print(f"  drift             : {drift}")
    print()

    if drift and not args.allow_config_drift:
        print("CONFIG DRIFT -- refusing to sweep.")
        print("  The head was trained under a different configuration, so this")
        print("  selection is not reproducible. Re-train, or pass")
        print("  --allow-config-drift to proceed knowingly (the artifact will")
        print("  record that the result is not a frozen-config selection).")
        return 3
    if drift:
        print("  [!] config drift acknowledged; this is NOT a frozen-config run")
        print()

    # -- dataset -----------------------------------------------------------
    hr(f"SWEEP SET ({args.split})")
    try:
        items = load_levir_dataset(
            args.data_root, splits=(args.split,), limit=args.limit, seed=seed
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2
    except SatQueryError as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 2

    if not items:
        print(f"no items in the {args.split!r} split")
        print("  Expected layout: <root>/{A,B,label}/%s_*.png" % args.split)
        return 2

    n_scenes = len({i.get("group_key") or i.get("image_key") or i["sample_id"]
                    for i in items})
    print(f"  tiles            : {len(items):,}")
    print(f"  scenes           : {n_scenes:,}")
    print()

    # -- forward once, score many ------------------------------------------
    hr("SCORING")
    try:
        model = load_trained_change_model(ckpt_path, device=device)
    except SatQueryError as exc:
        print(f"FAILED to load the detector: {type(exc).__name__}: {exc}")
        return 2
    print(f"  detector : {model}")

    started = datetime.now(timezone.utc)
    try:
        pairs = collect_change_predictions(
            model,
            items,
            device=device,
            batch_size=args.batch_size,
            tile_size=tile_size,
        )
        sweep = sweep_thresholds(pairs, thresholds, select_by=args.select_by)
    except SatQueryError as exc:
        # Deliberately NO `except ValueError` here. A threshold outside [0,1] is
        # rejected by `_build_thresholds` before any compute, and
        # `ChangeTrainingError` (unknown `select_by`, empty list) IS a
        # `SatQueryError`, so every *input* failure is already caught above.
        # A ValueError reaching this point is therefore a defect in the scoring
        # path, not bad input -- it must surface as a traceback so the bug can be
        # located, not be mislabelled as an invalid threshold and exit 2.
        print(f"FAILED to score: {type(exc).__name__}: {exc}")
        return 2
    seconds = (datetime.now(timezone.utc) - started).total_seconds()

    # -- the curve ---------------------------------------------------------
    # The row nearest the SHIPPED threshold, whatever the grid actually contains.
    # With the default grid that is the 0.50 row; with an explicit grid that
    # excludes 0.50 it is the nearest point, and `baseline_threshold` records
    # which one so the artifact never implies it was 0.50 when it was not.
    baseline = min(
        sweep.rows, key=lambda r: abs(r["threshold"] - SHIPPED_THRESHOLD)
    )
    baseline_threshold = baseline["threshold"]
    selected = sweep.selected
    delta = {
        "pooled_iou": round(selected["pooled"]["iou"] - baseline["pooled"]["iou"], 4),
        "macro_iou": round(selected["macro"]["iou"] - baseline["macro"]["iou"], 4),
        "pooled_f1": round(selected["pooled"]["f1"] - baseline["pooled"]["f1"], 4),
    }

    print()
    hr("THRESHOLD CURVE")
    print(f"  {'thr':>5}  {'pooled_iou':>10}  {'macro_iou':>10}  "
          f"{'pooled_f1':>10}  {'precision':>9}  {'recall':>9}")
    for row in sweep.rows:
        pooled, macro = row["pooled"], row["macro"]
        mark = "  <== selected" if row["threshold"] == selected["threshold"] else ""
        print(f"  {row['threshold']:>5.2f}  {pooled['iou']:>10.4f}  "
              f"{macro['iou']:>10.4f}  {pooled['f1']:>10.4f}  "
              f"{pooled['precision']:>9.4f}  {pooled['recall']:>9.4f}{mark}")
    print()
    print(f"  tiles with change : {sweep.n_images_with_change} / {sweep.n_pairs}")
    print(f"  selected          : threshold {selected['threshold']:.2f} "
          f"by {sweep.select_by}")
    print(f"  baseline          : threshold {baseline_threshold:.2f} "
          f"(nearest grid point to the shipped {SHIPPED_THRESHOLD:.2f})")
    print(f"  DELTA vs baseline : pooled_iou {delta['pooled_iou']:+.4f}  "
          f"macro_iou {delta['macro_iou']:+.4f}  "
          f"pooled_f1 {delta['pooled_f1']:+.4f}")
    print()

    # -- artifact ----------------------------------------------------------
    note = (
        "Threshold selected on the validation split only. The shipped test "
        "result at artifacts/change/eval_test/eval_result.json was scored at "
        "threshold 0.50 and has NOT been re-scored at the selected threshold. "
        "Re-scoring test would spend the one-shot benchmark and is a separate, "
        "explicit decision."
    )
    payload = {
        "artifact": "change_threshold_sweep",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "select_by": sweep.select_by,
        "n_pairs": sweep.n_pairs,
        "n_images_with_change": sweep.n_images_with_change,
        "thresholds": sweep.thresholds,
        "rows": sweep.rows,
        "selected": sweep.selected,
        "baseline": baseline,
        "baseline_threshold": baseline_threshold,
        "delta_vs_baseline": delta,
        "forced_overwrite": bool(args.force and out_path.exists()),
        "config_hash": cfg.hash,
        "checkpoint": str(ckpt_path),
        "checkpoint_config_hash": ckpt_config_hash,
        "checkpoint_config_hash_checked": drift_checked,
        "checkpoint_embedded_config": embedded_config,
        "config_drift": drift,
        "config_drift_acknowledged": bool(drift and args.allow_config_drift),
        "device": device,
        "tile_size": tile_size,
        "environment": env,
        "seconds": round(seconds, 3),
        "test_split_touched": False,
        "note": note,
    }
    if args.limit is not None:
        payload["smoke_run"] = True
        payload["smoke_run_note"] = (
            f"--limit {args.limit}: this is a smoke run, not a result"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    hr("WROTE")
    print(f"  {out_path}")
    if args.limit is not None:
        print()
        print(f"  [!] --limit {args.limit}: this is a SMOKE run, not a result.")
    hr("=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
