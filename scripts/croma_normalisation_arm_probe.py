"""Measure whether the DEV-2 normalisation arms actually differ at the encoder.

This is a **precursor** to the gated A/B experiment specified in
`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` section 4 -- it is NOT that
experiment, and its numbers do NOT decide anything.

Why it exists
-------------
The gated experiment's deciding metric is *fusion-head validation accuracy*. That
requires a trained head and a paired optical-SAR dataset.

> **Correction (2026-09-20).** This paragraph previously read *"neither exists yet
> (`training/fusion/` is empty and no optical-SAR corpus is on disk)"*. **Both clauses
> were stale.** `training/fusion/` now holds `train.py` (42,780 B), `extract.py`
> (70,222 B) and `reben_adapter.py` (16,927 B), with the original 0-byte `.gitkeep`
> still beside them; and the paired corpus **is** on disk at
> `data/bigearthnet_v2/reben/` (`BigEarthNet-S1`, `BigEarthNet-S2`, `Reference_Maps`).
> What remains true is the part that matters to this script's argument: **no trained
> fusion head exists**, because the gated DEV-2 A/B has not been run. So the deciding
> experiment still cannot run — for the head, not for the data. Factual correction
> only: no arm, metric or ruling is changed. Corrected here rather than left as a
> docstring defect because a probe that misstates the repository's state is a trap for
> the next reader.

But the spec also pre-registers a **secondary, non-deciding** metric:

    "mean per-channel activation statistics entering the encoder (to evidence
     that A and B genuinely differ in distribution, rather than differing only
     by noise)"

That part needs only the encoder, which now works. It answers a genuine
prerequisite question that is worth settling *before* anyone invests in the full
experiment:

    **If arm A and arm B drive CROMA to near-identical representations, the
    experiment is measuring noise and should not be run at all.**

Arms
----
    A  control  : per-channel mean +/- 2*std -> use_8_bit -> [0,1]   (the ruling)
    B  variant  : no encoder-input stretch (raw values, as stage 1 would leave them)

`C` (A with use_8_bit=False) is included because the code already supports it and
it isolates the quantisation decision for free.

What this CANNOT tell you
-------------------------
* Whether A is *better* than B. That is a task-accuracy question and needs a head.
* Anything about seed-to-seed variance of accuracy. Different quantity entirely.

Usage
-----
    python scripts/croma_normalisation_arm_probe.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _divergence(a: np.ndarray, b: np.ndarray) -> dict:
    """How different are two representation arrays, several ways."""
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    denom = (np.linalg.norm(a64) * np.linalg.norm(b64)) or 1.0
    cos = float(np.dot(a64, b64) / denom)
    l2 = float(np.linalg.norm(a64 - b64))
    rel = float(l2 / (np.linalg.norm(a64) or 1.0))
    return {
        "cosine_similarity": cos,
        "l2_distance": l2,
        "relative_l2": rel,
        # A cheap, interpretable "are these the same vector" test.
        "allclose_rtol1e-3": bool(np.allclose(a, b, rtol=1e-3, atol=1e-4)),
    }


def _make_inputs(batch: int, res: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Structured synthetic input in sensor-DN-like ranges.

    Deliberately NOT in [0, 1]: the whole question is what happens when raw
    sensor values are fed to a model expecting a stretched range. Reflectance
    x10000 and SAR in linear power are the realistic analogue.
    """
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0.0, 1.0, res * res, dtype=np.float32).reshape(res, res)
    optical = np.empty((batch, 12, res, res), dtype=np.float32)
    for b in range(batch):
        for c in range(12):
            base = ramp * 10_000.0 * (0.3 + 0.7 * (c + 1) / 12)
            optical[b, c] = base + rng.normal(0, 120.0, (res, res)).astype(np.float32)
    sar = np.empty((batch, 2, res, res), dtype=np.float32)
    for b in range(batch):
        for c in range(2):
            sar[b, c] = 2_000.0 * ramp * (0.5 + 0.5 * c) + rng.normal(
                0, 40.0, (res, res)
            ).astype(np.float32)
    return np.clip(optical, 0, None), np.clip(sar, 0, None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=120)
    ap.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "optical_sar" / "croma_arm_probe.json",
    )
    args = ap.parse_args(argv)

    from core.config import load_config
    from specialists.optical_sar.croma import build_encoder

    cfg = load_config()
    encoder = build_encoder(cfg)

    optical, sar = _make_inputs(args.batch_size, args.resolution)
    report: dict = {
        "phase": "11.5 (precursor only)",
        "decides_anything": False,
        "purpose": (
            "evidence that arms A and B differ in the distribution entering the "
            "encoder; NOT a task-accuracy comparison"
        ),
        "config_hash": cfg.hash,
        "input_dn_ranges": {
            "optical_min": float(optical.min()),
            "optical_max": float(optical.max()),
            "sar_min": float(sar.min()),
            "sar_max": float(sar.max()),
        },
        "arms": {},
    }

    reps: dict[str, dict[str, np.ndarray]] = {}

    # Arm A: the ruling -- normalise, 8-bit round trip.
    enc_a = encoder.encode(optical, sar)
    reps["A"] = {
        "optical_gap": enc_a.optical_gap,
        "sar_gap": enc_a.sar_gap,
        "joint_gap": enc_a.joint_gap,
    }
    report["arms"]["A"] = {
        "description": "per-channel mean +/- 2std -> use_8_bit=True (the ruling)",
        "input_normalisation": True,
        "use_8_bit": True,
    }

    # Arm B: no encoder-input stretch.
    # `normalize_input` is a *constructor* kwarg, not an `encode` parameter --
    # the encode signature is pinned by C-1 and must not change.
    from specialists.optical_sar.croma import CROMAEncoder

    enc_b = CROMAEncoder(
        encoder.model,
        resolution=encoder.resolution,
        device=encoder.device,
        normalize_input=False,
    ).encode(optical, sar)
    reps["B"] = {
        "optical_gap": enc_b.optical_gap,
        "sar_gap": enc_b.sar_gap,
        "joint_gap": enc_b.joint_gap,
    }
    report["arms"]["B"] = {
        "description": "no encoder-input stretch (raw sensor-range values)",
        "input_normalisation": False,
        "use_8_bit": None,
    }

    # Arm C: the ruling without the 8-bit quantisation.
    enc_c = CROMAEncoder(
        encoder.model,
        resolution=encoder.resolution,
        device=encoder.device,
        use_8_bit=False,
    ).encode(optical, sar)
    reps["C"] = {
        "optical_gap": enc_c.optical_gap,
        "sar_gap": enc_c.sar_gap,
        "joint_gap": enc_c.joint_gap,
    }
    report["arms"]["C"] = {
        "description": "per-channel mean +/- 2std -> use_8_bit=False",
        "input_normalisation": True,
        "use_8_bit": False,
    }

    # What the encoder actually received, per arm -- the spec's secondary metric.
    for arm, enc in (("A", enc_a), ("B", enc_b), ("C", enc_c)):
        if enc.radiometry is None:
            report["arms"][arm]["received_optical"] = "not normalised"
            continue
        w = enc.radiometry.optical.windows
        report["arms"][arm]["received_optical"] = {
            "normalised_channels": list(enc.radiometry.optical.normalised_channels),
            "unavailable_channels": list(enc.radiometry.optical.unavailable_channels),
            "first_window": (
                None if not w or w[0].lower is None
                else {"lower": w[0].lower, "upper": w[0].upper}
            ),
        }

    # Pairwise divergence of the representations.
    report["divergence"] = {}
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        report["divergence"][f"{x}_vs_{y}"] = {
            key: _divergence(reps[x][key], reps[y][key])
            for key in ("optical_gap", "sar_gap", "joint_gap")
        }

    # The headline question.
    ab = report["divergence"]["A_vs_B"]["joint_gap"]
    report["arms_are_distinguishable"] = not ab["allclose_rtol1e-3"]
    report["interpretation"] = (
        "A and B drive the encoder to materially different representations; the "
        "gated experiment is therefore measuring a real difference and is worth "
        "running once a fusion head and paired data exist."
        if report["arms_are_distinguishable"]
        else "A and B are indistinguishable at the encoder; the gated experiment "
        "would be measuring noise and should NOT be run."
    )

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 72)
    print("CROMA NORMALISATION ARM PROBE  (precursor -- decides nothing)")
    print("=" * 72)
    print(f"input DN ranges : optical [{report['input_dn_ranges']['optical_min']:.0f}"
          f", {report['input_dn_ranges']['optical_max']:.0f}]"
          f"  sar [{report['input_dn_ranges']['sar_min']:.0f}"
          f", {report['input_dn_ranges']['sar_max']:.0f}]")
    print()
    for pair, d in report["divergence"].items():
        j = d["joint_gap"]
        print(f"  {pair:<8} joint_GAP  cos={j['cosine_similarity']:+.4f}  "
              f"relL2={j['relative_l2']:.4f}  identical={j['allclose_rtol1e-3']}")
    print()
    print(f"arms distinguishable : {report['arms_are_distinguishable']}")
    print(f"interpretation       : {report['interpretation']}")
    print()
    print(f"report written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
