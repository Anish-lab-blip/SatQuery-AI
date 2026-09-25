"""Execute the first REAL CROMA forward pass and record what it produced.

Phase 11, step 11.4. Until this runs, every statement about CROMA's behaviour in
this project is a claim about the README. This script replaces those claims with
measured tensors.

What it does NOT do: train anything, or write to any frozen artefact. It loads the
pinned checkpoint, runs inference, and writes one JSON report.

Design notes
------------
* It uses the **production** load path (`build_encoder(config)`, no explicit
  `checkpoint_path`) so that the pinned revision and `hf_hub_download` are
  exercised, not bypassed. Passing a local path would be easier and would test
  less.
* Every assertion is something that would be *false* under a plausible defect.
  "The forward pass ran" is not evidence; "`joint_GAP` is (2, 768) and finite and
  differs between the two samples" is.
* DEV-3 (225 patches) is checked here for the first time against a real model.
  `docs/ARCHITECTURE_CHANGE_CROMA.md` records it as CONSISTENT but UNVERIFIED.

Usage
-----
    python scripts/verify_croma_forward.py
    python scripts/verify_croma_forward.py --json artifacts/optical_sar/croma_forward.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The expected encoder width, from the frozen contract.
VERIFIED_ENCODER_DIM = 768
# DEV-3: 120 / 8 = 15, 15^2 = 225. Asserted here against a real model.
EXPECTED_PATCHES_120 = 225
# Frozen fusion width: 3 * 768 + 12 + 2.
EXPECTED_FUSION_DIM = 2318


def _sha256(path: Path, *, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _rss_bytes() -> int | None:
    """Current RSS of this process, in bytes.

    `resource` is POSIX-only, so on Windows it is unavailable and the first
    version of this script silently reported `null`. `psutil` is the portable
    route and is already a project dependency. Returns None only if neither
    works -- and None is recorded as such rather than quietly becoming 0.
    """
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001
        pass
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value) if platform.system() == "Windows" else int(value) * 1024
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "optical_sar" / "croma_forward.json",
    )
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--resolution", type=int, default=120)
    args = ap.parse_args(argv)

    report: dict = {
        "phase": "11.4",
        "purpose": "first real CROMA forward pass",
        "usable_as_evidence": False,
    }

    # ------------------------------------------------------------------ torch
    import numpy as np
    import torch

    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    report["cuda_device_count"] = int(torch.cuda.device_count())

    from core.config import load_config

    cfg = load_config()
    device = cfg.device_preference
    report["device"] = str(device)
    report["config_hash"] = cfg.hash
    report["frozen_hash_intact"] = cfg.hash == "78f1e3700da15aa1"

    # -------------------------------------------------------- production load
    from specialists.optical_sar.croma import build_encoder

    t0 = time.perf_counter()
    encoder = build_encoder(cfg)  # no checkpoint_path: exercises the Hub path
    load_seconds = time.perf_counter() - t0
    report["load_seconds"] = round(load_seconds, 3)

    model = encoder.model
    n_params = sum(p.numel() for p in model.parameters())
    report["model_parameters"] = int(n_params)
    report["model_class"] = type(model).__name__
    report["model_modality"] = getattr(model, "modality", None)
    report["model_size"] = getattr(model, "size", None)
    report["model_image_resolution"] = getattr(model, "image_resolution", None)

    # The checkpoint actually used, and its identity.
    from huggingface_hub import hf_hub_download

    ckpt = Path(
        hf_hub_download(
            cfg.get("croma.checkpoint_repo"),
            cfg.get("croma.checkpoint_file"),
            revision=cfg.get("croma.checkpoint_revision"),
        )
    )
    report["checkpoint_path"] = str(ckpt)
    report["checkpoint_bytes"] = ckpt.stat().st_size
    report["checkpoint_sha256"] = _sha256(ckpt)
    report["checkpoint_revision_configured"] = cfg.get("croma.checkpoint_revision")

    # -------------------------------------------------- DEV-3: patch geometry
    # Read the real module's own constants rather than trusting arithmetic.
    s1 = getattr(model, "s1_encoder", None)
    s2 = getattr(model, "s2_encoder", None)
    report["patch_geometry"] = {
        "s1_patch_size": getattr(s1, "patch_size", None),
        "s2_patch_size": getattr(s2, "patch_size", None),
        "s1_num_heads": getattr(s1, "num_heads", None),
        "s2_num_heads": getattr(s2, "num_heads", None),
        "s1_dim": getattr(s1, "dim", None),
        "s2_dim": getattr(s2, "dim", None),
        "s1_in_channels": getattr(s1, "in_channels", None),
        "s2_in_channels": getattr(s2, "in_channels", None),
        "s1_depth": getattr(s1, "depth", None),
        "s2_depth": getattr(s2, "depth", None),
    }
    # The joint cross-encoder is the SAR-conditioned one; verify GAP asymmetry.
    report["gap_ffn_present"] = {
        "s1": hasattr(model, "GAP_FFN_s1"),
        "s2": hasattr(model, "GAP_FFN_s2"),
        "joint": hasattr(model, "GAP_FFN_joint"),
    }

    # CROMA-base is deliberately ASYMMETRIC: the SAR encoder is built with
    # `depth=int(encoder_depth/2)`. Measured here so that the "depth 12" figure
    # carried in the project brief is not mistaken for a symmetric contract.
    s1_depth = getattr(s1, "depth", None)
    s2_depth = getattr(s2, "depth", None)
    report["encoder_depth"] = {
        "declared_encoder_depth": getattr(model, "encoder_depth", None),
        "s1_depth_measured": s1_depth,
        "s2_depth_measured": s2_depth,
        "asymmetric": s1_depth is not None and s2_depth is not None and s1_depth != s2_depth,
    }

    # What the checkpoint actually contains -- evidence that the weights loaded
    # are the ones the loader expects, not an empty or partial state dict.
    try:
        import torch as _torch

        _ck = _torch.load(ckpt, map_location="cpu", weights_only=True)
        report["checkpoint_keys"] = sorted(_ck) if isinstance(_ck, dict) else None
        report["checkpoint_is_state_dict_bundle"] = isinstance(_ck, dict)
    except Exception as exc:  # noqa: BLE001
        report["checkpoint_keys"] = None
        report["checkpoint_key_probe_error"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------- real input
    rng = np.random.default_rng(0)
    B, R = args.batch_size, args.resolution
    # In-range, per the DEV-2 ruling: the encoder expects [0, 1] after the
    # radiometry stage. Random but *structured* (a gradient plus noise) so that a
    # degenerate all-constant input cannot accidentally look fine.
    ramp = np.linspace(0.0, 1.0, R * R, dtype=np.float32).reshape(R, R)
    optical = np.empty((B, 12, R, R), dtype=np.float32)
    for b in range(B):
        for c in range(12):
            optical[b, c] = np.clip(ramp * (0.5 + 0.5 * (c + 1) / 12) + rng.normal(
                0, 0.02, (R, R)
            ).astype(np.float32), 0.0, 1.0)
    sar = np.empty((B, 2, R, R), dtype=np.float32)
    for b in range(B):
        for c in range(2):
            sar[b, c] = np.clip(ramp * (0.7 + 0.3 * c) + rng.normal(
                0, 0.02, (R, R)
            ).astype(np.float32), 0.0, 1.0)
    report["input_shapes"] = {"optical": list(optical.shape), "sar": list(sar.shape)}
    report["input_ranges"] = {
        "optical_min": float(optical.min()),
        "optical_max": float(optical.max()),
        "sar_min": float(sar.min()),
        "sar_max": float(sar.max()),
    }

    # ------------------------------------------------------------- forward
    rss_before = _rss_bytes()
    t0 = time.perf_counter()
    enc = encoder.encode(optical, sar)
    forward_seconds = time.perf_counter() - t0
    rss_after = _rss_bytes()

    report["forward_seconds"] = round(forward_seconds, 3)
    report["rss_before_bytes"] = rss_before
    report["rss_after_bytes"] = rss_after
    report["rss_delta_bytes"] = (
        None if rss_before is None or rss_after is None else rss_after - rss_before
    )
    report["n_patches_reported"] = int(enc.n_patches)
    report["resolution_reported"] = int(enc.resolution)
    report["dim_reported"] = int(enc.dim)

    def _describe(arr: np.ndarray) -> dict:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": bool(np.all(np.isfinite(arr))),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
        }

    report["outputs"] = {
        "optical_gap": _describe(enc.optical_gap),
        "sar_gap": _describe(enc.sar_gap),
        "joint_gap": _describe(enc.joint_gap),
        "optical_tokens": _describe(enc.optical_tokens),
        "sar_tokens": _describe(enc.sar_tokens),
        "joint_tokens": _describe(enc.joint_tokens),
    }

    # ------------------------------------------------------------- assertions
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    check(
        "encoder_dim_is_768",
        enc.dim == VERIFIED_ENCODER_DIM,
        f"dim={enc.dim}, expected {VERIFIED_ENCODER_DIM}",
    )
    check(
        "DEV-3 patch_count_is_225_at_120px",
        enc.n_patches == EXPECTED_PATCHES_120,
        f"n_patches={enc.n_patches}, expected {EXPECTED_PATCHES_120}",
    )
    check(
        "gap_shapes_are_B_by_768",
        all(
            getattr(enc, k).shape == (B, VERIFIED_ENCODER_DIM)
            for k in ("optical_gap", "sar_gap", "joint_gap")
        ),
        f"{enc.optical_gap.shape}, {enc.sar_gap.shape}, {enc.joint_gap.shape}",
    )
    check(
        "token_shapes_are_B_by_225_by_768",
        all(
            getattr(enc, k).shape == (B, EXPECTED_PATCHES_120, VERIFIED_ENCODER_DIM)
            for k in ("optical_tokens", "sar_tokens", "joint_tokens")
        ),
        f"optical_tokens={enc.optical_tokens.shape}",
    )
    check(
        "all_outputs_finite",
        all(
            bool(np.all(np.isfinite(getattr(enc, k))))
            for k in (
                "optical_gap",
                "sar_gap",
                "joint_gap",
                "optical_tokens",
                "sar_tokens",
                "joint_tokens",
            )
        ),
        "no NaN/inf in any output tensor",
    )
    check(
        "gap_vectors_are_not_degenerate",
        all(float(getattr(enc, k).std()) > 1e-6 for k in
            ("optical_gap", "sar_gap", "joint_gap")),
        "every GAP vector has non-trivial variance",
    )
    check(
        "joint_gap_differs_from_sar_and_optical",
        not np.allclose(enc.joint_gap, enc.sar_gap)
        and not np.allclose(enc.joint_gap, enc.optical_gap),
        "joint is a genuinely third representation, not a copy",
    )
    check(
        "joint_gap_has_no_ffn_head",
        not hasattr(model, "GAP_FFN_joint"),
        "matches upstream: joint_GAP is a raw mean, s1/s2 GAPs go through FFNs",
    )
    check(
        "sar_encoder_is_half_depth",
        report["encoder_depth"]["s1_depth_measured"] == 6
        and report["encoder_depth"]["s2_depth_measured"] == 12,
        f"s1={report['encoder_depth']['s1_depth_measured']} "
        f"s2={report['encoder_depth']['s2_depth_measured']} "
        "(upstream builds s1 with depth=int(encoder_depth/2)); "
        "CROMA-base is asymmetric by design",
    )
    check(
        "samples_are_independent",
        B < 2 or not np.allclose(enc.joint_gap[0], enc.joint_gap[1]),
        "two different inputs give two different joint_GAP vectors",
    )
    check(
        "fusion_input_dim_composes",
        enc.dim * 3 + 12 + 2 == EXPECTED_FUSION_DIM,
        f"3*{enc.dim} + 12 + 2 = {enc.dim * 3 + 12 + 2}",
    )
    check(
        "frozen_config_hash_intact",
        cfg.hash == "78f1e3700da15aa1",
        f"cfg.hash={cfg.hash}",
    )
    check(
        "radiometry_summary_present",
        enc.radiometry is not None,
        "DEV-2 stage 3.1.4: the applied transform reached the trace",
    )

    report["checks"] = checks
    report["checks_passed"] = sum(1 for c in checks if c["passed"])
    report["checks_total"] = len(checks)
    report["all_checks_passed"] = all(c["passed"] for c in checks)
    # A forward pass that produced real, finite, correctly-shaped tensors is
    # *model smoke* evidence -- one rung up from "the loader imports".
    report["usable_as_evidence"] = bool(report["all_checks_passed"])
    report["evidence_rung"] = "model smoke (real weights, real tensors, no training)"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # ------------------------------------------------------------- console
    print("=" * 72)
    print("CROMA FIRST REAL FORWARD PASS")
    print("=" * 72)
    print(f"torch            : {report['torch_version']}  cuda={report['cuda_available']}")
    print(f"device           : {report['device']}")
    print(f"model            : {report['model_class']} "
          f"size={report['model_size']} modality={report['model_modality']} "
          f"res={report['model_image_resolution']}")
    print(f"parameters       : {n_params:,}")
    print(f"checkpoint       : {ckpt.name}  {report['checkpoint_bytes']:,} bytes")
    print(f"checkpoint sha256: {report['checkpoint_sha256']}")
    print(f"load time        : {load_seconds:.2f} s")
    print(f"forward time     : {forward_seconds:.2f} s  (batch={B}, {R}px)")
    print()
    print(f"n_patches        : {enc.n_patches}  (DEV-3 expects {EXPECTED_PATCHES_120})")
    for key in ("optical_gap", "sar_gap", "joint_gap",
                "optical_tokens", "sar_tokens", "joint_tokens"):
        d = report["outputs"][key]
        print(f"  {key:<16} {tuple(d['shape'])!s:<22} {d['dtype']:<8} "
              f"finite={d['finite']} std={d['std']:.5f}")
    print()
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")
    print()
    print(f"RESULT: {report['checks_passed']}/{report['checks_total']} checks passed")
    print(f"report written to {args.json}")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
