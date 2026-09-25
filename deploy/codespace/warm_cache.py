"""Warm the Hugging Face model cache for the four pinned specialists.

Running this before the first request means the initial `/v1/analyze` does not
pay a cold download. It is idempotent and safe to re-run: every download targets
the same HF_HOME cache layout the serving controller actually reads, so a second
run is a no-op (the blob is already on disk).

Per-model status is reported as OK / SKIPPED / FAILED and one model failing never
aborts the others (degraded serving is an accepted deployment state). The script
always exits 0 so a cache miss during Codespace build does not fail the build.

Model ids + revisions are transcribed VERBATIM from configs/base.yaml:
  vlm       HuggingFaceTB/SmolVLM-500M-Instruct    rev a7da5b986cb5
  grounding chendelong/RemoteCLIP  (RemoteCLIP-ViT-B-32.pt)  rev bf1d8a3ccf2d
  router    sentence-transformers/all-MiniLM-L6-v2 rev 1110a243fdf4
  croma     antofuller/CROMA  (CROMA_base.pt)       rev 0dd28e3d633b

The two `.pt` checkpoints (RemoteCLIP, CROMA) are fetched with `hf_hub_download`
— the exact call the runtime's `resolve_checkpoint_path` makes — while the two
library models (SmolVLM, MiniLM) are fetched with `snapshot_download`, which is
what `transformers`/`sentence-transformers` `from_pretrained` consumes.

ENVIRONMENT
  SATQUERY_WARM_OFFLINE / HF_HUB_OFFLINE  -> skip everything (status SKIPPED)
  SATQUERY_WARM_SKIP="croma,grounding"    -> skip the named models (or "all")
"""

from __future__ import annotations

import os

from huggingface_hub import hf_hub_download, snapshot_download

# key -> (repo_id, revision, filename-or-None, human label)
MODELS: dict[str, tuple[str, str, str | None, str]] = {
    "vlm": (
        "HuggingFaceTB/SmolVLM-500M-Instruct",
        "a7da5b986cb5",
        None,
        "SmolVLM-500M-Instruct (VLM)",
    ),
    "grounding": (
        "chendelong/RemoteCLIP",
        "bf1d8a3ccf2d",
        "RemoteCLIP-ViT-B-32.pt",
        "RemoteCLIP (grounding)",
    ),
    "router": (
        "sentence-transformers/all-MiniLM-L6-v2",
        "1110a243fdf4",
        None,
        "all-MiniLM-L6-v2 (router)",
    ),
    "croma": (
        "antofuller/CROMA",
        "0dd28e3d633b",
        "CROMA_base.pt",
        "CROMA_base (optical-SAR)",
    ),
}


def _warm_one(key: str, spec: tuple[str, str, str | None, str]) -> tuple[str, str]:
    """Download one model; return (status, detail)."""
    repo, revision, filename, label = spec
    try:
        if filename is None:
            # Library model: full snapshot, mirrors from_pretrained().
            path = snapshot_download(repo_id=repo, revision=revision)
        else:
            # Checkpoint: exact call the runtime resolver uses.
            path = hf_hub_download(repo=repo, filename=filename, revision=revision)
        return "ok", f"{label} -> {path}"
    except Exception as exc:  # noqa: BLE001 - report, never crash the warm step
        return "failed", f"{label}: {type(exc).__name__}: {exc}"


def main() -> int:
    skip_all = bool(
        os.environ.get("SATQUERY_WARM_OFFLINE") or os.environ.get("HF_HUB_OFFLINE")
    )
    skip_keys = {
        k.strip().lower()
        for k in os.environ.get("SATQUERY_WARM_SKIP", "").split(",")
        if k.strip()
    }

    results: list[tuple[str, str, str]] = []
    for key, spec in MODELS.items():
        if skip_all:
            results.append((key, "skipped", "offline mode (HF_HUB_OFFLINE / SATQUERY_WARM_OFFLINE)"))
            continue
        if key in skip_keys or "all" in skip_keys:
            results.append((key, "skipped", "explicitly skipped via SATQUERY_WARM_SKIP"))
            continue
        status, detail = _warm_one(key, spec)
        results.append((key, status, detail))

    width = max((len(k) for k, _, _ in results), default=0)
    print("=" * 64)
    print("SatQuery AI — Hugging Face model cache warm-up")
    print("=" * 64)
    for key, status, detail in results:
        print(f"[{status.upper():7}] {key:<{width}}  {detail}")
    print("-" * 64)
    n_ok = sum(1 for _, s, _ in results if s == "ok")
    n_skip = sum(1 for _, s, _ in results if s == "skipped")
    n_fail = sum(1 for _, s, _ in results if s == "failed")
    print(f"done: {n_ok} ok, {n_skip} skipped, {n_fail} failed")

    # Best-effort: a cache miss must not fail the Codespace build.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
