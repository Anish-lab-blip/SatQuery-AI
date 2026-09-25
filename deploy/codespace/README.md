# deploy/codespace — GitHub Codespace inference target

This directory turns a GitHub Codespace into the **live FastAPI inference server**
that the production frontend calls via the Render orchestrator. The Codespace is
the production inference target, not a dev sandbox.

## What is here

- `serve.py` — the serve entrypoint. Builds the app via
  `app.space_app.build_space_app()` and binds it with uvicorn on `$PORT`
  (default `8000`). This is the launcher that `app/space_app.py` is missing:
  that module exposes `build_space_app()` / `main()` but has no `__main__` block
  and never binds a port.
- `requirements.txt` — lean CPU-only inference deps. torch is pinned to the
  `+cpu` wheel so Codespaces do not download the multi-GB CUDA build, and gradio
  is omitted (the Space serves JSON only).
- `warm_cache.py` — pre-downloads the four pinned HF models into the HF cache so
  the first `/v1/analyze` is fast. Idempotent, per-model OK/SKIPPED/FAILED status,
  never aborts on a single miss.
- `post_create.sh` — `set -euo pipefail`; installs the requirements then warms
  the cache. Runs automatically via `postCreateCommand`.

## Run locally

```bash
PORT=8000 python deploy/codespace/serve.py
```

The server answers `GET /v1/health`, `GET /v1/capabilities`,
`POST /v1/assets`, `POST /v1/analyze`.

## CPU adaptation

The owner chose a CPU-first deployment. `serve.py` leaves device resolution to
the serving controller, which reads `SATQUERY_DEVICE=cpu` (set in
`.devcontainer/devcontainer.json`). On a CPU host the controller auto-resolves
`device=None` to `cpu` anyway.

## Port visibility

`.devcontainer/devcontainer.json` forwards `8000` as **public** so Render can
reach it. The public URL GitHub emits is
`https://<codespace-name>-8000.app.github.dev`; Render discovers the live URL
via the GitHub Codespaces API (handled in a separate workstream).

## Frozen config — do not edit

`configs/base.yaml` and `configs/deploy.yaml` are frozen. `Config.hash ==
78f1e3700da15aa1` is a sha256 over the whole registry; editing either file moves
the hash and invalidates the frozen Phase-9 benchmark. The model ids and
revisions used here are transcribed verbatim from `configs/base.yaml` and must
stay in sync with it — change them there first, never here in a way that diverges.
