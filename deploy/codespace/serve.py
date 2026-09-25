"""SatQuery AI — Codespace serve entrypoint (production inference target).

This is the launcher the GitHub Codespace runs to expose the FastAPI inference
server that the production frontend reaches through the Render orchestrator.

The Codespace checks out the repo, so the `app` package is importable when the
working directory is the repo root. `build_space_app()` is cheap to import:
FastAPI is imported inside it and no model is loaded at module scope, so this
file stays import-safe on a CPU host with no GPU and no weights present.

The serving controller (via `app.serving.build_serving_controller`) resolves
`device` from the `SATQUERY_DEVICE` env var; set it to `cpu` for the CPU-first
adaptation. See deploy/codespace/README.md.
"""

import os

from app.space_app import build_space_app
import uvicorn

app = build_space_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
