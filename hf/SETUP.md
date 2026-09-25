# Publishing SatQuery AI to Hugging Face

The three files in this `hf/` directory are the **Hugging Face presence** for
SatQuery AI. SatQuery does not own any model weights, so these are a *project
presence* plus pinned model references — not model cards for weights we don't
have, and no large artifacts are included.

## Files

- `hf/README.md` — the Hugging Face **model card** / landing page.
- `hf/MODEL_REFERENCES.md` — the exact pinned `repo_id` / `revision` metadata.
- `hf/SETUP.md` — this file.

## To publish

1. Create a Hugging Face **Model** repo named `satquery-ai/SatQuery-AI`.
2. Upload `hf/README.md` as the repo's model card (the `README.md` at the repo
   root).
3. Upload `hf/MODEL_REFERENCES.md` and `hf/SETUP.md` as supporting metadata
   files.
4. Do **not** upload any weights or binaries.

> **No push was performed.** This environment has no Hugging Face credentials, so
> the files are only staged here under `hf/`. Publishing must be done by the
> maintainer with `huggingface_hub` or the web UI.
