# Model References

Exact model identities pinned in `configs/base.yaml`. These strings are the
source of truth for reproducibility — SatQuery AI ships **no weights** for any
of them; they are third-party models pinned by `repo_id` + `revision`.

| Role | repo_id | revision | Load call (from code) |
| --- | --- | --- | --- |
| VLM captioning / VQA | `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb5` | `from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct", revision="a7da5b986cb5")` |
| Remote-sensing grounding | `chendelong/RemoteCLIP` | `bf1d8a3ccf2d` | `from_pretrained("chendelong/RemoteCLIP", revision="bf1d8a3ccf2d")` |
| Router embedding | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | `from_pretrained("sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4")` |
| Optical-SAR fusion | `antofuller/CROMA` | `0dd28e3d633b` | `from_pretrained("antofuller/CROMA", revision="0dd28e3d633b")` |

## Source mapping in `configs/base.yaml`

- **VLM** — `vlm.checkpoint: HuggingFaceTB/SmolVLM-500M-Instruct`,
  `vlm.revision: a7da5b986cb5`
- **Grounding** — `grounding.checkpoint_repo: chendelong/RemoteCLIP`,
  `grounding.checkpoint_revision: bf1d8a3ccf2d`
  (checkpoint file `RemoteCLIP-ViT-B-32.pt`)
- **Router** — `router.model: sentence-transformers/all-MiniLM-L6-v2`,
  `router.revision: 1110a243fdf4`
- **CROMA** — `croma.checkpoint_repo: antofuller/CROMA`,
  `croma.checkpoint_revision: 0dd28e3d633b`
  (checkpoint file `CROMA_base.pt`)

All four revisions are present and verified reachable as of 2026-09-16 per the
configuration registry. No revision was invented or guessed.
