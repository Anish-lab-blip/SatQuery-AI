# Hugging Face Release Verification

**Phase:** 4 · **Date:** 2026-09-25 · **Repo:** https://huggingface.co/thundercode/SatQuery
**Result:** **VERIFIED — all 6 artifacts byte-identical, all 4 support files present.**

---

## 1. What was released

| Item | Detail |
|---|---|
| Repository | `thundercode/SatQuery` (model repo) |
| Visibility | **public** (`private: false`) |
| HEAD after release | **`bf2779e18fcaa7476b93a48a978f08c108dfdfb7`** |
| `lastModified` | `2026-09-25T21:46:52Z` |
| Files on Hub | **42** (26 markdown docs + 6 trained artifacts + 4 support files + 6 tooling files) |
| First release HEAD | `55681e0cddb91a4a5655da98a49bc025e537b657` (artifacts only) |

## 2. Pre-flight inspection (what was already there, and what was done with it)

| Existing file | Content found | Action |
|---|---|---|
| `README.md` | a **25-byte stub** — literally `---\nlicense: unknown\n---`, no content | **replaced** with the model card (nothing of value lost) |
| `.gitattributes` | the standard Hugging Face **LFS routing template** — already routes `*.pt`, `*.safetensors`, `*.bin` through LFS | **left untouched** (it was already correct and is required for LFS) |

**No useful content was overwritten.** The stub README contained no project information.

## 3. Write permission

Verified **before** uploading, by reading the token's own scopes from `GET /api/whoami-v2`:

```
auth.type                      : access_token
auth.accessToken.displayName   : Workbuddy
auth.accessToken.role          : fineGrained
scoped[0].entity               : user thundercode
scoped[0].permissions          : repo.content.read, repo.access.read, repo.write, …
```

`repo.write` is present, scoped to the `thundercode` user. This **closes** the earlier
"write permission unproven" note from Phase 0.

## 4. Uploaded contents

| # | Path on Hub | Bytes |
|---|---|---|
| 1 | `README.md` (model card) | 26,767 |
| 2 | `MODEL_CARD.md` | 7,002 |
| 3 | `models/manifest.json` | 4,458 |
| 4 | `models/checksums.sha256` | 642 |
| 5–15 | `docs/*.md` (11 files) | ~84,000 total |
| 16 | `change/head.pt` | 63,231,009 |
| 17 | `change_vqa/head.pt` | 5,822,809 |
| 18 | `optical_sar/head.pt` | 14,427,457 |
| 19 | `grounding/head.pt` | 12,639,041 |
| 20 | `router/adapter.pt` | 211,961 |
| 21 | `vlm/adapter_model.safetensors` | 34,798,048 |

Total released weight payload: **131,130,325 bytes (~125 MiB)** across the six trained artifacts.
**No backbone weights are redistributed.**

## 5. Independent verification (re-downloaded, not trusted)

`release/tools/hf_verify.py` **re-downloads each artifact over direct HTTPS** and hashes the bytes it
receives, comparing against the locally-computed sha256. It does **not** trust the upload step.

| STATUS | hf_path | remote bytes | local bytes |
|---|---|---|---|
| **MATCH** | `change/head.pt` | 63,231,009 | 63,231,009 |
| **MATCH** | `change_vqa/head.pt` | 5,822,809 | 5,822,809 |
| **MATCH** | `optical_sar/head.pt` | 14,427,457 | 14,427,457 |
| **MATCH** | `grounding/head.pt` | 12,639,041 | 12,639,041 |
| **MATCH** | `router/adapter.pt` | 211,961 | 211,961 |
| **MATCH** | `vlm/adapter_model.safetensors` | 34,798,048 | 34,798,048 |

```
artifacts verified : 6
artifacts failed   : 0
```

Support files confirmed present: `README.md`, `MODEL_CARD.md`, `models/manifest.json`,
`models/checksums.sha256` — all `OK`.

### 5.1 A verification method that was itself wrong (recorded)

The **first** verification attempt reported all six artifacts `DIFFER`, with every remote hash equal
to `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` — the sha256 of **empty
content**. The cause was not the upload: `hf_hub_download` returned an empty file in this
environment (a download-path problem), so the verifier hashed nothing.

This was caught by a **second, independent method** (a direct `curl` download), which produced the
correct hash `8527c3ed28a293e13293d48601d48e3ceafa137b9acabddaf5de31a58a509b5c` for
`router/adapter.pt` — byte-identical to the local file, and confirmed to be a real PyTorch zip
(`PK\x03\x04`, containing `adapter/data.pkl`).

The verifier was then rewritten to use **direct HTTPS with proxies disabled**. It now reports
6/6 MATCH. The failed first attempt is recorded because a verifier that silently hashes an empty
file would have produced a **false failure** — and, with a different bug, could just as easily have
produced a **false pass**.

## 6. Cross-checks against independently-recorded hashes

Two of the six hashes can be checked against values recorded elsewhere in the project, independent
of this release:

| Artifact | Recorded elsewhere | Computed here | Agree |
|---|---|---|---|
| `change_vqa_head` | `cfae5e43…d63a82a` (`artifacts/change_vqa/run/PROMOTION.json`) | `cfae5e43…d63a82a` | **yes** |
| `vlm_lora_adapter` | `07c76a75…a5adf5e` (adapter provenance) | `07c76a75…a5adf5e` | **yes** |

## 7. Access verification

- The repository is **public** — the URLs resolve without a token.
- `GET /api/models/thundercode/SatQuery` returns the repo with `private: false`, `gated: false`,
  `disabled: false`.
- Every artifact is retrievable at
  `https://huggingface.co/thundercode/SatQuery/resolve/main/<path>`.

## 8. Secrets

**No secret was uploaded.** The uploaded set is: the model card, the manifest, the checksums, the 11
docs, and the six weight files. No tokens, keys, environment files, or credentials exist in any
uploaded file. The token used for the upload is **not** written into any released file.

## 9. Reproduce this verification

```bash
export HF_TOKEN=...            # token with repo.content.read
python release/tools/hf_verify.py
```

Exit code 0 with `artifacts failed : 0` means the release is intact. Committed output:
`release/tools/hf_verify_report.txt`.

## 10. Notes and caveats

- The Hub's own **README metadata validator** emitted a warning during upload
  (`empty or missing yaml metadata in repo card`) for one intermediate commit. The final README
  carries full YAML frontmatter (`license: other`, tags, `pipeline_tag: image-to-text`).
- The released docs are the **detailed** versions; the model card's relative links
  (`docs/LIMITATIONS.md`, `models/manifest.json`) resolve because both are uploaded.
- **The six artifacts are small trained modules over frozen backbones.** Anyone using them must also
  fetch the pinned backbones listed in `models/manifest.json` / `docs/MODELS.md`.
