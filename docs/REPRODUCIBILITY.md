# Reproducibility

**Status tags:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED` · `NOT RUN` · `BLOCKED` ·
`DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

This document states **exactly what a third party can reproduce, with what, and — just as
importantly — what they cannot**. It is written to the same depth as the rest of this documentation
set: real commands, real file paths, real hashes, real expected output, and an explicit status for
every claim.

It is deliberately **not** a "quickstart". A quickstart tells you the happy path and hides the
edges. This document records the edges, because in this project the edges are where the real
debugging time went — a dead sandbox proxy, a pytest that only exists in one virtualenv, a Chrome
that silently drops synthetic key events, a Hugging Face helper that returned an *empty file* and
would have produced a **false failure**. Those are documented in §10 so that a third party does not
rediscover them.

> **The one rule this document obeys, like every document in `docs/`:** *do not fabricate*. Every
> claim below is grounded in a file that was read (`core/config.py`, `configs/base.yaml`,
> `release/tools/*.py`, `artifacts/**`, `docs/**`). Where the evidence does not establish something,
> the text says `UNKNOWN — not established from the available evidence`.

---

## Table of contents

1. [The reproducibility contract](#1-the-reproducibility-contract)
2. [The frozen configuration](#2-the-frozen-configuration)
3. [Reproduce the metric verification (cheap, no GPU)](#3-reproduce-the-metric-verification-cheap-no-gpu)
4. [Reproduce the environment](#4-reproduce-the-environment)
5. [Reproduce the tests](#5-reproduce-the-tests)
6. [Reproduce a live run](#6-reproduce-a-live-run)
7. [Reproduce and verify the released artifacts](#7-reproduce-and-verify-the-released-artifacts)
8. [What "reproduce" means for the two externally-trained artifacts](#8-what-reproduce-means-for-the-two-externally-trained-artifacts)
9. [What is NOT reproducible from this release — exhaustive](#9-what-is-not-reproducible-from-this-release--exhaustive)
10. [Environment traps recorded for reproducibility](#10-environment-traps-recorded-for-reproducibility)
11. [What is NOT RUN / OPEN / BLOCKED for this topic](#11-what-is-not-run--open--blocked-for-this-topic)
12. [Where the evidence lives](#12-where-the-evidence-lives)

---

## 1. The reproducibility contract

Reproducibility here is not a slogan; it is a set of **enforced guarantees**. Each row below states a
guarantee, how it is enforced in code or in a tool, and how a third party can independently check
it.

| Guarantee | How it is enforced | How to check it |
|---|---|---|
| **Frozen configuration** | All tunables live in `configs/base.yaml`; there are no magic numbers in Python. The loader validates invariants and computes a hash. | Read `configs/base.yaml`; run the loader; compare the hash. |
| **Frozen config hash** | `Config.hash` = `sha256(json.dumps(data, sort_keys=True, default=str))[:16]` (`core/config.py:76-80`). Every evaluation run and every artifact records the hash it was produced against. | `Config.hash` returns `78f1e3700da15aa1`; the same string appears in `models/manifest.json` and each artifact. |
| **Pinned backbones** | Every backbone is pinned by `repo_id` **and** `revision` (a commit prefix, not a floating tag) in `configs/base.yaml`. | The four pins are listed in §4.3; each resolves to an exact Hub commit. |
| **Seed** | `project.seed: 42` (`core/config.py:82-84`). | Read the key; it is in the registry. |
| **Immutable public test** | `evaluation.immutable_public_test: true` and `evaluation.hidden_data_access: false` (`configs/base.yaml:273-277`). | Read the two keys; the corpus module refuses to mutate a sealed corpus. |
| **Leakage isolation** | `evaluation.leakage_split_key: scene_id` — splits are assigned by *scene*, never by *tile*. | Read the key; `evaluation/leakage.py` audits manifests. |
| **No official aggregate** | `evaluation.official_aggregate_weights: null`. There is no weighted composite score. | Read the key; `normalize.py` raises on aggregate requests. |
| **Byte-verified artifacts** | Every released artifact ships with a sha256 in `models/checksums.sha256`, generated from the actual files. | `sha256sum -c models/checksums.sha256`. |
| **Verified metrics** | Every number quoted in the README is checked against its source artifact by `release/tools/verify_readme_metrics.py`. | Run the tool; it prints `ALL CLAIMS VERIFIED`. |
| **Independent Hub verification** | `release/tools/hf_verify.py` re-downloads every uploaded file over direct HTTPS and re-hashes the bytes it received. | Run the tool; it prints `artifacts failed : 0`. |
| **Negative results preserved** | Rejected and open rulings stay in the record (`PHASE7_RESOLUTION_DECISION.md`, `phase6_closure.json`, `threshold_sweep_val.json`). | Read the artifacts; they contain the rejections, not scrubbed versions. |

### 1.1 What the contract is *not*

The contract does **not** promise:

- a single command that retrains every artifact from scratch (§8);
- an end-to-end system-level accuracy number (§9 — no such benchmark exists);
- that the two externally-trained artifacts can be regenerated inside this repository (§8);
- bit-identical *training* on arbitrary hardware (only bit-identical *released artifacts* are
  guaranteed, by hash).

### 1.2 The reproducibility ladder

Reproduction here comes in four tiers, from cheapest to most expensive. A reviewer who wants to
spend five minutes gets tier 1; a reviewer who wants to audit the whole thing walks all four.

| Tier | What it proves | Cost | Section |
|---|---|---|---|
| **1. Metric verification** | Every quoted number matches its artifact | seconds, CPU | §3 |
| **2. Environment + tests** | The code installs and the suites behave as recorded | minutes, CPU | §4–§5 |
| **3. Live run** | The deployed stack answers correctly on unseen imagery | minutes (plus cold start), network | §6 |
| **4. Artifact + Hub verification** | The released weights are byte-identical to what was uploaded | minutes, network | §7 |

---

## 2. The frozen configuration

### 2.1 What "frozen" means here

`configs/base.yaml` is the **single registry** for every tunable in the system. The design rule is:
*no magic numbers in Python*. If a value can change behaviour — an image size, a channel count, a
threshold, a precision, a split key — it lives in the registry, not in a function body.

This matters for reproducibility because it means the **entire behavioural surface of the system is
captured by one file**, and that file has a hash.

### 2.2 The hash and how it is computed

```python
@property
def hash(self) -> str:
    """Stable hash of the whole registry. Recorded in every evaluation run."""
    blob = json.dumps(self._data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
```

— `core/config.py:76-80`.

Three properties make this hash **stable**:

1. **`sort_keys=True`** — key order in the YAML file does not change the hash.
2. **`default=str`** — non-JSON-native values (dates, paths) are stringified deterministically.
3. **`[:16]`** — the first 16 hex characters are used as the human-readable identifier.

The frozen value is:

```
78f1e3700da15aa1
```

It is recorded in `models/manifest.json` (`"config_hash": "78f1e3700da15aa1"`), in each artifact's
metadata, and in the live evidence (each live run's panel shows the hash). This is the single
identifier that ties "the thing that produced this number" to "the configuration a reviewer is
reading".

### 2.3 The enforced invariants, with exact arithmetic

Editing `configs/base.yaml` moves the hash and **invalidates every artifact keyed to it**. To make
that failure loud rather than silent, the loader rejects a config that violates a set of recorded
invariants. Two of them exist specifically because their violation is a *silent* error that torch
only surfaces much later.

#### Invariant 1 — fusion input dimension (finding C-1)

```
fusion.input_dim == 3 * croma.encoder_dim + croma.optical_channels + croma.sar_channels
```

With the frozen values `encoder_dim = 768`, `optical_channels = 12`, `sar_channels = 2`:

```
fusion.input_dim == 3 * 768 + 12 + 2
                 == 2304      + 12 + 2
                 == 2318
```

This is the concatenation of CROMA's three 768-d GAP vectors (`optical_GAP`, `SAR_GAP`,
`joint_GAP`) plus the 12 optical channels and 2 SAR channels of the sensor adapter. The loader
recomputes `expected` from the config and raises `ConfigError` if `fusion.input_dim` disagrees
(`core/config.py:154-163`). The docstring is explicit about *why*: "CROMA emits optical/SAR/joint
GAP vectors; the [declared value must match]". The upstream contract was independently reproduced
in `docs/PHASE14_OPTICAL_SAR_DECISIONS.md §1`, where the fusion input is recomputed at runtime
(`fusion_head.py:74`, `:311`) rather than hardcoded.

Two related guards fire alongside it (`core/config.py:165-168`):

```
croma.optical_channels == 12   # CROMA s2_channels is fixed
croma.sar_channels     == 2    # CROMA s1_channels is fixed
```

#### Invariant 2 — grounding head feature dimension (finding P7-1)

```
grounding_head.feature_dim == 4 * grounding.encoder_projected_dim
```

With `encoder_projected_dim = 512`:

```
grounding_head.feature_dim == 4 * 512 == 2048
```

The loader's own comment explains why this is **load-bearing**:

> A mismatch here is a SILENT shape error. torch only raises at the [similarity step], after patch
> features are already cached. It is therefore rejected at load time.

— `core/config.py:176-195`.

The "4×" arises because the grounding head consumes four projected feature streams; the projected
dimension is declared in config precisely so this guard can exist without hardcoding 512.

#### Further load-time guards

The `_validate()` method (`core/config.py:94`) also enforces:

| Guard | Rule | Finding |
|---|---|---|
| CROMA resolution | `croma.image_resolution` is an int and a multiple of 8 (CROMA asserts `% 8 == 0`) | C-7 |
| Training precision | `training.precision ∈ {fp16, bf16, fp32}` | C-6 |
| ZeroGPU compile ban | `deployment.torch_compile` must not be `True` — ZeroGPU does not support `torch.compile` | C-8 |
| VLM prompt template | `vlm.prompt_must_use_chat_template` must be `true` — SmolVLM raises otherwise | — |

These are the invariants a reviewer can verify by reading one method. They are the reason a
*plausible-looking* config edit fails at load instead of producing a wrong number three hours later.

### 2.4 The evaluation invariants

```yaml
evaluation:
  immutable_public_test: true
  hidden_data_access: false
  official_aggregate_weights: null
  leakage_split_key: scene_id
```

— `configs/base.yaml:273-277`.

| Key | Value | Meaning |
|---|---|---|
| `immutable_public_test` | `true` | The public-test corpus is sealed; it cannot be mutated by an evaluation. |
| `hidden_data_access` | `false` | No code path may read hidden data during a scored run. |
| `official_aggregate_weights` | `null` | **There is no composite score.** A single "overall accuracy" is forbidden by construction. |
| `leakage_split_key` | `scene_id` | Splits are assigned by scene, so tiles from one scene never straddle train/test. |

The evaluation-honesty rules that these keys serve are stated in full in
[`EVALUATION.md`](EVALUATION.md) §2.

### 2.5 The deployment section is frozen paperwork

```yaml
deployment:
  platform: huggingface-spaces
  sdk: gradio
  zerogpu: true
  torch_compile: false
  ...
```

— `configs/base.yaml:282-295`.

The deployment target described here is the **superseded** HF-Space/ZeroGPU design. The active
topology is Cloudflare Pages → Render → Codespace (`docs/DEPLOYMENT.md §1`). This section is
retained as **frozen paperwork**: editing it would move the config hash, and the hash is what every
artifact is keyed to. No Gradio runtime exists in code. The `torch_compile: false` line is the
C-8 guard and remains meaningful regardless of host.

### 2.6 The hash-exempt path

One decision (DEV-2, the CROMA input-normalisation question) needed to become inspectable **without**
adding a `base.yaml` key — because adding a key would have moved the hash away from `78f1e3700da15aa1`
and invalidated every artifact. The resolution was a hash-exempt path
(`docs/PHASE14_OPTICAL_SAR_DECISIONS.md`, item 7):

```
SATQUERY_CROMA_USE_8_BIT (env)  →  croma.use_8_bit (config)  →  default True
```

It is inspectable at runtime via `CROMAEncoder.describe()["input_normalisation"]`. This is recorded
because it is the one place where "frozen" and "configurable" were reconciled deliberately, and a
reviewer should know the escape hatch exists and is documented rather than hidden.

### 2.7 What invalidates the hash

Anything that changes `self._data` when serialised with `sort_keys=True`. Concretely:

- editing any value in `configs/base.yaml`;
- adding or removing a key;
- changing a value's *type* such that its `default=str` rendering changes.

The consequence is not merely a different hash; it is that **every artifact keyed to
`78f1e3700da15aa1` no longer describes the running configuration**, and the honest action is to
re-run the evaluation rather than reinterpret the old numbers.

---

## 3. Reproduce the metric verification (cheap, no GPU)

This is the cheapest, highest-value reproduction in the whole release. It takes seconds, needs no
GPU and no network, and it is the check that a skeptical reviewer should run **first**.

### 3.1 The verifier

```bash
python release/tools/verify_readme_metrics.py
```

`release/tools/verify_readme_metrics.py` is read-only. Its own docstring states its contract:

> Verify every metric quoted in the release README against its source artifact. Read-only. Each
> claim is compared at the precision at which the README states it.

The tool is **not** a re-computation of the metrics. It is a **provenance check**: it reads the
artifact that the README cites, resolves a dotted key into it, and compares the value against the
literal string the README prints — at the README's own precision.

### 3.2 The 20 claims and their exact key paths

The claim list is not assumed; the key names were **discovered by walking the artifacts**. Several
live under nested paths, and the tool's docstring says so:

> Key names were discovered by walking the artifacts, NOT assumed: several live under nested paths
> (e.g. change metrics are `metrics.pooled.iou`, grounding is `results.head_threshold.mean_best_iou`,
> VLM is `why_usable_verified.adapted_test.*`).

| # | Claim | Artifact | Dotted key | README value |
|---|---|---|---|---|
| 1 | change pooled IoU | `artifacts/change/eval_test/eval_result.json` | `metrics.pooled.iou` | `0.8122` |
| 2 | change macro IoU | `artifacts/change/eval_test/eval_result.json` | `metrics.macro.miou` | `0.8457` |
| 3 | change pooled F1 | `artifacts/change/eval_test/eval_result.json` | `metrics.pooled.f1` | `0.8964` |
| 4 | grounding canonical head_threshold mean_best_IoU | `artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json` | `results.head_threshold.mean_best_iou` | `0.2838` |
| 5 | grounding canonical head_threshold recall@0.5 | `…/eval_result_canonical.json` | `results.head_threshold.recall.0.50` | `0.2198` |
| 6 | grounding matched6 head_threshold mean_best_IoU | `…/eval_result_matched6.json` | `results.head_threshold.mean_best_iou` | `0.2566` |
| 7 | grounding matched6 head_threshold recall@0.5 | `…/eval_result_matched6.json` | `results.head_threshold.recall.0.50` | `0.1938` |
| 8 | grounding head_argmax mean_best_IoU (canonical) | `…/eval_result_canonical.json` | `results.head_argmax.mean_best_iou` | `0.1215` |
| 9 | grounding zero-shot baseline IoU (canonical) | `…/eval_result_canonical.json` | `results.zero_shot_matched.mean_best_iou` | `0.0972` |
| 10 | optical-SAR fusion accuracy | `artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json` | `accuracy` | `0.931` |
| 11 | optical-SAR fusion macro_F1 | `…/pre_registered_115_metric.json` | `macro_f1` | `0.434161` |
| 12 | change_vqa test accuracy | `artifacts/change_vqa/run/PROMOTION.json` | `verification.test_accuracy` | `0.697626` |
| 13 | change_vqa test macro_F1 | `artifacts/change_vqa/run/PROMOTION.json` | `verification.test_macro_f1` | `0.378373` |
| 14 | change_vqa test2 accuracy | `artifacts/change_vqa/run/PROMOTION.json` | `verification.test2_accuracy` | `0.651469` |
| 15 | change_vqa test2 macro_F1 | `artifacts/change_vqa/run/PROMOTION.json` | `verification.test2_macro_f1` | `0.372309` |
| 16 | router overall ungated accuracy | `artifacts/router/threshold_sweep_val.json` | `overall_ungated_accuracy` | `0.965116` |
| 17 | calibration ECE before scaling | `artifacts/calibration_v001.json` | `metrics.ece_before` | `0.013755` |
| 18 | calibration ECE after scaling | `artifacts/calibration_v001.json` | `metrics.ece_after` | `0.014929` |
| 19 | VLM adapter exact_match | `artifacts/vlm/phase6_closure.json` | `why_usable_verified.adapted_test.exact_match` | `0.963` |
| 20 | VLM adapter F1 | `artifacts/vlm/phase6_closure.json` | `why_usable_verified.adapted_test.f1` | `0.96432` |

Note rows 1–3, 6–7, 12–15, and 16: these are the rows that keep the honesty rules honest. The
change numbers are quoted as **pooled and macro together** (rule 4); grounding is quoted under
**both protocols** (rule 2); change-VQA is quoted under **both test sets** (rule 3); the router
number is quoted with its **validation-only** caveat (rule 5). A verifier that checked only one
number per task would be reproducible and *wrong*.

### 3.3 The status assertions

Metrics are not the only claims. After the 20 numeric comparisons, the tool asserts four statuses
that the README also states (`verify_readme_metrics.py:117-131`):

```
VLM headline contains ACCEPTANCE-REJECTED : True
VLM status                               : CLOSED
router corpus_limited                    : True
router n_val                             : 86
calibration temperature (temperature_scaling.temperature) : 0.9772731820958189
calibration ece_improvement              : -0.001174  (negative => calibration did NOT help)
```

These are the checks that prevent the documentation from quietly upgrading a status. In particular:

- the **VLM** metric is usable but the artifact's own headline contains `ACCEPTANCE-REJECTED` — so
  `USABLE ≠ ACCEPTED` (rule 7);
- the **router** number is flagged `corpus_limited: true` with `n_val: 86` — so it can never be
  presented as a test number (rule 5);
- the **calibration** `ece_improvement` is **negative**, i.e. calibration made ECE *worse*, and the
  tool prints that interpretation inline (rule 6).

### 3.4 The nested-key resolution trick

Two subtleties in the resolver are worth recording, because they are the kind of thing that turns a
verifier into a **false-pass** machine if done naively (`verify_readme_metrics.py:69-90`):

1. **Keys that themselves contain dots.** The grounding recall dictionary is keyed by the string
   `"0.10"`, `"0.25"`, `"0.50"`. A naive `dotted.split(".")` walk would break
   `results.head_threshold.recall.0.50` into `…recall → 0 → 50` and fail. The resolver therefore
   tries the **longest matching key at each step first**, so `0.50` stays intact.
2. **Precision matching.** The comparison is
   `round(float(val), decimals(claimed)) == float(claimed)` — i.e. the artifact value is rounded to
   the number of decimals the README prints, then compared. This is why row 15 shows the artifact
   value `0.372308516` matching the README's `0.372309`: the README states six decimals and the
   artifact rounds to it.

A verifier that compared raw floats would report false differences on every rounded value. A
verifier that split dots naively would report false differences on the recall rows. Both failure
modes are **silent false failures** — the same class of bug discussed in §7.3 and §10.

### 3.5 The committed output

```
STATUS   claim                                                  artifact       readme  source
----------------------------------------------------------------------------------------------------------------------
MATCH    change pooled IoU                                        0.8122       0.8122  artifacts/change/eval_test/eval_result.json#metrics.pooled.iou
MATCH    change macro IoU                                         0.8457       0.8457  artifacts/change/eval_test/eval_result.json#metrics.macro.miou
MATCH    change pooled F1                                         0.8964       0.8964  artifacts/change/eval_test/eval_result.json#metrics.pooled.f1
MATCH    grounding canonical head_threshold mean_best_IoU         0.2838       0.2838  artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json#results.head_threshold.mean_best_iou
...
MATCH    VLM adapter exact_match                                   0.963        0.963  artifacts/vlm/phase6_closure.json#why_usable_verified.adapted_test.exact_match
MATCH    VLM adapter F1                                          0.96432      0.96432  artifacts/vlm/phase6_closure.json#why_usable_verified.adapted_test.f1

=== status assertions ===
  VLM headline contains ACCEPTANCE-REJECTED : True
  VLM status                               : CLOSED
  router corpus_limited                    : True
  router n_val                             : 86
  calibration temperature (temperature_scaling.temperature) : 0.9772731820958189
  calibration ece_improvement              : -0.001174  (negative => calibration did NOT help)

RESULT: ALL CLAIMS VERIFIED
```

— `release/tools/readme_metrics_report.txt` (committed).

The tool **exits 0** and prints `ALL CLAIMS VERIFIED` only when every numeric comparison is `MATCH`
and no artifact is missing or lacks the key. Any `NOFILE`, `NOKEY`, or `DIFFER` line increments the
failure counter, the final line reads `N CLAIM(S) FAILED`, and the process exits 1. This makes the
check usable in CI: a non-zero exit is a hard failure.

### 3.6 Why this is a real check and not a tautology

A skeptic might object: "you wrote the README from the artifacts, so of course they match." The
answer is that the check is not that the README was *derived* from the artifacts, but that the
README's published claims remain **mechanically traceable** to files a third party also has. If
someone edits the README to state a nicer number, the tool fails. If someone edits an artifact to a
different number, the tool fails. The check binds the published prose to the recorded evidence, and
it does so without trusting either.

---

## 4. Reproduce the environment

### 4.1 What you need

- **Python 3.11+** (`README.md` §Installation: "Python 3.11+ and a CPU are sufficient").
- A **CPU is enough**. There is no CUDA requirement.
- Network access on first use, to fetch the pinned backbones from the Hugging Face Hub.

### 4.2 The two install profiles

`requirements.txt` documents two profiles explicitly:

```text
#   CPU (local dev / schema / geo / unit tests):
#       pip install -r requirements.txt
#   GPU (Kaggle T4x2 / HF ZeroGPU): torch is preinstalled on both.
#       Do NOT pin torch here — Kaggle and HF ship their own builds.
```

— `requirements.txt` (header comment).

The pins, with the contract each one carries (`requirements.txt`):

| Package | Pin | Why |
|---|---|---|
| `numpy` | `>=1.26,<3` | core |
| `pyyaml` | `>=6.0` | config loading |
| `pydantic` | `>=2.6,<3` | schema |
| `rasterio` | `>=1.3` | geospatial IO |
| `pyproj` | `>=3.6` | CRS handling |
| `opencv-python-headless` | `>=4.9` | image ops |
| `transformers` | `>=4.52` | SmolVLM via `AutoModelForImageTextToText`; `AutoModelForVision2Seq` is deprecated (finding **C-2**) |
| `open-clip-torch` | `>=2.24` | RemoteCLIP loaded via `pretrained=<path>` so `load_checkpoint()` runs its state-dict fixups (**C-4**) |
| `sentence-transformers` | `>=2.7` | router embedding |
| `peft` | `>=0.10` | LoRA adapter |
| `huggingface_hub` | `>=0.23` | pin RemoteCLIP / SmolVLM / CROMA revisions |
| `safetensors` | `>=0.4` | adapter serialisation |
| `einops` | `>=0.7` | **required** by the vendored `use_croma.py` (`from einops import rearrange`) |
| `gradio` | `>=4.44` | UI (frozen deployment target) |
| `reportlab` | `>=4.1` | reporting |
| `pytest` | `>=8.0` | tests |
| `pytest-cov` | `>=5.0` | coverage |

The comment block closes with an instruction a reviewer should heed: *"do not 'fix' by changing these
pins"* — the pins encode verified contracts, not preferences.

### 4.3 The `einops` lesson — a dependency no document declared

`requirements.txt` carries an unusually explicit note about `einops`:

> REQUIRED by the vendored `specialists/optical_sar/vendor/use_croma.py` (`from einops import
> rearrange`). Found missing on 2026-09-18 by executing the vendored module: it raised
> `ModuleNotFoundError`, so the CROMA path was blocked on a dependency that no document declared.
> Not optional: without it the encoder cannot be imported at all.

This is recorded verbatim because it is a reproducibility fact: a missing dependency that no
document mentioned was found **by executing the code**, not by reading a spec. A third party
building the environment will hit the same wall if they install from a partial list.

### 4.4 Create the environment

```bash
git clone https://github.com/Anish-lab-blip/SatQuery-AI
cd SatQuery-AI
python -m venv .venv
source .venv/Scripts/activate      # Windows git-bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

— `README.md` §Installation.

> **Trap (Windows, recorded in §10.2).** In the authoring environment, `pytest` exists **only** in
> the repository virtualenv. Invoking the system `pytest` fails or resolves to a different
> interpreter. Always invoke the venv interpreter explicitly:
> `.venv/Scripts/python.exe -m pytest …`.

### 4.5 Backbones are fetched, pinned by revision, and never redistributed

The four third-party backbones are pinned by `repo_id` **and** `revision` (`README.md`
§Reproducibility item 2; `models/manifest.json` per-artifact `backbone` fields):

| Model | Revision | Role |
|---|---|---|
| `HuggingFaceTB/SmolVLM-500M-Instruct` | `a7da5b986cb5` | VQA + captioning backbone |
| `chendelong/RemoteCLIP` | `bf1d8a3ccf2d` | remote-sensing grounding encoder |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4` | router embedding |
| `antofuller/CROMA` | `0dd28e3d633b` | optical/SAR fusion encoder |

**No backbone weights are redistributed** in this release (`release/HF_RELEASE_VERIFICATION.md §4`).
The released artifacts are small trained modules *over* these frozen backbones; anyone using them
must fetch the backbones too. Pinning by commit prefix (not by a floating tag like `main`) is what
makes "the same backbone" reproducible months later.

### 4.6 Device selection

Device is selected via `SATQUERY_DEVICE`, and **all placement is `.to(device)`, never `.cuda()`**
(`README.md` §Installation; `core/config.py:86-91`):

```python
@property
def device_preference(self) -> str:
    override = os.environ.get("SATQUERY_DEVICE")
    if override:
        return override
    return "cuda" if _torch_cuda_available() else "cpu"
```

This is why a CPU-only reviewer can run everything except the training paths. The env override is
read **without importing torch** in the deployment context (`docs/DEPLOYMENT.md §3.2`), so a
misconfigured device cannot crash the service at import time.

---

## 5. Reproduce the tests

### 5.1 The suites and their expected results

| Suite | Command | Expected | Status |
|---|---|---|---|
| Frontend live-wiring | `python -m pytest tests/unit/test_frontend_live_wiring.py -q` | **106 passed** | `VERIFIED` |
| Doc/frontend suite | `pytest` on the 5 doc/frontend files | **183 passed** | `VERIFIED` |
| Full unit suite | `python -m pytest tests/unit` | **5–6 environmental/ordering failures**, rest pass; a re-run passes 137 | `MEASURED` |

The three rows tell a story a single number would hide: the two **targeted** suites are clean, and
the **full** suite has 5–6 failures that are environmental or ordering-related, not regressions. The
next subsections reproduce each.

### 5.2 Frontend live-wiring regression suite — 106 passed

```bash
python -m pytest tests/unit/test_frontend_live_wiring.py -q
```

— `README.md` §Local development.

This suite is the regression net for the frontend↔backend wiring: the query box, the intent panel,
the task tags, the run identifiers, and the evidence rendering. It is the suite that would catch a
re-introduction of the router defect described in §6.4.

**Expected: 106 passed.** Note the historical figure, because a reviewer may see both numbers:
`docs/FINAL_DELIVERY_REPORT.md §7` records **94 passed** for this file, while
`CURRENT_RELEASE_STATE.md:121` and `README.md` §The test suites record **106 passed** ("re-run this
session"). The difference is the suite growing after the delivery report was written — the
regression tests added for the router fix took this file from 100 → 106 tests
(`DELIVERY_REPORT_2026-09-25.md`, regression-tests note). This document quotes **106**, the
re-run-this-session figure, and cites both. (The exact count of tests *collected* in a single
full-suite invocation is `UNKNOWN — not established from the available evidence`; the per-suite
counts are what the evidence establishes.)

### 5.3 Doc/frontend suite — 183 passed

The documentation-and-frontend suite is **five files**, named explicitly in
`docs/FINAL_DELIVERY_REPORT.md §7`:

- `test_frontend_guide_doc`
- `test_frontend_live_wiring`
- `test_api_contract_doc`
- `test_runbook_doc`
- `test_deploy_config`

Together they report **183 passed**. These are the checks that the docs' internal links resolve, the
frontend assets are present, the API contract document matches the code, the runbook is consistent,
and the deploy config is coherent.

The same three-row test table is published in this release at `README.md` §The test suites
(`README.md:1063-1074`), so a public reader can see the counts without the project-internal report.
The underlying record is `docs/FINAL_DELIVERY_REPORT.md §7` (project-internal); the frontend count
was re-run this session to **106** (`release/CURRENT_RELEASE_STATE.md:121`). See also
[`EVALUATION.md`](EVALUATION.md) §7.2.

### 5.4 Full `tests/unit` — the 5–6 failures, reported honestly

Running the entire unit tree trips **5–6** failures. They are reported here rather than hidden, and
they are classified by cause exactly as `docs/FINAL_DELIVERY_REPORT.md §7` records them:

| # | Failure | Cause | Regression? |
|---|---|---|---|
| 1–4 | `test_safe_delete_shim` (×4) | the sandbox's bulk-**delete guard** (Windows verbatim-path behaviour) | **No** |
| 5 | one ordering flake in the **router route test** | passes in isolation; order/collection-dependent | **No** |
| 6 | one **stale adapter test** | asserts `optical_sar` absent when CROMA is *unshipped* — **CROMA is now shipped** | **No** |

**Why these are not regressions.** They were isolated by a re-run of the affected files **together**,
which passes **137** tests (`docs/FINAL_DELIVERY_REPORT.md §7`;
[`EVALUATION.md`](EVALUATION.md) §7.3–7.4). The `test_safe_delete_shim` failures are Windows-specific
verbatim-path behaviour that does not occur on the deployment host; the ordering flake disappears when
the router route test runs in isolation; the stale adapter test encodes an assumption (CROMA not yet
shipped) that the codebase has since outgrown. None of them is a behavioural defect in shipped code.

> **Honesty note.** The precise full-suite *collected* count is
> `UNKNOWN — not established from the available evidence`. What the evidence establishes is: the two
> targeted suites pass (106, 183), the full suite has 5–6 environmental/ordering failures, and a
> re-run of the affected files passes 137.

### 5.5 The evidence-engine determinism suite

The evidence engine (the component that turns specialist output into a *proof*) has its own suite —
`tests/unit/test_evidence_engine.py` (73 tests) — covering **determinism** and **purity**: the same
input produces the same evidence, and computing evidence does not mutate inputs. This is what makes
the "evidence is independent of the verdict" property in §6.5 hold.

### 5.6 Invocation traps

- **Use the venv interpreter.** `.venv/Scripts/python.exe -m pytest …` on Windows.
- **The full-suite run trips a bulk-delete guard** in the authoring sandbox (§10.3). Run the
  targeted suites to avoid it.
- **Do not pipe pytest through `grep`** in the authoring sandbox — output is block-buffered and a
  killed pipeline swallows it (§10.6). Redirect to a file instead.

---

## 6. Reproduce a live run

### 6.1 The deployed stack

The live topology is **Cloudflare Pages → Render → outbound tunnel → GitHub Codespace**
(`docs/DEPLOYMENT.md §1`; `README.md` §Deployment):

| Layer | Role | Host |
|---|---|---|
| Cloudflare Pages | static frontend | `https://satquery.pages.dev` |
| Render | orchestrator / API gateway (`/api/*`, CORS, wake flow) | `https://<backend-host>` |
| GitHub Codespace | FastAPI inference host, CPU, port 8000 | `potential-space-trout-r4ppw969w45j2pvvw` |

Deployed revisions (VERIFIED):

| Component | Revision |
|---|---|
| Frontend | `2d7ae53b482d` |
| Backend / orchestrator | `89d80eaddec5` |
| Inference | `5a0936ace491` |

> **Trap.** The deployment sources are **private** repositories, separate from this release. The
> `deploy/` directory inside the monorepo working copy is **stale and untracked** — it is *not* the
> deployed source (`docs/DEPLOYMENT.md §1`). Reproducing a live run means driving the deployed
> endpoints, not rebuilding from `deploy/`.

### 6.2 Health and capabilities

```bash
curl --noproxy '*' https://<backend-host>/api/health
curl --noproxy '*' https://<backend-host>/api/capabilities
```

The `/api/health` payload (recorded live, `docs/DEPLOYMENT.md §2`):

```json
{"status":"ok","service":"satquery-orchestrator",
 "tunnel":{"agent_connected":true,"agent_id":"codespaces-fd1038","pending":0,"completed":97},
 "config":{"codespace_name":"potential-space-trout-r4ppw969w45j2pvvw\n","codespace_port":8000,
           "transport_mode":"auto","tunnel_timeout_s":150.0,"wake_timeout_s":120.0,
           "upstream_timeout_s":90.0,"device":"cpu","has_github_token":true}}
```

`/api/capabilities` returns **six tasks, all `available: true`**. A live run requires
`tunnel.agent_connected: true`; if the Codespace is stopped, the request parks until the tunnel
timeout (`docs/DEPLOYMENT.md §5`).

> **`--noproxy '*'` is not optional in the authoring sandbox** (§10.1). The sandbox proxy is dead;
> without the flag, the request fails before reaching Render. In a normal environment the flag is
> harmless.

### 6.3 What was run, and how to re-run it

The behavioural validation is **3 passes × 8 cases = 24 live runs**, each pass **8/8**, with **0
mock nodes** and a live trace bar at **94.4444 %** (`.workbuddy-ai/scratch/live_validation/
LIVE_VALIDATION_POSTFIX.md`). The eight cases exercise all six specialist tasks plus the two
router-defect queries (B1, B2).

| pass | HEAD | harness | result |
|---|---|---|---|
| 1 | `ff46eba42b18`+`d413d3672311` | v1 (`fill_input`) | 8/8 |
| 2 | `2d7ae53b482d` | v2 asserting | 8/8 |
| 3 | `2d7ae53b482d` | v2 asserting (pre-discriminator-fix) | 8/8 (recomputed) |

To re-run: drive `https://satquery.pages.dev` in a **headed** browser (the headed requirement is
explained in §6.4), upload **one asset per case** for single-image tasks and a **pair** for the
temporal tasks, submit the query, and read the run identifier, the `mock_nodes` count, the trace
fill, and the answer's `[task]` tag. The eight expected cases are tabulated in
[`EVALUATION.md`](EVALUATION.md) §6.2.

### 6.4 The Chrome focus trap — why the harness asserts

Pass 1's harness drove the query box with `fill_input()`, which types with **real CDP key events**.
A re-run attempt failed on case 1 with `run_id=0002`, `mock_nodes=9`, `answer="No answer yet"`, and
only a `capabilities` call — the **mock path**. Root cause:

> Chrome **drops synthesized key events when the browser window does not hold OS focus**; the harness
> had **no assertion**, so it clicked Run with the page's *default* query still in the box.

Measured directly: with Chrome backgrounded, `press_key("Z")` left `#qtext.value` unchanged, while
`type_text("Q")` (CDP `Input.insertText`, not focus-gated) inserted fine
(`LIVE_VALIDATION_POSTFIX.md`, "Why this pass needed a new harness").

This is the single most important trap for anyone reproducing a live run, because it produces a
**silent false pass** — the pipeline "works", it just answered a different question. The fix was to
**assert the input state before dispatching**, with three pre-dispatch assertions:

- `q_ok` — the query box really held the query;
- `obs_ok` — `#obsTail == 'ready'`;
- `t0_ok` — both frames present where required.

The full account, including the check that pass 1 was **not** infected, is in
[`EVALUATION.md`](EVALUATION.md) §6.4–6.5. In short: pass 1's intents are query-specific (A1 reads
`taskvqa…temporalnone`, not the default's `taskchange…temporalrequired`), their answers embed the
query text, and A6's answer proves two files were uploaded — so pass 1's results are clean.

### 6.5 The recompute-verdicts safety property

Pass 3 was launched with a harness build that still carried two *discriminator* bugs (the answer
`[task]` tag exists only for region tasks; the intent panel is a concatenated string needing a
non-greedy match). Its raw output says `SUMMARY 0/8`; the verdicts are recomputed from the recorded
evidence by `recompute_verdicts.py`, giving 8/8. This is not a workaround — it is the intended
safety property:

> The recorded evidence (run id, `mock_nodes`, intent, answer, assertions) is **independent of the
> verdict computation**, so a harness bug can never silently turn a real failure into a pass, and
> never costs a re-run to correct.

— `LIVE_VALIDATION_POSTFIX.md`, Pass 3.

Two further harness bugs were found and fixed, both causing **false failures** (not false passes):
the answer `[task]` tag exists only for region tasks, and the intent panel needs a non-greedy match.
The harness now computes the dispatched task as `answer_tag` when present, else the reading. Both
are recorded because a false failure and a false pass are different risks and the project tracks
them separately.

### 6.6 Cold start and tunnel caveats

- **Cold start.** Render's free tier sleeps and the Codespace may be stopped. The first request can
  exceed the client timeout while weights are fetched; a retry a few seconds later normally
  succeeds. Warm the stack before any demonstration and confirm `agent_connected: true`
  (`README.md` §Deployment caveats).
- **Tunnel gaps (B-07, `OPEN`).** The tunnel agent can be briefly absent; a request during a gap may
  hang or return 504. **This is not fixed in production.** Root cause: in `auto` mode a tunnel
  timeout falls through to the forwarded-port path, spending the 120 s wake timeout on a 302 — the
  observed ~249 s failure (`docs/DEPLOYMENT.md §6`). A patch exists and was deliberately **not**
  deployed.
- **Never retry `POST /api/infer` at the gateway** — a retry consumes inference twice
  (`docs/DEPLOYMENT.md §7`).

---

## 7. Reproduce and verify the released artifacts

### 7.1 The manifest and checksums are generated, not typed

`models/manifest.json` and `models/checksums.sha256` are produced by
`release/tools/generate_model_manifest.py`, which reads the actual files:

> Generated by reading the files. No byte count or hash is typed by hand. Backbones are NOT
> redistributed; they are fetched from the Hugging Face Hub, pinned by revision.

— `models/manifest.json` (`"note"` field).

Verify the local weights:

```bash
sha256sum -c models/checksums.sha256
```

`models/checksums.sha256` (committed):

```
c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa  change/head.pt
cfae5e43b97ca930f568dc5b8ae4f36b24e9ff717af226159802206ffd63a82a  change_vqa/head.pt
785815729a3a39fc34dc41894efaf00d8739365d970a3f830a326e68ae888dab  optical_sar/head.pt
93432f7034be91a8ffd9c1a84e3eeec00bed7832c043fe7f83d2be230284c6bb  grounding/head.pt
8527c3ed28a293e13293d48601d48e3ceafa137b9acabddaf5de31a58a509b5c  router/adapter.pt
07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e  vlm/adapter_model.safetensors
```

The manifest records, per artifact: `id`, `task`, `kind`, `path` (in the project), `hf_path` (on the
Hub), `backbone`, `architecture`, `source_metric_artifact`, `config_hash`, `bytes`, `sha256`, and
`status`. The manifest's own header records `config_hash: 78f1e3700da15aa1` and
`artifact_count: 6`, generated `2026-09-25T18:15:38+00:00`.

### 7.2 Independent Hub verification

```bash
export HF_TOKEN=...            # token with repo.content.read
python release/tools/hf_verify.py
```

`release/tools/hf_verify.py` is an **independent** check: it does not trust the upload step. It
reads the manifest, **re-downloads each artifact from the Hub over direct HTTPS**, hashes the bytes
it received, and compares against the locally-computed sha256. Its docstring is explicit that it
uses direct HTTPS *deliberately*:

> Note: `hf_hub_download` is deliberately NOT used here — in this environment it returned an empty
> file (sha256 e3b0c442…), which would have produced a false FAIL. Direct HTTPS is the honest check.

Recorded result (`release/tools/hf_verify_report.txt`):

```
repo            : thundercode/SatQuery
private         : False
sha (HEAD)      : 55681e0cddb91a4a5655da98a49bc025e537b657
lastModified    : 2026-09-25T18:21:52.000Z
files on Hub    : 22

STATUS  hf_path                                    remote bytes   local bytes
----------------------------------------------------------------------------------
MATCH   change/head.pt                               63,231,009    63,231,009
MATCH   change_vqa/head.pt                            5,822,809     5,822,809
MATCH   optical_sar/head.pt                          14,427,457    14,427,457
MATCH   grounding/head.pt                            12,639,041    12,639,041
MATCH   router/adapter.pt                               211,961       211,961
MATCH   vlm/adapter_model.safetensors                34,798,048    34,798,048

artifacts verified : 6
artifacts failed   : 0

  OK  README.md
  OK  MODEL_CARD.md
  OK  models/manifest.json
  OK  models/checksums.sha256
```

Total released weight payload: **131,130,325 bytes (~125 MiB)** across the six artifacts
(`release/HF_RELEASE_VERIFICATION.md §4`). Exit code 0 with `artifacts failed : 0` means the release
is intact.

### 7.3 A verification method that was itself wrong — the empty-file false FAIL

This is the most instructive reproducibility incident in the release, and it is recorded because a
verifier that silently hashes an **empty file** is a verifier that can lie in *both* directions.

The **first** verification attempt reported all six artifacts `DIFFER`, with every remote hash equal
to:

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
```

— the sha256 of **empty content**. The cause was **not** the upload: `hf_hub_download` returned an
empty file in the authoring environment (a download-path problem), so the verifier hashed nothing.

It was caught by a **second, independent method** — a direct `curl` download — which produced the
correct hash `8527c3ed28a293e13293d48601d48e3ceafa137b9acabddaf5de31a58a509b5c` for
`router/adapter.pt`, byte-identical to the local file and confirmed to be a real PyTorch zip
(`PK\x03\x04`, containing `adapter/data.pkl`). The verifier was then rewritten to use direct HTTPS
with proxies disabled (`ProxyHandler({})`), and now reports 6/6 `MATCH`.

`release/HF_RELEASE_VERIFICATION.md §5.1` states the lesson directly:

> The failed first attempt is recorded because a verifier that silently hashes an empty file would
> have produced a **false failure** — and, with a different bug, could just as easily have produced
> a **false pass**.

**What a third party should take from this.** Do not use `hf_hub_download` as the *sole* verification
path. The honest check is a direct HTTPS download with proxies disabled, hashing the bytes received,
and comparing sizes as well as hashes. `hf_verify.py` does all three
(`size_ok = remote_bytes == local_bytes == len(content)`; `hash_ok = remote_hash == local_hash`).

### 7.4 Cross-checks against independently-recorded hashes

Two of the six hashes can be checked against values recorded **elsewhere** in the project,
independent of this release (`release/HF_RELEASE_VERIFICATION.md §6`):

| Artifact | Recorded elsewhere | Computed here | Agree |
|---|---|---|---|
| `change_vqa_head` | `cfae5e43…d63a82a` (`artifacts/change_vqa/run/PROMOTION.json`) | `cfae5e43…d63a82a` | **yes** |
| `vlm_lora_adapter` | `07c76a75…a5adf5e` (adapter provenance) | `07c76a75…a5adf5e` | **yes** |

This is a stronger form of verification than self-consistency: it means the released bytes match
hashes that were recorded by the training/export process, not merely by the release step.

### 7.5 The evidence archive

`release/tools/build_archive.py` builds the Phase-7 evidence archive. Per owner decision **G3**,
everything under `artifacts/` is included **verbatim** — including duplicates and caches — so the
archive is **~4 GB** and requires ZIP64. The layout:

```
SatQuery_AI_Final_Archive_2026-09-25/
  README_ARCHIVE.md              (what is inside, what was excluded and why)
  release/                       (the curated public release tree, verbatim)
  artifacts/                     (VERBATIM, per owner decision G3)
  evidence/
    live_validation/             (3 validation passes + screenshots)
    delivery/                    (delivery report + handoff)
  verification/                  (state, manifests, verification reports)
```

Weights and caches are stored with `ZIP_STORED` (no recompression) to avoid wasting hours on
already-compressed data. The archive **excludes** secret/token files, OS junk, virtualenvs,
`node_modules`, the HF hub cache, and temp browser profiles — documented in `README_ARCHIVE.md`,
never silent. It is verified by `release/tools/verify_archive.py`, which extracts to a separate temp
directory and checks the CRC of every member plus the sha256 of every artifact against the live
files.

---

## 8. What "reproduce" means for the two externally-trained artifacts

Six trained artifacts are released. **Four were trained inside the project's own pipeline; two were
trained outside this repository** on external GPU. This distinction is the heart of the
reproducibility contract, so it is stated plainly rather than implied.

| Artifact | Where it trains | Guide |
|---|---|---|
| `router/adapter.pt` | local CPU | `configs/base.yaml` §router |
| `grounding/head.pt` | local | `configs/base.yaml` §grounding_training |
| `change/head.pt` | local | `configs/base.yaml` §change |
| `optical_sar/head.pt` | local, seed sweep | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` |
| `change_vqa/head.pt` | **external GPU (Kaggle)** | `docs/R02_KAGGLE_TRAINING_GUIDE.md` |
| `vlm/adapter_model.safetensors` | **external GPU** | `configs/base.yaml` §training |

### 8.1 What the repository reproduces for the external artifacts

For the two externally-trained artifacts, this repository reproduces:

1. **The promotion gate** — the checks the returned checkpoint must pass before it is accepted
   (byte-identity, sha256 match against the recorded digest, zero non-finite tensors).
2. **The evaluation** — the scoring path that turns a checkpoint into the recorded metrics.
3. **The serving wiring** — the code path that loads the artifact and routes to it.

It does **not** ship a one-command retrain for those two artifacts. That is stated, not implied.

### 8.2 The change-VQA path (Kaggle)

`docs/R02_KAGGLE_TRAINING_GUIDE.md` is the authoritative guide; it states its own scope:

> **Audience:** the person running the external GPU job. **Scope:** everything needed to go from
> this repository to a returned, reviewable checkpoint. **Not in scope:** deciding whether R-02 is
> done — that happens after the artifact comes back.

Its status is `IMPLEMENTATION_READY_FOR_EXTERNAL_TRAINING`, and it is explicit that *"training
produces an artifact, not a verified capability, and the run record says `TRAINED_UNVERIFIED`."*

The reproduction is **byte-scoped**: the upload set is defined by a digest so the external run can be
shown to have started from exactly this code.

| Item | Value |
|---|---|
| Upload set | **249 files**, at the root of the archive |
| Digest-defining subset | **248 files, 67,225,247 bytes** |
| Manifest hash | `e98d3854db5f3edec76c1748c0c1bc3041d006ae5858debee31701539af7d0d6` |
| Archive | `satquery-ai.zip` — 250 entries, 60.0 MB compressed |
| Inside the archive | `UPLOAD_MANIFEST.txt` — every file with size and SHA256, plus the digest |

The digest covers **248 files, not 249**, for a principled reason the guide states: the guide itself
lives in `docs/`, so it is part of the upload set — but it also *records* the digest, and a document
cannot contain the hash of itself. The digest is therefore defined over the upload set **excluding
this one file**; that is the only definition that can be reproduced. For the same reason the byte
total for all 249 files is **not** quoted in the guide — it changes whenever the guide is edited; the
authoritative value is in `UPLOAD_MANIFEST.txt`.

The frozen STANet checkpoint the head depends on is verified by size and SHA256 in the notebook:

```
checkpoint : <CODE_ROOT>/artifacts/change/levir_change_v001/head.pt
bytes      : 63,231,009
sha256     : c5ef31277b67aa01a593aec0eac503eeaccc6d674349fda20ca44c9cc6f8e9fa
checkpoint verified against the frozen digest
```

Training knobs are fixed, not tuned-to-pass: `HARD_STOP_SECONDS = 3 * 3600`, `EPOCHS = 40`,
`BATCH_SIZE = 256`, `DEVICE = "cuda"`, `SEED = 42` (the trainer's own defaults are
`epochs=40, batch_size=128, seed=42, patience=6, time_limit=10800s`). The guide instructs: *"Do not
change these to get past an error."*

Split integrity is measured, not assumed:

```
measured corpus ground truth (unique scenes / resolved questions):
  Train    1600 scenes   65967 questions
  Val       400 scenes   16441 questions
  Test      968 scenes   39686 questions
  Test2     968 scenes   31036 questions
integrity clean : True
scenes/split    : {'Train': 1600, 'Val': 400}
overlap         : {'Train|Val': 0}
```

The two test sets (Test 39686, Test2 31036) are why change-VQA is quoted under **two** test sets
(rule 3) — and why the promotion artifact carries both.

### 8.3 The VLM path

The VLM artifact is a PEFT LoRA adapter (`r=16, alpha=32, dropout=0.05`) on the text-model
projections of `HuggingFaceTB/SmolVLM-500M-Instruct` (`models/manifest.json`). Its acceptance status
is **`ACCEPTANCE-REJECTED`** — the metrics are usable (`exact_match 0.963`, `f1 0.96432`) but the
adapter was **not promoted**. `USABLE ≠ ACCEPTED` (rule 7). Reproducing this artifact's *training*
requires the external GPU environment described in `configs/base.yaml` §training; reproducing its
*evaluation* and its *serving wiring* is done by this repository. The rejection is preserved in the
record (`artifacts/vlm/phase6_closure.json`), not scrubbed.

### 8.4 The optical-SAR seed sweep

The optical-SAR head was trained locally with a **seed sweep**; the selected production artifact is
`fusion_head_production_v001`, and its pre-registered metric is the 11.5 target recorded in
`docs/PHASE12_115_METRIC_COMPUTED.md` and `artifacts/optical_sar/fusion_head_production_v001/
pre_registered_115_metric.json`. The input dimension is **derived at runtime** (not hardcoded) —
`2318 = 3×768 + 12 + 2` (`fusion_head.py:74`, `:311`), independently confirmed in
`docs/PHASE14_OPTICAL_SAR_DECISIONS.md §1`. The ruling is **`OPEN`**, and the accuracy (`0.931`) is
never quoted without the macro-F1 (`0.434161`) — rare classes are poorly handled (rule 4).

### 8.5 What is *not* one-command

There is no script that takes this repository and produces all six artifacts in one invocation. The
four locally-trained artifacts have local training paths, and the two external ones require an
external GPU environment. This is a deliberate boundary: the release ships **frozen artifacts with
provenance**, not a retraining harness.

---

## 9. What is NOT reproducible from this release — exhaustive

This section is the honest counterweight to §1. Every item below is something a reader might
reasonably expect to reproduce and **cannot** from this release alone.

### 9.1 Not reproducible because the source is not released

| Item | Reason |
|---|---|
| The three **private deployment repositories** (`SatQuery-Frontend`, `SatQuery-Backend`, `SatQuery-Inference`) | Private by design; their links 404 for an outside audience (`docs/DEPLOYMENT.md §6`). |
| The **live deployment** itself | Requires the private sources plus the Render/Codespace/Cloudflare accounts. |
| The **monorepo working copy** | Local only, no remote; 334 dirty entries (`docs/DEPLOYMENT.md §1`). |

### 9.2 Not reproducible because it does not exist or was not run

| Item | Reason |
|---|---|
| **System-level end-to-end benchmark** | **No such benchmark exists.** Per-specialist metrics are real; a single end-to-end number is `NOT RUN` (`README.md` §Known limitations 1). |
| **Router test-split number** | **The test split was never run.** The only router number is validation, ungated, `n = 86` (rule 5). |
| **Captioning benchmark** | `IMPLEMENTED` but the benchmark is `NOT RUN` ([`EVALUATION.md`](EVALUATION.md) §4.8). |
| **A composite / "overall accuracy"** | Forbidden by construction: `official_aggregate_weights: null`; `normalize.py` raises on aggregate requests (rule 8). |

### 9.3 Not reproducible because the data is large or external

| Item | Reason |
|---|---|
| **CDVQA / SECOND imagery** | Public but large. The release documents the acquisition + name-verification procedure, not the data. |
| **BigEarthNet full corpus** | **Not downloaded** — only a 28k S2 subset was used. |
| **LEVIR-CD-256 / VRSBench raw data** | Not redistributed; the splits (7120/1024/2048, and VRSBench `n=16159`) are documented, not shipped. |
| **Fusion held-out test (n=4000)** | Documented, not shipped. |
| **Feature caches** | Reproducible but included only in the verbatim archive, not in the release. |

### 9.4 Not reproducible because it was external-GPU training

| Item | Reason |
|---|---|
| **change-VQA retrain** | External GPU (Kaggle); the repo reproduces the *promotion gate*, *evaluation*, and *serving wiring*, not a one-command retrain (§8.2). |
| **VLM LoRA retrain** | External GPU; and the adapter is `ACCEPTANCE-REJECTED` anyway (§8.3). |

### 9.5 Not reproducible because it is frozen paperwork

| Item | Reason |
|---|---|
| **The historical ZeroGPU / Gradio deploy target** | `configs/deploy.yaml` describes the old HF-Space/ZeroGPU target and is left undisturbed as frozen paperwork; **no Gradio runtime exists in code** (`docs/DEPLOYMENT.md §9`). Editing it would move the config hash. |

### 9.6 Not reproducible *as a bit-identical training run*

The released **artifacts** are byte-identical (guaranteed by hash). The **training runs** that
produced them are not claimed to be bit-reproducible on arbitrary hardware: floating-point
non-determinism, different accelerators, and library-version drift all affect a training run. What is
guaranteed is: the *config* is frozen, the *seed* is recorded (`42`), and the *artifact* is hashable.

### 9.7 Explicit non-claims

To remove any ambiguity, this release does **not** claim:

- that the live system achieves any single accuracy number;
- that the router's `0.965116` is a test result (it is validation-only, `n = 86`);
- that the VLM adapter is accepted (it is rejected);
- that calibration improves ECE (it worsens it, `0.013755 → 0.014929`);
- that optical-SAR fusion is production-ready (ruling `OPEN`, macro-F1 `0.434161`);
- that change-VQA is settled (ruling `OPEN`, two test sets);
- that grounding is solved (`0.2838` canonical IoU — useful, not solved);
- that B-07 tunnel gaps are fixed (patch prepared, `NOT deployed`).

---

## 10. Environment traps recorded for reproducibility

These are the edges that cost real debugging time in the authoring environment. They are recorded so
a third party does not rediscover them — and, more importantly, so that a reviewer can distinguish
"this is a documented environment quirk" from "this is a defect in the released code".

### 10.1 The dead sandbox proxy needs `--noproxy '*'` / `ProxyHandler({})`

**Symptom.** Outbound HTTP calls fail or hang in the authoring sandbox.
**Root cause.** The sandbox proxy is dead; requests are routed to it and never reach the target.
**Fix.** Disable proxies for the call:

```bash
curl --noproxy '*' https://<backend-host>/api/health
```

```python
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
```

`release/tools/hf_verify.py:37-38` does exactly this (`opener_no_proxy()`). In a normal environment
the flag is harmless; in the sandbox it is mandatory.

### 10.2 pytest exists only in the repository virtualenv

**Symptom.** Invoking the system `pytest` fails or resolves to a different interpreter.
**Root cause.** pytest is installed only in `.venv`.
**Fix.** Always invoke the venv interpreter explicitly:

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_frontend_live_wiring.py -q
```

### 10.3 The full-suite pytest trips a bulk-delete guard

**Symptom.** Running the whole `tests/unit` tree trips 4× `test_safe_delete_shim` failures.
**Root cause.** Windows **verbatim-path** defects in the sandbox's bulk-delete guard. The precise
condition under which the shim intermittently triggers on Windows is
`UNKNOWN — not established from the available evidence`.
**Fix / workaround.** Run the targeted suites (106 and 183 pass cleanly); treat the 4 shim failures
as environmental, not regressions (§5.4).

### 10.4 Cloudflare 308-redirects `X.html` → `/X`

**Symptom.** A request to `something.html` returns a 308 redirect.
**Root cause.** Cloudflare Pages 308-redirects `X.html` → `/X`.
**Fix.** Reference the **extensionless** path (`docs/DEPLOYMENT.md §7`).

### 10.5 Chrome drops synthetic CDP key events without OS focus

**Symptom.** A browser-driven run silently answers the *default* query; the query box looks
untouched; `mock_nodes` is non-zero.
**Root cause.** Chrome **drops synthesized key events when the browser window does not hold OS
focus**. `press_key`/`fill_input` (real CDP key events) are focus-gated; `Input.insertText`
(`type_text`) is not.
**Fix.** Use `type_text` (not `fill_input`), and **assert the input state before dispatch**
(`q_ok`/`obs_ok`/`t0_ok`). This is the single most dangerous trap because it produces a **silent
false pass** (§6.4; [`EVALUATION.md`](EVALUATION.md) §6.4).

### 10.6 Buffering hides liveness (two separate causes)

**Symptom 1.** A redirected browser-harness log stays at 0 bytes until the process exits —
indistinguishable from a stall.
**Root cause 1.** `browser-use` block-buffers stdout even when redirected.
**Fix 1.** Force line-buffering and print a per-case marker (`CASE_START <id>`).

**Symptom 2.** Piping the harness through `grep` swallows all output if the pipeline is killed.
**Root cause 2.** `grep` block-buffers when piped.
**Fix 2.** Redirect to a file instead of piping through `grep`.

### 10.7 `hf_hub_download` returned an EMPTY file → a false FAIL

**Symptom.** Every artifact hashes to `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
(empty content) and the verification reports all `DIFFER`.
**Root cause.** `hf_hub_download` returned an empty file in the authoring environment (a download-path
problem), so the verifier hashed nothing. The upload was fine.
**Fix.** Use a **direct HTTPS download with proxies disabled**, hash the bytes received, and compare
**size as well as hash**. `release/tools/hf_verify.py` is the honest check (§7.3).

### 10.8 `codespace_name` carries a trailing `\n`

**Symptom.** `/api/health` reports `"codespace_name":"potential-space-trout-r4ppw969w45j2pvvw\n"`.
**Root cause.** B-02; the wake path strips it, so it is **cosmetic**.
**State.** `OPEN (cosmetic)`; the fix is not deployed (`docs/DEPLOYMENT.md §2, §6`).

### 10.9 A forwarded Codespace port returns 302 for a private repo

**Symptom.** A forwarded Codespace port returns HTTP 302 instead of serving.
**Root cause.** Private-repo port forwarding returns 302 — which is *why* the outbound tunnel exists
(`docs/DEPLOYMENT.md §7`).
**Fix.** Use the tunnel, not the forwarded port.

### 10.10 The tunnel agent must be started by the devcontainer

**Symptom.** A restarted Codespace comes up with `agent_connected: false`.
**Root cause.** The tunnel agent is started by the devcontainer `postStartCommand`; if it does not
run, the agent is absent.
**Fix.** Ensure `.devcontainer/` starts the agent on start (`docs/DEPLOYMENT.md §7`).

---

## 11. What is NOT RUN / OPEN / BLOCKED for this topic

Per the documentation standard, this section lists, for the topic of reproducibility itself, what is
not settled.

| Item | Status | Note |
|---|---|---|
| End-to-end system benchmark | **NOT RUN** | No such benchmark exists; nothing to reproduce. |
| Router **test**-split number | **NOT RUN** | Only validation (`n = 86`, ungated) exists. |
| Captioning benchmark | **NOT RUN** | Implemented; not benchmarked. |
| VLM adapter acceptance | **REJECTED** | Metrics usable; not promoted. `USABLE ≠ ACCEPTED`. |
| Optical-SAR fusion ruling | **OPEN** | Accuracy without macro-F1 is never quoted. |
| Change-VQA ruling | **OPEN** | Two test sets. |
| Grounding acceptance | **OPEN** | Two protocols; modest IoU. |
| Calibration improvement | **REJECTED (retained in frozen config)** | ECE worsened `0.013755 → 0.014929`. |
| B-07 tunnel gaps | **OPEN** | Patch prepared, **NOT deployed**. |
| B-02 trailing `\n` | **OPEN (cosmetic)** | Not deployed. |
| Licence | **OPEN** | **No LICENSE file exists**; owner decision. |
| Exact full-suite *collected* count | **UNKNOWN** | Per-suite counts are known (106, 183, 137 re-run); the single collected total is not established. |
| Exact Windows trigger for the delete-shim flake | **UNKNOWN** | The intermittent condition is not established from the available evidence. |
| Bit-identical training on arbitrary hardware | **NOT CLAIMED** | Only artifacts are guaranteed byte-identical, by hash. |

---

## 12. Where the evidence lives

| Evidence | Location |
|---|---|
| The 20-claim metric verifier | `release/tools/verify_readme_metrics.py` |
| Its committed output | `release/tools/readme_metrics_report.txt` (`ALL CLAIMS VERIFIED`) |
| The Hub verifier | `release/tools/hf_verify.py` |
| Its committed output | `release/tools/hf_verify_report.txt` (6/6 `MATCH`) |
| Hub release verification write-up | `release/HF_RELEASE_VERIFICATION.md` |
| Model manifest (generated) | `models/manifest.json` |
| Checksums (generated) | `models/checksums.sha256` |
| Manifest generator | `release/tools/generate_model_manifest.py` |
| Archive builder / verifier | `release/tools/build_archive.py`, `release/tools/verify_archive.py` |
| Config loader + invariants | `core/config.py` (hash `:76-80`; `_validate` `:94`) |
| Frozen config registry | `configs/base.yaml` (§evaluation `:273`, §deployment `:282`) |
| Dependency manifest + profiles | `requirements.txt` |
| Live topology, env vars, traps | `docs/DEPLOYMENT.md` |
| Live validation (3 passes, 24 runs) | `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md`, `results_final.json`, `results_pass3.json`, `run_final2.txt`, `run_final3.txt` |
| Verdict recomputation | `recompute_verdicts.py` |
| Kaggle training guide (change-VQA) | `docs/R02_KAGGLE_TRAINING_GUIDE.md` |
| Optical-SAR decisions | `docs/PHASE14_OPTICAL_SAR_DECISIONS.md` |
| Pre-registered 11.5 metric | `docs/PHASE12_115_METRIC_COMPUTED.md` |
| Grounding resolution decision | `docs/PHASE7_RESOLUTION_DECISION.md` |
| Delivery test results (94/183/5–6/137) | `docs/FINAL_DELIVERY_REPORT.md §7` |
| Frontend suite re-run count (106) | `release/CURRENT_RELEASE_STATE.md:121`, `README.md` §The test suites |
| Evaluation protocols and honesty rules | [`EVALUATION.md`](EVALUATION.md) |
| Benchmarks and their statuses | [`BENCHMARKS.md`](BENCHMARKS.md) |
| Model cards and artifact details | [`MODELS.md`](MODELS.md) |
| Datasets and splits | [`DATASETS.md`](DATASETS.md) |
| Training paths | [`TRAINING.md`](TRAINING.md) |
| Known limitations | [`LIMITATIONS.md`](LIMITATIONS.md) |
