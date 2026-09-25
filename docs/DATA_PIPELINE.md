# Data pipeline — deep reference

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED`
· `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

Every statement in this document was produced by reading a file in `C:/Users/anish/satquery-ai/`. Where a
fact is not established from the available evidence, this document writes
`UNKNOWN — not established from the available evidence`. Where a stage is *declared* in configuration but
read by no code, that is stated explicitly rather than presented as a working step. Where a cache spec
hash is cited, it was **recomputed** from the code that produces it, not copied from a report.

This chapter is the reference for the **end-to-end data path**: from an uploaded asset to the tensor a
specialist consumes, including every decode, every normalisation, every cache, and every intermediate
artefact. It is written to be sufficient to reconstruct the pipeline from the document alone.

**Sibling documents:** [`GEOSPATIAL.md`](GEOSPATIAL.md) (the raster contract, CRS, coordinate systems, the
sensor adapter, the tiling policy and the 224 decision), [`architecture/03-request-lifecycle.md`](architecture/03-request-lifecycle.md)
(the nine controller states), [`architecture/05-specialists.md`](architecture/05-specialists.md) (the six
specialists), [`architecture/07-configuration-freeze.md`](architecture/07-configuration-freeze.md) (the
frozen config identity), [`DATASETS.md`](DATASETS.md) (the corpora), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
(reproduction recipes), [`TRAINING.md`](TRAINING.md) (how each corpus is consumed).

---

## Table of contents

0. [How to read this document](#0-how-to-read-this-document)
1. [The end-to-end path](#1-the-end-to-end-path)
2. [Validation and decoding](#2-validation-and-decoding)
3. [Modality inference, and where it is consumed](#3-modality-inference-and-where-it-is-consumed)
4. [The normalisation stages](#4-the-normalisation-stages)
5. [Tiling and tile selection](#5-tiling-and-tile-selection)
6. [The change-VQA data path](#6-the-change-vqa-data-path)
7. [The optical-SAR data path](#7-the-optical-sar-data-path)
8. [The router data path](#8-the-router-data-path)
9. [The grounding data path](#9-the-grounding-data-path)
10. [The VQA / caption data path](#10-the-vqa--caption-data-path)
11. [Determinism and reproducibility](#11-determinism-and-reproducibility)
12. [Where intermediate artefacts live](#12-where-intermediate-artefacts-live)
13. [What is NOT part of the pipeline](#13-what-is-not-part-of-the-pipeline)
14. [What is NOT RUN / OPEN / BLOCKED for this topic](#14-what-is-not-run--open--blocked-for-this-topic)
15. [Where the evidence lives](#15-where-the-evidence-lives)

---

## 0. How to read this document

The pipeline has **two distinct halves**, and almost every confusion about it comes from conflating them:

1. **The serving path** — an uploaded asset becomes a specialist result, request by request. This is what
   runs in production. It is *stateless* with respect to data: it reads a raster, prepares it, and calls
   a model. No cache is consulted except the router's, and no intermediate artefact is required for a
   result to be produced.
2. **The preparation / training path** — a corpus on disk becomes a *cache* of frozen features, and a head
   is trained over that cache. This runs offline, once per corpus revision. It is where the
   `mean ± 2·std` radiometric stretch actually executes, where the `change_feat_v1` feature vector is
   computed, and where the arm (A/B) is baked in.

The caches are the bridge: the training path produces them, and the **serving path does not read them**
(except the router's corpus cache, which is a training-time artefact, and the change-VQA text cache, which
is likewise training-time). A reader who assumes the serving path is cache-driven will misread the system;
the caches exist so that a *frozen* encoder is not re-run across an entire corpus on every training
experiment.

Three further distinctions are load-bearing and are stated at each point below:

- **Declared versus executed.** `configs/base.yaml` declares the tiling policy, the optical percentile
  stretch and the SAR dB clip. All three are read by no code on the serving path.
- **Deterministic versus seeded.** Some stages are pure functions of their input (the display stretch, the
  quality gate, the radiometric stretch). Others are stochastic and take a seeded RNG (channel dropout).
  The distinction determines what "reproducible" means for each.
- **Cache-hit versus cache-miss.** Every cache in the project is keyed by a **spec hash** or a
  **fingerprint**, and a mismatch is a **miss** or a **refusal**, never a silent stale read.

---

## 1. The end-to-end path

### 1.1 The nine controller states

The controller runs a fixed nine-state machine, declared in `configs/base.yaml` under `agent.states` and
mirrored by the `ControllerState` enum in `core/schemas.py:79`:

```
RECEIVE → PARSE → VALIDATE → PLAN → PREPROCESS → EXECUTE → AGGREGATE → VERIFY → RESPOND
```

The division of labour is stated once and holds throughout: *the router understands, the policy decides,
the specialists compute, the VLM explains, and the evidence proves.* The three states that matter for the
data pipeline are VALIDATE, PREPROCESS and EXECUTE.

### 1.2 The path, drawn

```
POST /v1/assets                          (gateway; five-type allowlist, two-layer size cap)
   │
   └─ asset_id (opaque, 128-bit; never a path)
        │
        ▼
POST /v1/analyze { assets: [...], query }
   │
   ├─ RECEIVE   ── resolve handles to server-side paths
   │
   ├─ PARSE     ── router: query text → Intent (task, modality, temporal, spatial, language)
   │
   ├─ VALIDATE  ── core/controller.py::_inspect_asset
   │                 └─ preprocessing.raster.inspect_raster(path, max_pixels=..., ...)
   │                      header-only: dims → bands → dtype → CRS → transform → bounds
   │                      → nodata → modality → AssetMetadata
   │
   ├─ PLAN      ── policy selects a workflow (which specialists, in what order)
   │
   ├─ PREPROCESS ── per-specialist preparation (this chapter's core)
   │                 ├─ decode: read_bands → (C, H, W); load_image_array → (H, W, 3) uint8
   │                 ├─ sensor adapter: band map + zero-fill + availability mask
   │                 ├─ normalisation: display stretch / encoder-input stretch
   │                 └─ resize: to the model's canonical resolution
   │
   ├─ EXECUTE   ── specialist forward pass → SpecialistResult
   │
   ├─ AGGREGATE ── combine specialist results
   ├─ VERIFY    ── consistency checks
   └─ RESPOND   ── ResultEnvelope { run_id, result, trace }
```

`architecture/03-request-lifecycle.md` records a fact worth repeating here: the **PREPROCESS state is
never recorded** as a distinct `TraceStep` in the execution trace; the work happens inside each
specialist's `execute` rather than as a controller-level step. So the trace shows VALIDATE and EXECUTE but
not the preparation between them, and the preparation's parameters reach the trace through the
specialist's **evidence payloads** (the availability masks, the radiometry report, the decode metadata)
rather than through a PREPROCESS step.

### 1.3 What each state reads and writes

| State | Reads | Writes | Cache consulted |
|---|---|---|---|
| RECEIVE | the uploaded asset handle | a server-side path | none |
| PARSE | the query text | an `Intent` | router corpus cache (training only) |
| VALIDATE | raster **header** only | `AssetMetadata` | none |
| PLAN | the `Intent`, the `AssetMetadata` list | a workflow plan | none |
| PREPROCESS | raster **pixels** | model-ready arrays | none |
| EXECUTE | model-ready arrays | a `SpecialistResult` | none (serving) |
| AGGREGATE / VERIFY / RESPOND | specialist results | the envelope + trace | none |

The single most important cell in that table is VALIDATE's "raster header only": the inspection that
decides whether an asset is usable **does not decode it**. `inspect_raster`'s docstring states this —
*"Validate and describe a raster without loading its pixel data"* — and it is what keeps a 25-megapixel
rejection cheap.

---

## 2. Validation and decoding

### 2.1 Header-only validation — `inspect_raster`

Covered in full in [`GEOSPATIAL.md`](GEOSPATIAL.md) §1.3. The pipeline-relevant points:

- It reads dimensions, band count, dtypes, CRS, transform, bounds, nodata, resolution, driver and a tiled
  flag, and constructs a `GeoMetadata` plus an `AssetMetadata`.
- The **pixel budget** is enforced here: `width * height > max_pixels` raises `OversizedImageError`, which
  is recoverable — the caller may downscale. `max_pixels` comes from `image.max_pixels` (25,000,000).
- The **modality** is assigned here by `infer_modality` (§3).
- The **SHA-256** is computed here only when `compute_hash=True` (streaming, 1 MiB chunks) — it is needed
  for manifests and dedup detection, and it is the field the change specialist uses to detect a repeated
  acquisition.

### 2.2 Band decoding — `read_bands`

`preprocessing/raster.py::read_bands(path, *, indexes=None, out_dtype=None)` returns
`(bands, H, W)` plus a profile that retains `crs`, `transform` and `bounds`. The profile is the source of
georeferencing for every downstream write (change maps, masks) so *"downstream code never loses
georeferencing."* It accepts an optional `indexes` list and an optional `out_dtype` cast, and wraps every
failure as `RasterReadError`.

Consumers:

- `preprocessing/imagery.py::load_image_array` — for the VQA, caption and grounding paths.
- `specialists/optical_sar/specialist.py::execute` — reads the optical and SAR arrays raw, then hands them
  to the sensor adapter.
- `specialists/change/specialist.py::_write_artifacts` — reads T1's profile so the change map is written
  with T1's georeferencing.

### 2.3 Display decoding — `load_image_array`

`preprocessing/imagery.py::load_image_array` turns a raster into a displayable `(H, W, 3)` uint8 array:

1. `read_bands` → `(bands, H, W)`.
2. Transpose to `(H, W, bands)`; take the first three bands as RGB; repeat a single band three times; or
   stack a 2-D array three times.
3. Per-**scene** 2/98 percentile stretch over finite values, clip to `[0, 1]`, scale to uint8.

Its docstring is explicit about what it does **not** do: *"It does not resample, crop, or reproject. Those
change the pixel grid, and the grounding specialist converts normalized boxes to pixel coordinates using
the ORIGINAL raster's dimensions — a silent resize here would put every box in the wrong place."* And
about determinism: *"The percentile stretch is deterministic: the same file always yields the same array,
so a grounding box and a VQA answer describe identical pixels."*

`load_pil_image` wraps the same array in a `PIL.Image`, and notes that georeferencing is not lost because
the caller's `AssetMetadata` already carries it.

### 2.4 The deterministic quality gate

`preprocessing/quality.py` is *"a deterministic input-quality gate"* that sits **upstream of the model**.
It exists because of a measured failure: a loaded SmolVLM-500M-Instruct, given uniform random noise and a
prompt that explicitly said *"answer only from what is visible"*, produced *"a fluent, specific, entirely
fabricated scene description."* The gate's rationale is stated directly: *"A 500M-parameter VLM will
describe *something* for any input it is given, and asking it to self-assess reliably is asking it to do
the thing it just failed at."*

The discriminating signal is **lag-1 spatial autocorrelation**, not variance. The measured separation:

| Input | Autocorrelation |
|---|---|
| Real remote-sensing imagery | 0.6 – 0.99 |
| Uniform random noise | ~0.00 |
| A constant (blank) image | undefined; variance ~0 |

Variance alone cannot separate a flat desert scene from an all-zero tile (both have near-zero variance);
autocorrelation is high for both uniform-but-real imagery and textured imagery, and collapses only for
noise.

The constants:

| Constant | Value | Meaning |
|---|---|---|
| `MIN_AUTOCORRELATION` | `0.10` | Below this, the array is not spatially coherent |
| `FLAT_STD_EPSILON` | `1e-6` | Below this normalised std, the array is constant |
| `NOISE_ENTROPY_BITS` | `7.8` | Shannon entropy above which the histogram is noise-like |
| `BLOCKING_VERDICTS` | `{NOISE, INVALID_VALUES}` | Verdicts that must block a VLM call |

`assess_image_quality` returns an `ImageQuality` with one of five verdicts:

| Verdict | Meaning | Effect |
|---|---|---|
| `STRUCTURED` | Spatial structure consistent with real imagery | Safe to analyse |
| `FLAT` | Effectively constant | Usable, but `is_degraded` — lower confidence |
| `NOISE` | Uncorrelated, near-uniform | **Blocks the VLM call** |
| `TOO_SMALL` | Fewer than 16 px on a side | Usable, flagged |
| `INVALID_VALUES` | NaN or infinite values present | **Blocks the VLM call** |

Two behaviours are worth quoting. First, the non-finite rule is deliberately **tolerance-free**: *"a fixed
fraction is size-dependent, so one NaN in a 64x64 array (0.99976) would pass while one NaN in a 10x10
array (0.99) would fail. The same defect must not be tolerated or rejected depending on image
dimensions."* Any non-finite value is disqualifying. Second, the entropy check is described as a
**redundant second signal**: *"MEASURED, and the margin here is thin: structured ramp+texture gives 7.581,
uniform noise gives 7.988. That is a 0.22-bit gap against a 7.8 threshold. The autocorrelation check is
what actually carries this gate (0.952 vs -0.008, a 0.96 margin against a 0.10 threshold)."*

The gate is a pure function — *"Nothing here is learned, sampled, or probabilistic. Same input, same
verdict."* A known limitation is recorded: a float GeoTIFF whose nodata sentinel is NaN lands in
`INVALID_VALUES` and is refused, and the comment records that *"Filling nodata from the raster profile
belongs in the tiling work"* — i.e. the fix is deferred to a stage that does not exist (§5).

### 2.5 Error types

Every failure in the validation and decoding path is a typed `SatQueryError` subclass. The relevant ones
for this pipeline:

| Error | Raised by | Meaning |
|---|---|---|
| `RasterReadError` | `inspect_raster`, `read_bands`, `write_raster` | The file is not a readable raster |
| `OversizedImageError` | `inspect_raster` | Exceeds the pixel budget; **recoverable** |
| `UnsupportedBandsError` | `inspect_raster`, the sensor adapter | Band count is zero, or bands cannot be placed |
| `MissingCRSError` | `parse_crs` | A CRS string is malformed |
| `CoordinateError` | `geospatial/transform.py` | A conversion lacks the context it needs |
| `SpecialistError` | specialists | A forward pass or preparation failed |

The design rule from `preprocessing/raster.py` applies throughout: *"Never raise a bare exception. Every
failure is a typed SatQueryError."*

---

## 3. Modality inference, and where it is consumed

`infer_modality(band_count, explicit=None)` (`preprocessing/raster.py:58`) is covered in full in
[`GEOSPATIAL.md`](GEOSPATIAL.md) §6. Its pipeline role is as the **first consumer** of the band count that
VALIDATE read:

- `inspect_raster` calls it and stores the result on `AssetMetadata.modality`.
- The optical-SAR specialist's `_assess_pair` reads the **declared** modality first and falls back to
  `infer_modality(asset.geo.band_count)` only when it is `UNKNOWN`, appending a warning that names the
  heuristic.
- The change specialist does **not** infer modality: it requires exactly two assets and compares their
  CRS and registration, without asserting a modality for either.

The heuristic's limits are stated in its own comment: *"These are heuristics, not ground truth — the
sensor adapter is authoritative when a sensor descriptor is supplied."* A band count is not a sensor
declaration, and the adapter is the only component that can map a band to a canonical channel.

---

## 4. The normalisation stages

There are **three** distinct normalisation transforms in this codebase, plus one **declared but
unimplemented** stage. Conflating them is the most common misreading, so they are laid out side by side.

### 4.1 The display stretch — per-scene 2/98

**Where:** `preprocessing/imagery.py::load_image_array` (and its grayscale cousin in
`specialists/change/postprocess.py::to_grayscale_float`).

**What:** one `(lo, hi)` computed over **all finite values across all bands** of one scene, then
`(arr - lo) / (hi - lo)`, clipped to `[0, 1]`.

**Output:** uint8 `(H, W, 3)` for display (or float32 grayscale for registration).

**Purpose:** presentation — so a VQA answer and a grounding box describe identical pixels.

**Determinism:** pure; same file → same array.

**Not** the CROMA transform. `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §1.1 names three disqualifying
differences: per-scene rather than per-channel; uint8 3-band rather than float 12-channel; presentation
rather than radiometry. Reusing it for CROMA is explicitly **forbidden** by that ruling (§3.3).

### 4.2 The encoder-input stretch — per-channel `mean ± 2·std`

**Where:** `specialists/optical_sar/radiometry.py::normalise_for_croma`.

**What:** for each channel `c` of each sample, `lower = mean_c - 2·std_c`, `upper = mean_c + 2·std_c`,
then `(x - lower) / (upper - lower)`, optionally through a uint8 round-trip (`×255`, clip, quantise,
`/255`). Computed **per sample** (not per batch) with `ddof=1`.

**Output:** float32 of the same shape, in `[0, 1]`.

**Purpose:** the transform the CROMA authors' own README instructs users to apply before the frozen
encoder.

**Determinism:** *"pure and deterministic: no sampling, no learned statistics, no RNG, and no dependence
on batch composition. Same input -> same output."*

**Where it runs:** the **extraction** path (`training/fusion/extract.py`), not the serving path (§7.5).
The extraction module applies it *"because the arm's `use_8_bit` has to be applied at exactly one
place"*, and it refuses an encoder that would stretch a second time.

**The zero-channel rule:** an all-zero channel (the sensor adapter's unavailability convention) has
`std = 0` and would divide by zero. Such channels are **skipped and left at exactly zero**, with status
`unavailable`; a present-but-constant channel is skipped with status `degenerate`. This is the C-1
discipline applied to radiometry — *"the same rule that forbids inventing a band forbids inventing a
dynamic range for a band that does not exist."*

### 4.3 The declared-but-unimplemented conditioning stage

**Where declared:** `configs/base.yaml`:

```yaml
optical:
  normalization: percentile
  lower_percentile: 2
  upper_percentile: 98
sar:
  representation: db
  clip_min_db: -30
  clip_max_db: 5
```

**Status:** `DECLARED (not read)`. No code applies a percentile/dB conditioning stage. The radiometry
module's docstring states the consequence: *"the two `optical.*` and three `sar.*` config keys are still
read by no code. That is recorded, not fixed."* `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §6 item 3
adds: *"A config key that no code reads is worse than a missing key: it reads as a satisfied
requirement."*

### 4.4 The side-by-side

| Transform | Granularity | Output | Purpose | Implemented | Runs on serving path |
|---|---|---|---|---|---|
| Display 2/98 | per-scene, all bands | uint8 (H,W,3) | Presentation | ✅ | ✅ |
| Encoder-input `mean ± 2·std` | per-channel, per-sample | float32 `[0,1]` | Frozen-encoder interface | ✅ | ❌ (extraction only) |
| Change grayscale 2/98 | per-scene, grayscale | float32 `[0,1]` | Registration correlation | ✅ | ✅ |
| Percentile / dB conditioning | per-scene, per config | unbounded rescale | `base.yaml`'s intent | ❌ | ❌ |

---

## 5. Tiling and tile selection

Covered in full in [`GEOSPATIAL.md`](GEOSPATIAL.md) §10. The pipeline-relevant summary:

- `image.max_pixels: 25000000` **is enforced**, at `inspect_raster` time, raising `OversizedImageError`.
- `image.tile_size: 512`, `image.tile_overlap: 128`, `image.max_tiles: 64`, `image.top_k_tiles: 4` are
  **declared and read only by** `core/config.py`'s single relationship check
  (`top_k_tiles <= max_tiles`). No serving-path code slices a raster into tiles or selects a top-K.
- `change.tile_size: 256`, `change.tile_overlap: 0` are likewise declared and not consumed as a tiling
  step; 256 is the change model's training resolution.

**Consequence for the pipeline:** a raster within the pixel budget is handed to a specialist **whole**.
The only size-reduction steps that actually run are per-specialist resizes to a model's canonical
resolution (e.g. `resize_to_canonical` to 120×120 for CROMA, and the change feature extractor's bilinear
resize to 256×256). Neither is tiling: both resample the whole image rather than selecting a region.

The VLM path's `processor_longest_edge: 512` is a **control against** the processor's default 2048, not a
tiling step. `core/config.py` enforces `processor_longest_edge <= image.tile_size`, with the measured
rationale recorded: the default 2048 upscales a 512 tile 4× and splits it into ~17 sub-images, *"the real
figure is ~17x"* against the plan's estimated 4×.

---

## 6. The change-VQA data path

The change-VQA path (`Task.CHANGE_VQA`, R-02) is the most elaborate in the project: it has a frozen
detector, two frozen feature extractors, a trained head, and four cache artefacts. It is the clearest
illustration of the serving/preparation split.

### 6.1 The dataset and the preprocessing version tag

`training/change_vqa/dataset.py` defines the identity constants:

```python
DATASET_ID = "cdvqa"
PREPROCESSING_VERSION = "change_vqa_preproc_v1"
SPLIT_MAP: dict[str, str] = {
    "Train": "train",
    "Val": "val",
    "Test": "test",
    "Test2": "test",
}
```

`PREPROCESSING_VERSION` is *"bumped when the record shape or the target definition changes. Recorded in
every evaluation so two numbers computed under different definitions can never be silently compared."*
It is the **preprocessing version tag** for this path: it names the *record and target* definition, and is
distinct from the *feature spec* hashes below, which name the *tensor* definition.

`SPLIT_MAP` is a leakage control, not a convenience: `Test2` maps to `test` because Test2 is *"a SECOND
QUESTION SET over the SAME 968 scenes as Test, not an independent sample; it must never be pooled."*
`docs/PHASE10_CDVQA_DATA_STATUS.md` §4 measured this: Test and Test2 share **100% of their images** (968 /
968), while Train/Val/Test are cleanly disjoint.

`LABEL_PALETTE` decodes the six change classes (NVG surface, trees, low vegetation, water, buildings,
playgrounds) from a fixed colour palette; white `(255,255,255)` is a **shared background**, not a class.

`SceneTargets` derives supervision from `label1`/`label2` with four definitions, all fractions of total
pixels: `class_mag[c]` (magnitude), `class_delta[c]` (signed), `class_ratio[c]` (per-class ratio) and
`total_changed`. The `total_changed` definition was **validated against gold answers**: over 40 real
`change_ratio` rows, `count(label1 != label2) / total` reproduced the annotated bin **34/40 = 85%** of the
time, versus **1/40** for the competing definition. The residual 15% is recorded as *"annotator/map
disagreement, which is the noise floor of the target"* rather than smoothed away.

### 6.2 The change feature extractor

`training/change_vqa/features.py` defines the frozen change feature. The module explains why the features
are frozen: the plan budgets CDVQA at ≤ 3 h on T4×2, and training end-to-end through STANet would
re-learn a representation the project already has. Freezing it drops the trainable parameter count from
~15.8 M to ~1.5 M.

`CHANGE_FEATURE_DIM = 1045`, and it is **three things**:

| Part | Dim | Composition | Answers |
|---|---|---|---|
| `level_pooled` | 1024 | 4 levels × 128 ch × {mean, max} | *"what does the change look like locally?"* |
| `change_stats` | 5 | mean, std, frac>0.3, frac>0.5, frac>0.7 of the change map | *"how much of the scene changed?"* |
| `change_grid` | 16 | adaptive 4×4 pool of the same map | *"WHERE did it change?"* |

The dimensions are **derived, never typed** — *"a literal here would drift silently from the pooling code
below and produce a shape error three files away"*:

```python
N_LEVELS = 4
LEVEL_CHANNELS = 128
LEVEL_POOLED_DIM = N_LEVELS * LEVEL_CHANNELS * 2      # 1024
CHANGE_STATS_DIM = 2 + len(CHANGE_STAT_THRESHOLDS)    # 5
CHANGE_GRID_DIM = CHANGE_GRID * CHANGE_GRID           # 16
CHANGE_FEATURE_DIM = LEVEL_POOLED_DIM + CHANGE_STATS_DIM + CHANGE_GRID_DIM  # 1045
```

The spatial parts are load-bearing: *"Without `change_grid` the head would see a scene-global average and
could not distinguish 'buildings changed in one corner' from 'buildings changed everywhere', which is
precisely the difference between `largest_change` and `smallest_change`."*

The default working resolution is `DEFAULT_IMAGE_SIZE = 256` — *"the change model's OWN training
resolution (`configs/base.yaml: change.tile_size: 256`; LEVIR-CD patches are 256x256). Feeding the native
512 would push the frozen encoder outside the distribution it was trained on."* 512 is offered as an
explicit, recorded alternative.

`ChangeFeatureExtractor.extract(t1, t2)` re-runs the detector's own submodules to capture the intermediate
fused levels that `STANetStyleChangeDetector.forward` does not return. That is *"a second implementation
of one forward pass, so it is a drift hazard — and it is closed by a test that asserts the probability map
produced here is **bit-identical** to `STANetStyleChangeDetector.forward(...).probabilities`."*

### 6.3 The text feature extractor

`TextFeatureExtractor` wraps `sentence-transformers/all-MiniLM-L6-v2` — the same model the router uses, so
*"the text side of the reasoning head adds no new model to the deployment."* It produces `TEXT_FEATURE_DIM
= 384` normalised vectors and raises `FeatureExtractionError` if the encoder's output width disagrees with
the declared constant.

### 6.4 The cache spec hashes

Two independent hash functions key the two caches. Both are `sha256(...)[:16]`.

`feature_spec_hash(image_size, *, extractor)` hashes the feature definition:
`{spec, image_size, extractor, levels, level_channels, thresholds, grid, text_encoder}`, sorted keys.

`text_spec_hash(*, encoder, revision=None)` hashes the question-feature definition:
`{spec: "change_vqa_text_v1", encoder, revision, dim, normalised}`, sorted keys. The separate hash is
deliberate: *"One combined hash would force a full re-extraction of both whenever either changed, and —
worse — would let a stale text cache pass a check it should fail."*

**Recomputed values** (this document computed them from the functions above, with the constants as read
from the source):

| Input | Spec hash |
|---|---|
| `feature_spec_hash(256, extractor="stanet:trained")` | `c801326f85a185f8` |
| `feature_spec_hash(256, extractor="stanet:untrained")` | `714efac5b0e6a6d1` |
| `text_spec_hash(encoder="sentence-transformers/all-MiniLM-L6-v2", revision="1110a243fdf4")` | `d2801ea1a314354a` |
| `text_spec_hash(encoder="sentence-transformers/all-MiniLM-L6-v2")` (unpinned) | `f87f5226c4c4a69b` |

So the change-feature cache spec for the trained detector at 256 px is **`c801326f85a185f8`**, and the
question-feature cache spec with the pinned MiniLM revision (`configs/base.yaml: router.revision:
1110a243fdf4`) is **`d2801ea1a314354a`**. The spec hash **encodes trained-vs-untrained**, which is how the
serving specialist detects a detector-state mismatch (§6.6).

### 6.5 Cache layout, resume, and the artefacts

`scripts/prepare_change_vqa.py` is the producer. Its default output directory is
`DEFAULT_OUT = REPO_ROOT / "artifacts" / "change_vqa"`, and it writes:

| Artefact | Shape | Content |
|---|---|---|
| `<out>/scene_manifest.jsonl` | one record per `(native split, scene)` | identity, integrity, split, question counts |
| `<out>/scene_targets.jsonl` | one record per scene | label-derived `SceneTargets` |
| `<out>/change_features.npz` | `(n_scenes, 1045)` + spec hash + extractor config | frozen STANet features, keyed by `scene_key` |
| `<out>/text_features_<split>.npz` | `(n_questions, 384)` + spec hash | question features, keyed by `question_id` |

The manifest is **scene-level, deliberately**: *"153,130 question rows would produce a ~50 MB JSONL that
duplicates text already authoritative in `annotations/*.json`. The manifest therefore records the **2,968
scenes** ... and the question-level records are derived from the annotations on demand."*

Resume is spec-guarded. `ChangeFeatureCache.read(path, *, expect_spec_hash=...)` **refuses** a cache built
under a different spec:

```python
if expect_spec_hash is not None and cache.spec_hash != expect_spec_hash:
    raise FeatureExtractionError(
        f"change feature cache at {p} was built under spec "
        f"{cache.spec_hash!r} but the caller expects {expect_spec_hash!r}; "
        f"re-extract rather than training on a representation that does not match serving"
    )
```

`TextFeatureCache.read` applies the same guard. The producer calls
`ChangeFeatureCache.read(cache_path, expect_spec_hash=extractor.spec_hash)` when resuming, so a
resolution change or a detector-state change invalidates the cache instead of silently mixing two
representations.

A scene whose imagery cannot be read is **skipped and reported** through `on_progress`, never silently
dropped: *"a scene missing from the cache is a training sample the model never sees, and the caller must be
able to count how many that was."*

### 6.6 The serving path, and the spec-mismatch refusal

`ChangeVQASpecialist` (`specialists/change/vqa_specialist.py`) assembles three pieces: a trained head, a
change feature extractor, and a question text encoder. Its `execute`:

1. Validates exactly two assets, both present.
2. Calls `unavailable_reason()`; if anything is missing or mismatched, returns `_unavailable_result` with
   **`answer=""`** and `degraded=True`.
3. Otherwise loads the two images, extracts `(vector, facts)` and the text vector, resolves the question
   type and temporal reference, and calls `predict_answers`.

**Why degraded produces no answer.** The module docstring is explicit: *"a random change map is visibly
noise, whereas an untrained 19-way classifier still emits a fluent, confident-looking `yes`. A user cannot
tell the second from a real answer, so the untrained case returns **no answer at all**."* The
`SpecialistResult` validator independently marks an empty change-VQA answer as degraded.

**The spec-mismatch refusal.** `feature_spec_mismatch()` compares the head's trained spec
(`head_metadata["change_cache_spec"]`) with the serving extractor's `spec_hash`. A mismatch is a
**refusal, not a warning**: *"A head is only meaningful for the feature distribution it was fitted on."*
The measured reason this matters: `configs/base.yaml`'s `change:` section carries **no `checkpoint_path`
key**, so the registry builds the change feature extractor with `checkpoint_path=None` and gets an
**untrained** STANet — while `scripts/prepare_change_vqa.py` defaults to the trained LEVIR checkpoint.
Training and serving would therefore consume different representations, and *"the head would still emit a
fluent `yes`."* The spec hash covers this because it encodes trained-vs-untrained, so the refusal catches
the detector-state mismatch as well as a resolution mismatch.

Other serving constants: `LOW_CONFIDENCE_THRESHOLD = 0.40` (a reporting threshold, not a calibration) and
`TEMPORAL_ORDER_NOTE` (T1=pre, T2=post; `label1=pre/label2=post` **proven** at agreement 1.0000 over 2,968
scenes; `im1=pre/im2=post` **supported statistically, not proven**).

**A config fact worth recording.** The specialist reads `change_vqa.image_size` (default 256),
`change_vqa.apply_type_mask` (default `True`) and `change_vqa.head_path` via `config.get(...)`, but
**`configs/base.yaml` has no `change_vqa` block at all**. Those keys therefore always take their code
defaults, and the frozen config does not name them. This is the same pattern as the grounding head's
`DEFAULT_HEAD_PATH` (a code constant rather than a config key, to avoid moving `Config.hash`).

---

## 7. The optical-SAR data path

### 7.1 The serving path, step by step

`OpticalSarSpecialist.execute` (`specialists/optical_sar/specialist.py:320`):

1. **Validate.** Exactly two assets; `_assess_pair` establishes one optical and one SAR (§9.2 of
   [`GEOSPATIAL.md`](GEOSPATIAL.md)).
2. **Descriptors.** `_descriptor_for` returns the asset's `SensorDescriptor` if present, else builds one
   from the band count — `build_optical_adapter` for optical, `positional_fallback_descriptor` for SAR —
   and appends a warning naming the assumption.
3. **Read pixels.** `read_bands(optical_asset.path)` and `read_bands(sar_asset.path)`.
4. **Pipeline.** `run_pipeline(...)` (below).
5. **Confidence.** `_confidence_components` computes the four plan-section-26 components.
6. **Answer + evidence.** `_compose_answer` builds prose only from computed facts; `_build_evidence`
   emits the availability masks, the view items, the fused-representation item and the decision.

### 7.2 `run_pipeline`

`specialists/optical_sar/inference.py::run_pipeline` is the seam between "an image on disk" and "three GAP
vectors plus two masks":

```
optical raster -> sensor adapter -> (12, H, W) + mask[12]
SAR raster     -> sensor adapter -> ( 2, H, W) + mask[2]
resize to CROMA's 120x120
CROMA(SAR_images=..., optical_images=...)     <- no mask, ever
assemble_fusion_input                         <- mask enters HERE
fusion head -> logits -> probabilities
```

Two properties are load-bearing:

- **Degradation is a first-class outcome.** `run_pipeline` *"NEVER raises for a missing model: the
  degradation is reported, because 'we have no trained head' is a fact about the deployment, not a failure
  of the request."* It raises only for a caller error — arrays that cannot be canonicalised at all.
- **The sensor side always runs.** *"The masks, the canonical channel placement, the availability
  statistics and the modality validation are all real work that requires no weights, and they are exactly
  the facts that tell an operator why a result is or is not trustworthy on Cartosat-2S + RISAT."*

`resize_to_canonical(array, resolution)` bilinearly resizes a `(C, H, W)` array to
`(C, resolution, resolution)`. Its docstring notes that nearest-neighbour would be wrong for continuous
optical reflectance channels, while for zero-filled channels *"interpolating zeros gives zeros."* It uses
`cv2.INTER_LINEAR` when OpenCV is available, falling back to `torch.nn.functional.interpolate`.

### 7.3 The mask path

`build_masks` converts the two `SensorAdapterOutput` masks to `(1, C)` float32, ready for concatenation.
`mask_availability_stats` computes the fractions and counts that feed both the confidence components and
the evidence. The mask **never reaches CROMA** (finding C-1) — the specialist's evidence item records
`"croma_received_mask": false` and `"mask_consumed_by": "fusion_head"`.

### 7.4 The fusion cache — `fusion_features`

`training/fusion/train.py` and `training/fusion/extract.py` implement the Phase 12 frozen-feature trainer
and its missing producer. The pipeline:

```
paired patches -> normalise -> resize -> CROMA (frozen) -> FusionFeature
               -> write_feature_cache -> train_fusion_head
```

The cache constants:

| Constant | Value | Meaning |
|---|---|---|
| `CACHE_VERSION` | `"v1"` | Bumped when the encoder, resolution or stored fields change |
| `DEFAULT_FEATURE_CACHE_DIR` | `artifacts/optical_sar/fusion_features` | Under `artifacts/`, scoped to this specialist, **not** under the frozen `artifacts/change/` tree |

`feature_cache_path(cache_dir=None, tag="")` returns
`base / f"croma_features_{CACHE_VERSION}{suffix}.npz"`, where the suffix is `_<tag>` when a tag is given
(e.g. a split name). The Phase-12 CLI (`scripts/extract_fusion_features.py`) instead takes an explicit
`--out-cache`, documented in its own usage as `artifacts/optical_sar/fusion_features/train.npz`, and writes
a `.npz` plus a `.json` sidecar. So both naming conventions are in play: the training loop's default is
`croma_features_v1_<tag>.npz`, and the CLI's documented example is `<split>.npz`. Both are
`artifacts/optical_sar/fusion_features/` files, and `verify_feature_cache` checks `row_counts_agree` and
`sidecar_n_matches_rows`.

**The write is atomic-ish.** `write_feature_cache` writes both files to unique temporary names and renames
them into place with `os.replace`, so *"no reader ever sees a half-written `.npz`."* The two renames cannot
be one operation; a crash in the window leaves a complete-but-stale pair, which `verify_feature_cache`
**reports** rather than letting a truncated array load cleanly and be silently wrong. A crashed run may
leave a `.<name>.tmp-<pid>-<uuid>.npz` behind, and the function deliberately never deletes it — *"a delete
is not a safe operation to perform on someone else's filesystem."*

**The multi-label collapse is an explicit policy.** BigEarthNet v2.0 is multi-label; the head is a
single-label 19-class softmax. The collapse is a parameter, `label_policy: Sequence[str] -> int | None`,
and with no policy: exactly one label → its index; more than one → `AmbiguousLabelError`; zero →
`None` → the sample is **skipped and counted**, never mapped to class 0 (*"'we do not know' and 'arable
land' are different facts and must not be merged"*).

**Resume is provenance-guarded.** Re-running with an existing cache loads it, skips the `sample_id`s
already present and appends the rest (`existing + new`, never a replacement). But *"appending rows from a
different arm, resolution or config would produce a cache whose metadata describes only half its rows"* —
so `check_resume_provenance` refuses that and names the field and both values; `resume=False` rebuilds.

**The dry run makes the same decision, not a similar one.** `plan_extraction` calls the same
`select_samples` the run calls, so `plan.n_selected` is `result.n_encoded` and `plan.n_would_write` is
`result.n_total`.

**Exit codes.** `scripts/extract_fusion_features.py` returns `0` on success (including a dry run), `2` for
a pre-flight or input error (nothing was encoded — a missing corpus, a leaked split, an unwritable
`--out-cache`, a missing CROMA checkpoint, an ambiguous label under `require_single_label`), and `3` for a
run that **started and then failed** with a typed error (e.g. a cached feature that is not 2318 wide). The
comment on the `FusionTrainingError` branch states the distinction: *"the run started, so this is 3, not a
pre-flight 2."*

### 7.5 The normalisation arm — A / B

The Phase 12 trainer selects the normalisation arm through the **existing hash-exempt environment
channel** `SATQUERY_CROMA_USE_8_BIT`, **not** by editing `configs/base.yaml`:

```python
ARM_CONTROL = "A"
ARM_VARIANT = "B"

ARMS: dict[str, Arm] = {
    ARM_CONTROL: Arm(ARM_CONTROL, True,  "... use_8_bit=true ..."),
    ARM_VARIANT: Arm(ARM_VARIANT, False, "... use_8_bit=false ..."),
}
```

The reason for the env channel is stated: *"Adding a `fusion_training:` block would move the frozen
`Config.hash` off `78f1e3700da15aa1` and detach the Phase 9 benchmark from its config."* `apply_arm` writes
the value and returns it so a caller can record it.

**A warning that must be repeated, not summarised.** The `B` in this dict is **NOT** the arm B registered
in `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §4. Both arms run the **same** per-channel `mean ± 2·std`
encoder-input stretch; they differ **only** on whether the result then takes the uint8 round-trip. The
axis is the **uint8 axis**, not the registered "stretch vs no-stretch" axis. The originally preregistered
arm B (*"percentile/dB conditioning only, no encoder-input stretch"*) is **NOT CONSTRUCTIBLE** from this
repository: stage 1 is unimplemented, and `Arm` exposes no field that can express "skip the encoder-input
stretch" — the stretch in `training/fusion/extract.py` is applied unconditionally. There is **no arm C**.

**Blocker recorded, not crossed.** The arm is baked into the cached features, and
`artifacts/optical_sar/fusion_features/` holds **only the Arm-A cache** (`train.npz` and `train.json`;
`val.npz` and `test.npz` were pending as of the Phase-14 correction notes). Arm B therefore needs its own
feature cache before it can be trained.

### 7.6 The extraction module's double-stretch guard

`training/fusion/extract.py` applies the DEV-2 stretch itself, and refuses an encoder that would apply it
again:

> `CROMAEncoder` applies the same stretch inside `encode` when `normalize_input=True` (its default).
> Passing such an encoder here would stretch twice — a silent distribution shift that produces a perfectly
> well-shaped cache full of wrong numbers. `_assert_raw_encoder` refuses it by name rather than trusting
> the caller; `build_extraction_encoder` constructs the raw encoder this module expects.

The CLI prints `normalize_input : <value>  (must be False)`, so the guard is visible in the run output as
well as enforced in code.

---

## 8. The router data path

### 8.1 The frozen encoder

`router/encoder.py::FrozenEncoder` wraps `SentenceTransformer` and holds it frozen. The verified contract
(Phase 4 probe, 2026-09-16, sentence-transformers 6.0.1) is recorded at the top of the module:

```
SentenceTransformer(
    model_name_or_path='sentence-transformers/all-MiniLM-L6-v2',
    revision='1110a243fdf4',      # pinned, verified reachable
    device='cpu',
)
.get_sentence_embedding_dimension()  -> 384
.tokenizer.model_max_length          -> 256     (finding F4-1)
.max_seq_length                      -> 256
params                               -> 22,713,216
encode(64 queries, CPU)              -> 0.118 s
```

Two constants are declared: `VERIFIED_TOKENIZER_MAX_LENGTH = 256` and `VERIFIED_EMBEDDING_DIM = 384`. The
constructor **refuses** a `max_length` above 256 — *"truncation would be a silent no-op"* — and sets
`self._model.max_seq_length = max_length` so the configured truncation (128, from
`router.max_length`) is *"enforced rather than merely documented."* The encoder returns detached numpy
arrays with no autograd history, which is what makes cached-embedding training possible (finding F4-2).

### 8.2 The corpus embedding cache

`router/train.py` is the consumer. `CORPUS_CACHE_VERSION = "v1"`, and `_corpus_fingerprint(corpus,
encoder_id, max_length)` hashes:

```python
payload = {
    "version": CORPUS_CACHE_VERSION,
    "encoder": encoder_id,          # f"{encoder.model_name}@{encoder.revision}"
    "max_length": max_length,
    "texts": sorted(e.text for e in corpus),
}
return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
```

`embed_corpus_cached` writes two files into the cache directory:

| File | Content |
|---|---|
| `embeddings_<fingerprint[:16]>.npy` | the `(n, 384)` float32 embedding matrix |
| `embeddings_<fingerprint[:16]>.json` | `{fingerprint, encoder, max_length, count, dim, seconds}` |

The cache key includes the encoder id **and** the corpus text set, so *"changing either invalidates it. A
stale cache silently training on the wrong vectors would be worse than no cache."* A cache hit requires
both `cached.shape[0] == len(corpus)` **and** `meta.get("fingerprint") == fingerprint`. A bad cache is a
**miss, not an error**: `except Exception: pass` falls through to re-encoding.

**This cache is a training-time artefact.** The serving path does not read it. `embed_corpus_cached` is
defined and used only in `router/train.py` (defined at `:232`, used at `:518`), while the serving path
calls `self.router.route(request.query)` (`core/controller.py:476`), which encodes the query live — a
single short string, measured at 0.118 s for 64 queries on CPU, so one query is sub-millisecond.

### 8.3 The router's training discipline

The module records two findings that shape the data path:

- **F4-2 (frozen encoder ⇒ cached training).** *"The encoder is FROZEN. Embeddings are therefore a pure
  function of the query text. So we embed the whole corpus ONCE, cache the vectors to disk, and train the
  50,822-parameter adapter on the cached matrix."* Router training needs **no GPU**.
- **F4-3 (splits by group, never by example).** *"Splitting by example would put 'show me the water body'
  in train and 'show me the road' in val — same template, one token apart — and report a fake accuracy."*
  The train function **refuses to proceed** if the split report is not clean.

The router's reported accuracy is **0.965116 on validation, ungated, n = 86**; the **test split was NOT
RUN**. This is a data-pipeline fact: the test split exists and is unused.

---

## 9. The grounding data path

### 9.1 The path

`GroundingSpecialist.execute`:

1. Validate exactly one image and a non-empty phrase; check the path exists.
2. `load_pil_image(asset.path)` — the deterministic display decode (§2.3).
3. `encoder.encode_image(image)` and `encoder.encode_text([query])` — RemoteCLIP ViT-B/32 at 224 px,
   producing `(49, 512)` patch tokens and a 512-d text embedding.
4. Decode: `_decode_with_head` when a trained head is loaded, else `_decode_zero_shot`.
5. Confidence from measurable objectness statistics.
6. Evidence: `BOUNDING_BOX` items in `NORMALIZED_0_1`, a `GEOLOCATION` item in `GEO` when the CRS allows,
   a `STATISTIC` item, and a `STATISTIC` record of any dropped degenerate candidates.

### 9.2 There is no feature cache on the grounding serving path

Unlike the change-VQA and optical-SAR paths, grounding has **no serving-time feature cache**. The encoder
runs live on each request. The Phase 7 resolution experiment and the Phase 8 evaluation used **cached**
patch features (the resolution experiment's artifacts are `per_sample_224_full.jsonl`,
`per_sample_448_full.jsonl` and `resolution_experiment_full.json` — 16,159 lines each), and the single
decode implementation exists precisely so a cached-feature path and a live path *"cannot disagree"*:

`decode_candidates_from_features` (`specialists/grounding/inference.py:151`) is *"THE single
implementation of the zero-shot decode. `ground_phrase` encodes an image then calls this; the evaluation
script reads a feature cache then calls this."* The module records the defect this closed: the Phase 8
eval originally built its own single-box baseline with `argmax_candidate` while Phase 7 measured through
`ground_phrase` — same 16,159 records, same metric, same cached features, but a different **decode**:

```
Phase 7 via ground_phrase  : mean best IoU 0.0972
eval via argmax_candidate  : mean best IoU 0.0092
```

A 10× gap, from a duplicated decode. *"The duplication was the defect, so the duplication is what is
removed: there is now exactly one place that turns similarity into boxes."*

### 9.3 The decode

`similarity_map` L2-normalises both sides before the dot product, because *"without that the dot product is
dominated by whichever vector happens to have the larger norm, which is a property of the features rather
than of the match."* `decode_candidates_from_features` then produces a threshold box (patches within
`DEFAULT_DELTA = 0.02` of the peak) plus up to `top_k` local maxima (a patch must beat its 4-neighbours,
and adjacent already-accepted peaks are skipped). If nothing qualifies, it falls back to `argmax_candidate`.

The head path uses `decode_cell_relative` + `sigmoid(objectness)` + `nms(iou_threshold=0.50)`, capped at
the specialist's `max_candidates` (6 by default, not the config's 20). The degenerate-box guard
(`DEGENERATE_AREA_FRACTION = 0.9`) applies to the zero-shot path only.

---

## 10. The VQA / caption data path

The VQA and caption paths share the display decode and the deterministic quality gate:

1. `load_image_array(asset.path)` → `(H, W, 3)` uint8.
2. The quality gate (`assess_image_quality`) — a `NOISE` or `INVALID_VALUES` verdict **blocks the VLM
   call**; a `FLAT` or `TOO_SMALL` verdict is usable but degraded. The gate is **wired into the serving
   path**, not merely defined: `specialists/vqa/inference.py:190` computes it and returns
   `self._refuse(...)` when `not quality.is_usable`, *before* the model is constructed or called. The
   comment there states the reason — *"The VLM will produce a fluent, specific, entirely fabricated scene
   description for ANY input it is handed, including uniform noise, even when the prompt explicitly
   instructs it to decline. Measured, not theorised. The guard therefore has to sit here, before the
   model."*
3. The processor builds `pixel_values` with `processor_longest_edge: 512` pinned, so a 512-px tile is
   **not** upscaled and split. The measured contrast is recorded in `configs/base.yaml`: default →
   `pixel_values (1, 17, 3, 512, 512)`, 1142 prompt tokens; pinned → `pixel_values (1, 1, 3, 512, 512)`.
4. Prompts **must** go through `processor.apply_chat_template()` — hand-written prompt strings raise
   `ValueError` because SmolVLM requires one `<image>` token per image (finding F5-3).
5. `max_images_per_call: 1` — the VLM sees one image per call.

The VLM adapter is loaded from `SATQUERY_VLM_ADAPTER` when set (the resolution order is: explicit
argument, then that env var, then none). The adapter's metrics are **usable** (exact_match 0.963) but its
status is **ACCEPTANCE-REJECTED** — USABLE ≠ ACCEPTED.

---

## 11. Determinism and reproducibility

### 11.1 What is deterministic

| Stage | Determinism | Mechanism |
|---|---|---|
| `load_image_array` | **Deterministic** | Pure percentile stretch; no sampling |
| `to_grayscale_float` | **Deterministic** | Pure percentile stretch |
| `assess_image_quality` | **Deterministic** | *"No model, no randomness, no thresholds learned from data"* |
| `normalise_for_croma` | **Deterministic** | *"No sampling, no learned statistics, no RNG, and no dependence on batch composition"* |
| `infer_modality` | **Deterministic** | Pure function of band count and explicit label |
| Sensor adapter | **Deterministic** | Pure band placement + zero-fill |
| `assemble_fusion_input` | **Deterministic** | Concatenation with asserted order |
| `resize_to_canonical` | **Deterministic** | Bilinear interpolation |
| Registration measurement | **Deterministic** | `cv2.phaseCorrelate`; no sampling |
| Post-processing (morphology, components) | **Deterministic** | `cv2` fixed operations |
| `channel_dropout` | **Seeded** | Takes an `rng`; *"Seeded deliberately: an unseeded training transform makes a run unreproducible"* |
| Router training | **Seeded** | `project.seed: 42` |
| Fusion training | **Seeded** | Explicit seeding, per `train_change_head`'s discipline |

### 11.2 The global seed

`configs/base.yaml` declares `project.seed: 42`. `Config.seed` returns
`int(self.get("project.seed", 42))`. Training loops take it from the config.

### 11.3 The config hash

`Config.hash` is `sha256(json.dumps(self._data, sort_keys=True, default=str).encode())[:16]`. The frozen
value is **`78f1e3700da15aa1`**. It is recorded in every evaluation run, and `scripts/eval_change.py`
**refuses to score on drift** (exit 3). This is why several pipeline decisions live in code constants or
environment channels rather than in `configs/base.yaml`: adding a key moves the hash and detaches the
published benchmark from its config.

### 11.4 Environment channels — two distinct classes

There are two **different** kinds of environment override, and conflating them would be wrong:

**(a) Pre-hash overrides — these DO change `Config.hash`.** `load_config` applies these into the data
dictionary **before** `Config(data)` is constructed:

| Variable | Effect |
|---|---|
| `SATQUERY_PRECISION` | Sets `training.precision` |
| `SATQUERY_TORCH_COMPILE` | Sets `deployment.torch_compile` |

Because they are merged before hashing, a run using either has a **different** `Config.hash` from the
frozen one.

**(b) Hash-exempt overrides — these do NOT change `Config.hash`.** They are read at use time, outside the
config object:

| Variable | Read by | Purpose |
|---|---|---|
| `SATQUERY_DEVICE` | `core/config.py::Config.device_preference` | Device selection |
| `SATQUERY_CROMA_USE_8_BIT` | `specialists/optical_sar/radiometry.py::resolve_use_8_bit` | The uint8 arm (A/B) |
| `SATQUERY_CROMA_CHECKPOINT` | `specialists/optical_sar/croma.py::resolve_checkpoint_path` | CROMA checkpoint path (first in resolution order) |
| `SATQUERY_VLM_ADAPTER` | `specialists/vqa/model.py` | VLM adapter path |

The distinction matters for reproducibility: a run that sets `SATQUERY_DEVICE` is comparable to a run that
does not (same config identity), while a run that sets `SATQUERY_PRECISION` is **not** (different hash).

### 11.5 Checkpoint resolution order

`specialists/optical_sar/croma.py::resolve_checkpoint_path` resolves the CROMA checkpoint
**offline-first**: `SATQUERY_CROMA_CHECKPOINT` (when set and non-empty, and the path must exist —
otherwise it raises naming the variable), then the Hugging Face Hub cache. The module records that the
`use_8_bit` resolution order is *"`SATQUERY_CROMA_USE_8_BIT` and then `croma.use_8_bit`."*

### 11.6 What "reproducible" means per artefact

| Artefact | Reproducible? | How |
|---|---|---|
| Display array from a raster | ✅ byte-identical | Deterministic stretch |
| Availability mask | ✅ byte-identical | Deterministic placement |
| `normalise_for_croma` output | ✅ byte-identical | Pure function |
| Change feature cache | ✅ given the same spec hash and detector state | Spec-guarded |
| Text feature cache | ✅ given the same encoder + revision | Spec-guarded |
| Fusion feature cache | ✅ given the same arm, resolution and config | Provenance-guarded |
| Router corpus cache | ✅ given the same encoder id + text set | Fingerprint-guarded |
| A trained head | ❌ not byte-reproducible across hardware | Seeded, but GPU nondeterminism applies |

`UNKNOWN — not established from the available evidence` whether bit-exact reproducibility of a trained
checkpoint is achieved across GPU runs; the code seeds explicitly but does not claim cross-device
determinism.

---

## 12. Where intermediate artefacts live

All generated artefacts live under `artifacts/`, which is *"the repository's generated-artefact root (not
source, not config, not `data/` inputs)."*

### 12.1 Caches (reproducible from a corpus)

| Artefact | Path | Producer | Key | Reproducible? |
|---|---|---|---|---|
| Router corpus embeddings | `<cache_dir>/embeddings_<fingerprint[:16]>.npy` + `.json` | `router/train.py::embed_corpus_cached` | corpus fingerprint (`v1` + encoder id + max_length + sorted texts) | ✅ |
| Change features | `artifacts/change_vqa/change_features.npz` | `scripts/prepare_change_vqa.py` | `feature_spec_hash(256, "stanet:trained")` = `c801326f85a185f8` | ✅ |
| Question features | `artifacts/change_vqa/text_features_<split>.npz` | `scripts/prepare_change_vqa.py` | `text_spec_hash(...)` = `d2801ea1a314354a` (pinned revision) | ✅ |
| Scene targets | `artifacts/change_vqa/scene_targets.jsonl` | `scripts/prepare_change_vqa.py` | `change_vqa_preproc_v1` | ✅ |
| Scene manifest | `artifacts/change_vqa/scene_manifest.jsonl` | `scripts/prepare_change_vqa.py` | `change_vqa_preproc_v1` | ✅ |
| Fusion features (Arm A) | `artifacts/optical_sar/fusion_features/train.npz` + `.json` | `scripts/extract_fusion_features.py` | `CACHE_VERSION="v1"` + arm + config + checkpoint digest | ✅ |
| Fusion features (Arm B) | — | — | — | **NOT PRODUCED** |

**A naming caution.** There is **no** `fusion_features_armB` directory (or any `armB`/`arm_b` literal) in
the repository — a grep for those spellings returns nothing. The arm is a field *inside* the cache's JSON
sidecar and the run record, not a directory name; the Arm-B cache would be a **separate cache built under
the same `artifacts/optical_sar/fusion_features/` root**, distinguished by its provenance record. Since
that cache has not been built, the root holds only the Arm-A `train.npz` + `train.json`.

### 12.2 Trained artefacts (not reproducible byte-for-byte)

| Artefact | Path | Notes |
|---|---|---|
| Grounding head | `artifacts/grounding/remoteclip_grounding_v001/head.pt` | The shipped default; `DEFAULT_HEAD_PATH` in code |
| Change detector | `artifacts/change/levir_change_v001/head.pt` | Test pooled IoU 0.8122; **not wired into serving by default** |
| Change-VQA head | `artifacts/change_vqa/run/head.pt` | `change_vqa_head_v1`, 1,453,912 params |
| Fusion head (production) | `artifacts/optical_sar/fusion_head_production_v001/` | `pre_registered_115_metric.json` |

### 12.3 Evidence artefacts

| Artefact | Path |
|---|---|
| CROMA forward pass | `artifacts/optical_sar/croma_forward.json` |
| CDVQA imagery verification | `artifacts/cdvqa/imagery_verification.json` |
| CDVQA/SECOND overlap | `artifacts/cdvqa/second_overlap.json` |
| CDVQA temporal order | `artifacts/cdvqa/temporal_order_evidence_v2.json` |
| Resolution experiment | `per_sample_224_full.jsonl`, `per_sample_448_full.jsonl`, `resolution_experiment_full.json` |
| Phase 12 selection manifest | `artifacts/phase12_selection/selection_manifest_seed10.jsonl` |

### 12.4 What the serving path writes

The serving path writes only **server-side diagnostic artefacts**, and their refs are **always `null`** to
the client (F-16): the change map (`change_map_<stem>.tif`) and the optical/SAR views
(`optical_view_<stem>.png`, `sar_view_<stem>.png`). *"The map is still WRITTEN. It is the operator's
diagnostic ... What changes is only that its location is a server-side fact, not a client-facing one."*
No `artifact://` URI is fabricated in its place.

---

## 13. What is NOT part of the pipeline

Each item below was checked against the code. Where a capability is absent, it is named rather than
implied.

| Capability | Status | Evidence |
|---|---|---|
| **Reprojection** | **NOT part of the pipeline** | `compare_crs` reports `requires_reprojection`; no code reprojects. |
| **Cloud masking** | **NOT part of the pipeline** | No cloud/QA/SCL handling anywhere. |
| **Atmospheric correction** | **NOT part of the pipeline** | Rasters are consumed as delivered. |
| **Mosaicking** | **NOT part of the pipeline** | No composite or seamline code. |
| **Tiling / top-K tile selection** | **DECLARED, not executed** | Read only by `core/config.py`'s `top_k_tiles <= max_tiles` check. |
| **Percentile / dB conditioning** | **DECLARED, not executed** | Read by no code; `radiometry.py` docstring; PHASE14 §1. |
| **Nodata filling** | **NOT part of the pipeline** | `quality.py` refuses non-finite input; the fix is *"deferred to the tiling work."* |
| **Pair-dimension reconciliation** | **NOT part of the pipeline** | The change path requires equal shapes; no reconciling resize. |
| **A serving-time feature cache** | **NOT part of the pipeline** | Only the router's training corpus cache and the change-VQA/fusion preparation caches exist; serving encodes live. |
| **Cross-step artefact hand-off** | **NOT part of the pipeline** | The change-VQA specialist *"accepts an optional `change_map` path in `request.params` for a future planner to populate. It is unused today"* — *"the controller does not hand artifacts between steps."* |
| **A single-pass change+language request** | **NOT part of the pipeline** | Routing a change+language request plans **both** a `change` and a `change_vqa` step, so *"the detector runs twice for one request"* — the weights load once (the registry caches the instance), but the forward pass runs twice. |
| **Image resolution decoding** | **NOT part of the pipeline** | CDVQA's `res_x`/`res_y` are the opaque strings `".1524m"` on every row; *"The field is of unknown semantics and is stored as an opaque string. It is not used anywhere."* |

---

## 14. What is NOT RUN / OPEN / BLOCKED for this topic

### NOT RUN

- **The router test split.** Reported accuracy is **0.965116 on validation, ungated, n = 86**; the test
  split was **NOT RUN**.
- **The optical-SAR forward pass with a trained head at serving time.** CROMA is not loaded locally and no
  trained head is wired; the specialist runs sensor-only.
- **The CROMA normalisation experiment (option c).** Not run; the arm set is frozen at A/B and arm B (the
  registered one) is non-constructible.
- **An end-to-end system benchmark.** No system-level accuracy is claimed.
- **A change+language request end-to-end.** The planner would run the detector twice; the artefact
  hand-off that would make it once is not implemented.

### OPEN

- **The percentile/dB conditioning stage.** Either implement it or retire the keys.
- **The arm-constructibility question.** A human decision, deliberately unresolved.
- **The optical-SAR ruling.** Accuracy 0.931 with macro-F1 0.434161; **OPEN**. Never quote the accuracy
  without the macro-F1.
- **The change-VQA metric ruling.** `OPEN` and owner-gated (two test sets: test 0.697626/0.378373 and
  test2 0.651469/0.372309).
- **Calibration.** ECE went **0.013755 → 0.014929 — worse**. Retained only because it is in the frozen
  config.
- **The VLM adapter.** Metrics usable (exact_match 0.963) but status **ACCEPTANCE-REJECTED**.

### BLOCKED

- **The Arm-B fusion feature cache.** `artifacts/optical_sar/fusion_features/` holds only the Arm-A cache;
  `val.npz` and `test.npz` were pending. Arm B cannot be trained without its own cache.
- **A clean change demo.** Blocked on a same-shape pair or a resize step.

### DEFERRED

- **`codespace_name` trailing `\n`** in the `/api/health` payload (B-02). Cosmetic; the wake path is safe.
- **Nodata filling from the raster profile.** Deferred to the tiling work, which is not implemented.

---

## 15. Where the evidence lives

**Source modules (the pipeline this chapter describes):**

| Path | What it defines |
|---|---|
| `preprocessing/raster.py` | `inspect_raster`, `read_bands`, `write_raster`, `infer_modality`, `file_sha256`, `pixel_area_m2` |
| `preprocessing/imagery.py` | `load_image_array`, `load_pil_image` — the display decode |
| `preprocessing/quality.py` | `assess_image_quality`, the deterministic gate |
| `specialists/optical_sar/sensor_adapter.py` | Band mapping, zero-fill, availability mask |
| `specialists/optical_sar/radiometry.py` | `normalise_for_croma`, `resolve_use_8_bit`, the zero-channel rule |
| `specialists/optical_sar/inference.py` | `run_pipeline`, `resize_to_canonical`, `build_masks` |
| `specialists/optical_sar/fusion_head.py` | `assemble_fusion_input`, `channel_dropout`, `expected_fusion_dim` |
| `specialists/optical_sar/specialist.py` | `execute`, `_descriptor_for`, `_confidence_components` |
| `specialists/change/specialist.py` | `_assess_pair`, `_predict`, `_write_artifacts` |
| `specialists/change/postprocess.py` | `measure_registration`, `to_grayscale_float`, `postprocess_change_map` |
| `specialists/change/vqa_specialist.py` | `feature_spec_mismatch`, `unavailable_reason`, `execute` |
| `specialists/grounding/specialist.py` | `execute`, `_decode_with_head`, `_to_geo_boxes` |
| `specialists/grounding/inference.py` | `decode_candidates_from_features`, `ground_phrase` |
| `training/change_vqa/features.py` | `feature_spec_hash`, `text_spec_hash`, `ChangeFeatureExtractor`, the caches |
| `training/change_vqa/dataset.py` | `DATASET_ID`, `PREPROCESSING_VERSION`, `SPLIT_MAP`, `LABEL_PALETTE` |
| `training/fusion/train.py` | `CACHE_VERSION`, `DEFAULT_FEATURE_CACHE_DIR`, `feature_cache_path`, `write_feature_cache`, `ARMS` |
| `training/fusion/extract.py` | The Phase 12 producer, the DEV-2 stretch, `_assert_raw_encoder` |
| `router/encoder.py` | `FrozenEncoder`, the verified MiniLM contract |
| `router/train.py` | `CORPUS_CACHE_VERSION`, `_corpus_fingerprint`, `embed_corpus_cached` |
| `core/config.py` | `load_config`, `Config.hash`, `device_preference`, the env overrides |
| `core/controller.py` | `_inspect_asset`, the nine-state machine |

**Scripts (the producers):**

- `scripts/prepare_change_vqa.py` — scene targets, manifest, change features, text features.
- `scripts/extract_fusion_features.py` — the Phase 12 fusion feature cache (exit codes 0/2/3).
- `scripts/train_fusion.py`, `scripts/train_change_vqa.py`, `scripts/train_grounding.py`,
  `scripts/train_router.py` — the training loops.
- `scripts/diagnose_feature_cache.py` — cache diagnosis.

**Configuration:**

- `configs/base.yaml` — `project.seed`, `image.*`, `optical.*`, `sar.*`, `croma.*`, `fusion.*`,
  `grounding.*`, `change.*`, `router.*`, `vlm.*`, `training.*`.

**Documents:**

- `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` — the two ordered stages, the Gate F arm amendment, the
  Arm-B blocker.
- `docs/PHASE12_ENTRY_GATE.md`, `docs/PHASE12_R08_TRUTHFUL_BACKFILL.md` — the Phase 12 run records and the
  `result_status` plumbing.
- `docs/PHASE10_CDVQA_DATA_STATUS.md` — the CDVQA/SECOND layout, the temporal semantics, the four adapter
  defects found by execution.
- `docs/PHASE7_RESOLUTION_DECISION.md` — the 224 decision (the cached-feature experiment).
- `docs/API_CONTRACT.md` §2.5 — the asset allowlist and the upload contract.
- `docs/FINAL_DELIVERY_REPORT.md` §4 (run ids), §5 (metrics), §6 (the 726²/736² degraded change demo).

**Sibling release chapters:**

- [`GEOSPATIAL.md`](GEOSPATIAL.md) — the raster contract, CRS, coordinate systems, sensor adapter, tiling
  policy, the 224 decision.
- [`architecture/03-request-lifecycle.md`](architecture/03-request-lifecycle.md) — the nine controller
  states.
- [`architecture/05-specialists.md`](architecture/05-specialists.md) — the six specialists.
- [`architecture/07-configuration-freeze.md`](architecture/07-configuration-freeze.md) — the frozen hash
  and every config key.
- [`DATASETS.md`](DATASETS.md), [`TRAINING.md`](TRAINING.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
