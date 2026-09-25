# Geospatial — deep reference

**Status tags used on every substantive claim:** `IMPLEMENTED` · `VERIFIED` · `MEASURED` · `ATTEMPTED`
· `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

Every statement in this document was produced by reading a file in `C:/Users/anish/satquery-ai/`. Where
a fact is not established from the available evidence, this document writes
`UNKNOWN — not established from the available evidence`. Where a transform is *declared* in configuration
but read by no code, that is stated explicitly rather than presented as a working feature. Where a
capability is absent, it is listed in [§13](#13-what-is-not-implemented-exhaustive) rather than implied.

This chapter is the reference for the **spatial contract**: how a raster becomes an analysed image, what
georeferencing survives, what coordinate system a box is in, how a sensor's bands become CROMA's canonical
channels, and which of those steps are actually implemented on the serving path. It is written to be
sufficient to reconstruct the subsystem from the document alone.

**Sibling documents:** [`DATA_PIPELINE.md`](DATA_PIPELINE.md) (the end-to-end data path, caches and
artefacts), [`architecture/03-request-lifecycle.md`](architecture/03-request-lifecycle.md) (where the
PREPROCESS and VALIDATE states sit), [`architecture/05-specialists.md`](architecture/05-specialists.md)
(the six specialists in full), [`architecture/07-configuration-freeze.md`](architecture/07-configuration-freeze.md)
(the frozen config identity and every key), [`DATASETS.md`](DATASETS.md) (the corpora and their measured
shapes).

---

## Table of contents

0. [How to read this document](#0-how-to-read-this-document)
1. [The raster input contract](#1-the-raster-input-contract)
2. [CRS handling](#2-crs-handling-geospatialcrs-py)
3. [Affine transforms and coordinate frames](#3-affine-transforms-and-coordinate-frames)
4. [The coordinate-system contract](#4-the-coordinate-system-contract)
5. [Benchmark box scale — 0–100 versus 0–1](#5-benchmark-box-scale--0100-versus-01)
6. [Band-count modality inference](#6-band-count-modality-inference)
7. [Radiometric normalisation and representation](#7-radiometric-normalisation-and-representation)
8. [The CROMA sensor adapter contract](#8-the-croma-sensor-adapter-contract)
9. [Pair-compatibility rules](#9-pair-compatibility-rules)
10. [Large-image tiling policy](#10-large-image-tiling-policy)
11. [Grounding resolution — the frozen 224 decision](#11-grounding-resolution--the-frozen-224-decision)
12. [Worked examples](#12-worked-examples)
13. [What is NOT implemented — exhaustive](#13-what-is-not-implemented-exhaustive)
14. [What is NOT RUN / OPEN / BLOCKED for this topic](#14-what-is-not-run--open--blocked-for-this-topic)
15. [Where the evidence lives](#15-where-the-evidence-lives)

---

## 0. How to read this document

Three distinctions are load-bearing throughout, and conflating them is the single most common way to
misread this system:

1. **Declared versus executed.** `configs/base.yaml` is the authoritative registry, and `core/config.py`
   validates and hashes it at load time. But *declaring* a policy and *executing* it are different
   claims. The tiling policy, the optical percentile stretch and the SAR dB clip are all declared in
   `configs/base.yaml` and are all read by no code on the serving path. This document says so at each
   point rather than presenting the key as a feature.
2. **Adapter versus encoder.** The CROMA sensor adapter (`specialists/optical_sar/sensor_adapter.py`)
   performs band mapping, zero-fill and availability masking. CROMA the encoder does **not** receive the
   availability mask (finding C-1). The mask is consumed by the fusion head. These are separate contracts
   and §8.9 treats the distinction in full.
3. **Presentation versus radiometry.** There are two different percentile stretches in this codebase:
   a per-*scene* 2/98 display stretch in `preprocessing/imagery.py` (and a grayscale cousin in
   `specialists/change/postprocess.py`), and a per-*channel* `mean ± 2·std` encoder-input stretch in
   `specialists/optical_sar/radiometry.py`. They serve different purposes and are **not**
   interchangeable. §7.5 is explicit about why.

The status vocabulary is used on every substantive claim. `MEASURED` means a number was produced by
executing code; `VERIFIED` means the claim was checked against a second independent source; `IMPLEMENTED`
means the code exists on the serving path; `DECLARED (not read)` means the value lives in
`configs/base.yaml` and no code reads it.

---

## 1. The raster input contract

### 1.1 What arrives, and what is accepted

An asset reaches the analysis path through `POST /v1/assets`, which mints an opaque `asset_id`. The
gateway enforces a **closed content-type allowlist of exactly five types**
(`docs/API_CONTRACT.md` §2.5):

| Content type | Notes |
|---|---|
| `image/tiff` | **The type the geospatial specialists need.** A client that uploads only PNG/JPEG can serve `vqa`, `caption` and `grounding`, but not `change` or `optical_sar` |
| `image/geotiff` | Accepted; media-type parameters are ignored, so `image/tiff; charset=binary` is accepted |
| `image/png` | Accepted |
| `image/jpeg` | Accepted |
| `application/octet-stream` | Accepted, but the raster reader still has to be able to open the bytes |

Two behaviours are specified rather than defaulted: a request with **no** declared content type is
**refused** rather than defaulted (defaulting is how a PDF reaches a raster reader), and a disallowed
type gets `415`. The stored filename is derived from the **content type**, never from the client's
filename, so a client-supplied `../../` cannot influence where bytes land (`docs/API_CONTRACT.md` §2.5).
The size cap is enforced at two layers — the gateway and the Space — and both now refuse an unparsable
or non-positive value rather than silently falling back to a default
(`docs/API_CONTRACT.md` §2.5, "Size cap").

`artifact_ref` on every `Evidence` item is **always `null` in v1** (`core/schemas.py:225`): v1 exposes no
artifact-serving endpoint, so a reference a client cannot retrieve is not fabricated. The schema field's
own description states this — *"Reference to an externally retrievable artifact. NEVER a filesystem path
(F-16, owner ruling 2026-09-23)"* — and the change and optical-SAR specialists both emit `null` on every
path (§4.3).

### 1.2 The validation chain

`preprocessing/raster.py` is the first thing every uploaded asset meets. Its module docstring names the
chain it implements, verbatim:

```
file -> dimensions -> bands -> dtype -> CRS -> transform -> bounds -> nodata
     -> modality -> temporal metadata
```

The module's three design rules are stated at the top of the file and are worth quoting because they
govern every failure mode below:

> * Never raise a bare exception. Every failure is a typed SatQueryError.
> * Never silently drop geospatial metadata. If the source had a CRS, the returned AssetMetadata says so.
> * A missing CRS degrades to non-geospatial mode; it does not abort.

### 1.3 `inspect_raster` — header-only validation

`inspect_raster` (`preprocessing/raster.py:96`) validates and describes a raster **without loading its
pixel data**. Its exact signature:

```python
def inspect_raster(
    path: str | Path,
    *,
    max_pixels: int | None = None,
    explicit_modality: str | None = None,
    compute_hash: bool = False,
    sensor: SensorDescriptor | None = None,
) -> AssetMetadata:
```

The behaviour, field by field:

- The path is checked with `Path(path).exists()` and `is_file()`; a missing path raises `RasterReadError`.
- `rasterio` is imported lazily; an `ImportError` raises `RasterReadError` rather than a bare error.
- Inside `rasterio.open(p)`, the reader extracts `width`, `height`, `count`, the sorted set of `dtypes`,
  the CRS as a string (`ds.crs.to_string() if ds.crs else None`), the affine transform truncated to its
  first six coefficients (`list(ds.transform)[:6] if ds.transform else None`), the bounds
  (`list(ds.bounds)`), `nodata`, `resolution` (`list(ds.res)`), the `driver` string, and a tiled flag
  read through `_safe_is_tiled`.
- `_safe_is_tiled` (`preprocessing/raster.py:42`) exists because `DatasetReader.is_tiled` is scheduled for
  removal; it suppresses the pending deprecation and **degrades to `False`** rather than aborting, since
  the flag is informational metadata only.
- Degenerate dimensions (`width <= 0 or height <= 0`) raise `RasterReadError`.
- If `max_pixels` is given and `width * height > max_pixels`, it raises `OversizedImageError` — and the
  docstring records that this is **recoverable**: *"the caller may downscale"*.
- A band count `<= 0` raises `UnsupportedBandsError`.
- Anything else rasterio raises is wrapped: `RasterReadError(f"could not open raster '{p}': {exc}")`.

The return is an `AssetMetadata` (see §1.4) with `modality` from `infer_modality` (§6) and, when
`compute_hash=True`, the streaming SHA-256 of the file (`file_sha256`, `preprocessing/raster.py:82`,
chunked at 1 MiB).

`inspect_raster` is called from `core/controller.py::_inspect_asset` during the VALIDATE state, with the
configured pixel budget passed through. That means the **header-only** nature of the inspection is a real
property of the serving path: a request never decodes a raster just to learn its shape.

### 1.4 `AssetMetadata` and `GeoMetadata`

`GeoMetadata` (`core/schemas.py:120`) is the geospatial detail carrier. It is `extra="allow"`, which is
deliberate — the optical-SAR specialist adds cross-sensor keys to it (§4.3):

| Field | Type | Meaning |
|---|---|---|
| `crs` | `str \| None` | CRS as a string (e.g. `EPSG:4326`), or `None` |
| `transform` | `list[float] \| None` | Affine coefficients, row-major, six values |
| `bounds` | `list[float] \| None` | `[minx, miny, maxx, maxy]` |
| `width`, `height` | `int \| None` | Raster dimensions in pixels |
| `band_count` | `int \| None` | Number of bands |
| `dtype` | `str \| None` | Comma-joined distinct dtypes when the bands disagree |
| `nodata` | `float \| None` | The nodata sentinel, or `None` |
| `resolution` | `list[float] \| None` | `(xres, yres)` |
| `has_crs` | `bool` | `crs_value is not None` |
| `is_georeferenced` | `bool` | `bool(crs_value and transform)` |

`AssetMetadata` (`core/schemas.py:153`) is `extra="forbid"` and carries:

| Field | Type | Meaning |
|---|---|---|
| `asset_id` | `str` | Minted `asset_<12 hex>`, opaque |
| `path` | `str` | Resolved local path |
| `modality` | `Modality` | From `infer_modality`, or declared |
| `sha256` | `str \| None` | Present only when `compute_hash=True` |
| `geo` | `GeoMetadata` | The block above |
| `sensor` | `SensorDescriptor \| None` | Present only when a sensor descriptor was supplied |
| `acquisition_date` | `str \| None` | Temporal metadata |
| `scene_id`, `dataset_id`, `split` | `str \| None` | Provenance for leakage control |

Note the distinction between `has_crs` and `is_georeferenced`: a raster can carry a CRS but no affine
transform (or vice versa), and both are recorded. Only when **both** are present is
`is_georeferenced=True`. `geospatial/transform.py::to_geo` requires the transform, so a raster with
`has_crs=True` but `transform=None` cannot be converted to geographic coordinates — it will raise
`CoordinateError("cannot convert to geo: raster has no affine transform")`.

### 1.5 Band reading, writing, and pixel area

`read_bands` (`preprocessing/raster.py:188`) returns `(bands, H, W)` plus a profile that **retains**
`crs`, `transform` and `bounds` as plain values, so downstream code never loses georeferencing. The
profile is a copy of `ds.profile` with those three keys overwritten from the live dataset.

`write_raster` (`preprocessing/raster.py:221`) writes an array using a profile captured by `read_bands`,
and re-applies `crs` and `transform` explicitly in the `rasterio.open(..., "w", ...)` call. It is used by
evidence generation so change maps stay georeferenced — **except** when the pair is suppressed for poor
registration, in which case the transform is deliberately dropped (§9.1).

`pixel_area_m2` (`preprocessing/raster.py:259`) returns the area of one pixel in square metres **when the
CRS is metric**, and `None` otherwise:

```python
def pixel_area_m2(meta: AssetMetadata) -> float | None:
    from geospatial.crs import is_metric
    if not is_metric(meta.geo.crs):
        return None
    if not meta.geo.resolution or len(meta.geo.resolution) < 2:
        return None
    xres, yres = meta.geo.resolution[0], meta.geo.resolution[1]
    return abs(xres * yres)
```

The docstring is explicit: *"Returns None for geographic CRS or unknown resolution — callers must not
guess."* A degree-based CRS has no meaningful pixel area in metres, so the function refuses rather than
returning a plausible-looking number.

---

## 2. CRS handling (`geospatial/crs.py`)

CRS comparison is kept **separate** from pixel arithmetic so that *"can these two rasters be compared at
all?"* is answerable without touching pixels (`geospatial/crs.py` docstring). The module depends on
`pyproj`.

### 2.1 `parse_crs`

```python
def parse_crs(value: str | None) -> CRS | None:
    if value is None or value == "":
        return None
    try:
        return CRS.from_user_input(value)
    except PyProjCRSError as exc:
        raise MissingCRSError(f"unparseable CRS '{value}': {exc}") from exc
```

Absent is `None`; malformed raises. That distinction matters: a missing CRS is a legitimate state
(degrades to non-geospatial), while a *malformed* one is a defect that must surface.

### 2.2 `describe_crs`

`describe_crs` returns a dict for the execution trace: `present`, `srs`, `name`, `is_geographic`,
`is_projected`, and `axis_units` (the first axis's `unit_name`). For an absent CRS it returns
`{"present": False}`.

### 2.3 `compare_crs` and `CRSCompatibility`

`CRSCompatibility` (`geospatial/crs.py:19`) is a frozen dataclass with `compatible`, `identical`,
`same_units`, `left`, `right`, `reason`, and a derived property:

```python
@property
def requires_reprojection(self) -> bool:
    return self.compatible and not self.identical
```

`compare_crs` (`geospatial/crs.py:59`) decides:

- **A missing CRS on either side is NOT compatible.** The reason string is
  `"one or both rasters lack a CRS; spatial comparison is unsafe"`. The docstring records that this is
  *"not fatal for non-spatial tasks"*.
- **Identical CRS** (`crs_l.equals(crs_r)`) returns `identical=True`, reason `"identical CRS"`.
- **Same kind** (both projected or both geographic) returns `compatible=True, identical=False`, reason
  `"reprojection required (<left name> -> <right name>)"`.
- **Mixed projected/geographic** returns `compatible=True, identical=False`, with a reason that adds
  *"reprojection required and should be verified"*. The comment records the rationale: *"Mixing a
  projected CRS with a geographic one is legal to reproject but signals a pipeline mistake."*

`same_units` compares the first axis's `unit_name` on each side (defaulting to `"unknown"` when there is
no axis info).

**Crucially: `compare_crs` never reprojects.** It reports whether reprojection is *required*; the actual
transform is not implemented anywhere (§13).

### 2.4 `is_metric`

```python
def is_metric(crs_value: str | None) -> bool:
    crs = parse_crs(crs_value)
    if crs is None:
        return False
    if not crs.is_projected:
        return False
    if not crs.axis_info:
        return False
    return crs.axis_info[0].unit_name in {"metre", "meter", "m"}
```

A geographic CRS is never metric here, and the unit check is a closed set of three spellings.

### 2.5 Where CRS comparison is used

The only consumer on the serving path is the change specialist's `_assess_pair`
(`specialists/change/specialist.py:251`):

```python
compat = compare_crs(t1.geo.crs, t2.geo.crs)
if not compat.compatible:
    warnings.append(
        f"CRS comparison unavailable ({compat.reason}); change is "
        f"reported in normalized pixel coordinates only"
    )
elif compat.requires_reprojection:
    warnings.append(
        f"T1 and T2 use different CRS ({compat.reason}); change is "
        f"valid only if the rasters already share a pixel grid"
    )
```

The comment above it states the policy precisely: *"No CRS on either side is NOT fatal for pixel-domain
change detection — the two rasters still tile to the same grid. It IS fatal for any claim about ground
coordinates, so the warning is recorded and the geospatial block is withheld downstream."* The
withholding happens in `ChangeSpecialist._geospatial`, which drops the transform when registration is
unusable (§9.1).

---

## 3. Affine transforms and coordinate frames

`geospatial/transform.py` is where a normalised 0–1 box, a pixel box and a geographic box are converted
between. Its module docstring states the non-negotiable rules (citing `docs/ARCHITECTURE_FREEZE.md`
§2.6):

> * CRS and affine transform are preserved, never silently stripped.
> * Conversions are explicit about their source and target frames.
> * A conversion that cannot be performed raises, rather than returning a plausible lie.

That last line is the module's governing principle, and it is why `to_normalized`, `to_pixel` and
`to_geo` all raise `CoordinateError` when they lack the context a conversion needs.

### 3.1 `PixelWindow` and `GeoBounds`

`PixelWindow` (`geospatial/transform.py:29`) is a frozen dataclass of `[col_min, row_min, col_max,
row_max]`. Its `__post_init__` **raises** on an inverted window:

```python
if self.col_max < self.col_min or self.row_max < self.row_min:
    raise CoordinateError(f"pixel window is inverted: {self.as_list()}")
```

It exposes `width`, `height` and `area` properties. `GeoBounds` (`geospatial/transform.py:60`) is the
geographic cousin: `(minx, miny, maxx, maxy)`.

### 3.2 Normalised ↔ pixel

`normalized_to_pixel(box, width, height)` multiplies each coordinate by the corresponding dimension. Its
docstring records a deliberate choice:

> Values are NOT clipped: a box slightly outside the frame is preserved so the caller can decide whether
> that is a bug or a legitimate edge case.

It raises `CoordinateError` if `len(box) != 4` or if either dimension is `<= 0`.
`pixel_to_normalized` is the inverse, dividing by the dimensions.

### 3.3 Pixel ↔ geographic

`pixel_to_geo(window, transform)` applies the affine transform to two corners. The docstring explains the
axis convention:

> rasterio/affine transforms map (col, row) -> (x, y). The y axis usually points down in pixel space, so
> the geographic miny comes from the BOTTOM row.

It uses the non-deprecated matmul form `transform @ (col, row)` and then takes `min`/`max` across the two
corners so the returned `GeoBounds` is ordered. It raises `CoordinateError` if the transform is `None`.

`geo_to_pixel(bounds, transform)` inverts the transform with `~transform`, raising `CoordinateError` if
the inverse cannot be computed (`"affine transform is not invertible: ..."`).

### 3.4 Schema-aware conversions

`to_normalized`, `to_pixel` and `to_geo` operate on `Box` objects and are schema-aware — they read
`box.coordinate_system` and return a `Box` with the target system set. The context each requires is
documented in the docstring:

- `pixel -> normalized` requires `width` and `height`.
- `geo -> normalized` requires `width`, `height` **and** the affine `transform`.

Each returns early when the box is already in the target frame, and otherwise uses
`box.model_copy(update={...})` to produce a new box with the new coordinates and the new
`coordinate_system`.

### 3.5 Intersection and IoU

`intersect(a, b)` returns the axis-aligned intersection of two `PixelWindow`s, or `None` when they are
disjoint. `iou(a, b)` computes intersection-over-union in `[0, 1]`, returning `0.0` for a disjoint pair
or a zero-area union. These are used by the grounding NMS path and by change-region overlap.

### 3.6 `affine_from_metadata`

```python
def affine_from_metadata(geo: GeoMetadata) -> Affine | None:
    if not geo.transform or len(geo.transform) != 6:
        return None
    a, b, c, d, e, f = geo.transform
    return Affine(a, b, c, d, e, f)
```

A `GeoMetadata` whose transform is missing or not exactly six values yields `None`; callers treat that as
"cannot georeference" rather than as an error. This is the bridge the grounding specialist uses in
`_to_geo_boxes` (§4.3).

---

## 4. The coordinate-system contract

### 4.1 The enum

`CoordinateSystem` (`core/schemas.py:57`) has exactly three members, and the class docstring is a warning
in itself — *"Never omit this. A bare box is meaningless without it. (C-5)"*:

```python
class CoordinateSystem(str, Enum):
    NORMALIZED_0_1 = "normalized_0_1"
    PIXEL = "pixel"
    GEO = "geo"
```

`docs/API_CONTRACT.md` §3.2 fixes the meaning of each value:

| Value | Meaning | Rendering |
|---|---|---|
| `normalized_0_1` | `[0, 1]`, origin **top-left** | Multiply by image width/height |
| `pixel` | Absolute pixel coordinates | Use directly |
| `geo` | CRS coordinates (usually EPSG:4326) | Requires a map, not a 2-D canvas |

The contract adds a hard warning: *"The frontend must read the `coordinate_system` field on each
`Box`/`Region` and must not assume one convention. A box drawn with the wrong assumption lands in
plausible-looking wrong places."* It also records that earlier drafts used shorthand (`normalized`,
`geographic`) and that those strings are **wrong** — they would fail schema validation, because the
models are `extra="forbid"` and the enum is closed.

### 4.2 Why it is mandatory — the schema validator

The `Box` model (`core/schemas.py:171`) defaults `coordinate_system` to
`CoordinateSystem.NORMALIZED_0_1`, and its `_ordered` validator rejects a box whose coordinates do not
satisfy `x2 >= x1` and `y2 >= y1`. `Region` (`core/schemas.py:189`) and `ChangeRegion`
(`core/schemas.py:202`) both carry the same field with the same default.

The enforcement that matters is on `Evidence`. `Evidence._spatial_needs_crs`
(`core/schemas.py:238`) **rejects a spatial evidence item that carries coordinates but no
coordinate system**:

```python
@model_validator(mode="after")
def _spatial_needs_crs(self) -> "Evidence":
    spatial = {
        EvidenceType.BOUNDING_BOX,
        EvidenceType.MASK,
        EvidenceType.CHANGE_MAP,
        EvidenceType.TILE,
        EvidenceType.IMAGE_CROP,
        EvidenceType.JOINT_FEATURE_REGION,
    }
    if self.type in spatial and self.coordinates and self.coordinate_system is None:
        raise ValueError(
            f"evidence type '{self.type.value}' carries coordinates "
            "but no coordinate_system"
        )
    return self
```

So a bare box cannot be published as evidence. The validator fires only when the item both is of a
spatial type **and** carries coordinates — a `STATISTIC` or `GEOLOCATION` item with no coordinates is
unaffected, and a spatial item with no coordinates (e.g. an availability-mask record) is also unaffected.
`GEOLOCATION` is deliberately **not** in the spatial set: a geolocation is a point reference, not a
rectangle, and the grounding specialist sets its `coordinate_system` to `GEO` explicitly anyway.

Note the interaction with `extra="forbid"`: `Evidence` forbids extra keys, so an author cannot smuggle a
convention through an undeclared field. `GeoMetadata`, by contrast, is `extra="allow"`, which is how the
optical-SAR specialist attaches cross-sensor keys (§4.3).

### 4.3 What each specialist emits

| Specialist | Boxes / regions | Evidence coordinate systems |
|---|---|---|
| `grounding` | `Box` and `Region` in `NORMALIZED_0_1` | `BOUNDING_BOX` in `NORMALIZED_0_1`; a `GEOLOCATION` item in `GEO` when the CRS allows; `STATISTIC` items with no coordinates |
| `change` | `ChangeRegion` and `Region` in `NORMALIZED_0_1` | `CHANGE_MAP` and `BOUNDING_BOX` in `NORMALIZED_0_1`; `STATISTIC` items with no coordinates |
| `optical_sar` | No `regions` (empty list) | `AVAILABILITY_MASK` items (no coordinates); `OPTICAL_VIEW` / `SAR_VIEW` with `coordinates=[0,0,1,1]` in `NORMALIZED_0_1`; a `JOINT_FEATURE_REGION` in `NORMALIZED_0_1`; `STATISTIC` items |

**Grounding.** `GroundingSpecialist.execute` (`specialists/grounding/specialist.py:253`) builds every box
with an explicit system:

```python
boxes = [
    Box(
        x1=d.box[0], y1=d.box[1], x2=d.box[2], y2=d.box[3],
        score=d.score, label=request.query,
        coordinate_system=CoordinateSystem.NORMALIZED_0_1,
    )
    for d in decoded
]
```

Geographic boxes are **derived, never replacements**: `_to_geo_boxes` (`specialists/grounding/specialist.py:431`)
converts the normalised boxes to geographic bounds *only* when the raster has a CRS, a transform and
dimensions, and emits a `GEOLOCATION` evidence item with `coordinate_system=CoordinateSystem.GEO`. When
the raster has no CRS/transform, the method appends a warning —
*"raster has no CRS/transform; boxes are reported in normalized coordinates only and cannot be placed on a
map"* — and returns an empty list. The result's boxes stay normalised; the geo boxes are additional
evidence for a consumer with a CRS. Its own comment states the rule: *"Pixel and geo coordinates are
DERIVED, never replacements."*

**Change.** `regions_to_schema` (`specialists/change/postprocess.py:443`) builds `ChangeRegion` objects in
the requested system. It supports `NORMALIZED_0_1` and `PIXEL` and **raises** for anything else:

```python
else:
    raise ValueError(
        f"regions_to_schema cannot produce {coordinate_system.value} "
        f"coordinates; convert after this call"
    )
```

The change specialist calls it with the default (`NORMALIZED_0_1`). Geographic conversion is a deliberate
post-step — the docstring records: *"Callers that want geographic coordinates convert afterwards via
`geospatial.transform`, which keeps this function free of rasterio and CRS concerns."*

**Optical-SAR.** The specialist emits no `regions`. Its view evidence items carry
`coordinates=[0.0, 0.0, 1.0, 1.0]` in `NORMALIZED_0_1` — the whole frame — because the view is a
presentation of the canonical tensor, not a localisation. Its `geospatial` block is the **optical**
asset's georeferencing (chosen because it is the asset a reader can navigate by), with cross-sensor keys
added via `GeoMetadata`'s `extra="allow"`: `optical_crs`, `sar_crs`, `sar_bounds`, `sar_resolution`,
`optical_availability`, `sar_availability`, `optical_sensor`, `sar_sensor`, and
`resolution_ratio_sar_to_optical` when both resolutions are known
(`specialists/optical_sar/specialist.py:597`). The comment records the reason the two footprints are
carried separately rather than fused: *"RISAT and Cartosat-2S are different missions with different
orbits."*

---

## 5. Benchmark box scale — 0–100 versus 0–1

This is a small contract with an outsized failure mode, and the codebase treats it as such.

### 5.1 The two conventions

- **VRSBench** (the grounding benchmark) states verbatim that *"all box coordinates are normalized to
  0-100"*.
- **This project** stores every box normalised to **0–1**.

Treating a 0–100 box as a pixel box, or as a 0–1 box, is a silent, catastrophic bug: the numbers are all
in range, the schema accepts them, and the box lands in the wrong place.

### 5.2 The conversion functions

`geospatial/transform.py:286` provides the conversion, and its docstring names the failure mode it exists
to prevent:

```python
def benchmark_boxes_to_normalized(
    boxes: Iterable[Sequence[float]], scale: float = 100.0
) -> list[list[float]]:
    """Convert benchmark boxes (e.g. VRSBench's 0-100) to our internal 0-1.

    Finding from Phase 0: VRSBench states verbatim that "all box coordinates are
    normalized to 0-100". Treating those numbers as pixels is a silent, catastrophic
    bug — this function exists so that mistake can be made once, explicitly.
    """
    if scale <= 0:
        raise CoordinateError(f"box scale must be positive, got {scale}")
    ...
    out.append([float(v) / scale for v in b])
```

`normalized_boxes_to_benchmark` is the exact inverse (multiply by `scale`). Both raise on a
non-positive scale and on a box that is not four values.

A second, independent conversion lives in the evaluation metrics:
`evaluation/metrics/grounding.py::benchmark_to_normalized(box, scale=100.0)` divides by the scale and
raises `ValueError(f"scale must be positive, got {scale}")` on a non-positive value. `docs/API_CONTRACT.md`
§3.2 names it explicitly as the site of the VRSBench conversion.

### 5.3 The declared scale

`configs/base.yaml` declares the conversion factor in the grounding block, with a comment that restates
the convention:

```yaml
grounding:
  ...
  # VRSBench stores boxes normalised to 0-100; we store 0-1.
  benchmark_box_scale: 100.0
  coordinate_system: normalized_0_1
```

`benchmark_box_scale: 100.0` is therefore the **declared** value, and `100.0` is also the **default
argument** in both conversion functions. The two agree.

### 5.4 The double-application risk

Because the scale appears in three places — the config key, the `geospatial.transform` default and the
`evaluation.metrics.grounding` default — there is a real hazard that a caller converts 0–100 → 0–1 and
then a downstream stage divides by 100 again, producing a box 100× too small in each coordinate. The
codebase's mitigations are structural rather than documentary:

1. **The conversion is named for its direction.** `benchmark_boxes_to_normalized` and
   `normalized_boxes_to_benchmark` are inverses with self-describing names, so a reader cannot easily
   apply the wrong one without noticing.
2. **The default is the correct scale, not `1.0`.** A caller that forgets to pass `scale` gets `100.0`,
   which is right for VRSBench and would produce an obviously wrong box (values > 1) for already-0–1
   input — a loud failure, not a silent one.
3. **The schema rejects out-of-range values indirectly.** A `Box` built from un-divided VRSBench
   coordinates would have `score` in range but coordinates in `[0, 100]`; the `_ordered` validator would
   pass it (the ordering is still valid), so the schema alone does **not** catch this. The catch is that
   the metrics module's own `benchmark_to_normalized` is the documented entry point, and the resolution
   experiment and the grounding evaluation both go through it.

**Residual risk — stated honestly.** There is no runtime assertion that a `Box` in
`NORMALIZED_0_1` actually has coordinates within `[0, 1]`. `Box._ordered` checks only ordering. A
mis-scaled box would therefore pass schema validation. `UNKNOWN — not established from the available
evidence` whether any production call site currently constructs a `Box` without going through one of the
two conversion functions; the serving path (`GroundingSpecialist.execute`) builds boxes from the decoder's
output, which is already in `[0, 1]`, so the serving path is not exposed. The risk is confined to
evaluation and experiment code that reads VRSBench directly.

---

## 6. Band-count modality inference

### 6.1 The heuristics

`preprocessing/raster.py` defines two closed sets:

```python
_OPTICAL_BAND_COUNTS = {3, 4, 8, 11, 12, 13}
_SAR_BAND_COUNTS = {1, 2}
```

with the comment: *"Band-count heuristics for modality detection when explicit metadata is absent. These
are heuristics, not ground truth — the sensor adapter is authoritative when a sensor descriptor is
supplied."*

### 6.2 `infer_modality`

```python
def infer_modality(band_count: int, explicit: str | None = None) -> Modality:
    if explicit:
        try:
            return Modality(explicit.lower())
        except ValueError:
            pass

    if band_count in _SAR_BAND_COUNTS:
        return Modality.SAR
    if band_count in _OPTICAL_BAND_COUNTS:
        return Modality.OPTICAL
    # 12-band optical and 2-band SAR are both plausible; default to optical only
    # when the count clearly favours it.
    if band_count >= 4:
        return Modality.OPTICAL
    return Modality.UNKNOWN
```

The decision order is worth reading carefully:

1. An **explicit** label wins if it parses to a `Modality` member. An explicit label that does *not* parse
   (e.g. `"radar"`) is silently ignored and inference proceeds — the `except ValueError: pass` is
   deliberate.
2. `{1, 2}` ⇒ `SAR`.
3. `{3, 4, 8, 11, 12, 13}` ⇒ `OPTICAL`.
4. `>= 4` (and not already matched) ⇒ `OPTICAL`.
5. Otherwise ⇒ `UNKNOWN`.

So the effective mapping is: 1–2 ⇒ SAR; 3–13 ⇒ optical (by set or by the `>= 4` fallback); 0 or a
non-positive count ⇒ `UNKNOWN` (though `inspect_raster` rejects `band_count <= 0` before this point).

### 6.3 The declared-wins rule

In the optical-SAR specialist, `_assess_pair` (`specialists/optical_sar/specialist.py:223`) uses the
asset's **declared** modality first, falling back to band-count inference only when it is `UNKNOWN`:

```python
for asset in assets:
    modality = asset.modality
    if modality is Modality.UNKNOWN:
        inferred = self._infer_modality(asset, warnings)
        resolved.append(inferred)
    else:
        resolved.append(modality)
```

`_infer_modality` appends a warning that names the heuristic for what it is: *"A band count is a
heuristic, not a sensor declaration — label the asset explicitly if this is wrong."* The docstring above
`_assess_pair` states the rationale: *"The declared value wins when present, because the caller may know
something the band count does not — a 2-band Cartosat stack, or a 12-band decomposition product."*

### 6.4 Why the heuristic is not authoritative

The comment on the sets and the warning text both point at the same thing: a band count does not identify
a *sensor*. A 2-band file could be a dual-pol SAR pair or a two-band optical stack; a 4-band file could be
RGB+NIR or a four-polarisation SAR product. The adapter (§8) is the authoritative path, because it maps
named bands onto CROMA's canonical channels and records what it could not place.

---

## 7. Radiometric normalisation and representation

This section covers a genuine specification gap and its partial resolution. It is presented in full
because the gap is load-bearing and the honest state is more useful than a tidy summary.

### 7.1 What configuration declares

`configs/base.yaml` declares two radiometric transforms:

```yaml
optical:
  normalization: percentile
  lower_percentile: 2
  upper_percentile: 98
  # CROMA expects exactly 12 optical channels.
  canonical_channels: 12

sar:
  representation: db
  clip_min_db: -30
  clip_max_db: 5
  # CROMA expects exactly 2 SAR channels (VV, VH).
  canonical_channels: 2
```

So the declared optical normalisation is a **percentile stretch at the 2nd and 98th percentiles**, and the
declared SAR representation is **decibels clipped to [−30, +5]**.

### 7.2 The DEV-2 ruling — two ordered stages

`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` records the ruling that governs this. The gap it confirmed:
CROMA's own example preprocessing is a **per-channel** dynamic-range stretch
(`mean ± 2·std → [0, 1]`, optionally through a uint8 round-trip), while `configs/base.yaml` specifies
something structurally different. The ruling's verification table records that a grep of
`specialists/optical_sar/` for `NORM_*`, `clip_min_db`, `lower_percentile`, `use_8_bit` returned **six
hits, all constant definitions/labels** at `sensor_adapter.py:125,126,210,323,547,548`, and **zero
application sites**.

The ruling resolves this by declaring the pipeline to have **two ordered stages**, and it states
explicitly that they are *not alternatives*:

| Stage | Transform | Owner | Status |
|---|---|---|---|
| **Radiometric conditioning** | per-scene percentile / dB-clip, per `configs/base.yaml` | preprocessing / sensor adapter | **specified in the ruling, must be implemented** |
| **Encoder-input normalisation** | per-channel `mean ± 2·std` → `[0,1]`, then optional `×255` uint8 → `/255` | a new function consumed by `croma.py` | **mandatory, immediate** |

The rationale for the CROMA stage being mandatory: *"it is the only stage that bounds the input range.
Without it the encoder receives unbounded input."* The `use_8_bit` branch is the safer default because
the uint8 round-trip **guarantees** `[0, 1]`, whereas the float path's clip applies to a stretch that is
unbounded by construction (`mean ± 2·std` is not a min/max clamp). The decision on `use_8_bit` is
**ENABLED**.

### 7.3 The implemented stage — `specialists/optical_sar/radiometry.py`

The **encoder-input** stage is implemented, in full, at `specialists/optical_sar/radiometry.py`. Its
module docstring states the boundary between what is and is not implemented:

> Implemented: the **encoder-input** stage — per-channel `mean +/- 2*std` -> `[0, 1]`, the transform the
> CROMA authors' own README instructs users to apply.
>
> NOT implemented: the percentile / dB **conditioning** stage that `base.yaml` names.

The transform is quoted verbatim from the CROMA README in the docstring:

```
min_value = x[:, c].mean() - 2 * x[:, c].std()
max_value = x[:, c].mean() + 2 * x[:, c].std()
img = (x[:, c] - min_value) / (max_value - min_value) * 255.0
img = clip(img, 0, 255).to(uint8)
# before the forward pass:
x = x.float() / 255
```

The module makes two **deliberate deviations** from the README, both documented:

1. **Per sample, not per batch.** The README computes `x[:, c].mean()` over the whole batch, which makes
   one image's encoding a function of its neighbours. This module computes the window **per sample**,
   because *"a serving system whose whole point is reproducibility"* must not produce a different answer
   because another request was batched alongside it. At `N = 1` the two are identical, and the README's
   own example runs at `N = 1`.
2. **Unbiased standard deviation (`ddof=1`).** Upstream is torch, whose `.std()` defaults to the unbiased
   estimator; NumPy's defaults to `ddof=0`. The module uses `ddof=1` to match the reference
   implementation, and states the numerical immateriality (~1.00003 factor at 120×120) so nobody has to
   guess which convention a number came from.

The key constants:

| Constant | Value | Meaning |
|---|---|---|
| `TRANSFORM_NAME` | `"per_channel_mean_pm_2std"` | Recorded on every trace |
| `WINDOW_SIGMAS` | `2.0` | The window half-width |
| `DEFAULT_USE_8_BIT` | `True` | The ruling's decision |
| `ENV_USE_8_BIT` | `"SATQUERY_CROMA_USE_8_BIT"` | Hash-exempt override channel |
| `MIN_FINITE_PIXELS` | `2` | Below this, a std is not meaningful |
| `STATUS_NORMALISED` / `STATUS_UNAVAILABLE` / `STATUS_DEGENERATE` | strings | Per-channel status |

`normalise_for_croma(array, *, use_8_bit, modality)` is **pure and deterministic** — the docstring
states: *"no sampling, no learned statistics, no RNG, and no dependence on batch composition. Same input
-> same output."* It accepts `(B, C, H, W)` or `(C, H, W)`, returns a `float32` array of the same shape
plus a `RadiometryReport`, and leaves skipped channels **exactly zero**.

### 7.4 The zero-channel rule

This is the load-bearing part, and it is the C-1 discipline applied to radiometry. An unavailable channel
is all zeros (the sensor adapter's convention). Such a channel has `std = 0`, so `max_value - min_value
== 0` and the stretch divides by zero. The module therefore **skips and leaves at exactly zero** any
channel with no dynamic range, and records a **different status** for each reason:

- `unavailable` — the channel is all zeros (`if not np.any(channel)`), i.e. the sensor did not measure it.
- `degenerate` — the channel is present but constant (`_channel_window` returns `None` because
  `upper - lower <= 0`), i.e. the sensor measured a flat band.
- `normalised` — the real case.

The docstring explains why the distinction matters: *"'the sensor did not measure this' is not the same
fact as 'the sensor measured a flat band', and the trace must not merge them."* The mechanism is the
degenerate-window test, which needs **no mask** — deliberately, because `CROMAEncoder.encode` has a
pinned signature of exactly `(self, optical, sar)` and finding C-1 says CROMA never receives a mask.
Inferring unavailability from the data keeps that guard intact.

A non-finite pixel (NaN/Inf) is *"a defective measurement, not a measurement of zero"*; it is pinned to
`0.0` rather than allowed to propagate a NaN into the frozen encoder, where *"one NaN would poison the
whole forward pass."* The `finite_pixels` count in the report keeps the gap detectable.

`resolve_use_8_bit(config)` resolves the flag in a documented order — environment variable first, then
`croma.use_8_bit` when the key exists and is a bool, then `DEFAULT_USE_8_BIT` — and returns
`(value, source)` where `source` is `"env"`, `"config"` or `"default"`. The reason it is not simply a
config key is recorded: adding `croma.use_8_bit` to `configs/base.yaml` would move `Config.hash` away
from the frozen `78f1e3700da15aa1` and detach the Phase 9 benchmark. The hash-exempt channel is the same
pattern already used for `SATQUERY_DEVICE`.

`summarise(optical_report, sar_report)` bundles both modalities under one `use_8_bit` and **asserts** they
agree, raising `ValueError` if they do not: *"upstream applies one transform to both."*

### 7.5 The display stretches — a different transform

Two other percentile stretches exist in the codebase. Neither is the CROMA transform, and the DEV-2
ruling names reusing one of them as **forbidden**.

**`preprocessing/imagery.py::load_image_array`** produces a displayable `(H, W, 3)` uint8 array:

```python
arr = array.astype(np.float32)
finite = arr[np.isfinite(arr)]
if finite.size:
    lo, hi = np.percentile(finite, (2, 98))
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
arr = np.clip(arr, 0.0, 1.0)
return (arr * 255).astype(np.uint8)
```

Its docstring is explicit about its scope: it *"does not resample, crop, or reproject. Those change the
pixel grid, and the grounding specialist converts normalized boxes to pixel coordinates using the
ORIGINAL raster's dimensions — a silent resize here would put every box in the wrong place."* The
determinism claim is stated: *"the same file always yields the same array, so a grounding box and a VQA
answer describe identical pixels."*

`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §1.1 records the three disqualifying differences between
this stretch and CROMA's: it is **per-scene** (one `(lo, hi)` over all bands) rather than **per-channel**;
it returns **uint8 3-band** rather than float 12-channel; and its **purpose** is presentation, not
radiometry.

**`specialists/change/postprocess.py::to_grayscale_float`** applies a per-scene 2/98 percentile stretch
to a grayscale collapse, so that *"a uint16 raster and a uint8 raster of the same scene correlate
identically."* Its comment records a measured defect fixed in place: `np.percentile` returns float64
scalars even for float32 input, so the array is promoted; without the explicit
`.astype(np.float32, copy=False)` the function returns float64 and `cv2.phaseCorrelate` fails its own
type assertion against a `CV_32F` Hanning window — *"Measured, not theorised."*

### 7.6 The unimplemented stage, and its consequence

The percentile/dB conditioning stage is **not implemented**. The radiometry module's docstring states the
consequence plainly:

> **Consequence: the two `optical.*` and three `sar.*` config keys are still read by no code.** That is
> recorded, not fixed.

`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §6 item 3 adds: *"A config key that no code reads is worse
than a missing key: it reads as a satisfied requirement."* The ruling's follow-on table assigns the
implementation to the engineer, with the alternative of explicitly retiring the keys.

The status of the whole chain is therefore:

| Stage | Declared | Implemented | Read by code |
|---|---|---|---|
| Optical percentile 2/98 conditioning | ✅ `optical.*` | ❌ | ❌ |
| SAR dB clip −30/+5 conditioning | ✅ `sar.*` | ❌ | ❌ |
| Per-channel `mean ± 2·std` → [0,1] | ✅ (ruling §3.1) | ✅ `radiometry.py` | ✅ (via `training/fusion/extract.py`; see below) |

**Where the implemented stretch actually runs.** The serving-path pipeline
(`specialists/optical_sar/inference.py::run_pipeline`) does **not** call `normalise_for_croma`. It
canonicalises, resizes to 120×120 and calls `encoder.encode(...)`. The stretch is applied by the
**extraction** path (`training/fusion/extract.py`), which states: *"This pipeline applies the DEV-2
encoder-input stretch itself (`radiometry.normalise_for_croma`), because the arm's `use_8_bit` has to be
applied at exactly one place."* `extract.py` also refuses an encoder that would stretch a second time:
`_assert_raw_encoder` *"refuses it by name rather than trusting the caller."* So the implemented
radiometric transform is a **training-time** transform that feeds the cached features, and the serving
path's input scaling is whatever `CROMAEncoder.encode` does internally (see §8.9 and the CROMA module).

`UNKNOWN — not established from the available evidence` whether the serving path's encoder applies the
same `mean ± 2·std` stretch internally; `specialists/optical_sar/croma.py::CROMAEncoder.encode` was
verified to have the pinned signature `(self, optical, sar)` with no mask parameter, and the extraction
module's guard implies the encoder *can* apply a stretch when `normalize_input=True` (its default), but
the exact serving configuration of that flag is not established from the files read for this chapter.

---

## 8. The CROMA sensor adapter contract

`specialists/optical_sar/sensor_adapter.py` is *"the one place where 'some sensor's bands' becomes
'CROMA's canonical channels', and it exists so that translation is explicit, inspectable and testable."*
Its module docstring states **the two hard rules** from the architecture plan:

> "No invented missing bands."
> "Do not fabricate spectral bands."

The failure mode the module prevents is described with unusual clarity:

> A 4-band Cartosat-2S scene maps cleanly onto canonical channels 1-4. Filling channels 5-12 with
> *something* — a copy of B3 as a fake "red-edge", a zero-order hold, an interpolation — produces a tensor
> whose shape is correct and whose statistics look plausible. CROMA would run. Nothing would raise. And
> every downstream number would be computed from channels that no sensor ever measured, presented as if
> they were measurements.

So: **missing channels are zero-filled, and the availability mask says which ones were real.**

### 8.1 `SensorDescriptor`

The schema object (`core/schemas.py:138`) is `extra="forbid"`, described as the *"Frozen sensor-adapter
contract (plan section 18)"*:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `sensor` | `str` | required | Sensor identifier, e.g. `"cartosat_2s"` |
| `available_bands` | `list[str]` | `[]` | The bands the sensor actually delivers |
| `band_map` | `dict[str, str]` | `{}` | Source band name → canonical band name |
| `normalization` | `str` | `"percentile"` | Identifier recorded for the trace |
| `availability_mask` | `list[bool]` | `[]` | Which canonical channels are real |
| `resolution` | `float \| None` | `None` | Ground sample distance in metres |
| `canonical_channels` | `int` | `0` | Target channel count (12 optical, 2 SAR) |
| `missing_channels_zero_filled` | `bool` | `True` | The zero-fill convention |

**Why `band_map` is canonical-channel-keyed.** The module docstring addresses a direction question: the
plan's example writes `band_map` sensor-keyed (`{"B1": "c1", ...}`), and the schema types it as
`dict[str, str]`. This module uses the schema's direction because the question asked at load time is
forward-looking: *"canonical channel 3 — did I get a real band for it, and if so which one?"*
`canonical_to_sensor` is exposed as the inverse for reporting.

### 8.2 The canonical channel orders

```python
OPTICAL_CANONICAL: tuple[str, ...] = (
    "B01",  # coastal aerosol
    "B02",  # blue
    "B03",  # green
    "B04",  # red
    "B05",  # red edge 1
    "B06",  # red edge 2
    "B07",  # red edge 3
    "B08",  # NIR
    "B8A",  # narrow NIR
    "B09",  # water vapour
    "B11",  # SWIR 1
    "B12",  # SWIR 2
)

SAR_CANONICAL: tuple[str, ...] = ("VV", "VH")
```

The comment above `OPTICAL_CANONICAL` states the contract: *"The ORDER is part of the contract: channel
index i must always mean the same physical band, or a pretrained encoder's weights would be applied to
the wrong input."* This is Sentinel-2 L2A with the cirrus band removed, per CROMA's README: *"Sentinel-2
must be 12 channels (remove the cirrus band if necessary)."* Note that `B10` (cirrus) is **not** in the
canonical order — that is the removal.

`SAR_CANONICAL` is described as *"Sentinel-1 dual-pol order CROMA was pretrained with. Positional fallback
ONLY."*

`_BAND_ALIASES` maps common spellings onto the canonical form: `B1`→`B01`, `BLUE`→`B02`, `NIR`→`B08`,
`SWIR1`→`B11`, `VVPOL`→`VV`, and so on. The comment is a caution: *"Kept small and explicit: a guess here
becomes a silently mislabelled band."* `_canonicalise` strips whitespace and underscores and uppercases
before lookup, and **never invents** a band — an unmapped name passes through unchanged and is later
rejected.

### 8.3 Building an optical adapter

`build_optical_adapter(sensor, *, available_bands, canonical_channels=12, normalization="percentile",
resolution=None)` (`specialists/optical_sar/sensor_adapter.py:205`):

- When `available_bands` is `None`, the sensor is looked up in `_KNOWN_OPTICAL_SENSORS`; an unknown
  sensor with no explicit band list raises `UnsupportedBandsError` — *"refusing to guess a band
  arrangement."*
- Each band is canonicalised. Any canonical name **not** in `OPTICAL_CANONICAL` raises
  `UnsupportedBandsError`: *"do not map onto the canonical Sentinel-2 order ... supply an explicit
  band_map rather than guessing a position."*
- Duplicate bands raise.
- More bands than `canonical_channels` raises.
- `band_map` is built as `{source: canonical for source, canonical in zip(bands, canonical)}` — keyed by
  the **source** name, because *"that is what an array row can actually be looked up by."*
- `availability` is `[canonical_name in band_map.values() for canonical_name in OPTICAL_CANONICAL[:canonical_channels]]`.

The returned descriptor has `missing_channels_zero_filled=True`.

`_KNOWN_OPTICAL_SENSORS` is *"deliberately tiny and explicit. Adding one is a code change, not a guess at
runtime"*:

| Sensor | Bands |
|---|---|
| `sentinel_2` | All 12 canonical |
| `cartosat_2s` | `["B1", "B2", "B3", "B4"]` |
| `cartosat_2` | `["B1", "B2", "B3", "B4"]` |
| `landsat_8` | `["B1", "B2", "B3", "B4"]` |

The comment on `cartosat_2s` is a standing instruction: *"Cartosat-2S is a 4-band VNIR imager. It is NOT a
Sentinel-2 clone and must not be treated as one: the 8 channels it lacks stay masked-off."*

### 8.4 Describing a SAR sensor — inspect, do not assume

`describe_sar(sensor, *, available_bands, canonical_channels=2, normalization="db", resolution=None)`
(`specialists/optical_sar/sensor_adapter.py:318`) implements the plan's instruction:

> "RISAT imagery may vary in acquisition/polarization characteristics... the SAR adapter must inspect
> actual available channels rather than assuming one fixed polarization pair."

The module docstring explains why a hardcoded `["VV", "VH"]` would be wrong: *"ISRO describes RISAT-1 as
a C-band SAR mission with multiple polarization configurations (single HH, single VV, dual HH+HV, dual
VV+VH, ...). A hardcoded `["VV", "VH"]` would silently mislabel an HH/HV scene, and the resulting tensor
would be a *confidently wrong* input rather than a recognisably broken one."*

When `available_bands` names the polarisations, they are honoured. When it does not, the **channel count**
is used to select the most likely convention, and **the choice is recorded on the descriptor's
`available_bands`** so the assumption appears in the trace rather than being invisible.

The **slot order is the sensor's own**:

```python
band_map: dict[str, str] = {source: source for source in bands}
availability = [i < len(bands) for i in range(canonical_channels)]
```

The comment explains: *"Forcing HH/HV into a VV/VH frame would mean either relabelling the polarisations
(a lie the mask would then contradict) or masking both off and discarding real data."* So the tensor holds
the radar measurements that exist, and `available_bands[i]` says which polarisation slot `i` is.

`_known_sar(sensor)` is described as *"Best-effort polarisation layout for a named SAR sensor, or `[]`"*:

| Sensor key | Returns |
|---|---|
| `sentinel_1`, `sentinel1` | `["VV", "VH"]` |
| `risat*` | `["VV", "VH"]` — **recorded as an ASSUMPTION, not a fact** |
| `alos_palsar`, `palsar`, `alos2` | `["HH", "HV"]` |
| anything else | `[]` — *"An empty return is honest: it says 'I do not know this sensor's polarisations'"* |

The `risat` case carries an explicit comment: *"RISAT-1's most common operational mode is dual-pol.
Recorded as an ASSUMPTION, not a fact — see the module docstring. Callers with real band metadata should
pass `available_bands` and override this."* This is an `ASSUMPTION` in the project's status vocabulary,
and it is the reason the hidden-set behaviour on RISAT cannot be claimed as measured.

### 8.5 Applying the canonical layout — `_apply_canonical`

This is *"the function the two hard rules live in"* (`specialists/optical_sar/sensor_adapter.py:428`). It
places real bands in canonical positions and zero-fills the rest:

```python
canonical = np.zeros((n_canonical, height, width), dtype=np.float32)
mask = np.zeros((n_canonical,), dtype=bool)

for source_index, source_name in enumerate(descriptor.available_bands):
    if source_index >= n_source:
        continue
    canonical_name = descriptor.band_map.get(source_name)
    if canonical_name is None or canonical_name not in order:
        continue
    channel_index = order.index(canonical_name)
    if channel_index >= n_canonical:
        continue
    canonical[channel_index] = array[source_index].astype(np.float32)
    mask[channel_index] = True
```

The docstring's key claim: *"There is no branch anywhere below that writes a value into an unavailable
channel."* The absence of such a branch **is** the mask. If no band could be placed while the array
carries bands, it raises `UnsupportedBandsError` rather than returning an all-zero tensor silently.

`_canonical_order(descriptor)` decides the slot order: for a descriptor whose `available_bands` are all
SAR polarisations, the order is the descriptor's own list (padded to canonical width); otherwise it is
`OPTICAL_CANONICAL`.

`adapt_optical` and `adapt_sar` are thin wrappers that call `_apply_canonical` with the right modality
label. `adapt_sar`'s docstring names the plan's internal representation exactly:
`canonical_sar[2, H, W]` + `sar_channel_mask[2]`.

### 8.6 `SensorAdapterOutput`

The dataclass returned by `adapt_*` carries `canonical` `(C, H, W)` float32, `mask` `(C,)` bool, and the
`descriptor`. It exposes `n_channels`, `missing_indices` (`[i for i, ok in enumerate(self.mask) if not
ok]`), `available_bands`, and a `to_dict()` that includes `n_available` and `missing_channels`.

### 8.7 `positional_fallback_descriptor`

Used when a raster has the right **channel count** for a modality but **no band labels**. The fallback
order is chosen from `_FALLBACK_SAR_ORDER` (`("VV", "VH")`, `("HH", "HV")`) for 1–2 channels, or the
canonical optical prefix for ≥3 channels. The fact that it was a fallback is **recorded in the `sensor`
field** so it *"cannot be mistaken for measured metadata."* The `complement=True` flag selects the other
convention, which lets a caller test both HH/HV and VV/VH orderings without asserting which is correct.

The optical-SAR specialist uses this in `_descriptor_for`
(`specialists/optical_sar/specialist.py:398`) and appends a warning that names the assumption: *"The
polarisation IDENTITY of each channel is NOT known from a band count alone — supply a sensor descriptor
to state it."*

### 8.8 Finding C-1 — the mask goes to the fusion head, not to CROMA

This is the single most important fact in this section. `core/schemas.py`'s module docstring lists it
among the findings baked into the schemas: *"C-1 channel-availability mask is a first-class fusion input,
never a CROMA input."*

`specialists/optical_sar/fusion_head.py` devotes a section to the rationale:

> CROMA is a masked autoencoder. Handing it an availability mask invites it to reconstruct the missing
> channels — which is precisely the fabrication the sensor adapter exists to prevent: the model would
> output plausible values for bands no sensor measured, and those values would then be treated as data.
> The mask is therefore consumed HERE, where it can only do one thing: tell the classifier which inputs
> to distrust.

Mechanically: `CROMAEncoder.encode` has a pinned signature of exactly `(self, optical, sar)` — the
radiometry module records that `tests/unit/test_optical_sar_croma.py` *"asserts the absence of any mask
parameter, because freeze finding C-1 says CROMA never receives one."* The mask instead enters at
`assemble_fusion_input` (§8.9). The optical-SAR specialist's `JOINT_FEATURE_REGION` evidence item records
the fact explicitly in its payload: `"mask_consumed_by": "fusion_head"` and `"croma_received_mask":
False` (`specialists/optical_sar/specialist.py:816`).

`EvidenceType` even has a dedicated member for the mask as trust evidence:
`AVAILABILITY_MASK = "availability_mask"   # C-1: modality trust evidence` (`core/schemas.py:76`). The
optical-SAR specialist emits one `AVAILABILITY_MASK` item per modality, carrying the sensor, canonical
channel count, available bands, band map, the mask itself, the missing indices,
`missing_channels_zero_filled`, `normalization` and `resolution`.

### 8.9 The frozen fusion concatenation

`specialists/optical_sar/fusion_head.py` defines the concatenation that the mask feeds into:

```
optical_GAP      (B, 768)
SAR_GAP          (B, 768)
joint_GAP        (B, 768)
optical_mask     (B,  12)     <- availability, from the sensor adapter
sar_mask         (B,   2)     <- availability, from the sensor adapter
                 ---------
concat           (B, 2318)
```

`expected_fusion_dim(encoder_dim=768, optical_channels=12, sar_channels=2, modalities_used=3)` returns
`3*768 + 12 + 2 = 2318`. The module recomputes the width and **refuses to build on a mismatch**, so *"a
config edit cannot silently reshape the first Linear layer into something that trains but means
nothing."* `core/config.py` recomputes it independently at load time from `croma.encoder_dim`,
`croma.modalities_used`, `croma.optical_channels` and `croma.sar_channels`, and rejects a config whose
`fusion.input_dim` disagrees (finding C-1 in the validator's own message).

`assemble_fusion_input` asserts the order rather than assuming it — *"a permutation here produces a tensor
of exactly the right shape that trains to a worse number — the hardest kind of bug to notice"* — and
validates that the three GAP vectors agree on batch size and that both masks agree with it.

`channel_dropout(features, mask, *, keep_probabilities, rng)` is the mechanism that *"teaches the head to
trust the availability mask"*. Its load-bearing line is:

```python
keep = draws & (msk > 0.0)
out_mask = keep.astype(np.float32)
out = out * out_mask
```

Dropped channels are **zeroed, not renormalised** — *"renormalising would fabricate a scale that the real
missing-channel case does not have"* — and the mask is updated in lockstep. `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md`
§5 independently verified this against freeze §2.5 and ruled: *"the engineer's implementation SATISFIES
§2.5's intent. No change entry required."* Its table records that the `& (msk > 0.0)` guard is the
load-bearing one: *"no training transform can resurrect a band the sensor did not measure."*

The schedule (plan section 20) is optical at `100/80/60/40%` and SAR at `100/50%`, declared in
`training/fusion/train.py` as `OPTICAL_DROPOUT_RATES = (1.0, 0.8, 0.6, 0.4)` and
`SAR_DROPOUT_RATES = (1.0, 0.5)`.

`mask_availability_stats(optical_mask, sar_mask)` returns `optical_available_fraction`,
`sar_available_fraction`, `optical_channels_present` and `sar_channels_present`. These feed the
confidence system: the two modality-confidence terms in the optical-SAR specialist are exactly the
availability fractions, which the specialist's docstring justifies: *"On the hidden set the optical side
is Cartosat-2S with 4 bands against a canonical 12, so 8 of 12 channels are always absent. A classifier
resting on 4 measured channels is not the same claim as one resting on 12."*

### 8.10 The optical-SAR confidence components

`OpticalSarSpecialist._confidence_components` (`specialists/optical_sar/specialist.py:481`) implements
plan section 26's four components with these weights:

```python
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "fusion_margin": 0.40,
    "optical_confidence": 0.20,
    "sar_confidence": 0.20,
    "cross_modal_agreement": 0.20,
}
```

`cross_modal_agreement` is `cosine(optical_GAP, SAR_GAP)`, rescaled from `[-1, 1]` to `[0, 1]` via
`AGREEMENT_FLOOR = -1.0`. `cross_modal_agreement` in `inference.py` returns `None` on any exception or a
shape mismatch, and the specialist treats `None` as a zero component. The composition is a weighted sum,
**floored to 0.0** when there is no prediction or no trained head — *"Both are signal GAPS, not weak
signals, so they resolve to 0.0 rather than merely discounting."* The components dict records
`has_croma`, `trained_head`, `no_prediction` and the raw availability counts, so *"the trace states every
reason the score is zero, not just the first one encountered."*

---

## 9. Pair-compatibility rules

Two workflows take a **pair** of assets, and both enforce compatibility before reading pixels.

### 9.1 Change — exactly two, temporally distinct, co-registered

`ChangeSpecialist.validate_request` (`specialists/change/specialist.py:190`) enforces **exactly two**
assets. The comment names both common mistakes: *"ONE asset is the common mistake: the caller treated
this as a single-image task. THREE is the other: they attached a time series. Both are rejected with the
same typed error."* The user-facing message is *"Change detection needs exactly two images: an earlier one
and a later one."*

`_assess_pair` then runs three checks in order:

1. **Temporal distinctness.** Identical `path` raises `TemporalPairError`. Identical `sha256` (when both
   carry one) also raises: *"a repeated acquisition is not a temporal pair."*
2. **CRS compatibility** via `compare_crs` (§2.5). Incompatible or reprojection-requiring pairs get a
   warning, not a refusal — pixel-domain change detection can still proceed, but geographic claims are
   withheld.
3. **Registration quality** via `measure_registration` (§9.3).

If registration is **not usable**, the specialist **suppresses spatial claims** while still computing and
returning the change maps:

> The maps are still returned (the caller may want to look at them), but the REGION CLAIMS are withheld:
> a region is an assertion about *where* something changed on the ground, and we cannot make it.

`ChangeSpecialist._geospatial` drops the transform in that case, returning a `GeoMetadata` with
`is_georeferenced=False` — *"a transform invites the caller to place a region on a map, and we have just
said the regions are not placeable."* `_write_artifacts` writes the map **without** georeferencing when
suppressed, and the comment explains why copying T1's transform would be wrong: *"Copying T1's transform
onto it would produce a file that unlocks exactly the placed-on-a-map reading the suppression exists to
forbid."* The confidence is driven to the floor via a **minimum** against the registration factor, so
alignment is a gate, not one vote among three.

The change confidence weights are:

```python
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "registration_quality": 0.50,
    "mean_change_probability": 0.30,
    "component_stability": 0.20,
}
```

with `STABILITY_HEADROOM = 0.5`, `DEFAULT_MAX_SHIFT_PX = 8`, `MIN_REGION_PIXELS = 32`.

### 9.2 Optical-SAR — exactly two, one of each modality

`OpticalSarSpecialist.validate_request` (`specialists/optical_sar/specialist.py:192`) enforces **exactly
two** assets **and** that they are one optical and one SAR. The docstring explains the case worth
explaining:

> Two optical images is not a malformed request in the way that three is — it is a plausible mistake,
> from an operator who uploaded a bi-temporal pair to a single-modality workflow, or who labelled a
> four-band Cartosat scene as SAR.
>
> If such a request were accepted, the adapter would place the second optical image into the 2-channel SAR
> slot, zero-fill, and produce a fused tensor of the correct shape. CROMA would run. The answer would be
> about one modality while claiming to fuse two.

So the modality check is done on the **assets, before any pixels are read**. `_assess_pair` raises
`InvalidRequestError` with a machine-readable `reason` of `"two_optical"`, `"two_sar"` or
`"indeterminate"`, and each carries a user-facing message.

### 9.3 The equal-dimensions requirement, and the 726²-vs-736² error

The change detector requires T1 and T2 to have **identical shapes**. The guard is in the model itself,
`specialists/change/stanet.py:415`:

```python
if t1.shape != t2.shape:
    raise SpecialistError(
        f"T1 and T2 must have the same shape; got {tuple(t1.shape)} "
        f"and {tuple(t2.shape)}",
        specialist="change",
    )
```

This is a **hard** requirement, and it is where the bundled demo pair fails. `docs/FINAL_DELIVERY_REPORT.md`
§6 records it as `DEGRADED`:

| Finding | Status | Detail |
|---|---|---|
| Change on the bundled EO pair | DEGRADED | `delta-growth-t0-1975.jpg` (726²) and `t1-2025.jpg` (736²) differ in shape; the change specialist errors (`T1 and T2 must have the same shape`). A same-shape pair, or a resize step, is needed for a clean change demo. |

There is **no resize step on the change path**. `load_image_array` explicitly does not resample
(§7.5) — its docstring states *"a silent resize here would put every box in the wrong place"* — and the
change specialist's `_predict` only pads for the encoder's stride-8 requirement, cropping back to the
original size:

```python
h, w = t1.shape[-2:]
if h % 8 or w % 8:
    ph, pw = (-h) % 8, (-w) % 8
    t1 = torch.nn.functional.pad(t1, (0, pw, 0, ph), mode="reflect")
    t2 = torch.nn.functional.pad(t2, (0, pw, 0, ph), mode="reflect")
```

That padding does **not** reconcile two different input sizes: T1 and T2 are each padded by their own
`(-dim) % 8`, so a 726² and a 736² input become 728² and 736² respectively — still different — and the
model raises. The reflect-pad is a *stride* accommodation, not a *pair* accommodation, and the two must
not be conflated.

**Consequence for operators.** The change workflow requires a same-shape pair, and the failure is
`IMPLEMENTED` (it raises a typed error with a readable message) rather than silent. A resize or
co-registration step that reconciles differing dimensions is **not implemented** (§13). The
`FINAL_DELIVERY_REPORT` records the demo as `DEGRADED` and prescribes *"a same-shape pair, or a resize
step"* — either would fix the demo; neither is currently in the code path.

### 9.4 Registration measurement

`measure_registration` (`specialists/change/postprocess.py:134`) uses `cv2.phaseCorrelate` on grayscale
collapses of the two images, returning a `RegistrationQuality` with `shift_x`, `shift_y`, `response`,
`max_shift_px`, `is_usable` and `reason`. Defaults: `max_shift_px=8`, `min_response=0.15`.

The module docstring is candid about the method's limit:

> Deliberately a TRANSLATION model. Real mis-registration includes rotation and warp, which phase
> correlation does not recover; a small residual after a translation correction is therefore not proof of
> good alignment. What the measurement can do honestly is flag the *bad* cases, and that is how it is
> used: as a gate, not a certificate.

`RegistrationQuality.confidence_factor()` applies two independent penalties and takes the worse:
a shift penalty (`1.0` at zero shift, `0.0` at `max_shift_px`) and a response factor (`0.0` at zero
response, `1.0` at response ≥ 0.15). A pair that is offset *and* uncorrelated takes the worse of the two —
*"the conservative choice."*

A failure to *measure* is treated as a measurement of failure — both mean the pair cannot be trusted
spatially — but the distinction is recorded in the warning: *"A failure to MEASURE is not the same as a
measurement of failure."*

Two registration-gate hypotheses were tested and both are recorded as **REJECTED** or
**INSUFFICIENT**: `docs/CHANGE_REGISTRATION_GATE_COHERENCE_TEST.md` records a rejected hypothesis (AUC S2
vs S3 ≤ 0.59), and `docs/CHANGE_REGISTRATION_GATE_TEXTURE_TEST.md` records a low-texture signal that was
**CONFIRMED but INSUFFICIENT**, with option 3 **ELIMINATED**. So the gate as implemented (phase
correlation) is the surviving method, and no additional coherence- or texture-based gate was adopted.

---

## 10. Large-image tiling policy

### 10.1 What is declared

`configs/base.yaml` declares the full tiling policy:

```yaml
image:
  max_pixels: 25000000
  tile_size: 512
  tile_overlap: 128
  max_tiles: 64
  # plan section 9.1 tile policy: whole-image thumbnail first, then top-K tiles.
  # max_tiles is the hard ceiling on tiles *examined*; top_k_tiles is how many
  # are actually sent through a specialist.
  top_k_tiles: 4
```

| Key | Value | Declared meaning |
|---|---|---|
| `max_pixels` | 25,000,000 | Pixel budget; exceeding it is recoverable (the caller may downscale) |
| `tile_size` | 512 | Tile edge in pixels |
| `tile_overlap` | 128 | Overlap between adjacent tiles |
| `max_tiles` | 64 | Hard ceiling on tiles **examined** |
| `top_k_tiles` | 4 | How many tiles are actually sent through a specialist |

The comment states the intended policy: *"whole-image thumbnail first, then top-K tiles."* The change
workflow declares a different tiling resolution, `change.tile_size: 256` with `change.tile_overlap: 0`
— 256 is *"the change model's OWN training resolution"* (LEVIR-CD patches are 256×256).

### 10.2 What is enforced

`core/config.py` validates the tiling policy at load time, but only one relationship:

```python
# --- tiling policy -------------------------------------------------
if self.get("image.top_k_tiles", 0) > self.get("image.max_tiles", 0):
    errors.append("image.top_k_tiles cannot exceed image.max_tiles")
```

So the config loader guarantees that the number of tiles sent through a specialist does not exceed the
number examined. It does not implement tiling.

The pixel budget **is** enforced, but at the raster-inspection layer rather than as a tiling step:
`inspect_raster` raises `OversizedImageError` when `width * height > max_pixels`, and the error is
`recoverable` — the docstring says *"the caller may downscale."* The controller passes the configured
budget into `_inspect_asset` during VALIDATE.

### 10.3 What is not implemented — the honest status

**The tiling policy is `DECLARED (not read)`.** `image.tile_size`, `image.tile_overlap`, `image.max_tiles`
and `image.top_k_tiles` appear only in `configs/base.yaml` and in the `core/config.py` validation above.
No code on the serving path slices a raster into tiles, selects the top-K, or stitches results.
`architecture/03-request-lifecycle.md` reaches the same conclusion: the tiling policy is referenced only
in config validation and not implemented on the serving path.

The practical consequence: a raster within the `max_pixels` budget is handed to a specialist **whole**.
The VLM path pins `processor_longest_edge: 512` so the processor does not upscale and split a tile, but
that pin assumes a 512-px input — it is a *control against* the processor's default 2048, not a tiling
step. `core/config.py` enforces the relationship (`processor_longest_edge` must not exceed
`image.tile_size`), with the measured rationale recorded: the default 2048 upscales a 512 tile 4× and then
splits it into ~17 sub-images, *"the real figure is ~17x"* against the plan's estimated 4×.

`UNKNOWN — not established from the available evidence` how a raster between the tile size and the pixel
budget is handled by the VLM path in practice; the config implies the tile policy would govern it, but
the policy is not implemented, so the behaviour is whatever the specialist does with the whole image.

---

## 11. Grounding resolution — the frozen 224 decision

Grounding runs at **224 px**, and this is one of the project's cleanest pre-registered results.
`configs/base.yaml`:

```yaml
grounding:
  ...
  # RESOLVED 2026-09-16. 448 did NOT earn its cost over all 16,159 VRSBench
  # eval records: mean best IoU -0.0147, Recall@0.5 -0.0022, and every recall
  # threshold lower (-0.0699 @0.10, -0.0243 @0.25), at 1.59x the latency.
  # Paired over identical samples: mean diff -0.0147, 95% CI [-0.0160,-0.0134],
  # t = -22.63. 448 was better on 8.5% of records, worse on 20.9%.
  # Pre-registered rule and the paired test AGREE on 224.
  # Evidence: docs/PHASE7_RESOLUTION_DECISION.md
  image_size: 224
  resolution_frozen: true
  nms_iou: 0.50
  max_candidates: 20
  confidence_threshold: 0.40
  benchmark_box_scale: 100.0
  coordinate_system: normalized_0_1
  encoder_projected_dim: 512
```

### 11.1 The pre-registered rule

`docs/PHASE7_RESOLUTION_DECISION.md` records the rule, **fixed before the result was seen**:

```
448 WINS  if Recall@0.5 improves by >= 0.05 absolute
          OR mean best IoU improves by >= 0.05 absolute
224 WINS  otherwise
INCONCLUSIVE if fewer than 30 samples were scored
```

The artifact records `rule_changed_since_preregistration: false`.

### 11.2 The result

```
224 WINS
  recall@0.5 gain 448/224 : -0.0022
  bestIoU    gain 448/224 : -0.0147
  latency          ratio  : 1.59x
```

Neither component came close to the +0.05 margin; both were **negative**. The measured detail:

| Metric | 224 | 448 | Delta |
|---|---|---|---|
| Token grid | 7 × 7 = 49 | 14 × 14 = 196 | 4.0× tokens |
| Mean best IoU | **0.0972** | 0.0825 | **−0.0147** |
| Recall@0.10 | **0.3298** | 0.2599 | **−0.0699** |
| Recall@0.25 | **0.1187** | 0.0944 | **−0.0243** |
| Recall@0.50 | **0.0234** | 0.0212 | **−0.0022** |
| Latency mean | **20.0 ms** | 31.8 ms | 1.59× |
| Latency p90 | **20.9 ms** | 32.9 ms | 1.57× |
| Peak VRAM | **592.1 MB** | 599.8 MB | +7.7 MB |

All 16,159 / 16,159 VRSBench eval records were scored at both resolutions on a Tesla T4.

### 11.3 The paired test

Because both resolutions scored the **same 16,159 samples**, the paired test is the stronger statistic:

```
paired samples     : 16159
mean 224           : 0.0972
mean 448           : 0.0825
mean paired diff   : -0.0147   (95% CI -0.0160 .. -0.0134)
t statistic        : -22.63
CI excludes zero   : True

448 better on      :  1371/16159 ( 8.5%)
448 worse on       :  3372/16159 (20.9%)
identical          : 11416/16159 (70.6%)
```

The paired test and the pre-registered rule **agree**, so there is *"no rule-versus-evidence disagreement
to escalate."* The win/loss split is informative on its own: 448 wins on only 8.5% and loses on 20.9% —
*"the finer grid is not merely neutral, it is actively harmful on a fifth of the corpus."*

The recall ladder narrows as the threshold rises (−0.0699 at 0.10, −0.0243 at 0.25, −0.0022 at 0.50),
which the document reads as *"the signature of a method that cannot reach high IoU either way."*

### 11.4 Why 448 did not help

The document's honest reading: the zero-shot method selects a patch by text similarity and returns that
patch's box. At 224 a box is 1/7 of the image; at 448 it is 1/14. A finer grid is only better if the
target is small **and** the similarity peak lands on the correct fine cell. Two things work against that:
the peak is not sharper at 448 (the similarity field on frozen features is smooth, so the argmax moves
around), and Recall@0.10 — the loosest threshold — degrades *most*, which means *"the fine grid is adding
positional noise rather than positional precision."*

The document is careful to attribute this to the zero-shot baseline, not to RemoteCLIP: *"A **learned**
head trained to regress boxes from these features may respond differently to resolution."*

### 11.5 What the decision does not establish

- **Whether the zero-shot baseline is good.** It is not: mean best IoU 0.0972 and Recall@0.5 0.0234 are
  *"weak"*. It is an ablation floor for the Phase 8 head.
- **Whether a trained head has the same resolution sensitivity.** Re-opening the question is legitimate
  **if** the head's validation curve suggests it, and would be *"a new pre-registered experiment, not a
  silent retune."*
- **Anything about hidden ISRO/SAC imagery.** *"VRSBench is overhead optical. The hidden set is
  Cartosat-2S + RISAT, a different distribution entirely."*

### 11.6 Downstream consequences of the frozen resolution

The frozen resolution propagates into concrete serving parameters:

- **The decode grid.** `build_grounding_specialist` sets `grid = resolution // 32`, so at 224 the grid is
  **7×7 = 49 cells**. The head emits `(1, 49, 5)` — `[tx, ty, tw, th, objectness]` per cell.
- **The projected dimension.** `grounding.encoder_projected_dim: 512` is the measured `visual.proj`
  output of RemoteCLIP ViT-B/32 (the transformer width is 768; `visual.proj` maps to 512). The
  per-cell feature is `concat([patch, text, patch*text, global_pool]) = 4 × 512 = 2048`, which
  `grounding_head.feature_dim: 2048` must equal — enforced at load time by `core/config.py`, which
  rejects any other value, because *"a mismatch here is a SILENT shape error"* that torch raises only at
  the similarity step, *"after the patch features are already cached."*
- **NMS.** `grounding.nms_iou: 0.50` is passed to `nms(boxes, scores, iou_threshold)`
  (`specialists/grounding/head.py:294`).
- **The serving candidate budget.** `GroundingSpecialist` defaults `max_candidates=6`, **not** the
  config's `20`. The class docstring records why: *"The Phase 8 decision measured that a 20-box budget
  inflates mean best IoU relative to the 5.99-box baseline; 6 is the decode-matched setting."* The builder
  reads `grounding.serving_max_candidates` with a default of 6.
- **The degenerate-box guard.** `DEGENERATE_AREA_FRACTION = 0.9` drops any zero-shot candidate covering
  more than 90% of the frame, *"the shape a failed localisation takes once it has been smoothed into a
  rectangle."* This guard applies to the **zero-shot fallback path only** — *"the trained-head decode is
  a regressed box with its own learned prior; filtering it would change the frozen Phase 8 benchmark
  numbers."*
- **The decode function.** `decode_cell_relative` (`specialists/grounding/head.py:204`) turns
  `(B, N, 4)` cell-relative parameters into `(B, N, 4)` normalised xyxy, clamped to `[0, 1]` and
  order-corrected by `enforce_order` (which swaps inverted corners rather than letting IoU go silently to
  zero). The decode is *"differentiable: the output feeds the box and GIoU losses directly, so the loss is
  computed on exactly the coordinates the metric measures."*

---

## 12. Worked examples

### 12.1 A 4-band Cartosat-2S optical scene

1. **Inspect.** `inspect_raster` reads `band_count=4`, no CRS or with CRS. `infer_modality(4)` returns
   `OPTICAL` (4 is in `_OPTICAL_BAND_COUNTS`).
2. **Descriptor.** With no `SensorDescriptor`, `_descriptor_for` maps the 4 bands positionally onto
   `OPTICAL_CANONICAL[:4]` = `B01, B02, B03, B04` and appends a warning naming the assumption. With a
   descriptor built by `build_optical_adapter("cartosat_2s")`, the bands are `["B1","B2","B3","B4"]`
   canonicalised to `B01..B04`, and `availability = [True, True, True, True, False × 8]`.
3. **Canonicalise.** `adapt_optical` returns a `(12, H, W)` tensor with channels 0–3 carrying the real
   bands and channels 4–11 exactly zero, plus a 12-element mask with four `True`.
4. **Resize.** `resize_to_canonical` bilinearly resizes to `120×120`. The zero channels stay zero.
5. **Encode.** `encoder.encode(optical_tensor[None], sar_tensor[None])` — **no mask** (C-1).
6. **Assemble.** `assemble_fusion_input` concatenates the three 768-d GAPs with the 12-element optical
   mask and the 2-element SAR mask → `(1, 2318)`.
7. **Head.** With a trained head, the classifier emits 19 logits → softmax → a label and a margin. With
   no head, `probabilities` stays `None`, `degraded=True`, and no label is invented.
8. **Evidence.** An `AVAILABILITY_MASK` item per modality records the sensor, the 4 real bands, the band
   map, the mask (`[true × 4, false × 8]`), and `missing_channels: [4,5,6,7,8,9,10,11]`. A
   `JOINT_FEATURE_REGION` item records `fusion_input_dim: 2318`, `mask_consumed_by: "fusion_head"`,
   `croma_received_mask: false`.
9. **Confidence.** `optical_confidence = 4/12 = 0.3333`; the SAR side contributes its own fraction; the
   fusion margin (if a head exists) is the largest weight. If no head, the raw confidence is **0.0** with
   `no_prediction: 1.0` recorded.

### 12.2 A RISAT SAR scene with no band labels

1. **Descriptor.** `positional_fallback_descriptor(stem, 2)` selects `("VV", "VH")` from
   `_FALLBACK_SAR_ORDER` and records the fallback in the `sensor` field. The specialist appends: *"The
   polarisation IDENTITY of each channel is NOT known from a band count alone."*
2. **Canonicalise.** `adapt_sar` returns `(2, H, W)` with both slots filled, mask `[True, True]`.
3. **Trace.** `available_bands = ["VV", "VH"]` is the **assumption**, not a measurement. A caller with
   real polarisation labels passes `available_bands=["HH","HV"]`, in which case the descriptor's own order
   becomes the slot order and the mask reflects it — no relabelling, no discarded data.

### 12.3 A change pair with differing dimensions

1. **Validate.** Exactly two assets, distinct paths/SHA-256, CRS compared.
2. **Register.** `measure_registration` collapses both to grayscale float32 and runs `phaseCorrelate`.
3. **Shape check.** If the two images differ in size, `STANetStyleChangeDetector.forward` raises
   `SpecialistError("T1 and T2 must have the same shape; got ...")` — the 726²/736² case (§9.3). No resize
   step exists to reconcile them.
4. **If shapes match but registration is poor:** the maps are still computed, the regions are withheld,
   the transform is dropped from the `geospatial` block, and the confidence is floored at 0.0 with
   `suppressed_by_registration: 1.0` recorded.

---

## 13. What is NOT implemented — exhaustive

Each item below was checked against the code. Where a capability is absent, it is named rather than
implied; where the absence cannot be confirmed from the files read, it is marked `UNKNOWN`.

| Capability | Status | Evidence |
|---|---|---|
| **Reprojection pipeline** | **NOT IMPLEMENTED** | `geospatial/crs.py::compare_crs` *reports* `requires_reprojection`; no code performs a reprojection. `preprocessing/imagery.py` explicitly does not reproject. |
| **Cloud masking** | **NOT IMPLEMENTED** | No cloud, cirrus, QA or `SCL` handling anywhere in `preprocessing/` or the specialists. `UNKNOWN — not established from the available evidence` whether any dataset adapter applies a cloud mask. |
| **Atmospheric correction** | **NOT IMPLEMENTED** | No atmospheric, surface-reflectance or `L2A`-processing code. Rasters are consumed as delivered. |
| **Mosaicking** | **NOT IMPLEMENTED** | No mosaicking, seamline or composite code exists. |
| **Tiling / top-K tile selection** | **DECLARED (not read)** | `image.tile_size`, `image.tile_overlap`, `image.max_tiles`, `image.top_k_tiles` are read only by `core/config.py`'s `top_k_tiles <= max_tiles` check. No serving-path tiling. |
| **Optical percentile 2/98 conditioning** | **DECLARED (not read)** | `optical.normalization`, `optical.lower_percentile`, `optical.upper_percentile` are read by no code (`radiometry.py` docstring; PHASE14 §1). |
| **SAR dB clip −30/+5 conditioning** | **DECLARED (not read)** | `sar.representation`, `sar.clip_min_db`, `sar.clip_max_db` are read by no code. |
| **Resampling / cropping on the display path** | **DELIBERATELY ABSENT** | `preprocessing/imagery.py` docstring: *"It does not resample, crop, or reproject."* |
| **Resize to reconcile differing pair dimensions** | **NOT IMPLEMENTED** | The change path requires equal shapes and has no reconciling resize (§9.3). |
| **Nodata filling** | **NOT IMPLEMENTED** | `quality.py` refuses any non-finite input outright; its comment records: *"Filling nodata from the raster profile belongs in the tiling work."* |
| **Area computation for geographic CRS** | **NOT IMPLEMENTED** | `pixel_area_m2` returns `None` for a non-metric CRS rather than approximating. |
| **GeoJSON / WKT / KML output** | **NOT IMPLEMENTED** | Coordinates are emitted as lists; no geometry serialisation exists. |
| **Multi-band raster → RGB band selection policy** | **PARTIAL** | `load_image_array` takes the first three bands as RGB (or repeats a single band); there is no configurable band combination. |
| **Sensor adapter for sensors beyond the known list** | **REFUSED BY DESIGN** | An unknown optical sensor with no explicit band list raises `UnsupportedBandsError`; `_known_sar` returns `[]` for an unknown SAR sensor. |

**A note on the "not implemented" list.** None of these omissions is hidden. The two most consequential —
the tiling policy and the radiometric conditioning — are declared in the frozen config and read by no
code, and both the config comments and the DEV-2 ruling name that state explicitly rather than letting the
key read as a satisfied requirement.

---

## 14. What is NOT RUN / OPEN / BLOCKED for this topic

### NOT RUN

- **The optical-SAR forward pass at serving time with a trained head.** `CROMA_base.pt` is not present
  locally and `use_croma.py` must be vendored; the specialist runs sensor-only and produces no label.
- **A system-level geospatial accuracy benchmark.** No end-to-end benchmark exists; no system-level
  accuracy is claimed anywhere.
- **Reprojection of a genuinely mismatched-CRS pair.** The code path that would consume a reprojection is
  absent, so the case has never been exercised.
- **Change on a bundled pair with equal dimensions.** The only bundled demo pair is 726² vs 736² and
  errors; a same-shape pair has not been run as a demo.

### OPEN

- **The `optical.*` / `sar.*` conditioning stage.** `OPEN` — either it gets an implementation or the keys
  are retired (`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §6 item 3).
- **The arm-constructibility question.** The §4-registered arm B (*"percentile/dB only, no encoder-input
  stretch"*) is **not constructible** while stage 1 is unimplemented; the arm now named B corresponds to
  the registered arm C. The ruling records this as *"a human decision and deliberately not resolved."*
- **The optical-SAR ruling.** Accuracy **0.931** with macro-F1 **0.434161**; ruling **OPEN**. Never quote
  the accuracy without the macro-F1.
- **The grounding resolution question after a trained head.** Re-opening 224 is legitimate only as a new
  pre-registered experiment.
- **The change-VQA metric ruling.** `OPEN` and owner-gated.
- **Whether any call site builds a `Box` outside the two conversion functions** (§5.4) — `UNKNOWN`.

### BLOCKED

- **A clean change demo.** Blocked on a same-shape pair or a resize step.
- **The CROMA normalisation experiment (option c).** Blocked on arm B being non-constructible and on the
  Arm-B feature cache, which does not exist (`artifacts/optical_sar/fusion_features/` holds only the
  Arm-A cache).
- **The gated experiment's sample size.** Cannot be specified until a variance-estimation pass is run;
  the ruling deliberately declines to invent a number.

### REJECTED

- **448 px grounding resolution.** `REJECTED` by the pre-registered rule and the paired test.
- **Option (b) — match `base.yaml` alone, accept the shift.** `REJECTED` as a standalone path
  (`docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` §2.1).
- **Reusing `preprocessing/imagery.py`'s stretch for CROMA.** `REJECTED` as `forbidden` (§3.3 of the
  ruling).
- **Coherence- and texture-based registration gates.** One `REJECTED`, one `CONFIRMED but INSUFFICIENT`.

---

## 15. Where the evidence lives

**Source modules (the code this chapter describes):**

| Path | What it defines |
|---|---|
| `preprocessing/raster.py` | The validation chain, `inspect_raster`, `read_bands`, `write_raster`, `infer_modality`, `file_sha256`, `pixel_area_m2` |
| `preprocessing/imagery.py` | `load_image_array`, `load_pil_image`, the per-scene 2/98 display stretch |
| `preprocessing/quality.py` | The deterministic quality gate, `MIN_AUTOCORRELATION=0.10`, `NOISE_ENTROPY_BITS=7.8` |
| `geospatial/crs.py` | `parse_crs`, `describe_crs`, `compare_crs`, `is_metric`, `CRSCompatibility` |
| `geospatial/transform.py` | `PixelWindow`, `GeoBounds`, `to_normalized`/`to_pixel`/`to_geo`, `intersect`, `iou`, `benchmark_boxes_to_normalized`, `affine_from_metadata` |
| `core/schemas.py` | `CoordinateSystem`, `GeoMetadata`, `SensorDescriptor`, `AssetMetadata`, `Box`, `Region`, `ChangeRegion`, `Evidence._spatial_needs_crs` |
| `core/config.py` | Config loading, validation, `Config.hash`, `device_preference` |
| `specialists/optical_sar/sensor_adapter.py` | `OPTICAL_CANONICAL`, `SAR_CANONICAL`, `build_optical_adapter`, `describe_sar`, `_apply_canonical`, `positional_fallback_descriptor` |
| `specialists/optical_sar/radiometry.py` | `TRANSFORM_NAME`, `normalise_for_croma`, `resolve_use_8_bit`, the zero-channel rule |
| `specialists/optical_sar/fusion_head.py` | `expected_fusion_dim`, `assemble_fusion_input`, `channel_dropout`, `build_fusion_head` |
| `specialists/optical_sar/inference.py` | `run_pipeline`, `resize_to_canonical`, `cross_modal_agreement`, `build_head_from_config` |
| `specialists/optical_sar/specialist.py` | `_assess_pair`, `_descriptor_for`, `_confidence_components`, `_geospatial` |
| `specialists/change/postprocess.py` | `RegistrationQuality`, `measure_registration`, `to_grayscale_float`, `regions_to_schema` |
| `specialists/change/specialist.py` | `_assess_pair`, `_predict`, `_geospatial`, the suppression path |
| `specialists/change/stanet.py:415` | The equal-shape guard |
| `specialists/grounding/specialist.py` | `_to_geo_boxes`, the `NORMALIZED_0_1` box construction, `DEGENERATE_AREA_FRACTION` |
| `specialists/grounding/head.py` | `decode_cell_relative`, `enforce_order`, `nms` |
| `specialists/grounding/inference.py` | `decode_candidates_from_features`, `ground_phrase` |
| `evaluation/metrics/grounding.py` | `benchmark_to_normalized(box, scale=100.0)` |

**Configuration:**

- `configs/base.yaml` — `image.*` (tiling), `optical.*`, `sar.*`, `croma.*`, `fusion.*`, `grounding.*`,
  `grounding_head.*`, `change.*`.

**Documents:**

- `docs/PHASE14_CROMA_NORMALISATION_CHANGE.md` — the DEV-2 ruling, the two ordered stages, the Gate F arm
  amendment.
- `docs/CROMA_NORMALISATION_UPSTREAM_EVIDENCE.md` — what is and is not established about CROMA's
  pretraining distribution.
- `docs/PHASE7_RESOLUTION_DECISION.md` — the 224/448 decision in full.
- `docs/CHANGE_REGISTRATION_GATE_COHERENCE_TEST.md`, `docs/CHANGE_REGISTRATION_GATE_TEXTURE_TEST.md` — the
  rejected and insufficient registration gates.
- `docs/API_CONTRACT.md` §2.5 (asset allowlist), §3.2 (coordinate systems).
- `docs/FINAL_DELIVERY_REPORT.md` §6 — the 726²/736² change demo recorded as DEGRADED.
- `docs/PHASE10_CDVQA_DATA_STATUS.md` — the CDVQA/SECOND pair layout and temporal semantics.
- `docs/PHASE11_CROMA_STATUS.md` — the reproduced CROMA forward pass.

**Sibling release chapters:**

- [`DATA_PIPELINE.md`](DATA_PIPELINE.md) — the end-to-end data path, caches and artefacts.
- [`architecture/03-request-lifecycle.md`](architecture/03-request-lifecycle.md) — the nine controller
  states, including VALIDATE and PREPROCESS.
- [`architecture/05-specialists.md`](architecture/05-specialists.md) — the six specialists.
- [`architecture/07-configuration-freeze.md`](architecture/07-configuration-freeze.md) — the frozen hash
  and every config key.
- [`DATASETS.md`](DATASETS.md) — the corpora and their measured shapes.
