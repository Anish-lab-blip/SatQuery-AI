# 07 — Configuration Freeze

**Parent:** [Architecture hub](README.md) · **Status tags:** `IMPLEMENTED` · `VERIFIED` ·
`MEASURED` · `OPEN`

---

## 0. Why this chapter exists, and what it is really about

This is not a chapter about a YAML file. It is a chapter about a **single decision** and its
consequences:

> every tunable number in SatQuery AI lives in **one** registry file, that file is **validated**
> against the frozen architecture at load time, and it is **hashed** so that every measurement the
> project has ever published can be traced back to the exact configuration that produced it.

Those three properties — one place, validated, hashed — are load-bearing for the whole project, and
they are in tension with each other. "One place" makes the config a magnet for edits. "Validated"
means a bad edit fails loudly instead of quietly. "Hashed" means a *good* edit still breaks
something, because it detaches published numbers from the configuration they were measured under.

The consequence a reader must internalise before touching anything in `configs/base.yaml`:

> **Editing the registry MOVES the config hash. Moving the hash INVALIDATES every artifact keyed to
> it.** The frozen value is `78f1e3700da15aa1`. A value that "looks wrong" is therefore not casually
> fixed — the fix has a cost that is not visible at the point of editing.

Everything else in this chapter — the loader pipeline, the twelve validation rules, the finding ids,
the `deploy.yaml` manifest, the frozen-value table, the open-for-tuning list — is elaboration of that
sentence.

| Property | Mechanism | Where |
|---|---|---|
| One place | `configs/base.yaml` is the only file the loader reads | `core/config.py:25` (`DEFAULT_CONFIG`) |
| Validated | `Config.__init__` calls `_validate()`; a bad config raises before any model loads | `core/config.py:46-49`, `:94-222` |
| Hashed | `Config.hash` = first 16 hex chars of a sha256 over the whole registry | `core/config.py:76-80` |

**Grounding note.** Every rule, key, number and finding id in this chapter was read out of the
repository at `core/config.py`, `configs/base.yaml`, `configs/deploy.yaml`, the `docs/` records it
cites, and the test suite. Where a claim could not be established from a file, it is written as
`UNKNOWN — not established from the available evidence` rather than guessed.

---

# Part A — The registry, and the rule

## 1. The registry

The registry is one file:

```
configs/base.yaml          # 294 lines, the authoritative registry
```

`core/config.py` names it once, at module scope:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"
```

(`core/config.py:24-25`)

That is the **only** path the loader resolves by default. `load_config()` with no argument reads
exactly this file, and `get_config()` — the singleton most of the codebase imports — reads exactly
this file. The loader **never globs** `configs/*.yaml`. This is not incidental; it is the property
that lets `configs/deploy.yaml` sit beside the registry without joining it (Part E).

### 1.1 What the registry contains

`configs/base.yaml` is organised into **eighteen** top-level blocks. Every one of them is loaded,
validated and hashed:

| Block | Purpose | Representative keys |
|---|---|---|
| `project` | identity, seed, schema version | `name`, `version`, `seed`, `schema_version` |
| `image` | input geometry and tiling policy | `max_pixels`, `tile_size`, `tile_overlap`, `max_tiles`, `top_k_tiles` |
| `optical` | optical band handling | `normalization`, `lower_percentile`, `upper_percentile`, `canonical_channels` |
| `sar` | SAR band handling | `representation`, `clip_min_db`, `clip_max_db`, `canonical_channels` |
| `router` | intent router (encoder + adapter + training) | `model`, `revision`, `max_length`, `embedding_dim`, `num_tasks`, `tasks`, `training.*` |
| `vlm` | SmolVLM contract | `checkpoint`, `revision`, `loader_class_preference`, `processor_longest_edge`, `prompt_must_use_chat_template` |
| `grounding` | RemoteCLIP grounding encoder | `checkpoint_repo`, `image_size`, `resolution_frozen`, `encoder_projected_dim` |
| `grounding_head` | the trainable head over the frozen encoder | `hidden_dim`, `dropout`, `feature_dim`, `positive_confidence_weight`, `decode` |
| `grounding_training` | grounding head training protocol | `learning_rate`, `epochs`, `box_loss_weight`, `giou_loss_weight`, `confidence_loss_weight` |
| `change` | Siamese change detection | `tile_size`, `threshold`, `encoder`, `sa_mode`, `bce_weight`, `dice_weight`, `levir_split.*` |
| `croma` | optical-SAR encoder | `checkpoint_repo`, `image_resolution`, `encoder_dim`, `optical_channels`, `sar_channels`, `modalities_used` |
| `fusion` | optical-SAR fusion head | `input_dim`, `hidden_dim`, `dropout`, `num_classes` |
| `evidence` | evidence aggregation bounds | `max_items`, `coordinate_system_default` |
| `confidence` | calibration switch + artifact | `temperature_scaling`, `calibration_file` |
| `agent` | controller policy | `max_specialists`, `timeout_seconds`, `unload_after_workflow`, `states` |
| `training` | VLM adaptation protocol | `precision`, `vlm_batch_size`, `lora_rank`, `gradient_checkpointing` |
| `evaluation` | evaluation isolation | `immutable_public_test`, `hidden_data_access`, `leakage_split_key` |
| `deployment` | frozen deployment declaration | `platform`, `sdk`, `zerogpu`, `torch_compile`, `gpu_duration_*`, `cpu_mode_required` |

(Every key above is present in `configs/base.yaml`; the block count is eighteen. The table is the
authoritative enumeration.)

## 2. The rule: "no magic numbers in Python"

The very top of `configs/base.yaml` states the rule in three lines:

```yaml
# SatQuery AI — authoritative configuration registry
# Architecture v1.0 (docs/ARCHITECTURE_FREEZE.md)
#
# RULE: no magic numbers anywhere in Python. Everything tunable lives here.
# Every value below is loaded, validated and hashed by core/config.py.
```

(`configs/base.yaml:1-5`)

The `core/config.py` module docstring restates the intent from the loader's side:

> *"One config system. No duplicated constants. Every value in `configs/base.yaml` is loaded,
> validated against the frozen architecture, and hashed so evaluation runs are reproducible."*
> (`core/config.py:1-9`)

### 2.1 Why the rule exists — the two reasons, and they are different

It is tempting to read "no magic numbers" as a style preference. It is not. It buys two specific
properties, and the second is the one that matters most.

**Reason 1 — one place to change.** A number that lives in a module is a number that can be
duplicated. `image.tile_size` is 512. If it were a literal in `preprocessing/tiling.py`, then the VLM
processor's `longest_edge` pin, the grounding tile geometry, and the tiling policy would each carry
their own copy, and the copies would drift. The loader's own comment on the F5-2 guard makes this
explicit — the pin is *"tied to `image.tile_size`"* so that *"the processor cannot silently start
upscaling tiles again"* (`core/config.py:129-130`). A cross-block invariant is only expressible if
both blocks are in the same registry and the loader can compare them.

**Reason 2 — one hash to pin.** This is the reason that turns a style rule into an architecture
decision. Because every value is in one file, the **whole file can be hashed**, and that hash can be
recorded alongside a measurement. `Config.hash` is that hash. `artifacts/change/levir_change_v001/model_metadata.json`
records the hash the shipped change head was trained under, and `scripts/eval_change.py` **refuses to
score** when the current hash drifts from it (exit `3`) — proven by the test docstring at
`tests/test_config.py:46-73`:

> *"The hash is a frozen CONTRACT, not merely a stability property. … A well-meaning config edit
> therefore silently invalidates the project's benchmark."*

If numbers lived in modules, there would be no single artifact to hash, and the reproduction check
would have to enumerate every module — which is the same as not having it. The "one place" rule is
what makes the "one hash" rule possible. They are one decision, not two.

### 2.2 What the rule does NOT cover — deployment state

The rule is "everything **tunable** lives here", and the corollary is that things that are **not
architecture** deliberately do not. The clearest worked example is the asset-upload store, whose
sizing lives in the environment, not the registry:

> *"Read from the environment rather than from `configs/base.yaml` on purpose: adding a key there
> moves `Config.hash` off `78f1e3700da15aa1` and invalidates the frozen Phase-9 benchmark. Sizing is
> deployment state."* (`app/space_app.py`, `_asset_max_files` docstring)

The same reasoning governs `SATQUERY_ASSET_MAX_FILES`, `SATQUERY_ASSET_TTL_S`,
`SATQUERY_MAX_FILE_BYTES`, `SATQUERY_MAX_BODY_BYTES`, `SATQUERY_ASSET_ENABLED` and
`SATQUERY_ASSET_DIR` (`docs/DEPLOYMENT_ARCHITECTURE.md` §4). And owner decision **D-4** records the
same pattern for a *model path*: rather than add `grounding_head.head_path` to `base.yaml`, the
default lives in code as `DEFAULT_HEAD_PATH`, *"because setting it would have moved the frozen config
hash"* (`docs/OWNER_DECISIONS_2026-09-23.md` D-4).

The pattern is a rule of thumb the reader can apply:

> If the value changes the **architecture**, it belongs in the registry and is hashed.
> If the value changes only **where or how large** the deployment is, it belongs in the environment.

Two facts make that rule checkable rather than aspirational: `change.checkpoint_path` and
`grounding_head.head_path` are both **absent** from `configs/base.yaml` (verified by reading all 294
lines), and `core/registry.py` declares them as *optional* config keys resolved at the call site
(`docs/OWNER_DECISIONS_2026-09-23.md` D-4; `docs/DEPLOYMENT_ARCHITECTURE.md` §5.6).

---

# Part B — The loader pipeline

## 3. `load_config` — the five stages

The entry point is `load_config`, 28 lines, and it does exactly five things in a fixed order:

```python
def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load, merge and validate the configuration registry."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if overrides:
        data = _deep_merge(data, overrides)

    # Environment overrides for the two values most likely to differ by host.
    if env_prec := os.environ.get("SATQUERY_PRECISION"):
        data.setdefault("training", {})["precision"] = env_prec
    if env_dev := os.environ.get("SATQUERY_TORCH_COMPILE"):
        data.setdefault("deployment", {})["torch_compile"] = env_dev.lower() == "true"

    return Config(data, source=str(cfg_path))
```

(`core/config.py:246-267`)

The stages, in order, with what each one is for:

| # | Stage | Code | What it establishes |
|---|---|---|---|
| 1 | **Resolve path** | `Path(path) if path else DEFAULT_CONFIG` | explicit path, or the one registry |
| 2 | **Fail fast if absent** | `if not cfg_path.exists(): raise ConfigError` | a missing registry is a named error, not a `FileNotFoundError` traceback |
| 3 | **Parse YAML** | `yaml.safe_load(fh) or {}` | a mapping; empty file → `{}`, never `None` |
| 4 | **Deep-merge overrides** | `_deep_merge(data, overrides)` | programmatic overrides that do not touch the file |
| 5 | **Env overrides** | `SATQUERY_PRECISION`, `SATQUERY_TORCH_COMPILE` | the two values most likely to differ by host |
| 6 | **Construct + validate** | `Config(data, source=...)` | `_validate()` runs in `__init__` |

Two ordering facts are load-bearing and easy to miss:

1. **Environment overrides win over `overrides`.** Stage 5 runs *after* stage 4, and it assigns
   directly into `data`. A caller that passes `overrides={"training": {"precision": "bf16"}}` while
   `SATQUERY_PRECISION=fp16` is set gets `fp16`. The environment is the outermost layer.
2. **Environment overrides still go through validation.** They are written into `data` *before*
   `Config(...)` is constructed, so `SATQUERY_TORCH_COMPILE=true` does **not** quietly enable
   `torch.compile` — it sets `deployment.torch_compile = True`, which the C-8 guard then rejects with
   a `ConfigError`. Likewise `SATQUERY_PRECISION=tf32` is rejected by the C-6 guard. The environment
   is a legitimate override surface, not a way around the guards.

Note also stage 3's `or {}`: `yaml.safe_load` returns `None` for an empty document, and the loader
coerces that to an empty mapping so the subsequent `data.setdefault(...)` calls cannot raise
`AttributeError`. A registry that parses to nothing then fails validation on every required key,
which is the correct outcome — but it fails with *named* errors, not a crash inside the loader.

### 3.1 `_deep_merge` — recursive, right-biased, and shallow-safe

```python
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
```

(`core/config.py:33-40`)

The semantics, exhaustively:

- **A new top-level key is copied verbatim** — `override` can *add* blocks the registry lacks. This
  is exercised by `test_unknown_key_in_override_is_kept_not_crashing` (`tests/test_config.py:345-348`):
  *"overrides may add keys; the registry is permissive, the guards are specific."*
- **A nested mapping merges recursively** — `{"fusion": {"hidden_dim": 1024}}` changes only
  `fusion.hidden_dim`, leaving `fusion.input_dim`, `fusion.dropout` and `fusion.num_classes` intact.
  This is the whole point of a deep merge.
- **A non-mapping value replaces** — a scalar, list, `None` or a mapping-over-scalar all overwrite.
  There is no list concatenation and no list merging; a list override is a replacement.
- **`base` is not mutated** — `out = dict(base)` copies the top level, and recursive calls copy each
  nested level they descend into. The caller's `data` dict from `yaml.safe_load` is safe to reuse.

The `isinstance(..., dict)` check on **both** sides is what makes the merge total: if `base[key]` is a
list and `override[key]` is a dict (or vice versa), the branch falls to `else` and the override wins
outright rather than raising.

The consequence for hash stability is direct: `test_hash_changes_when_config_changes`
(`tests/test_config.py:35-38`) asserts that `load_config(overrides={"fusion": {"hidden_dim": 1024}})`
produces a hash **different** from the base config. An override is a config change, and a config
change is a hash change. That is exactly why the registry's `builders=` override (Part D.4) is a
*call-site* mechanism and not a config key.

### 3.2 The two environment overrides, exactly

```python
if env_prec := os.environ.get("SATQUERY_PRECISION"):
    data.setdefault("training", {})["precision"] = env_prec
if env_dev := os.environ.get("SATQUERY_TORCH_COMPILE"):
    data.setdefault("deployment", {})["torch_compile"] = env_dev.lower() == "true"
```

(`core/config.py:261-265`)

| Variable | Writes | Coercion | Validation that then applies |
|---|---|---|---|
| `SATQUERY_PRECISION` | `training.precision` | none (verbatim string) | **C-6**: must be `fp16` / `bf16` / `fp32` |
| `SATQUERY_TORCH_COMPILE` | `deployment.torch_compile` | `.lower() == "true"` → bool | **C-8**: `True` is forbidden |

Three precise behaviours worth recording:

- **Truthiness gate, not presence gate.** `os.environ.get(...)` is used in a walrus with a truthiness
  test, so `SATQUERY_TORCH_COMPILE=""` (empty string) is *ignored* — the registry value stands. Only a
  non-empty value overrides. The comment calls these *"the two values most likely to differ by host"*.
- **The bool coercion is case-insensitive and exact.** `SATQUERY_TORCH_COMPILE=TRUE` → `True`;
  `SATQUERY_TORCH_COMPILE=1` → `False` (because `"1".lower() != "true"`); `SATQUERY_TORCH_COMPILE=yes`
  → `False`. Anything that is not `true` (case-insensitively) becomes `False`, which is the C-8-safe
  direction — an unparsable value cannot *enable* `torch.compile`.
- **`setdefault` makes the write safe on a partial registry.** `data.setdefault("training", {})` only
  creates the block if it is absent, so an override that already provided `training` is not
  clobbered at the block level (only the `precision` leaf is set).

These two are the *only* environment overrides in the loader. Every other environment variable in the
project — device selection, asset store sizing, gateway limits — is read at its point of use, not by
the registry loader. That split is deliberate: `SATQUERY_DEVICE` (Part B.6) must be readable *without
importing torch*, which the loader's validation path must not require.

## 4. `get_config()` — the process-wide singleton

```python
@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide singleton. Import this, do not re-read YAML."""
    return load_config()
```

(`core/config.py:270-273`)

Three properties follow from `lru_cache(maxsize=1)`:

1. **The YAML is read once per process.** The first caller pays the file read and the `_validate()`
   pass; every later caller gets the identical object. There is no re-read and no re-validation.
2. **The hash is computed on demand but the data is fixed.** `hash` is a `@property`
   (`core/config.py:76-80`), so it recomputes the sha256 each access — but it hashes the same frozen
   `_data`, so it is stable. It is not cached, and it does not need to be.
3. **`maxsize=1` with no arguments means exactly one cached entry.** Because `get_config()` takes no
   arguments, the cache can never hold more than one `Config`, which is what "process-wide singleton"
   means here. There is no key to vary.

**The import convention.** The docstring is an instruction, not a description: *"Import this, do not
re-read YAML."* A module that calls `load_config()` in a hot path would re-read the file and
re-validate on every call. `get_config()` is the intended surface, and the registry it returns is the
one every subsystem sees.

**One caveat about `Config`'s "immutable" docstring.** The class docstring calls itself an
*"Immutable, validated, hashable view over the YAML registry"* (`core/config.py:44`). Read precisely,
that means the class exposes **no mutating API**: there is no setter, no `__setitem__`, and the only
way to obtain a plain mutable copy is `as_dict()`, which returns a fresh structure via a JSON
round-trip. The underlying `self._data` dict is reachable through `__getitem__` and `get`, and no
code path in the repository mutates it. The practical guarantee a reader should rely on is therefore:
*treat the returned `Config` as read-only, and if you need a modified registry, call `load_config`
with `overrides` and accept that the hash moves.*

## 5. Access surfaces: `get`, `require`, `__getitem__`, `as_dict`, `seed`

Four access methods and two convenience properties make up the whole read surface.

### 5.1 `get(path, default=None)` — dotted-path traversal

```python
def get(self, path: str, default: Any = None) -> Any:
    """Dotted-path access: cfg.get('croma.image_resolution')."""
    node: Any = self._data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
```

(`core/config.py:57-64`)

Behaviour, exhaustively:

- The path is split on `"."`; each segment must be a key of a dict.
- A missing segment returns `default` — **never raises**.
- Traversing *through* a non-dict returns `default`. `cfg.get("project.name.oops")` returns `None`,
  because `project.name` is a string and `isinstance(node, dict)` fails on the next iteration.
- `default` defaults to `None`, so `get` cannot distinguish "key absent" from "key present and `None`"
  by itself. That ambiguity is what `require` exists to resolve.
- A single-segment path works: `cfg.get("project")` returns the whole block.

`test_dotted_access_and_require` (`tests/test_config.py:84-89`) pins the basics:
`cfg.get("change.tile_size") == 256`.

### 5.2 `require(path)` — a sentinel, so `None` is a legal value

```python
def require(self, path: str) -> Any:
    sentinel = object()
    value = self.get(path, sentinel)
    if value is sentinel:
        raise ConfigError(f"required config key missing: {path}")
    return value
```

(`core/config.py:66-71`)

The `sentinel = object()` is the whole design. If `require` called `self.get(path)` with the default
`None` and then tested `if value is None`, a key that is legitimately `None` — and the registry has
several, such as `evaluation.official_aggregate_weights: null` — would be reported as *missing*. A
fresh `object()` is unique, cannot appear in YAML, and therefore separates "absent" from "present and
null" exactly.

`require` raises `ConfigError` (a `WorkflowPlanError` subclass), so a missing required key surfaces as
the project's own error taxonomy, not a bare `KeyError`. `test_dotted_access_and_require` pins both
halves: `cfg.require("project.seed") == 42`, and `cfg.require("does.not.exist")` raises `ConfigError`.

### 5.3 `__getitem__` — top-level only, and it raises

```python
def __getitem__(self, key: str) -> Any:
    if key not in self._data:
        raise KeyError(key)
    return self._data[key]
```

(`core/config.py:52-55`)

Unlike `get`, `cfg["x"]` is a **top-level** lookup that raises `KeyError` when absent. It does not
understand dotted paths — `cfg["croma.image_resolution"]` raises `KeyError`, because there is no
top-level key with that literal name. The two surfaces are deliberately different: `[]` is for
"this must exist and I want it to fail loudly", `get` is for "this may not exist and I have a
fallback".

### 5.4 `as_dict()` — a JSON-safe deep copy

```python
def as_dict(self) -> dict[str, Any]:
    return json.loads(json.dumps(self._data, default=str))
```

(`core/config.py:73-74`)

The `json.dumps(..., default=str)` → `json.loads(...)` round-trip does two things: it produces a
**fresh** structure (no aliasing of the registry's nested dicts), and it coerces any non-JSON value
to its string form via `default=str`. `test_as_dict_roundtrips_to_json_safe_types`
(`tests/test_config.py:332-337`) asserts the result `json.dumps` cleanly and that
`payload["project"]["name"] == "satquery-ai"`. This is the surface used when a registry snapshot has
to be embedded in a response or a trace.

### 5.5 `seed` — the one convenience property for reproducibility

```python
@property
def seed(self) -> int:
    return int(self.get("project.seed", 42))
```

(`core/config.py:82-84`)

Note the fallback: if `project.seed` were absent, `seed` returns `42` rather than raising. The
registry declares `project.seed: 42` (`configs/base.yaml:10`), so the fallback is a safety net, not
the source of truth. `test_seed_is_int` (`tests/test_config.py:92-93`) pins `42`.

## 6. `device_preference` and `SATQUERY_DEVICE`

```python
@property
def device_preference(self) -> str:
    override = os.environ.get("SATQUERY_DEVICE")
    if override:
        return override
    return "cuda" if _torch_cuda_available() else "cpu"
```

(`core/config.py:86-91`)

```python
def _torch_cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415
        return bool(torch.cuda.is_available())
    except Exception:
        return False
```

(`core/config.py:237-243`)

Four facts about this property:

1. **Environment wins.** If `SATQUERY_DEVICE` is set to any non-empty string, that string is returned
   **verbatim** — no validation, no case-folding, no normalisation at this layer. The value is
   passed through as the caller wrote it.
2. **Otherwise it probes torch, and the probe is defensive.** `_torch_cuda_available()` wraps the
   import *and* the call in a single `try/except Exception` and returns `False` on any failure. On a
   machine with no torch installed — which is the CPU development environment and the base CI image —
   the property returns `"cpu"` rather than raising `ModuleNotFoundError`. The comment `# noqa: PLC0415`
   records that the local import is intentional: it keeps `import core.config` cheap.
3. **It is not part of the hash.** `device_preference` is a computed property over `os.environ` and a
   runtime probe. It does not read or write `self._data`, so it cannot move `Config.hash`. This is
   why the device can differ per host without invalidating a single published number.
4. **It is deliberately not read on the metadata path.** `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3
   records that `build_serving_registry()` *"via `Config.device_preference`, a `@property` that calls
   `_torch_cuda_available()`"* was one of two things that *did* import torch on the health path, and
   that both were removed. The test
   `test_the_metadata_path_does_not_import_torch` runs the import in a subprocess and asserts
   `torch imported: False`. So the property exists, is correct, and is simply not on the path that
   must stay torch-free.

**A note on validation of the served value.** `docs/DEPLOYMENT_ARCHITECTURE.md` §4 records finding
**F-8**: the *served* device value is validated downstream (the contract publishes a closed set
`cpu | cuda | mps | null`, and an unrecognised value yields `null` rather than being echoed). That
validation lives in the deployment adapter, not in `Config.device_preference`. The distinction is
worth keeping straight: the config layer passes the string through; the serving layer narrows it.

## 7. `ConfigError` and the error taxonomy

```python
class ConfigError(WorkflowPlanError):
    code = "config_error"
    user_message = "The system configuration is invalid."
```

(`core/config.py:28-30`)

A configuration failure is a **`WorkflowPlanError`** subclass carrying a machine `code`
(`config_error`) and a client-safe `user_message`. It is raised from exactly three places:

| Site | Trigger |
|---|---|
| `load_config` | the registry file does not exist (`config file not found: …`) |
| `require` | a required key is missing (`required config key missing: …`) |
| `_validate` | one or more frozen-architecture guards failed (a multi-line list) |

Because `_validate` collects *all* errors and raises **one** `ConfigError` listing every failure, a
bad registry reports every problem in a single startup message rather than one per run. That
collection behaviour is covered in Part C.13.

---

# Part C — Every validation rule

## 8. How `_validate` is structured

```python
def _validate(self) -> None:
    errors: list[str] = []
    ...
    if errors:
        raise ConfigError(
            "configuration failed frozen-architecture validation:\n  - "
            + "\n  - ".join(errors)
        )
```

(`core/config.py:94-95`, `:218-222`)

Every check **appends** to `errors`; nothing raises mid-pass. The single raise at the end means the
failure message is a bulleted list of every violated rule at once. `_validate` runs from
`Config.__init__` (`core/config.py:46-49`), so **a `Config` object that exists is a `Config` object
that passed every rule** — there is no unvalidated state to hold.

The rules are grouped by the finding they encode. The table below is the map; each rule gets its own
sub-section.

| # | Rule | Finding | Source |
|---|---|---|---|
| C.1 | `croma.image_resolution` is an int multiple of 8 | **C-7** | `core/config.py:97-105` |
| C.2 | `training.precision ∈ {fp16, bf16, fp32}` | **C-6** | `core/config.py:107-112` |
| C.3 | `deployment.torch_compile` must not be `True` | **C-8** | `core/config.py:114-119` |
| C.4 | `vlm.processor_longest_edge` is an int ≥ 1 and ≤ `image.tile_size` | **F5-2** (C-3) | `core/config.py:121-142` |
| C.5 | `vlm.prompt_must_use_chat_template` must be `True` | **F5-3** | `core/config.py:144-150` |
| C.6 | `fusion.input_dim == 3·768 + 12 + 2` | **C-1** | `core/config.py:152-162` |
| C.7 | `croma.optical_channels == 12`, `croma.sar_channels == 2` | **C-1** | `core/config.py:164-168` |
| C.8 | `grounding_head.feature_dim == 4 · grounding.encoder_projected_dim` | **P7-1** | `core/config.py:170-196` |
| C.9 | `router.tasks` includes `unsupported` | router ontology | `core/config.py:198-201` |
| C.10 | `router.num_tasks == len(router.tasks)` | router ontology | `core/config.py:202-206` |
| C.11 | `change.encoder` present; `change.sa_mode ∈ {BAM, PAM}` | change config | `core/config.py:208-212` |
| C.12 | `image.top_k_tiles ≤ image.max_tiles` | tiling policy | `core/config.py:214-216` |

## 9. C-7 — `croma.image_resolution` must be a multiple of 8

```python
# --- C-7: CROMA image_resolution must be a multiple of 8 -----------
croma_res = self.get("croma.image_resolution")
if croma_res is None:
    errors.append("croma.image_resolution is required")
elif not isinstance(croma_res, int) or croma_res % 8 != 0:
    errors.append(
        f"croma.image_resolution must be an int multiple of 8 "
        f"(CROMA asserts image_resolution % 8 == 0); got {croma_res!r}"
    )
```

(`core/config.py:97-105`)

**What it checks.** Three things at once: the key is present, it is an `int`, and it is divisible by
8.

**Why.** CROMA's own constructor contains `assert image_resolution % 8 == 0`
(`docs/PHASE0_CONTRACT_VALIDATION.md` §1.1). The `8` is CROMA's patch size: `num_patches =
int((image_resolution / 8) ** 2)`. A resolution that is not a multiple of 8 does not merely produce a
different patch count — it violates an assertion **inside the vendored model**, which means the
failure would surface at model construction, after the config has been accepted and the process has
started. The loader moves that failure to startup, and gives it a name.

The registry's value is `image_resolution: 120` with the comment *"finding C-7: `image_resolution %
8 == 0`. Native 120 → 225 patches."* (`configs/base.yaml:208-209`). The arithmetic:
`120 / 8 = 15`, and `15² = 225` patches — matching the verified `num_patches = 225 (15×15)` at
`docs/PHASE0_CONTRACT_VALIDATION.md` §1.3.

**Why the type check matters.** `not isinstance(croma_res, int)` rejects a float such as `120.0` even
though `120.0 % 8 == 0.0` would be falsy-safe — because CROMA's `int(...)` cast and its assertion are
written against an integer, and a float would change `num_patches`' arithmetic. It also rejects
`True`, since `isinstance(True, int)` is `True` in Python but `True % 8 != 0`, so the guard still
catches it.

**Pinned by:** `test_croma_resolution_must_be_multiple_of_8` (rejects `121`) and
`test_croma_native_resolution_is_valid` (accepts `120`, asserts `% 8 == 0`) —
`tests/test_config.py:123-131`.

## 10. C-6 — `training.precision` must be a viable precision, and T4 makes it `fp16`

```python
# --- C-6: precision must be viable on the target accelerator -------
precision = self.get("training.precision")
if precision not in {"fp16", "bf16", "fp32"}:
    errors.append(
        f"training.precision must be fp16|bf16|fp32, got {precision!r}"
    )
```

(`core/config.py:107-112`)

**What it checks.** Membership in a closed set of three strings. Note there is **no** type guard
beyond set membership — a non-string simply fails to be in the set and is rejected.

**Why three, and why `fp16` is the frozen choice.** The Phase-0 contract validation measured the
training accelerator and found the plan's declared `bf16` unusable:

| Source | Statement |
|---|---|
| `docs/PHASE0_CONTRACT_VALIDATION.md` §7 (contradiction register) | C-6: plan said `precision: bf16`; *"T4 is SM 7.5, no bf16 tensor cores"*; resolution *"fp16 + AMP default"* |
| `configs/base.yaml:254` (block header) | *"Training (finding C-6: T4 is SM 7.5 → fp16, NOT bf16)"* |
| `configs/base.yaml:257` | `precision: fp16` |
| `docs/ARCHITECTURE_FREEZE.md` §4 | `training.precision` = `fp16`, reason *"T4 = SM 7.5, no bf16 tensor cores (C-6)"* |

The rationale in one sentence: **bf16 requires hardware support that the target accelerator does not
have.** NVIDIA's T4 is compute capability **SM 7.5**, which predates bf16 tensor cores; requesting
bf16 there is not "slower", it is unsupported. `fp16` is the mixed-precision format that the T4 *does*
accelerate, so `fp16` is the frozen default rather than a preference.

`fp32` remains in the allowed set because a full-precision run is a legitimate (if slower) choice and
is not architecturally invalid; `bf16` remains in the set because the guard is about *internal
consistency of the value*, not about forbidding the format — the host-specific consequence is
documented, and the loader's job is to reject typos and invented formats like `tf32` or `bf16-amp`.

**Pinned by:** `test_precision_must_be_known` (rejects `tf32`) and `test_precision_defaults_to_fp16_for_t4`
(asserts `fp16`) — `tests/test_config.py:175-182`.

## 11. C-8 — `deployment.torch_compile` is forbidden

```python
# --- C-8: ZeroGPU forbids torch.compile ----------------------------
if self.get("deployment.torch_compile") is True:
    errors.append(
        "deployment.torch_compile=true is forbidden: ZeroGPU does not "
        "support torch.compile (finding C-8)"
    )
```

(`core/config.py:114-119`)

**What it checks.** `is True` — an identity test against the boolean `True`. Not truthiness. A value
of `1`, `"true"` (string) or `"yes"` does **not** trip this guard, because none of them *is* `True`.
The registry's `torch_compile: false` (`configs/base.yaml:287`) is a real YAML boolean, so the
comparison is boolean-to-boolean.

**Why.** `docs/PHASE0_CONTRACT_VALIDATION.md` §5.1 records the verified ZeroGPU constraint directly:
*"`torch.compile` | **not supported** (use AoTI, torch 2.8+)"*. Finding C-8's resolution states:
*"`torch.compile` is forbidden throughout."* Enabling it on a ZeroGPU Space is not a performance
regression — it is a configuration the platform does not support, so the loader makes it impossible
to ship.

The registry carries the same statement inline:

```yaml
deployment:
  platform: huggingface-spaces
  sdk: gradio
  zerogpu: true
  # finding C-8: ZeroGPU does not support torch.compile. Never enable.
  torch_compile: false
```

(`configs/base.yaml:282-287`)

**Interaction with the environment override.** As noted in Part B.3.2,
`SATQUERY_TORCH_COMPILE=true` sets this key to `True` and is then rejected here. The guard is the
backstop for the override, not a bypass of it.

**Pinned by:** `test_torch_compile_is_forbidden_on_zerogpu` — `tests/test_config.py:185-187`. The
deploy validator independently requires `torch_compile: False` in both `base.yaml` and `deploy.yaml`
(`scripts/validate_deploy_config.py`, `REQUIRED_BOOL`).

## 12. F5-2 — the VLM processor must not upscale our tiles

This is the longest guard in the file, and its comment block is the reason: it encodes a *measurement*,
not an assumption.

```python
# --- C-3 / F5-2: the VLM processor must not upscale our tiles -----
#
# MEASURED (docs/PHASE5_VLM_CONTRACT.md): the processor's default
# longest_edge is 2048. A 512 px tile is upscaled 4x and then split by
# do_image_splitting into 4x4 sub-images + 1 overview = 17 images and
# 1142 prompt tokens, versus 1 image when pinned. The plan estimated a
# 4x overrun; the real figure is ~17x.
#
# Tying this to image.tile_size makes the pin a control rather than a
# comment: the processor cannot silently start upscaling tiles again.
proc_edge = self.get("vlm.processor_longest_edge")
tile_size = self.get("image.tile_size")
if not isinstance(proc_edge, int) or proc_edge < 1:
    errors.append(
        f"vlm.processor_longest_edge must be an int >= 1, got {proc_edge!r}"
    )
elif isinstance(tile_size, int) and proc_edge > tile_size:
    errors.append(
        f"vlm.processor_longest_edge={proc_edge} exceeds "
        f"image.tile_size={tile_size}; the processor would upscale every "
        f"tile and then split it into ~17 sub-images (finding F5-2)"
    )
```

(`core/config.py:121-142`)

**What it checks.** Two things: `processor_longest_edge` is a positive integer, and — when
`image.tile_size` is an int — it does not exceed the tile size.

**Why — the measurement.** The VLM processor's default `longest_edge` is **2048**. A 512 px tile is
therefore upscaled 4×, and then `do_image_splitting=True` cuts the 2048 px image into 4×4 = 16
sub-images of 512 px each, plus one overview image. The measured result
(`docs/PHASE5_VLM_CONTRACT.md` §"The headline finding"):

```
INPUT: one 512×512 RGB tile

DEFAULT  (size.longest_edge = 2048, do_image_splitting = True)
    pixel_values  (1, 17, 3, 512, 512)     <- 17 images
    input_ids     (1, 1142)                <- 1142 tokens

PINNED   (size = {"longest_edge": 512})
    pixel_values  (1,  1, 3, 512, 512)     <-  1 image
```

The plan's finding **C-3** predicted a **4×** cost overrun. The measured figure is **~17×**, and the
same document states why the difference is not academic: *"the difference between '4×' and '17×' is
the difference between a cost you absorb and a cost that makes the deployment quota non-viable."* At
1142 prompt tokens per call, an un-pinned processor *"would spend a user's entire daily quota on one
or two queries"* against the 5 GPU-minutes/day free tier.

**Why the guard is *tied to the tile size* rather than merely asserting `512`.** The comment says it
plainly: *"Tying this to `image.tile_size` makes the pin a control rather than a comment: the
processor cannot silently start upscaling tiles again."* A hard-coded `512` check would pass even if
someone later changed `image.tile_size` to 1024, leaving the processor at 512 — now *downscaling*.
The relational check keeps the two numbers in the relationship the architecture requires, which is
only possible because both live in one registry.

**The `isinstance(tile_size, int)` guard on the second branch** is a defensive detail: if
`image.tile_size` were missing or non-int, the comparison is skipped rather than raising a `TypeError`
inside the validator. The first branch still enforces positivity on `proc_edge` independently.

**The registry values.** `processor_longest_edge: 512` with the F5-2 comment and the measured
before/after, and the note *"This value MUST be set explicitly on the processor at construction
time."* (`configs/base.yaml:98-105`). `image.tile_size: 512` (`configs/base.yaml:18`). The two are
equal, which is the strongest form of the constraint.

**Why `do_image_splitting` stays `true`.** The registry explains: *"Kept true so genuinely oversized
inputs still split, but at 512 a tile no longer exceeds `max_image_size` (also 512) and splitting does
not trigger."* (`configs/base.yaml:106-108`). The pin removes the *upscaling* that triggers splitting;
it does not remove the splitting capability.

**Pinned by** four tests (`tests/test_config.py:137-162`): `test_processor_longest_edge_must_be_positive_int`
(rejects `0`), `test_processor_must_not_upscale_tiles` (asserts `≤`), `test_processor_upscaling_is_rejected`
(rejects `4096`), and `test_processor_pin_is_tied_to_tile_size` (asserts **equality** with
`image.tile_size`). And the pin is confirmed in the real load path, not just the probe:
`docs/PHASE5_VLM_CONTRACT.md` records `max_images_seen = 1` as *"the F5-2 confirmation: the pin
survives model construction, processor construction, and generation. Unpinned it would read 17."*

## 13. F5-3 — prompts must go through the chat template

```python
# --- F5-3: prompts must go through the chat template --------------
if self.get("vlm.prompt_must_use_chat_template") is not True:
    errors.append(
        "vlm.prompt_must_use_chat_template must be true: SmolVLM raises "
        "ValueError on prompts lacking one <image> token per image "
        "(finding F5-3)"
    )
```

(`core/config.py:144-150`)

**What it checks.** `is not True` — so anything other than the boolean `True` (including a missing
key, `1`, `"true"`, or `None`) is rejected. This is the strictest form in the file, and deliberately
so: the setting is not a tunable, it is an invariant.

**Why.** SmolVLM requires **one `<image>` token per image** in the prompt. A hand-written prompt
string lacks them and raises `ValueError`. `docs/PHASE5_VLM_CONTRACT.md` §F5-3 records the finding,
and the registry records the fix as a permanent fact rather than a rediscovery:

```yaml
# Finding F5-3: SmolVLM requires one <image> token per image in the prompt.
# Hand-written prompt strings raise ValueError; always build prompts through
# processor.apply_chat_template(). Recorded so it is not rediscovered.
prompt_must_use_chat_template: true
```

(`configs/base.yaml:109-112`)

**Why it is a *config* guard at all.** The value does not parametrise anything — the prompt builder
either uses the chat template or it does not. Making it a registry key with a hard `is not True`
check converts an implementation convention into an enforced invariant: *"the chat-template path must
not be disableable"* (`tests/test_config.py:165-169`). A future refactor cannot quietly introduce a
hand-built prompt path, because the registry declares the invariant and the loader enforces it.

**Pinned by:** `test_chat_template_is_required` (rejects `False`) — `tests/test_config.py:165-169`.

## 14. C-1 — `fusion.input_dim` must match the verified CROMA concatenation

```python
# --- C-1: fusion input dim must match the verified concatenation ---
expected = self._expected_fusion_dim()
declared = self.get("fusion.input_dim")
if expected is not None and declared != expected:
    errors.append(
        f"fusion.input_dim={declared} disagrees with the verified CROMA "
        f"concatenation ({expected} = 3*encoder_dim + optical_channels + "
        f"sar_channels). CROMA emits optical/SAR/joint GAP vectors; the "
        f"availability mask is consumed by the fusion head, not by CROMA "
        f"(finding C-1)."
    )
```

(`core/config.py:152-162`)

**What it checks.** `fusion.input_dim` equals `_expected_fusion_dim()`, when that expectation is
computable.

### 14.1 The expected value is *derived*, not hard-coded

```python
def _expected_fusion_dim(self) -> int | None:
    dim = self.get("croma.encoder_dim")
    mods = self.get("croma.modalities_used") or []
    opt = self.get("croma.optical_channels")
    sar = self.get("croma.sar_channels")
    if None in (dim, opt, sar) or not mods:
        return None
    return len(mods) * int(dim) + int(opt) + int(sar)
```

(`core/config.py:224-231`)

The formula is `len(modalities_used) · encoder_dim + optical_channels + sar_channels`. It returns
`None` — meaning "cannot compute, skip the check" — if any input is missing. This is why the guard is
`if expected is not None and declared != expected`: a partial registry produces other, more specific
errors (missing channel counts, missing `encoder_dim`) rather than a spurious dimension mismatch.

### 14.2 The arithmetic, with the registry's values

| Term | Source | Value |
|---|---|---|
| `len(croma.modalities_used)` | `configs/base.yaml:214` → `[optical, sar, joint]` | **3** |
| `croma.encoder_dim` | `configs/base.yaml:210` | **768** |
| `croma.optical_channels` | `configs/base.yaml:211` | **12** |
| `croma.sar_channels` | `configs/base.yaml:212` | **2** |

```
expected = 3 · 768 + 12 + 2
         = 2304   + 14
         = 2318
```

And the registry declares `fusion.input_dim: 2318` (`configs/base.yaml:219`), with the derivation
written inline:

```yaml
fusion:
  # concatenated dim = 3 * 768 + 12 + 2 = 2318
  # finding C-1: the availability mask is consumed HERE, not by CROMA.
  input_dim: 2318
```

### 14.3 Why — C-1 is the project's most consequential contract finding

`docs/PHASE0_CONTRACT_VALIDATION.md` §1.2 records finding **C-1 (P0)**. CROMA's `ViT.__init__` builds
its patch embedding as a *fixed-shape* `nn.Linear`:

```python
pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
self.linear_input = nn.Linear(pixels_per_patch, self.dim)
```

and its `forward` signature is `def forward(self, SAR_images=None, optical_images=None):` — **no mask
parameter, no missing-channel token, no mask argument anywhere**. The finding is therefore:

> *"CROMA **cannot** consume a channel-availability mask. The master plan §19 … is **not
> implementable against the official code.**"*

The **resolution** is what fixes the fusion dimension: zero-fill the optical/SAR tensors to the
canonical channel counts (which *is* compatible with the fixed `Linear`), and route the availability
mask to the **fusion head** as a first-class input. The verified concatenation
(`docs/ARCHITECTURE_FREEZE.md` §2.5) is:

```
optical_GAP      (B, 768)
SAR_GAP          (B, 768)
joint_GAP        (B, 768)
optical_mask     (B, 12)     <- availability, from sensor adapter
sar_mask         (B,  2)     <- availability, from sensor adapter
                 ---------
concat           (B, 2318)
```

The three `GAP` vectors come from CROMA's verified six-key forward output
(`docs/PHASE0_CONTRACT_VALIDATION.md` §1.3); the two masks come from the sensor adapter, **not** from
CROMA. The guard's error message names exactly this: *"the availability mask is consumed by the
fusion head, not by CROMA (finding C-1)."*

### 14.4 Why it is enforced rather than documented

If `fusion.input_dim` disagreed with the real concatenation width, the fusion head's first `Linear`
would have the wrong `in_features`. PyTorch would raise a shape error — but only when the mismatched
tensor reached that layer, which is *after* CROMA has run and *after* the fusion head has been
constructed and loaded. The loader's check moves that failure to startup, before any model is
touched, and its message contains the derivation so the reader can see *why* 2318 is the right
number rather than merely that 2318 is expected.

**Pinned by:** `test_fusion_dim_matches_verified_croma_concatenation` (recomputes the formula and
asserts `2318`, then asserts `fusion.input_dim == expected`) and `test_fusion_dim_mismatch_is_rejected`
(rejects `999`) — `tests/test_config.py:99-110`.

## 15. C-1 (continued) — the fixed channel counts, 12 and 2

```python
# --- channel counts must match CROMA's fixed inputs ---------------
if self.get("croma.optical_channels") != 12:
    errors.append("croma.optical_channels must be 12 (CROMA s2_channels is fixed)")
if self.get("croma.sar_channels") != 2:
    errors.append("croma.sar_channels must be 2 (CROMA s1_channels is fixed)")
```

(`core/config.py:164-168`)

**What it checks.** Exact equality with `12` and `2`. Two independent guards, each naming the CROMA
attribute it mirrors.

**Why — these are *fixed*, not defaulted.** `docs/PHASE0_CONTRACT_VALIDATION.md` §1.1 quotes CROMA's
source directly:

| Attribute | Verified value | Evidence (verbatim) |
|---|---|---|
| SAR channels | **fixed 2** | `self.s1_channels = 2  # fixed at 2 SAR backscatter channels` |
| Optical channels | **fixed 12** | `self.s2_channels = 12  # fixed at 12 multispectral optical channels` |

Because CROMA's patch embedding is a fixed-shape `Linear` over
`patch_size² · in_channels`, the channel count is not a hyperparameter — it is baked into the weight
shapes. Feeding 8 optical channels to a model whose `linear_input` expects 12 is a shape error at the
input layer. The guards prevent a config from claiming a channel count the frozen encoder cannot
accept.

The registry states the same, with the canonical-order note:

```yaml
optical:
  normalization: percentile
  lower_percentile: 2
  upper_percentile: 98
  # CROMA expects exactly 12 optical channels.
  canonical_channels: 12
...
sar:
  representation: db
  clip_min_db: -30
  clip_max_db: 5
  # CROMA expects exactly 2 SAR channels (VV, VH).
  canonical_channels: 2
```

(`configs/base.yaml:26-38`)

Note the relationship to the `optical` and `sar` blocks: they each carry a `canonical_channels` key
**and** the `croma` block carries `optical_channels` / `sar_channels`. The guards check the **`croma`**
values, because those are the ones the fusion dimension formula consumes. The `optical` / `sar`
`canonical_channels` values are the sensor-adapter's statement of the same fact from its side.

`docs/ARCHITECTURE_FREEZE.md` §2.5 adds the operational detail: optical input is *"12 channels,
zero-filled to canonical order"* and SAR input is *"2 channels, zero-filled to canonical order"*.

**Pinned by:** `test_croma_channel_counts_are_pinned` (rejects `optical_channels: 8` and
`sar_channels: 3`) — `tests/test_config.py:113-117`.

## 16. P7-1 — `grounding_head.feature_dim == 4 · encoder_projected_dim`

This is the guard whose *enforcement* is the point, so it gets the longest comment in the file after
F5-2.

```python
# --- Phase 8: the head's assembled feature must match the encoder ---
#
# Per-cell feature = concat([patch, text, patch*text, global_pool]),
# i.e. 4 x the encoder's PROJECTED dim. Measured as 512, NOT the 768
# transformer width (finding P7-1): `visual.proj` is (768, 512).
#
# A mismatch here is a SILENT shape error. torch only raises at the
# similarity step, by which point the patch features have already been
# computed and cached — so the failure surfaces far from its cause.
#
# grounding.encoder_projected_dim is declared in config so this guard
# needs no torch import; specialists/grounding/remoteclip.py asserts
# the same value against the real model at load time.
projected = self.get("grounding.encoder_projected_dim")
head_dim = self.get("grounding_head.feature_dim")
if not isinstance(projected, int) or projected < 1:
    errors.append(
        f"grounding.encoder_projected_dim must be a positive int, "
        f"got {projected!r}"
    )
elif not isinstance(head_dim, int) or head_dim != 4 * projected:
    errors.append(
        f"grounding_head.feature_dim={head_dim} but the frozen encoder "
        f"projects to {projected}, so the assembled per-cell feature is "
        f"{4 * projected}-d. Expected feature_dim == 4 * "
        f"grounding.encoder_projected_dim (finding P7-1)."
    )
```

(`core/config.py:170-196`)

**What it checks.** `grounding.encoder_projected_dim` is a positive int, and
`grounding_head.feature_dim == 4 · encoder_projected_dim`.

### 16.1 The arithmetic

| Term | Source | Value |
|---|---|---|
| `grounding.encoder_projected_dim` | `configs/base.yaml:142` | **512** |
| `grounding_head.feature_dim` | `configs/base.yaml:158` | **2048** |

```
expected = 4 · 512 = 2048      ✓ matches the declared feature_dim
```

### 16.2 Why the factor is 4, and why 512 and not 768

The per-cell feature is the concatenation of **four** vectors:

```
per-cell = concat([patch, text, patch·text, global_pool])   ->  4 · projected_dim
```

The registry records the definition and the "4×" (`configs/base.yaml:150-158`), and the freeze states
the head structure. The trap the comment names is **768 vs 512**:

> *"the 768-vs-512 distinction is the one that bites. `visual.positional_embedding` is 768 wide and
> `visual.proj` is (768, 512); the embeddings the text tower can be compared against are the
> PROJECTED ones. Using 768 anywhere here would be a shape error that torch would only surface at the
> similarity computation, by which point the patch features have already been computed and cached."*
> (`specialists/grounding/remoteclip.py:15-19`)

So RemoteCLIP ViT-B/32 has transformer width **768** but projects to **512**. The *comparison space*
is 512-dimensional, so the head must assemble 4·512 = 2048, not 4·768 = 3072. `4 · 768 = 3072` would
be the "obvious" wrong answer, and it is a shape error rather than a type error — which is exactly
why the loader refuses it.

### 16.3 Why it is ENFORCED rather than documented — the silence argument

This is the load-bearing sentence in the guard's comment:

> *"A mismatch here is a SILENT shape error. torch only raises at the similarity step, by which point
> the patch features have already been computed and cached — so the failure surfaces far from its
> cause."*

Unpack the failure mode:

1. The head is constructed with `feature_dim = 3072` (say).
2. The encoder runs and produces patch features projected to 512.
3. The head assembles a 3072-wide feature from 512-wide inputs — this *may* still construct, if the
   assembly pads or the mismatch lands at a `Linear` whose `in_features` was set from `feature_dim`.
4. The error does not fire at construction. It fires at the **similarity step**, after every patch
   feature has been computed and cached.

The consequences that make this worse than an ordinary shape error:

- **It is far from its cause.** The traceback points at the similarity computation inside the encoder
  or head, not at the config value that was wrong.
- **It happens after expensive work.** Patch features are already computed and cached by the time it
  raises, so a training or inference run has paid the cost before failing.
- **It is easy to misattribute.** A reader seeing a shape error in the similarity step is likely to
  suspect the encoder or the head's forward, not a single integer in `configs/base.yaml`.

Enforcing the relation in the loader converts a distant, expensive, misattributable runtime failure
into a **startup** failure whose message contains the expected value and the derivation.

### 16.4 The guard needs no torch import — and a second assertion backs it at load

The comment records the design that makes this possible:

> *"`grounding.encoder_projected_dim` is declared in config so this guard needs no torch import;
> `specialists/grounding/remoteclip.py` asserts the same value against the real model at load time."*

So there are **two** enforcement points, and they check different things:

| Point | Checks | Against |
|---|---|---|
| `core/config.py` P7-1 guard | `feature_dim == 4 · encoder_projected_dim` | the **registry's declaration** |
| `specialists/grounding/remoteclip.py` | `visual.proj.shape[1] == VERIFIED_PROJECTED_DIM` | the **real loaded model** |

The config guard is arithmetic over declared numbers, so it is torch-free and runs at startup. The
specialist guard checks the *actual* model, so it catches a declaration that is internally consistent
but wrong about the checkpoint:

```python
#: Measured, not assumed. Asserted against at load time.
VERIFIED_PATCH_SIZE = 32
VERIFIED_TRANSFORMER_WIDTH = 768
VERIFIED_PROJECTED_DIM = 512
VERIFIED_PARAMETERS = 151_277_313
```

(`specialists/grounding/remoteclip.py:38-42`)

```python
proj = getattr(visual, "proj", None)
if proj is not None and proj.shape[1] != VERIFIED_PROJECTED_DIM:
    raise ModelLoadError(
        f"visual.proj maps to {proj.shape[1]}, expected "
        f"{VERIFIED_PROJECTED_DIM}",
        specialist="grounding",
    )
```

(`specialists/grounding/remoteclip.py:195-201`)

The two together mean: a registry that is self-consistent passes the config guard, and a model whose
`visual.proj` disagrees with the declaration fails at load. Neither alone is sufficient.

**Pinned by:** `test_head_feature_dim_matches_the_frozen_encoder` (imports `VERIFIED_PROJECTED_DIM`
and asserts `feature_dim == 4 · VERIFIED_PROJECTED_DIM`) and `test_head_feature_dim_mismatch_is_rejected`
(rejects `768`) — `tests/test_config.py:273-288`.

### 16.5 The neighbouring `grounding_head` value, and why it is *not* guarded

`grounding_head.positive_confidence_weight: 20.0` is in the same block but is **not** a validated
invariant — it is a tuning value. Its rationale is documented in the registry rather than enforced:

```yaml
# Objectness BCE sees 1 positive cell out of 49. Unweighted, the optimum is
# "no object" everywhere; this weight is what stops that collapse.
positive_confidence_weight: 20.0
```

(`configs/base.yaml:159-161`)

The distinction is the point of Part C: **relations that encode a contract are enforced; values that
encode a tuning choice are documented.** The test suite pins the *direction* of the tuning value
(`test_head_positive_confidence_weight_is_set` asserts `> 1.0`) without pinning the number, which is
the right granularity for a tunable.

## 17. Router ontology — `unsupported` must exist, and the count must match

```python
# --- router ontology ----------------------------------------------
tasks = self.get("router.tasks") or []
if "unsupported" not in tasks:
    errors.append("router.tasks must include 'unsupported'")
if self.get("router.num_tasks") != len(tasks):
    errors.append(
        f"router.num_tasks={self.get('router.num_tasks')} does not match "
        f"router.tasks length ({len(tasks)})"
    )
```

(`core/config.py:198-206`)

**What it checks.** Two invariants: the task list contains the literal `unsupported`, and the declared
count equals the list's length.

**Why `unsupported` must be present.** `unsupported` is the router's explicit *"this is not a
satellite-imagery question"* class (`docs/ARCHITECTURE_FREEZE.md` §2.1 lists the six classes as
`vqa | caption | grounding | change | optical_sar | unsupported`). It is the target of the
below-threshold fallback: *"Below `router.confidence_threshold` → deterministic lexical fallback →
else `unsupported`."* Removing it would leave the router with no way to decline a question, which
would force every query into a specialist — the exact failure the class exists to prevent. The guard
makes that impossible to configure away.

Note the `or []` on the first line: a missing or null `router.tasks` is treated as an empty list, so
`"unsupported" not in []` is `True` and the guard fires with the right message, rather than raising
`TypeError` on `in None`.

**Why `num_tasks` must equal `len(tasks)`.** `num_tasks` is the classifier's output width — the
number of logits the adapter's task head emits. The registry declares it **separately** from the list
because the two are consumed in different places:

```yaml
router:
  ...
  num_tasks: 6
  tasks:
    - vqa
    - caption
    - grounding
    - change
    - optical_sar
    - unsupported
```

(`configs/base.yaml:59-66`)

Two sources for one number is a drift hazard: a task list of six with a head of five would be a shape
mismatch between the ontology and the classifier, and it would fail only when the head was built. The
guard keeps them equal, so the ontology and the head width cannot disagree.

**The asymmetry a reader should know about.** `docs/architecture/01-system-overview.md` §2 records
that the router's six classes and the capabilities endpoint's six tasks are **different sixes** —
`unsupported` is in the router's label space but not the capability list, and `change_vqa` is the
reverse. The config's `router.tasks` is the **router's** ontology (with `unsupported`), and the guard
is scoped to that. `core/schemas.py::Task` carries all seven values.

**Pinned by:** `test_router_tasks_include_unsupported`, `test_router_num_tasks_matches_list`, and
`test_router_missing_unsupported_is_rejected` (rejects `["vqa", "caption"]`) —
`tests/test_config.py:197-208`.

## 18. Change config — `encoder` required, `sa_mode ∈ {BAM, PAM}`

```python
# --- change config -------------------------------------------------
if self.get("change.encoder") is None:
    errors.append("change.encoder is required")
if self.get("change.sa_mode") not in {"BAM", "PAM"}:
    errors.append("change.sa_mode must be BAM or PAM")
```

(`core/config.py:208-212`)

**What it checks.** `change.encoder` is present (specifically, not `None`), and `change.sa_mode` is
one of two exact strings.

**Why `encoder` is required.** The change specialist is a Siamese network: two encoders sharing
weights, plus a decoder. The encoder backbone is not optional — a change detector with no encoder
cannot produce a difference representation. The registry declares `encoder: resnet18`
(`configs/base.yaml:187`). The check is `is None`, not truthiness, so an empty string would pass;
the intent is "the key must be populated with a backbone name", and the registry supplies one.

**Why `sa_mode` is a closed set of two.** The spatial-attention module has exactly two verified modes.
`docs/PHASE0_CONTRACT_VALIDATION.md` §4.1 records the upstream STANet contract:

| Property | Verified value |
|---|---|
| Model variants | `CDF0` (base), `CDFA` (with SA module) |
| Attention modes | `--SA_mode BAM` or `--SA_mode PAM` |

The upstream training invocation shows the flag in use: `--model CDFA --SA_mode PAM`. The registry
declares:

```yaml
change:
  ...
  encoder: resnet18
  sa_mode: PAM             # BAM | PAM
  pretrained: true
```

(`configs/base.yaml:187-189`)

The guard's job is to reject anything else — a typo like `Pam`, `pan`, or `none` — because an
unrecognised mode would silently select neither attention variant. Note the set is **case-sensitive**:
`pam` is rejected. The comment `# BAM | PAM` in the registry is the human-readable form of the same
constraint.

**Why the freeze's change contract is what it is.** `docs/ARCHITECTURE_FREEZE.md` §2.4 records the
verified hyperparameters that go with this: *"lr = 1e-3, batch_size = 8, patch size 256×256
non-overlapping, loss 0.5·BCE + 0.5·Dice"*, all present in the `change` block
(`configs/base.yaml:190-193`). The encoder and SA mode are the two *structural* choices; the loader
guards those, and the test suite separately pins the hyperparameters
(`test_change_hyperparameters_match_verified_stanet`, `tests/test_config.py:228-233`).

**Pinned by:** `test_change_sa_mode_is_valid` and `test_change_sa_mode_rejects_junk` (rejects `XYZ`) —
`tests/test_config.py:236-242`.

## 19. Tiling policy — `top_k_tiles ≤ max_tiles`

```python
# --- tiling policy -------------------------------------------------
if self.get("image.top_k_tiles", 0) > self.get("image.max_tiles", 0):
    errors.append("image.top_k_tiles cannot exceed image.max_tiles")
```

(`core/config.py:214-216`)

**What it checks.** The number of tiles actually sent through a specialist does not exceed the hard
ceiling on tiles examined.

**Why.** The two values have distinct meanings, stated in the registry:

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

(`configs/base.yaml:16-24`)

- `max_tiles: 64` — the **hard ceiling** on tiles examined. This bounds the preprocessing work.
- `top_k_tiles: 4` — how many tiles are **actually sent through a specialist**. This bounds the
  expensive inference work.

`top_k_tiles > max_tiles` would mean "send more tiles to a specialist than were ever examined", which
is incoherent: the selection step cannot choose from a set larger than the set it built. The guard
makes that impossible.

**The `0` defaults are deliberate.** Both lookups use `self.get(key, 0)`, so a missing `top_k_tiles`
or `max_tiles` reads as `0` and the comparison is `0 > 0` → `False`. The guard does **not** fire for a
missing key; that is handled elsewhere (the key simply has no value, and downstream code that needs it
will fail with its own named error). The guard's scope is precisely "the ordering relation between two
present values", nothing more.

**The relationship to the VLM pin.** `image.tile_size: 512` is the value the F5-2 guard compares
against `vlm.processor_longest_edge`. So the `image` block participates in two guards — F5-2 and the
tiling check — which is a small illustration of why centralising the registry makes cross-block
invariants expressible.

**Pinned by:** `test_top_k_cannot_exceed_max_tiles` (rejects `top_k_tiles: 999`) —
`tests/test_config.py:214-216`. A companion test pins a *neighbouring* relation that the loader does
**not** guard: `test_tile_overlap_smaller_than_tile_size` asserts `image.tile_overlap < image.tile_size`
and `change.tile_overlap < change.tile_size` (`tests/test_config.py:219-222`) — a test-only invariant.

## 20. Rules the loader does NOT enforce (and the tests that pin them instead)

Not every frozen value is a loader guard. The distinction is worth stating explicitly, because a
reader who assumes "frozen ⇒ guarded" will overestimate what `_validate` protects.

| Value | Why it is not a loader guard | Where it is pinned |
|---|---|---|
| `grounding.image_size: 224` | the value is frozen by *measurement*, not by a structural relation; there is no arithmetic to check | `test_grounding_resolution_is_frozen_at_224`, `test_resolution_experiment_flag_is_gone` (`tests/test_config.py:248-267`) |
| `grounding.resolution_frozen: true` | a declaration of a decided question, not a constraint | same tests |
| `evaluation.hidden_data_access: false` | an isolation policy, asserted where evaluation runs | `test_hidden_data_access_is_disabled` (`tests/test_config.py:317-318`) |
| `evaluation.official_aggregate_weights: null` | *"the uploaded specification explicitly prohibits inventing an official aggregate formula"* (plan §63) | `test_no_official_aggregate_weights_are_invented` (`tests/test_config.py:321-322`) |
| `evaluation.leakage_split_key: scene_id` | a leakage policy, exercised by the leakage tests | `test_leakage_split_key_is_scene_level` (`tests/test_config.py:325-326`) |
| `change.learning_rate`, `change.batch_size`, `change.tile_size` | verified upstream hyperparameters, not structural relations | `test_change_hyperparameters_match_verified_stanet` (`tests/test_config.py:228-233`) |
| `deployment.cpu_mode_required: true` | a deployment declaration; the *validator* checks it, the loader does not | `test_cpu_mode_is_required` (`tests/test_config.py:190-191`); `scripts/validate_deploy_config.py` `REQUIRED_BOOL` |
| `grounding_training.*` | a training protocol, present rather than constrained | `test_grounding_training_exposes_run_protocol_keys` (`tests/test_config.py:300-306`) |
| `grounding.benchmark_box_scale: 100.0` | a unit conversion; wrongness is a data error, not a shape error | `test_vrsbench_box_scale_is_declared` (`tests/test_config.py:309-311`) |

The general principle: **`_validate` guards the values whose wrongness produces a failure far from
its cause** (a shape error in a model, a silent upscale, a forbidden platform feature). Values whose
wrongness is locally visible — a bad learning rate, an incorrect box scale — are pinned by tests and
documented in the registry, but not enforced at load.

## 21. The validation failure mode: collect all, raise once

```python
if errors:
    raise ConfigError(
        "configuration failed frozen-architecture validation:\n  - "
        + "\n  - ".join(errors)
    )
```

(`core/config.py:218-222`)

Because every check appends and nothing raises mid-pass, a registry with three problems reports
**all three** in one message. The rendered shape is:

```
configuration failed frozen-architecture validation:
  - croma.image_resolution must be an int multiple of 8 (CROMA asserts image_resolution % 8 == 0); got 121
  - training.precision must be fp16|bf16|fp32, got 'tf32'
  - image.top_k_tiles cannot exceed image.max_tiles
```

Two properties of this design:

1. **Fix-once feedback.** An operator editing a config sees every violation in one run rather than
   discovering them one per attempt.
2. **The message is a contract surface.** Each line names the key and the expectation, and the
   finding-carrying rules name their finding id (`C-8`, `F5-2`, `C-1`, `P7-1`). A reader who hits a
   guard can search the id and find the measurement that produced it — which is the whole reason the
   ids are in the messages.

**The error is a `WorkflowPlanError` subclass** (Part B.7), so a config failure is catchable as the
project's own taxonomy and carries `code = "config_error"` and a client-safe `user_message`.

---

# Part D — The hash

## 22. `Config.hash` — the definition

```python
@property
def hash(self) -> str:
    """Stable hash of the whole registry. Recorded in every evaluation run."""
    blob = json.dumps(self._data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
```

(`core/config.py:76-80`)

Decomposed, exactly:

| Step | Code | Effect |
|---|---|---|
| Serialise | `json.dumps(self._data, sort_keys=True, default=str)` | the **whole** registry as canonical JSON |
| Canonicalise key order | `sort_keys=True` | key insertion order cannot change the hash |
| Coerce non-JSON types | `default=str` | any non-serialisable value becomes its string form |
| Encode | `.encode()` | UTF-8 bytes |
| Digest | `hashlib.sha256(blob)` | a 64-hex-character sha256 |
| Truncate | `.hexdigest()[:16]` | the **first 16 hex characters** |

Three consequences worth stating plainly:

1. **It hashes the whole registry, not a subset.** Every block, every key, every value. `sort_keys=True`
   means the hash depends on the *content*, not on the file's key ordering, so reformatting the YAML
   without changing values does **not** move the hash. Adding, removing or editing a value does.
2. **`default=str` makes the hash total.** A value the JSON encoder cannot handle is stringified
   rather than raising, so hashing never fails on an exotic type.
3. **16 hex characters is 64 bits.** It is a fingerprint, not a cryptographic commitment: it is
   chosen to be short enough to print in a manifest and a log line. Two *different* registries
   colliding in 64 bits is not a concern for this use — the hash is a drift detector, not a security
   primitive.

`repr(Config)` surfaces it: `<Config source=… hash=…>` (`core/config.py:233-234`).

## 23. The frozen value: `78f1e3700da15aa1` — `VERIFIED`

**The frozen config hash is `78f1e3700da15aa1`.**

This is stated in three independent places and was **re-verified by execution** while writing this
chapter — `get_config().hash` was run in the repository's virtual environment and returned exactly
`78f1e3700da15aa1`:

| Source | Statement |
|---|---|
| `tests/test_config.py:43` | `FROZEN_CONFIG_HASH = "78f1e3700da15aa1"` |
| `docs/PHASE9_FREEZE.md` §3 | *"Config hash | `78f1e3700da15aa1`"* |
| `docs/OWNER_DECISIONS_2026-09-23.md` §cross-cutting rule 4 | *"`Config.hash = 78f1e3700da15aa1` is verified untouched."* |
| `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3 requirement 5 | the entrypoint *"must not … otherwise move `Config.hash` off `78f1e3700da15aa1`"* |

The test that pins it is explicit that the value — not merely the stability — is the contract:

```python
def test_config_hash_matches_the_shipped_checkpoint_record() -> None:
    """The hash is a frozen CONTRACT, not merely a stability property.

    `test_config_hash_is_stable` only checks `a.hash == b.hash`; it never pins
    the VALUE. But the value is load-bearing: `scripts/eval_change.py` reads
    `config_hash` from `artifacts/change/levir_change_v001/model_metadata.json`
    and REFUSES TO SCORE when the current hash drifts (exit 3). A well-meaning
    config edit therefore silently invalidates the project's benchmark.
    """
    assert load_config().hash == FROZEN_CONFIG_HASH, (...)
```

(`tests/test_config.py:46-81`)

There are two distinct hash tests, and the difference between them is the whole point:

| Test | Asserts | Catches |
|---|---|---|
| `test_config_hash_is_stable` | `a.hash == b.hash`, `len == 16` | non-determinism |
| `test_config_hash_matches_the_shipped_checkpoint_record` | `hash == "78f1e3700da15aa1"` | **any** value drift |

`test_config_hash_is_stable` would pass even if every value in the registry changed, as long as the
hash stayed deterministic. Only the second test pins the value, and it is the one that guards the
published benchmark.

### 23.1 The sibling artifact hash

`configs/base.yaml` also has a **file-level** sha256, distinct from the registry hash. `docs/PHASE9_FREEZE.md`
§2 records it as a frozen artifact:

| Path | Bytes | SHA256 (first 16) |
|---|---|---|
| `configs/base.yaml` | 10,637 | `88434f7f8f78e2b8` |

The two are related but not identical: the file hash covers the file's **bytes** (including comments
and formatting), while `Config.hash` covers the parsed **data**. Reformatting the YAML moves the file
hash and leaves `Config.hash` untouched. Both are frozen; neither substitutes for the other.

## 24. The rule: editing the config MOVES the hash and INVALIDATES keyed artifacts

This is the central operational fact of the chapter.

### 24.1 The dependency chain

```
configs/base.yaml
      │  (parsed)
      ▼
   Config._data ──► Config.hash ──► recorded in artifacts/change/levir_change_v001/model_metadata.json
      │                                      │
      │ (validated by _validate)             ▼
      ▼                              scripts/eval_change.py compares
  the running system                 current hash vs recorded hash
                                              │
                              drift ──────────┴────────── match
                                │                            │
                                ▼                            ▼
                          exit 3: REFUSES TO SCORE     scoring proceeds
```

### 24.2 The practical consequence

> **A value that "looks wrong" is not casually fixed.**

Concretely, the project's own most-tempting edit is the worked example. `docs/PHASE9_FREEZE.md` §6
records it:

> *"Editing `base.yaml` to point at the checkpoint **moves** the config hash away from
> `78f1e3700da15aa1` — which trips the drift guard and makes `scripts/eval_change.py` exit `3`. The
> frozen benchmark number would be detached from its config."*

The edit in question — populating `change.checkpoint_path` so the trained change head is reachable
from serving — is *architecturally correct*. It is the right thing for the system to do. It is still
refused, because it would detach the published `0.8122` test number from the configuration it was
measured under. The resolution was **not** to edit the config; it was the registry's `builders=`
override, a **call-site** mechanism that injects the checkpoint **without moving the hash**
(`docs/PHASE9_FREEZE.md` §6; `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3 requirement 2).

Owner decision **D-4** records the same discipline for the grounding head: the default head path
*"lives in code"* as `DEFAULT_HEAD_PATH` rather than in `base.yaml`, *"because setting it would have
moved the frozen config hash."* The cross-cutting rule is stated in
`docs/OWNER_DECISIONS_2026-09-23.md`:

> *"Frozen artefacts stay frozen. `Config.hash = 78f1e3700da15aa1` is verified untouched. **New
> defaults live in code, not in the config registry.**"*

### 24.3 There is no single "drifted value" — five renderings, five hashes

A subtlety that catches people who try to *report* the drift: because the hash covers the **whole**
registry, the moved hash depends on the exact string written in. The test docstring records a
measurement of five renderings of the *same logical change*:

| Rendering of the same path | Resulting hash |
|---|---|
| backslashes / a `Path` object | `f4487e1a2cf13733` |
| forward slashes | `f571a5f85372ff54` |
| relative | `b1d8637d6ab733d9` |
| bare filename | `4fcc6a4a4f681c87` |
| directory with a trailing slash | `f5a6ec3db8217218` |

(`tests/test_config.py:59-68`)

The consequence, stated in the test: *"**Do NOT quote a single 'drifted value' — it does not exist.**
… Only the UNMODIFIED hash above is stable and quotable."* A doc that printed one drifted hash would
be printing a fact about one particular string, not about the change.

### 24.4 Why the drift guard is a *guard* and not an inconvenience

`scripts/eval_change.py` exiting `3` on drift is the mechanism that makes the freeze real. Without
it, a config edit would silently change the system under test, and the recorded `0.8122` would be
re-reported as if nothing had changed. `docs/PHASE9_FREEZE.md` §1 states the freeze's meaning:

> *"This freeze is a **claim about the past**: 'these are the numbers this run produced, and here is
> the evidence they have not moved.'"*

The hash is how "they have not moved" is checkable. `docs/PHASE9_FREEZE.md` §3 lists the config hash
among the **frozen invariants** alongside the scene split, the decision threshold and the model
identity — and each of those invariants names where it is enforced.

---

# Part E — `configs/deploy.yaml`, and why it is left undisturbed

## 25. What `configs/deploy.yaml` is

`configs/deploy.yaml` is a **45-line Hugging Face Spaces packaging manifest** from Phase 18. It is
**not** part of the configuration registry, and the file says so in its own header, in capitals:

```yaml
# SatQuery AI — Hugging Face Spaces packaging manifest (Phase 18).
#
# THIS FILE IS NOT PART OF THE CONFIG REGISTRY.
#
# `core/config.py` reads exactly ONE file — `configs/base.yaml` (the module's
# `DEFAULT_CONFIG`) — through a single `yaml.safe_load`. It never globs
# `configs/*.yaml`, so this file is never read by the loader. That matters
# because `Config.hash` is a sha256 over the WHOLE registry; merging any key from
# this file into the registry would move the hash off the recorded benchmark
# value `78f1e3700da15aa1`. The `registry: false` marker below makes the
# non-membership explicit and machine-readable, and
# `scripts/validate_deploy_config.py` asserts it.
```

(`configs/deploy.yaml:1-12`)

It carries a single top-level marker and a `deployment:` block that is a **byte-for-byte copy** of
`configs/base.yaml` lines 282-295:

```yaml
registry: false

deployment:
  platform: huggingface-spaces
  sdk: gradio
  zerogpu: true
  # finding C-8: ZeroGPU does not support torch.compile. Never enable.
  torch_compile: false
  # ZeroGPU free tier = 5 GPU-min/day. Declared duration reserves quota up front.
  gpu_duration_vqa: 20
  gpu_duration_grounding: 45
  gpu_duration_change: 30
  gpu_duration_optical_sar: 45
  cpu_mode_required: true
  lazy_load: true
  cache_max_models: 1
```

(`configs/deploy.yaml:29-45`)

The file's header states the reason the copy is identical:

> *"The `deployment:` block below is a byte-for-byte copy of `configs/base.yaml` lines 282-295. It is
> kept identical on purpose so the two cannot silently drift; `scripts/validate_deploy_config.py`
> enforces the equality key for key."*

## 26. Why it is FROZEN PAPERWORK

`docs/DEPLOYMENT_TOPOLOGY.md` §3.4 states the conclusion, and `docs/DEPLOYMENT_DECISION.md` §4 gives
the reasoning. The reasoning is structural, not sentimental:

> *"the legacy `configs/deploy.yaml` still describes an HF Space + Gradio + ZeroGPU target. That
> manifest is **frozen paperwork** — no Gradio runtime exists in code, and editing it would move
> `Config.hash`. It is left undisturbed."* (`docs/DEPLOYMENT_TOPOLOGY.md` §3.4)

Two independent reasons, and **both paths are closed**:

### 26.1 Reason 1 — `Config.hash` cannot move

The registry hash is `78f1e3700da15aa1` (Part D). `configs/deploy.yaml` carries `registry: false` and
is never read by the loader, so editing the *file* does not move the hash directly. But
`scripts/validate_deploy_config.py` **hard-fails** if the `deployment:` block in `deploy.yaml` differs
key-for-key from `base.yaml`'s:

> *"So changing `zerogpu: true` → `false` in `deploy.yaml` alone fails the validator, and moving
> `base.yaml` to match moves the frozen hash. **Both paths are closed.** The files stay as they
> are."* (`docs/DEPLOYMENT_DECISION.md` §4)

The validator's check is symmetric and names the mismatch in either direction
(`scripts/validate_deploy_config.py`, `_check_deployment_blocks_equal`): a key present in one and
missing from the other is reported, and a key present in both with different values is reported with
both values. So there is no quiet edit: either the file fails the validator, or the registry hash
moves.

The validator's own docstring is careful about what it does and does not prove:

> *"This is a NECESSARY condition for the file being outside the registry — NOT a proof against a
> merge: the registry is built FROM `configs/base.yaml`, so a key merged into `base.yaml` would appear
> on both sides and compare equal. The real guarantee against a merge is the `Config.hash` regression
> guard, `tests/unit/test_deploy_config.py::test_config_hash_regression_guard`, which pins the
> recorded hash."*

That is the honest statement: the key-equality check catches *divergence between the two files*; the
hash regression test catches *a merge*. Neither alone is sufficient, and the design uses both.

### 26.2 Reason 2 — there is no Gradio runtime to conflict with

The manifest declares `sdk: gradio`, but no Gradio application exists in the code:

> *"No `import gradio`, no `gr.Blocks`, no `gr.Interface` and no Gradio entrypoint exists anywhere.
> Gradio appears only as `requirements.txt:36` and the manifest value `sdk: gradio`
> (`configs/base.yaml:284`)."* (`docs/DEPLOYMENT_DECISION.md` §4)

The one ZeroGPU code path sits inside a function that is **never applied to a route**:

```python
def decorate_gpu(task: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    ...
    spaces = _spaces_module()
    if spaces is None or not hasattr(spaces, "GPU"):
        def _identity(fn): return fn
        return _identity
    return spaces.GPU(duration=duration)
```

(`app/space_app.py:144-165`)

The routes use plain `@api.get` / `@api.post` (`app/space_app.py:521`, `:549`, `:555`, `:661`), and
the real entrypoint is FastAPI's `build_space_app()` (`app/space_app.py:409`). The manifest therefore
describes a Space that does not exist in code:

> *"Conclusion: the frozen contract describes a Gradio Space that does not exist in code. It is frozen
> paperwork, not a competing deployment."* (`docs/DEPLOYMENT_DECISION.md` §4)

Because there is nothing for it to conflict with, there is no benefit to editing it — and a real cost
(reason 1). So it stays.

### 26.3 What the manifest deliberately does NOT specify

The header lists the absences explicitly, so a reader does not mistake them for oversights:

> *"Deliberately ABSENT (the freeze does not specify these): a hardware SKU, an SDK version pin, a
> requirements filename, the Space entrypoint/`app_file`, Space visibility/owner, and a Python
> version."* (`configs/deploy.yaml:24-27`)

## 27. The `gpu_duration_*` values and the 5 GPU-minute budget

The `deployment` block's four durations encode a quota constraint measured in Phase 0:

| Task | `gpu_duration_*` | Source |
|---|---|---|
| `vqa` / `caption` | 20 s | `configs/deploy.yaml` |
| `grounding` | 45 s | `configs/deploy.yaml` |
| `change` | 30 s | `configs/deploy.yaml` |
| `optical_sar` | 45 s | `configs/deploy.yaml` |
| `change_vqa` | 30 s | follows `change` — shares the STANet detector |

(`docs/DEPLOYMENT_ARCHITECTURE.md` §3.4)

**Why the durations exist at all.** `docs/PHASE0_CONTRACT_VALIDATION.md` §5.1 records the verified
ZeroGPU constraints:

| Constraint | Verified value |
|---|---|
| SDK | **Gradio only** (Docker/Static cannot schedule onto ZeroGPU) |
| Decorator | `@spaces.GPU(duration=N)`, default **60 s** |
| Free account quota | **5 GPU-minutes/day** |
| Quota window | 24 h from first use, not calendar day |
| `torch.compile` | **not supported** (use AoTI, torch 2.8+) |

Finding **C-8** records the operational consequence the plan omitted:

> *"the plan never states the operational consequence: at 5 min/day, a single VQA call declaring the
> 60 s default consumes **1/5 of a user's entire daily budget**."*
> (`docs/PHASE0_CONTRACT_VALIDATION.md` §5.2)

The resolution is that *"every GPU-decorated function declares an explicit, realistic `duration`"* —
hence the four frozen values. The registry states the same rationale inline: *"ZeroGPU free tier = 5
GPU-min/day. Declared duration reserves quota up front."* (`configs/base.yaml:288`).

## 28. The `change_vqa` detail — a duration that has no key of its own

`app/space_app.py` transcribes the durations into a Python dict:

```python
#: ZeroGPU duration per task, from `configs/deploy.yaml` -- the FROZEN values,
#: not new guesses. `change_vqa` has no key of its own and reuses `change`,
#: because adding a key would move `Config.hash` off `78f1e3700da15aa1`
#: (`docs/DEPLOYMENT_ARCHITECTURE.md` section 3.4).
GPU_DURATIONS: dict[str, int] = {
    "vqa": 20,
    "caption": 20,
    "grounding": 45,
    "change": 30,
    "optical_sar": 45,
    "change_vqa": 30,
}
```

(`app/space_app.py:105-116`)

Note the shape: **six tasks, four `gpu_duration_*` keys.** `vqa` and `caption` share the `vqa` budget;
`change_vqa` shares the `change` budget. And `change_vqa` is the interesting one, because the reason
is the hash:

> *"`change_vqa` has **no** `gpu_duration_change_vqa` key, and adding one would move `Config.hash`.
> Repurposing the `change` budget is the choice that avoids that, and it is recorded here as a
> decision rather than presented as a plan fact."* (`docs/DEPLOYMENT_ARCHITECTURE.md` §3.4)

The justification for the *repurposing itself* is also recorded: `change_vqa` *"follows `change` — it
shares the STANet detector."* So the shared budget is not arbitrary; it reflects that the two tasks
run the same detector. The config-freeze reasoning and the engineering reasoning agree here, which is
why this is a good example rather than a tension.

`decorate_gpu` refuses an undeclared task rather than guessing a duration:

```python
if task not in GPU_DURATIONS:
    raise KeyError(
        f"no gpu_duration_* is declared for {task!r}; add it to "
        f"configs/deploy.yaml (which moves Config.hash) or map it to an "
        f"existing task. Do not guess a duration."
    )
```

(`app/space_app.py:152-157`)

The error message names both options and the cost of the first — a small, self-documenting instance of
the chapter's central rule.

## 29. The ZeroGPU decoration has never executed

The honest status of the ZeroGPU path is stated in the entrypoint's own docstring:

> *"`spaces` (the ZeroGPU decorator package) is not installed in this environment … The decoration is
> therefore applied **conditionally** … The consequence is recorded in
> `docs/PHASE19_FINAL_HARDENING.md`: the ZeroGPU decoration has **never executed** here. It is
> specified from finding C-8 and the frozen `gpu_duration_*` values, and that is all it is."*
> (`app/space_app.py:31-42`)

This is `IMPLEMENTED (not run)`: the code exists, the durations are frozen and transcribed, and the
decoration has never been exercised. `docs/architecture/02-deployment-topology.md` §15 lists
"ZeroGPU decoration execution | `NOT RUN`" in the same terms. The conditional application is itself
correct rather than a workaround, because `configs/deploy.yaml` sets `cpu_mode_required: true` — a CPU
run must work, and it does, via the identity decorator.

---

# Part F — The frozen values a reader might want to change

This is the table the chapter exists to provide. Every row is a value that a reader, seeing it in
`configs/base.yaml`, might reasonably want to change — and the reason it is frozen.

## 30. Frozen values, and why

| Key | Frozen value | Why it is frozen | Enforced by |
|---|---|---|---|
| `croma.image_resolution` | `120` | CROMA asserts `% 8 == 0`; native resolution → 225 patches. A non-multiple of 8 violates an assertion inside the vendored model. | **C-7 guard** (`core/config.py:97-105`) |
| `croma.optical_channels` | `12` | CROMA's `s2_channels` is fixed at 12; the patch embedding is a fixed-shape `Linear`. | **channel guard** (`core/config.py:164-166`) |
| `croma.sar_channels` | `2` | CROMA's `s1_channels` is fixed at 2 (VV, VH). | **channel guard** (`core/config.py:167-168`) |
| `fusion.input_dim` | `2318` | Must equal `3·768 + 12 + 2`; a mismatch is a `Linear` shape error after CROMA has run. | **C-1 guard** (`core/config.py:152-162`) |
| `training.precision` | `fp16` | The target accelerator (T4, SM 7.5) has no bf16 tensor cores. | **C-6 guard** (`core/config.py:107-112`) |
| `vlm.processor_longest_edge` | `512` | The processor default (2048) upscales and splits every tile into 17 sub-images and 1142 tokens (measured). | **F5-2 guard** (`core/config.py:121-142`) |
| `vlm.prompt_must_use_chat_template` | `true` | SmolVLM raises `ValueError` without one `<image>` token per image. | **F5-3 guard** (`core/config.py:144-150`) |
| `grounding.encoder_projected_dim` | `512` | RemoteCLIP's `visual.proj` is (768, 512); the comparison space is the **projected** dim, not the transformer width. | **P7-1 guard** (`core/config.py:183-196`) + `remoteclip.py` load-time assert |
| `grounding_head.feature_dim` | `2048` | `4 · 512`; a mismatch is a **silent** shape error at the similarity step, after patch features are cached. | **P7-1 guard** (`core/config.py:190-196`) |
| `deployment.torch_compile` | `false` | ZeroGPU does not support `torch.compile` (use AoTI, torch 2.8+). | **C-8 guard** (`core/config.py:114-119`); deploy validator |
| `deployment.cpu_mode_required` | `true` | CPU must work; ZeroGPU is an accelerator, not a dependency. | deploy validator (`REQUIRED_BOOL`) |
| `grounding.image_size` | `224` | **RESOLVED by measurement**: 448 lost on mean best IoU (−0.0147), every recall threshold, and latency (1.59×) over 16,159 VRSBench records. | test + `resolution_frozen: true` |
| `grounding.resolution_frozen` | `true` | Asserts the resolution question is *decided*; a stale "experiment" flag would invite a re-run. | test |
| `change.sa_mode` | `PAM` | Upstream STANet's two verified modes are `BAM` and `PAM`; an unrecognised mode selects neither. | **change guard** (`core/config.py:211-212`) |
| `change.tile_size` | `256` | STANet's verified convention: 256×256 **non-overlapping** patches. | test |
| `change.bce_weight` + `change.dice_weight` | `0.5 + 0.5` | STANet's verified loss is `0.5·BCE + 0.5·Dice`. | test (sums to 1.0) |
| `image.tile_size` | `512` | The tile size the VLM processor pin is tied to, and the unit of the tiling policy. | referenced by **F5-2 guard** |
| `image.top_k_tiles` | `4` | Plan §9.1 tile policy: whole-image thumbnail first, then top-K tiles. | **tiling guard** (`≤ max_tiles`) |
| `router.tasks` (incl. `unsupported`) | 6 classes | `unsupported` is the router's "not a satellite question" outcome; removing it forces every query into a specialist. | **ontology guard** (`core/config.py:198-206`) |
| `router.num_tasks` | `6` | The classifier's output width; must equal the ontology length. | **ontology guard** (`core/config.py:202-206`) |
| `evaluation.official_aggregate_weights` | `null` | The uploaded specification prohibits inventing an official aggregate formula. | test |
| `evaluation.hidden_data_access` | `false` | Evaluation isolation. | test |
| `evaluation.leakage_split_key` | `scene_id` | Scene-level leakage isolation is a non-negotiable. | test |
| `project.seed` | `42` | Reproducibility. | `Config.seed` |
| `deployment.gpu_duration_*` | `20/45/30/45` | Derived from the verified 5 GPU-min/day ZeroGPU budget. | deploy validator (positive int) |

## 31. The hash itself

| Item | Frozen value | Why |
|---|---|---|
| `Config.hash` | **`78f1e3700da15aa1`** | Recorded in `artifacts/change/levir_change_v001/model_metadata.json`; `scripts/eval_change.py` exits `3` on drift. |
| `configs/base.yaml` file sha256 (first 16) | `88434f7f8f78e2b8` | A frozen Phase-9 artifact (`docs/PHASE9_FREEZE.md` §2); 10,637 bytes, mtime 2026-09-16 19:30. |

---

# Part G — What is NOT frozen: open for tuning

The freeze is a claim about **which** values are decided, not a claim that nothing may ever change.
The master plan draws the line explicitly, in two adjacent sections, and the repository's own
structure follows it.

## 32. The plan's own split

The plan separates immutable decisions from tunables in **§67**, **§68** and **§69**.

### 32.1 §67 — Immutable decisions

The implementation model **must not redesign** these (plan §67):

```
[ ] modular-monolith architecture
[ ] tiny NLP intent router
[ ] deterministic policy engine
[ ] common specialist interface
[ ] SmolVLM VLM layer
[ ] RemoteCLIP grounding path
[ ] STANet-style change path
[ ] CROMA optical-SAR path
[ ] shared evidence engine
[ ] calibrated confidence
[ ] common result schema
[ ] execution trace
[ ] leakage isolation
```

These are **structural**. Note that the list is about *components and interfaces*, not numbers. The
config registry is the mechanism by which the numbers those components depend on are pinned.

### 32.2 §68 — Variables open for tuning

The plan's tunable list (plan §68):

```
LoRA rank
VLM learning rate
router adapter dimension
router confidence threshold
grounding head architecture
grounding learning rate
change threshold
change loss weighting
CROMA fusion head width
tile size
tile overlap
top-K tile count
confidence calibration temperature
```

### 32.3 §69 — Variables requiring experimental optimization

The plan gives explicit ranges for the tunables it expects to be searched (plan §69):

| Component | Variable | Range |
|---|---|---|
| Router | hidden dimension | `64–256` |
| Router | dropout | `0–0.3` |
| Router | confidence | `0.60–0.90` |
| Grounding | head width | `256–1024` |
| Grounding | learning rate | `5e-5–2e-4` |
| Grounding | NMS | `0.4–0.6` |
| Change | threshold | `0.30–0.70` |
| Change | minimum component | `16–128 px` |
| CROMA | fusion width | `256–1024` |
| CROMA | dropout | `0–0.3` |

The plan's selection rule is one line: **"Selection: validation only."**

### 32.4 Where the registry's values sit against the plan's ranges

The registry's tunables fall inside (or beside) the plan's ranges. Cross-referencing them is
instructive, because it shows the registry is *the plan's tunables, realised*:

| Plan variable | Registry key | Registry value | In plan range? |
|---|---|---|---|
| Router hidden dimension | `router.hidden_dim` | `128` | ✅ within `64–256` |
| Router dropout | `router.dropout` | `0.10` | ✅ within `0–0.3` |
| Router confidence | `router.confidence_threshold` | `0.70` | ✅ within `0.60–0.90` |
| Grounding head width | `grounding_head.hidden_dim` | `512` | ✅ within `256–1024` |
| Grounding learning rate | `grounding_training.learning_rate` | `0.0001` (1e-4) | ✅ within `5e-5–2e-4` |
| Grounding NMS | `grounding.nms_iou` | `0.50` | ✅ within `0.4–0.6` |
| Change threshold | `change.threshold` | `0.50` | ✅ within `0.30–0.70` |
| Change min component | `change.min_component_pixels` | `32` | ✅ within `16–128 px` |
| CROMA fusion width | `fusion.hidden_dim` | `512` | ✅ within `256–1024` |
| CROMA dropout | `fusion.dropout` | `0.2` | ✅ within `0–0.3` |
| LoRA rank | `training.lora_rank` | `16` | (no range given) |
| VLM learning rate | `training.vlm_learning_rate` | `0.0002` | (no range given) |
| Tile size | `image.tile_size` | `512` | (no range given) |
| Tile overlap | `image.tile_overlap` | `128` | (no range given) |
| Top-K tile count | `image.top_k_tiles` | `4` | (no range given) |
| Calibration temperature | — | `UNKNOWN — not established from the available evidence` | the artifact is `calibration_v001.json`; no temperature is declared in `base.yaml` |

Two honest notes on that table:

1. **The plan's tunables are the registry's values.** Every range-bearing tunable the plan names has
   a registry key with a value inside the stated range. This is the plan's intent realised, not a
   coincidence — the registry was written from the freeze, which was written from the plan.
2. **The calibration temperature is not in the registry.** `confidence.temperature_scaling: true` and
   `confidence.calibration_file: calibration_v001.json` are the registry's confidence keys
   (`configs/base.yaml:231-233`); the fitted temperature itself lives in the artifact file. Whether
   that temperature is a "tunable" in the plan's §68 sense is `UNKNOWN — not established from the
   available evidence`; the artifact is covered in
   `docs/architecture/06-evidence-and-confidence.md` §8–§10, including the measured result that
   calibration made ECE **worse** (0.013755 → 0.014929) and is retained only because it is in the
   frozen config.

## 33. The real tension: tunable ≠ free

Here is the part a reader must not miss. **A value can be on the plan's tunable list and still be
frozen in practice.**

The plan's §68 list says `change threshold` is open for tuning. The registry has
`change.threshold: 0.50`. But `docs/PHASE9_FREEZE.md` §5 records that the threshold lever is
**CLOSED**:

> *"The val-only sweep scored 19 thresholds. Pooled IoU peaks at 0.35–0.40 (**0.8239**) against
> **0.8232** at 0.50 — a gain of **+0.0007**. … the last five validation epochs span 0.8213–0.8232, a
> spread of **0.0019**, so the entire available threshold gain is **0.37× the epoch-to-epoch noise**.
> A gain smaller than the run's own variance is not a finding. **0.50 is retained. This hypothesis is
> eliminated, not deferred.**"*

So there are **three** states a value can be in, not two:

| State | Meaning | Example |
|---|---|---|
| **Structurally frozen** | a guard rejects any other value | `fusion.input_dim`, `training.precision` |
| **Measured-frozen** | a value was chosen by measurement and the question is closed | `grounding.image_size` (224), `change.threshold` (0.50) |
| **Open for tuning** | genuinely searchable, with a validation-only selection rule | the plan's §69 ranges |

And a fourth, operational state that overrides all three:

| State | Meaning | Example |
|---|---|---|
| **Hash-pinned** | editing it moves `Config.hash` and detaches a published number | *any* key in `configs/base.yaml` |

The last row is the reason this chapter exists. A tunable that is "open" is still an edit to
`configs/base.yaml`, and an edit to `configs/base.yaml` still moves the hash. The supported way to
change a value without detaching a benchmark is **not** to edit the registry — it is to pass
`overrides` to `load_config`, or to use a call-site mechanism like the registry's `builders=`
override, and to accept that the resulting run has a **different** hash and is therefore a **different**
measurement.

`docs/OWNER_DECISIONS_2026-09-23.md` states this as a cross-cutting rule:

> *"**Frozen artefacts stay frozen.** `Config.hash = 78f1e3700da15aa1` is verified untouched. New
> defaults live in **code**, not in the config registry."*

## 34. The override surfaces that DO exist

If the registry is frozen, what is not? Three legitimate surfaces:

| Surface | Mechanism | Effect on the hash | Example |
|---|---|---|---|
| **Programmatic overrides** | `load_config(overrides={...})` | **moves it** — a different registry, honestly different | `load_config(overrides={"fusion": {"hidden_dim": 1024}})` |
| **Environment overrides** | `SATQUERY_PRECISION`, `SATQUERY_TORCH_COMPILE` | **moves it** — they write into `data` before hashing | `SATQUERY_PRECISION=fp32` |
| **Call-site injection** | the registry's `builders=` override | **leaves it bit-identical** | `build_serving_registry()` wiring the change head |

The first two are *config changes* and the hash moves with them — which is correct, because the run
really is different. The third is the one that changes **behaviour** without changing the
**configuration**, and it is the mechanism the project uses when a trained artifact needs to be wired
in without detaching the benchmark (`docs/PHASE9_FREEZE.md` §6;
`docs/DEPLOYMENT_ARCHITECTURE.md` §3.3 requirement 2).

**Environment variables outside the loader** are a fourth, weaker surface: `SATQUERY_DEVICE` (Part
B.6) changes the device without touching the registry, and the asset-store variables (Part A.2.2)
change deployment sizing. These are not config-registry values and therefore cannot move the hash —
they are the "deployment state" half of the Part A.2.2 rule of thumb.

---

# Part H — Honest boundaries

## 35. What is `NOT RUN` / `OPEN` / `UNKNOWN` for this topic

| Item | Status | Note |
|---|---|---|
| `Config.hash` frozen value `78f1e3700da15aa1` | **`VERIFIED`** | re-run by execution while writing this chapter; matches `tests/test_config.py:43` |
| Every `_validate` guard firing on a bad value | **`VERIFIED`** | each guard has a `pytest.raises` test in `tests/test_config.py` |
| `configs/deploy.yaml` left undisturbed | **`VERIFIED`** | `registry: false`; validator enforces key-for-key equality |
| ZeroGPU decoration execution | **`NOT RUN`** | *"the ZeroGPU decoration has **never executed** here"* (`app/space_app.py`) |
| The `5 GPU-min/day` quota against a live Space | **`NOT RUN`** | no live Space; the budget is a verified platform constraint, not a measured consumption |
| `SATQUERY_TORCH_COMPILE=true` observed at runtime | **`NOT RUN`** | the guard is tested in-process; no deployment was run with the variable set |
| `SATQUERY_PRECISION` used to switch precision in a real run | **`UNKNOWN — not established from the available evidence`** | the override is implemented; no run record was found that exercised it |
| The calibration temperature as a tunable | **`UNKNOWN — not established from the available evidence`** | the value lives in the artifact, not the registry |
| Whether `change.checkpoint_path` ever appeared in `base.yaml` | **absent** | verified by reading all 294 lines; the registry declares it optional and it resolves to `None` |
| A live config with `image.top_k_tiles` as the binding constraint | **`UNKNOWN — not established from the available evidence`** | the guard is tested; no run record was found |
| `Config.hash` computed on a registry other than `configs/base.yaml` in production | **`UNKNOWN — not established from the available evidence`** | `get_config()` reads the default path; no production override was found |
| The `Config` docstring's "immutable" claim as a hard guarantee | **`IMPLEMENTED`** | no mutating API exists; `_data` is reachable via `__getitem__`, and no code path mutates it |
| The CROMA mask contradiction (C-1) as resolved in code | **`RESOLVED`** (design) | the mask routes to the fusion head; the fusion dimension is guarded. Whether the fusion head has been trained is a Phase-12 question (`docs/OWNER_DECISIONS_2026-09-23.md` D-1) |
| `deployment.platform: huggingface-spaces` describing the live deployment | **superseded** | the active topology is Render + Codespace + Cloudflare Pages (`docs/DEPLOYMENT_TOPOLOGY.md`); the manifest is frozen paperwork |

## 36. Where the evidence lives

| Claim | Source |
|---|---|
| The loader, all guards, the hash, the access methods | `core/config.py` (275 lines; guards at `:94-222`) |
| The registry and every value | `configs/base.yaml` (294 lines) |
| `_deep_merge` semantics | `core/config.py:33-40` |
| Env overrides `SATQUERY_PRECISION` / `SATQUERY_TORCH_COMPILE` | `core/config.py:261-265` |
| `get_config()` singleton | `core/config.py:270-273` |
| `device_preference` + `SATQUERY_DEVICE` | `core/config.py:86-91`, `:237-243` |
| `ConfigError` taxonomy | `core/config.py:28-30` |
| C-7 `% 8` rule | `core/config.py:97-105`; `docs/PHASE0_CONTRACT_VALIDATION.md` §1.1 |
| C-6 precision / T4 SM 7.5 | `core/config.py:107-112`; `docs/PHASE0_CONTRACT_VALIDATION.md` §7; `configs/base.yaml:254-257` |
| C-8 `torch.compile` forbidden | `core/config.py:114-119`; `docs/PHASE0_CONTRACT_VALIDATION.md` §5.1 |
| F5-2 / C-3 processor upscaling, the 17 sub-images and 1142 tokens | `core/config.py:121-142`; `docs/PHASE5_VLM_CONTRACT.md` §headline finding |
| F5-3 chat template | `core/config.py:144-150`; `docs/PHASE5_VLM_CONTRACT.md` §F5-3 |
| C-1 fusion dim 2318, mask to the fusion head | `core/config.py:152-162`, `:224-231`; `docs/PHASE0_CONTRACT_VALIDATION.md` §1.2–§1.3; `docs/ARCHITECTURE_FREEZE.md` §2.5 |
| CROMA fixed channel counts 12 / 2 | `core/config.py:164-168`; `docs/PHASE0_CONTRACT_VALIDATION.md` §1.1 |
| P7-1 feature dim 2048, the silence argument | `core/config.py:170-196`; `specialists/grounding/remoteclip.py:15-19`, `:38-42`, `:195-201` |
| Router ontology guards | `core/config.py:198-206`; `docs/ARCHITECTURE_FREEZE.md` §2.1 |
| Change `encoder` / `sa_mode` guards | `core/config.py:208-212`; `docs/PHASE0_CONTRACT_VALIDATION.md` §4.1 |
| Tiling guard | `core/config.py:214-216`; `configs/base.yaml:16-24` |
| The frozen hash `78f1e3700da15aa1` | `tests/test_config.py:43`; `docs/PHASE9_FREEZE.md` §3; `docs/OWNER_DECISIONS_2026-09-23.md`; `docs/DEPLOYMENT_ARCHITECTURE.md` §3.3 |
| The five drifted-hash renderings | `tests/test_config.py:59-68` |
| The `configs/base.yaml` file sha256 `88434f7f8f78e2b8` | `docs/PHASE9_FREEZE.md` §2 |
| `configs/deploy.yaml` is not registry material | `configs/deploy.yaml:1-12`; `scripts/validate_deploy_config.py` |
| Why the manifest is frozen paperwork | `docs/DEPLOYMENT_TOPOLOGY.md` §3.4; `docs/DEPLOYMENT_DECISION.md` §4 |
| ZeroGPU duration mapping + `change_vqa` reuses `change` | `docs/DEPLOYMENT_ARCHITECTURE.md` §3.4; `app/space_app.py:105-116`, `:144-165` |
| ZeroGPU verified constraints (5 GPU-min/day, Gradio-only, no `torch.compile`) | `docs/PHASE0_CONTRACT_VALIDATION.md` §5.1–§5.2 |
| New defaults live in code, not the registry (D-4) | `docs/OWNER_DECISIONS_2026-09-23.md` D-4 and cross-cutting rule 4 |
| The threshold lever is closed | `docs/PHASE9_FREEZE.md` §5 |
| The plan's immutable / tunable / experimental split | plan §67, §68, §69 |
| The guard tests | `tests/test_config.py` (348 lines) |
| The deploy validator's required values | `scripts/validate_deploy_config.py` (`REQUIRED_BOOL`, `REQUIRED_INT`, `REQUIRED_STR`) |

---

**Next:** [08 API contract](08-api-contract.md) — the four endpoints, the `ResultEnvelope` and error
envelopes, the error codes, and the transport headers.
