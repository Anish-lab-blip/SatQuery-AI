# SatQuery AI — Testing, Verification and Evidence Integrity

**Status of this document.** This is the testing chapter of the public SatQuery AI release. It enumerates
the actual suites that exist, states what each one pins, and reports the full-suite result **including its
failures**. Every test file, count, command and code excerpt below was read from the repository. Where a
value could not be established it is written
`UNKNOWN — not established from the available evidence`.

**Status vocabulary** (see `DOCS_STYLE_GUIDE.md` §2): `IMPLEMENTED` · `VERIFIED` · `MEASURED` ·
`ATTEMPTED` · `NOT RUN` · `BLOCKED` · `DEFERRED` · `REJECTED` · `OPEN` · `RESOLVED` · `CLOSED`.

**The one rule this chapter follows without exception:** a green suite is not evidence about a property
nobody wrote an assertion for, and a failure is never reclassified as a pass. The project learned this the
hard way twice — a guard that pinned a *defect* as the specification
(`docs/DEPLOYMENT_ARCHITECTURE.md` §5.5) and a guard whose assertion was satisfied by an upstream fix so it
could not see the downstream site (`F-15b`). Both are recorded below rather than smoothed over.

---

## Table of contents

- [0. How to read this document](#0-how-to-read-this-document)
- [1. Testing philosophy: tests pin contracts, not implementation](#1-testing-philosophy-tests-pin-contracts-not-implementation)
- [2. The suite inventory](#2-the-suite-inventory)
- [3. The doc-guard tests: keeping documentation and code in sync](#3-the-doc-guard-tests-keeping-documentation-and-code-in-sync)
- [4. The evidence engine: purity, determinism and the reproducibility property](#4-the-evidence-engine-purity-determinism-and-the-reproducibility-property)
- [5. The frontend live-wiring suite](#5-the-frontend-live-wiring-suite)
- [6. The doc/frontend suite](#6-the-docfrontend-suite)
- [7. The full `tests/unit` result, and the 5–6 environmental failures](#7-the-full-testsunit-result-and-the-56-environmental-failures)
- [8. The live-validation harness and its integrity discipline](#8-the-live-validation-harness-and-its-integrity-discipline)
- [9. Verdict-independent evidence: recomputing from raw records](#9-verdict-independent-evidence-recomputing-from-raw-records)
- [10. How to run the tests](#10-how-to-run-the-tests)
- [11. What is NOT tested](#11-what-is-not-tested)
- [12. Open items and evidence index](#12-open-items-and-evidence-index)

---

## 0. How to read this document

### 0.1 The three evidence classes

The repository encodes an evidence class on every test via a pytest marker (`pytest.ini`):

```ini
markers =
    unit: fast, no I/O, no server, no network. Evidence about a component in isolation.
    integration: exercises two or more real components wired together in-process.
    smoke: a minimal end-to-end path run against real local artifacts, not a mock.
    real_inference: ran the real model on real inputs in this environment.
```

The comment above the markers states the purpose exactly, and it is the reason the markers exist rather
than being decorative:

> *"The markers exist to make a test's EVIDENCE CLASS explicit: a unit test never claims a deployment was
> exercised, and no test may be reported as 'real inference' unless it ran the real model on real data in
> this environment."*

And one deliberate non-marker, quoted in full because it is a subtle but important choice:

> *"`environment_blocked` is not a test marker: a blocked path is reported in the STEP 7 report rather than
> encoded as a permanently-skipped test, because **a skip can be mistaken for coverage**."*

That sentence is the spine of this chapter. A skip that looks like coverage is the failure mode the marker
design is built to avoid.

### 0.2 What "verified" means here

| Term | Means | Example |
|---|---|---|
| **Unit** | one component, no I/O, no server, no network | `tests/unit/test_evidence_engine.py` |
| **Integration** | two or more real components wired in-process | `tests/integration/test_step7_backend_chain.py` |
| **Smoke** | a minimal end-to-end path over real local artifacts | `tests/unit/phase6_smoke.py` (standalone) |
| **Real inference** | the real model on real inputs, in this environment | — see §11 |
| **Live validation** | the deployed stack driven by a browser | §8 |

The distinction matters most at the boundary between **integration** and **live**. `docs/API_CONTRACT.md`
§8 states the boundary plainly: *"no test in this repository dials a network address"*, and it points at
`docs/ITEM5_INTEGRATION_SUITE_SCOPE.md`, which *"records what the 26-test integration suite proves (the
app's boundary, in-process) and what only a live deployment can prove (reachability, cold start, memory
ceilings)."* A green `tests/integration` run is therefore **not** evidence about a deployment.

### 0.3 The two defects that shaped the discipline

Two incidents are referenced repeatedly below, because they are why this chapter is written the way it is:

1. **A guard that pinned the defect.** The test
   `tests/unit/test_controller.py::test_trace_records_inputs_and_query` asserted
   `envelope.trace.inputs == list(request.assets)` — which, after the upload handler rewrote handles into
   paths, pinned a **filesystem-path disclosure as the specification**. The lesson, recorded in
   `docs/DEPLOYMENT_ARCHITECTURE.md` §5.3: *"A guard is only as strong as the behaviour it pins, and this
   one pinned the defect — a reminder that a green suite is not evidence about a property nobody wrote an
   assertion for."*
2. **A guard that could not fail.** The `to_trace()` seam's scrub had **no covering test**: falsifying it
   alone left the producer-side guard **GREEN**, because the producer had already scrubbed the value the
   guard inspected. Recorded as F-15b in `docs/DEPLOYMENT_ARCHITECTURE.md` §5.5, and remedied by
   `test_to_trace_scrubs_a_detail_that_arrives_from_any_constructor`, which constructs the entry
   **directly** through the constructor.

Both are the same shape: *a guard whose assertion is satisfied by something other than the code under
test.*

---

## 1. Testing philosophy: tests pin contracts, not implementation

### 1.1 The philosophy, as the suites state it

The repository's test docstrings state a consistent philosophy, and the consistency is the point. Four
representative statements, each quoted from a module docstring:

**Doc-guards pin examples, not prose** (`tests/unit/test_api_contract_doc.py`):

> *"These tests validate the EXAMPLES, not the prose. They cannot detect a wrong sentence (e.g. a
> mis-stated latency budget). What they guarantee is that a frontend developer who copies an example
> verbatim gets a request the server accepts -- which is the failure mode that actually costs time."*

**Guards pin the route, not the validity** (`tests/unit/test_frontend_live_wiring.py`):

> *"It also pins the `chang` word-boundary defect: `\bchang\b` cannot match 'changed', so the page's own
> default question ('What changed here?') routed to `vqa` instead of the change path. That is invisible to
> any test that only checks 'is this a valid enum value' — both branches produce a valid value. The test
> therefore asserts the ROUTE, not just the validity."*

**Tests read shipped source when there is no importable symbol** (`tests/unit/test_frontend_live_wiring.py`):

> *"Why these are read from the source text: the frontend is plain ES5 with no build step and no module
> exports, so there is no importable symbol. Parsing the source is the only way to assert the shipped
> bytes. The parsing is anchored on named tokens rather than line numbers, so it does not silently pass if
> the file is restructured."*

**A runbook must not assert facts the repository contradicts** (`tests/unit/test_runbook_doc.py`):

> *"A runbook is worse than useless if a copied command fails or a quoted number was never measured: the
> operator concludes the *system* is broken rather than the *document*."*

### 1.2 The three anti-patterns the suites are written to avoid

| Anti-pattern | What it looks like | How the suites avoid it |
|---|---|---|
| **Pinning the defect** | an assertion that encodes today's buggy behaviour as correct | guards are **falsified** before being trusted: the fix is reverted and the guard is shown to fail (§1.3) |
| **Vacuous guards** | an assertion satisfied by an upstream layer, so the downstream site is untested | the seam is tested by constructing the object **directly** through its constructor (F-15b) |
| **Measurement by prose** | a text search that is tripped by documentation *about* a defect | guards walk the **parsed** examples, not the document text (`test_api_contract_doc.py::test_no_contract_example_fabricates_an_artifact_uri`) |

The third is worth quoting in full, because it is a rule about what a guard must fail on
(`tests/unit/test_api_contract_doc.py`):

> *"This walks the PARSED examples rather than the document text on purpose. The prose in section 2.4
> deliberately quotes the old `artifact://run/9f2c.../change_map.png` example when explaining why it was
> removed, so a text search would be tripped by documentation *about* the removal -- which is measuring
> the wrong thing. A guard must fail on the defect, not on a description of the defect."*

### 1.3 Falsification is part of the discipline

A guard is not trusted until it has been shown to **fail** against the defect it guards. The pattern is
recorded with measurements throughout the project. The clearest instance is the F-15 four-site
falsification (`docs/DEPLOYMENT_ARCHITECTURE.md` §5.5), where each site was reverted individually and the
guards re-run:

| Site reverted | `test_a_construction_failure_publishes_no_filesystem_path` | `test_to_trace_scrubs_a_detail_that_arrives_from_any_constructor` |
|---|---|---|
| `registry.py:597` (`_failure_entry`) | **FAILS** | passes *(correctly — it tests the seam, not the producer)* |
| `registry.py:309` (`to_trace`) | **passes — blind** | **FAILS** |

The "passes — blind" cell is the F-15b finding: the producer guard cannot see the seam, because the
producer has already scrubbed the value. The two guards are therefore **not duplicates**, and the table is
the evidence.

A second falsification table covers the two controller sites (`docs/DEPLOYMENT_ARCHITECTURE.md` §5.5):

| Site reverted | `test_a_failed_step_publishes_no_filesystem_path` — assertion that fires |
|---|---|
| `controller.py:801` (`_failure_evidence`) | line 590, `payload["message"]` |
| `controller.py:969` (`_warnings`) | line 582, `warnings[]` |

And a third, for the trace-path fix (`docs/DEPLOYMENT_ARCHITECTURE.md` §5.4):

| Step | Result |
|---|---|
| Revert only the `PARSE` site, run the module | **1 failed, 55 passed** at `tests/unit/test_controller.py:1153` |
| Restore byte-exact, verify hash | `0c00ae6f54300668122728dc706344044bc19d48328ceead30323c7147e04397` |
| Module re-run | **56 passed** |
| Is the guard vacuous? | **No** — the shape check only discriminates because the fixture writes real GeoTIFFs, and the name-equality check rejects the revert independently |

The "restore byte-exact, verify hash" row is the anti-tamper step: after reverting a site to falsify a
guard, the file is restored and its hash checked, so the falsification itself cannot leave the tree in a
modified state.

### 1.4 Determinism is asserted, not hoped for

Several suites pin **determinism** as a first-class property rather than a nice-to-have:

| Suite | Determinism property |
|---|---|
| `tests/unit/test_evidence_engine.py` | the same claims produce a byte-identical, identically-ordered collection even when uuids differ between processes (§4) |
| `tests/unit/test_router_threshold_sweep.py` | validation-only sweep contracts |
| `tests/unit/test_change_threshold_sweep.py` | validation-only sweep contracts |
| `tests/unit/test_report_generator.py` | *"pure, deterministic, fixtures only"* |
| `tests/unit/test_vlm_evaluate.py` | the predeclared acceptance rule |
| `tests/unit/test_fusion_seed_variance.py` | the predeclared seed-variance pass |

The determinism claim that matters most is the evidence engine's, because it is what makes the live
validation's verdict-recomputation property hold (§9).

---

## 2. The suite inventory

### 2.1 The layout

`pytest.ini` sets the roots:

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -q --tb=short
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::rasterio.errors.NotGeoreferencedWarning
```

`tests/` contains eight directories and two top-level modules:

| Location | Contents | Evidence class |
|---|---|---|
| `tests/unit/` | 98 Python files — 94 collected test modules + `__init__.py`, `conftest.py`, `_qa_probe_tokens.py`, `phase6_smoke.py` | `unit` (mostly) |
| `tests/e2e/` | `test_demo.py`, `test_gate4_e2e.py`, `_demo_probe.py` | end-to-end |
| `tests/geospatial/` | `test_transform.py` | `unit` |
| `tests/integration/` | `test_step7_backend_chain.py`, `test_step7_r02_serving.py` | `integration` |
| `tests/leakage/` | `test_leakage.py` | `unit` |
| `tests/model/` | `test_vlm_contract.py`, `test_vqa_quality_gate.py` | `unit` / `integration` |
| `tests/routing/` | `test_router.py` | `unit` |
| `tests/` | `test_config.py`, `test_schemas.py` | `unit` |

`tests/unit` totals **49,673 lines** across its 98 Python files. Two files there are **deliberately not
collected**:

| File | Why it is not collected |
|---|---|
| `tests/unit/_qa_probe_tokens.py` | *"QA scratch probe (NOT collected by pytest -- underscore prefix)."* |
| `tests/unit/phase6_smoke.py` | *"Phase 6 real CPU smoke test -- standalone, NOT collected by pytest."* |
| `tests/unit/conftest.py` | *"Shared fixtures for the Phase 6 VLM unit tests."* — a fixture module, not a test module |
| `tests/unit/__init__.py` | package marker |

### 2.2 The full file inventory, with what each covers

The descriptions below are the **first line of each module's own docstring** — the module's statement of
what it pins, not a paraphrase.

#### 2.2.1 Gateway, deployment and configuration

| File | What it covers (module docstring) |
|---|---|
| `test_gateway_app.py` | *"The gateway ASGI layer: its contract, its allowlist, and real HTTP execution."* |
| `test_gateway_assets.py` | *"The ephemeral asset store: opaque handles, three obligations, real lifetimes."* |
| `test_gateway_policy.py` | *"The gateway's decisions must be correct, cheap, and never reach the Space on a rejection."* |
| `test_gateway_responsibilities.py` | *"STEP 7 §C — the eighteen gateway responsibilities, as UNIT tests."* |
| `test_space_app.py` | *"The Space entrypoint must be cheap, honest, and hash-preserving."* |
| `test_deployment_adapter.py` | *"The deployment capability adapter: registry-authoritative, contract-shaped."* |
| `test_deploy_config.py` | *"Tests for the Phase 18 deploy packaging manifest and its validator."* |
| `test_deploy_requests.py` | *"Phase 18 §79 DEPLOYMENT: cold-start and sequential-request behaviour."* |
| `test_config.py` | *"Phase 1 tests — configuration registry and frozen contract guards."* |
| `test_code_revision.py` | *"Tests for `core.code_revision` — provenance defect section 6b."* |
| `test_packaging_script.py` | *"Regression tests for `scripts/package_kaggle_code.py`."* |

#### 2.2.2 Controller, planner, registry and schemas

| File | What it covers |
|---|---|
| `test_controller.py` | *"Tests for `core.controller` — dispatch, partial failure and assembly (Phase 15)."* |
| `test_controller_seam.py` | *"The controller's asset-resolution seam — regression tests for a real defect."* |
| `test_planner.py` | *"Tests for `core.planner` — the deterministic policy planner (Phase 15)."* |
| `test_registry.py` | *"Tests for `core.registry` — the specialist capability registry (Phase 15)."* |
| `test_schemas.py` | *"Phase 1 tests — canonical schema contract."* |
| `test_errors.py` | *"Tests for `core.errors` — the taxonomy, and the F-15 path scrubber."* |
| `test_evidence_engine.py` | *"Tests for the evidence engine — the aggregator (Phase 13)."* (§4) |
| `test_report_generator.py` | *"Unit tests for `reports.generator` — pure, deterministic, fixtures only."* |

#### 2.2.3 Router

| File | What it covers |
|---|---|
| `test_router_threshold_sweep.py` | *"Phase 4/13 — contracts for the validation-only ROUTER threshold sweep."* |
| `tests/routing/test_router.py` | *"Phase 4 tests — the intent router."* |

#### 2.2.4 Change detection and change-VQA (the largest subsystem by file count)

| File | What it covers |
|---|---|
| `test_change.py` | *"Phase 9 tests — change detection model, dataset, and post-processing."* |
| `test_change_levir_real_layout.py` | *"Phase 9 — the LEVIR-CD loader against the REAL dataset layout."* |
| `test_change_phase9_decontamination.py` | *"Phase 9 -- pins that stop the change-detection module from being re-contaminated."* |
| `test_change_specialist.py` | *"Phase 9 — the change specialist serving contract."* |
| `test_change_threshold_sweep.py` | *"Phase 9 — contracts for the validation-only threshold sweep."* |
| `test_change_train_script_contract.py` | *"Phase 9 — contracts for `scripts/train_change.py`."* |
| `test_change_vqa_amp.py` | *"R-02 — the mixed-precision training contract."* |
| `test_change_vqa_dataset.py` | *"R-02 test areas E-J — the CDVQA dataset layer."* |
| `test_change_vqa_eval_cli.py` | *"R-02 — the evaluation CLI's empty-split diagnosis."* |
| `test_change_vqa_features.py` | *"R-02 test areas K-L — frozen change features and the feature caches."* |
| `test_change_vqa_head.py` | *"R-02 test areas M-Q — batching, the reasoning head, and the metrics."* |
| `test_change_vqa_integration.py` | *"R-02 test area R — the change-VQA specialist and its planner/registry wiring."* |
| `test_change_vqa_kaggle_notebook.py` | *"R-02 — the Kaggle notebook's discovery contract."* |
| `test_change_vqa_prepare_script.py` | *"R-02 — `prepare_record.json` must be the provenance of the whole directory."* |
| `test_change_vqa_promotion.py` | *"STEP 1 — the promoted R-02 change-VQA head at its canonical serving path."* |
| `test_change_vqa_smoke.py` | *"R-02 — the local smoke training run, and the artifact it leaves behind."* |
| `test_change_vqa_vocab.py` | *"R-02 test areas A-D — the CDVQA answer space and question ontology."* |
| `test_cdvqa_adapter.py` | *"Tests for the CDVQA reader, join and example construction."* |
| `test_cdvqa_second_overlap.py` | *"Tests for scripts/check_cdvqa_second_overlap.py."* |

#### 2.2.5 Optical-SAR

| File | What it covers |
|---|---|
| `test_optical_sar_croma.py` | *"CROMA wrapper — the interface verified against the real model."* |
| `test_optical_sar_croma_geometry.py` | *"Phase 11 geometry guards — the four requirements that were *measured but…"* |
| `test_optical_sar_fusion_head.py` | *"Optical-SAR fusion head — the frozen concatenation and channel dropout."* |
| `test_optical_sar_prompts.py` | *"Optical-SAR prompts — the narration boundary."* |
| `test_optical_sar_radiometry.py` | *"CROMA encoder-input radiometry — the DEV-2 ruling, as executable guards."* |
| `test_optical_sar_sensor_adapter.py` | *"Optical-SAR sensor adapter — the two hard rules, enforced."* |
| `test_optical_sar_specialist.py` | *"Phase 11/12 — the optical-SAR specialist serving contract."* |
| `test_app_serving_optical_sar.py` | *"Regression guards for the optical-SAR serving wiring (Phase 14 / Pass 16)."* |
| `test_eval_fusion_115.py` | *"Tests for the pre-registered 11.5 metric tool."* |
| `test_fusion_extraction.py` | *"Tests for the Phase 12 frozen-feature extraction pipeline (fixtures only)."* |
| `test_fusion_seed_variance.py` | *"Tests for the predeclared seed-variance pass (Gate F D-03 / D-05)."* |
| `test_fusion_training.py` | *"Tests for the Phase 12 fusion-head training loop (fixtures only)."* |
| `test_reben_adapter.py` | *"Tests for the reBEN v2 -> PairedSample adapter (Phase 12)."* |
| `test_analyze_reben_labels.py` | *"Tests for the reBEN label analyser."* |

#### 2.2.6 Grounding

| File | What it covers |
|---|---|
| `test_grounding_dataset.py` | *"Tests for the Phase 8 grounding dataset."* |
| `test_grounding_degenerate_fallback.py` | *"Regression tests for the degenerate fallback-box guard (Phase 8)."* |
| `test_grounding_feature_extraction.py` | *"Tests for the frozen-feature cache."* |
| `test_grounding_head_wiring.py` | *"Wiring the trained RemoteCLIP grounding head into production (work order §5)."* |
| `test_grounding_metrics.py` | *"Tests for the grounding metrics."* |
| `test_grounding_nms.py` | *"Unit tests for grounding NMS (specialists/grounding/head.py::nms)."* |
| `test_vrsbench_loader.py` | *"Tests for the VRSBench referring-expression loader."* |
| `test_vrsbench_lazy_images.py` | *"Tests for the lazy-image mode of the VRSBench loader."* |

#### 2.2.7 VLM training and evaluation

| File | What it covers |
|---|---|
| `test_vlm_artifact.py` | *"Tests for the Phase 6 adapter artifact (`training.vlm.artifact`)."* |
| `test_vlm_collate.py` | *"Tests for Phase 6 batch assembly (`training.vlm.collate`)."* |
| `test_vlm_config.py` | *"Tests for the Phase 6 training recipe (`training.vlm.config`)."* |
| `test_vlm_dataset.py` | *"Tests for the Phase 6 BigEarthNet -> SmolVLM instruction corpus."* |
| `test_vlm_evaluate.py` | *"Tests for the predeclared acceptance rule (`training.vlm.evaluate`)."* |
| `test_vlm_evaluate_cache.py` | *"Tests for the resumable per-question answer cache in `training.vlm.evaluate`."* |
| `test_vlm_evaluate_decision_split.py` | *"Tests for `decide_acceptance(..., decision_split=...)`."* |
| `test_vlm_formatting.py` | *"Tests for Phase 6 prompt construction and loss masking (`training.vlm.formatting`)."* |
| `test_vlm_lora.py` | *"Tests for Phase 6 LoRA injection (`training.vlm.lora`)."* |
| `tests/model/test_vlm_contract.py` | *"Phase 5 tests — SmolVLM contract, prompts, and the VQA specialist."* |
| `tests/model/test_vqa_quality_gate.py` | *"Integration tests: the F5-5 quality gate blocks the VLM before it runs."* |

#### 2.2.8 Datasets, metrics and evaluation

| File | What it covers |
|---|---|
| `test_bigearthnet_blocks.py` | *"Tests for the T2 spatial-block scene key (Phase 12)."* |
| `test_bigearthnet_pairing.py` | *"Tests for the BigEarthNet optical<->SAR pairing step (Phase 12)."* |
| `test_bigearthnet_prep.py` | *"Tests for the BigEarthNet reader and instruction-pair construction."* |
| `test_prepare_script.py` | *"Tests for the BigEarthNet preparation CLI."* |
| `test_caption_metrics.py` | *"Caption metrics (ruling R-16): BLEU, ROUGE-L, CIDEr, BERTScore."* |
| `test_vqa_metrics.py` | *"Tests for the Tier-1 VQA / Change-VQA text metrics (`evaluation.metrics.vqa`)."* |
| `test_eval_normalize.py` | *"Tests for `evaluation.normalize` — the plan section 63 metric normaliser."* |
| `test_evaluation_runner.py` | *"STEP 5 — the evaluation runner."* |
| `test_calibration.py` | *"STEP 2 — the calibration fitter, the artifact, and the consumer wiring."* |
| `test_quality_gate.py` | *"Tests for the deterministic input-quality gate (finding F5-5)."* |
| `test_benchmark_adapters.py` | *"STEP 5 — benchmark adapters: contract, registry, and the non-invention guard."* |
| `test_benchmark_adapters_real.py` | *"The four real-corpus benchmark adapters: correctness against synthetic fixtures."* |
| `test_public_test_corpus.py` | *"STEP 5 — the immutable public-test corpus (plan section 37, Test Set Firewall)."* |
| `test_manifest_freeze.py` | *"Tests for the frozen dataset-manifest record (`evaluation/manifest_freeze.json`)."* |
| `test_prompt_freeze.py` | *"Tests for the frozen prompt-set record (`evaluation/prompt_freeze.json`)."* |
| `test_run_manifest_population.py` | *"Tests for STEP 4 — runtime population of the run manifest."* |
| `test_provenance_fixes.py` | *"Tests for the R-02 provenance fixes: sections 6a, 6b and 6c."* |
| `test_diagnose_feature_cache.py` | *"Tests for `scripts/diagnose_feature_cache.py`'s exit-code contract."* |

#### 2.2.9 Serving composition and the app boundary

| File | What it covers |
|---|---|
| `test_app_serving.py` | *"Regression tests for `app.serving` — the public serving composition root."* |
| `test_phase6_closure.py` | *"Tests for the Phase 6 closure record."* |

#### 2.2.10 Contract-conformance, frontend and documentation guards

| File | What it covers |
|---|---|
| `test_api_contract_doc.py` | *"The API contract must not drift from the schemas it documents."* |
| `test_frontend_guide_doc.py` | *"The contract documents must stay consistent with each other and the schemas."* |
| `test_frontend_live_wiring.py` | *"Regression tests for the frontend <-> orchestrator live wiring."* (§5) |
| `test_runbook_doc.py` | *"The deployment runbook must not assert facts the repository contradicts."* |
| `test_deploy_config.py` | *"Tests for the Phase 18 deploy packaging manifest and its validator."* |
| `test_step7_api_contract_conformance.py` | *"STEP 7 section D -- the API contract, tested against the schema that implements it."* |
| `test_safe_delete_shim.py` | *"The Windows safe-delete shim: verbatim prefixes, and the shell's return code."* (§7) |

#### 2.2.11 The other directories

| File | What it covers |
|---|---|
| `tests/geospatial/test_transform.py` | *"Phase 2 tests — geospatial transforms and the raster input contract."* |
| `tests/leakage/test_leakage.py` | *"Phase 3 tests — Gate 1: dataset manifests and scene-level leakage isolation."* |
| `tests/integration/test_step7_backend_chain.py` | *"STEP 7 sections H, I, M -- the real inference service, driven over HTTP."* |
| `tests/integration/test_step7_r02_serving.py` | *"STEP 7 section I -- the R-02 head, verified through the real serving path."* |
| `tests/e2e/test_demo.py` | *"End-to-end tests for the Phase-19 demonstration driver (`demo.run_demo`)."* |
| `tests/e2e/test_gate4_e2e.py` | *"Gate-4 end-to-end test — the full control tier in its real local state."* |
| `tests/e2e/_demo_probe.py` | *"Subprocess probe: run the demo in a FRESH interpreter and report which…"* (helper) |

### 2.3 What the inventory says about coverage shape

Three observations a reader should draw from the inventory:

1. **The largest single cluster is change detection and change-VQA** (19 files), which matches the
   project's stated engineering priority ordering (`docs/MASTER_ARCHITECTURE_PLAN.md` §2.1: optical-SAR
   first, change second).
2. **Every capability has a *serving contract* test, separate from its *model* tests.** For example
   `test_change.py` covers the model/dataset/post-processing while `test_change_specialist.py` covers
   *"the change specialist serving contract"*; the same split exists for optical-SAR
   (`test_optical_sar_fusion_head.py` vs `test_optical_sar_specialist.py`) and grounding. The split is
   deliberate: a model that works in a notebook and a specialist that serves the contract are different
   claims.
3. **Documentation is tested as code.** Five of the 94 modules are doc-guards (§3), and they are counted in
   the 183-passed doc/frontend suite (§6) rather than being an afterthought.

### 2.4 The integration suite's documented scope

`docs/ITEM5_INTEGRATION_SUITE_SCOPE.md` is the record of what the integration suite proves. The contract
summarises it (`docs/API_CONTRACT.md` §8):

> *"`docs/ITEM5_INTEGRATION_SUITE_SCOPE.md` records what the 26-test integration suite proves (the app's
> *boundary*, in-process) and what only a live deployment can prove (reachability, cold start, memory
> ceilings). Read it before treating a green `tests/integration` run as evidence about a deployment —
> **no test in this repository dials a network address**, including this contract's own `/v1/*`
> examples."*

The exact number of tests **collected** in a single `tests/integration` invocation is
`UNKNOWN — not established from the available evidence`. The documented figure is 26; the evidence this
chapter can establish is the two module files and their stated scope.

---

## 3. The doc-guard tests: keeping documentation and code in sync

### 3.1 Why documentation is tested

Five modules treat documentation as a **tested artefact**. The rationale is stated in each module's
docstring, and the common thread is that a documentation defect costs a real debugging session:

| Guard | The failure it prevents |
|---|---|
| `test_api_contract_doc.py` | a frontend builds against an example that does not validate, and the cause looks like a frontend bug |
| `test_frontend_guide_doc.py` | a value a frontend must send or read is documented wrong, producing a `422` and a wasted session |
| `test_runbook_doc.py` | an operator copies a command that fails, or quotes a number that was never measured, and concludes the *system* is broken |
| `test_deploy_config.py` | the deploy manifest silently moves the frozen `Config.hash` |
| `test_step7_api_contract_conformance.py` | the contract and the schema that implements it drift apart |

`test_api_contract_doc.py` records what the guards actually catch, with a list of four real defects the
act of writing the contract produced:

> *"Writing this contract produced exactly that failure four times over -- `Box` fields were nested instead
> of flat, `Evidence` used `summary`/`value`/`source` instead of
> `coordinates`/`score`/`source_specialist`, `CoordinateSystem` values were shorthand rather than the real
> `normalized_0_1`/`pixel`/`geo`, and `ExecutionTrace.inputs`/`outputs` were dicts rather than lists. Each
> was caught by validating the examples against the real models, which is what these tests do
> permanently."*

### 3.2 `test_api_contract_doc.py`

**Scope:** validate every fenced ```json block in `docs/API_CONTRACT.md` against the real Pydantic models.

| Mechanism | Detail |
|---|---|
| Fixture `contract_text` | asserts the document exists, then reads it |
| Fixture `json_blocks` | regex-extracts every ```json block and parses it; a block that is not valid JSON raises `AssertionError` naming the block index |
| `_find(blocks, predicate)` | locates an example by shape rather than by position |

| Test | What it pins |
|---|---|
| `test_every_json_block_in_the_contract_parses` | *"A malformed example is unusable; this fails loudly rather than silently."* |
| `test_the_result_envelope_example_validates` | the envelope example validates against `ResultEnvelope`; `schema_version == SCHEMA_VERSION`; `confidence.method in ("uncalibrated", "temperature_scaling")` |
| `test_no_contract_example_fabricates_an_artifact_uri` | no parsed example contains an `artifact://` URI or a filesystem path in `change_map` / `artifact_ref` (walks the **parsed** examples, §1.2) |
| (further tests) | documented task/coordinate-system values are the real ones; every error code documented; the frozen config hash is recorded; the calibration caveat is recorded |

The `confidence.method` assertion is the honest-signal guard: it pins that the contract's example cannot
claim a calibration method the engine does not have. §4 shows the engine side of the same rule.

### 3.3 `test_frontend_guide_doc.py`

**Scope:** `docs/API_CONTRACT.md` and `docs/FRONTEND_INTEGRATION.md` must agree with each other **and** with
`core/schemas.py`.

It imports the real models — `Box, CoordinateSystem, Evidence, EvidenceType, Region, Task` — and compares
the prose against them. The docstring lists the five defects that motivated it:

> *"Writing them produced, in order: `CoordinateSystem` documented as `normalized`/`geographic` (real:
> `normalized_0_1`/`geo`), `Box` geometry documented as nested (real: flat), `Evidence` fields named
> `summary`/`value`/`source` (real: `coordinates`/`score`/`source_specialist`),
> `ExecutionTrace.inputs`/`outputs` as objects (real: lists), and `Task.unsupported` missing from both
> tables."*

The representative test asserts a **bidirectional** requirement:

```python
def test_every_task_is_documented_in_both_documents(api_text, frontend_text):
    for task in Task:
        assert f"`{task.value}`" in api_text, f"Task.{task.value} missing from API contract"
        assert f"`{task.value}`" in frontend_text, (
            f"Task.{task.value} missing from the frontend guide; a value the client "
            f"may receive but cannot find documented is a crash waiting to happen"
        )
```

The message on the second assertion is the rationale: *"a value the client may receive but cannot find
documented is a crash waiting to happen."*

The suite also pins: no login screen, no secrets in the browser, the confidence-property trap, and
serialized analyses (`docs/EVALUATION.md` §7.2).

### 3.4 `test_runbook_doc.py`

**Scope:** `docs/BACKEND_DEPLOYMENT_RUNBOOK.md` must not assert facts the repository contradicts.

The module names **two failure modes**:

> 1. *"**Invented measurements.** The runbook quotes command output. If it quotes a byte size, a
>    capability list, a function name or an exit code that the repository does not produce, the quote is
>    fabricated. Where a fact is cheap to verify locally, this module verifies it."*
> 2. *"**Invented API surface.** The runbook calls into the codebase (`build_serving_registry`,
>    `available()`, `CHANGE_CHECKPOINT`, `CHANGE_VQA_HEAD`). A runbook that names a function the code does
>    not export is a runbook whose commands raise `AttributeError` on the operator's first copy-paste."*

It reads three documents — the runbook, `docs/DEPLOYMENT_ARCHITECTURE.md` and `docs/API_CONTRACT.md` — so
that a claim contradicted *across* documents is caught, not only a claim contradicted by code.

**The most important class of assertion in this module** is stated in its docstring:

> *"The absence of a deployment cannot be tested here, so these tests assert the runbook's *local* claims
> and, separately, assert that the document itself labels its unimplemented sections as unimplemented. The
> second class matters more than it looks: a runbook that describes a live deployment without having
> performed one is the exact overclaim the work order forbids."*

So the guard pins **honesty about status**, not merely factual accuracy: an "undecided" item must be
declared undecided, and a "designed" item must not be described as "verified" (`docs/EVALUATION.md` §7.2).

### 3.5 `test_deploy_config.py`

**Scope:** the Phase 18 deploy manifest and its validator, and the **config-hash regression guard**.

The docstring states the two properties it proves:

> *"These tests prove the manifest is INERT — it is never merged into the config registry and cannot move
> `Config.hash` — and that the validator rejects each failure mode. Negative cases use in-memory dicts via
> `validate_documents`, so the real config files are never mutated."*

Key elements:

| Element | Detail |
|---|---|
| `REGISTRY_MARKER` | `test_deploy_manifest_exists_and_declares_the_non_registry_marker` asserts `doc[REGISTRY_MARKER] is False` |
| `test_validator_reports_no_errors_on_the_real_files` | `assert validate() == []` |
| `test_validator_cli_exits_zero_as_a_standalone_script` | drives the real entry point **in a child interpreter** (the same one running the suite) *"to prove the module's own `sys.path` bootstrap makes `core` importable WITHOUT pytest's `pythonpath = .` shortcut"* |
| `GPU_DURATION_KEYS` | `gpu_duration_vqa`, `gpu_duration_grounding`, `gpu_duration_change`, `gpu_duration_optical_sar` |
| `FROZEN_CONFIG_HASH` | imported from `tests.test_config` rather than re-declared — *"We import it rather than re-declare it, so the two cannot drift."* |

The `FROZEN_CONFIG_HASH` import is the anti-drift move applied to the guard itself: the deploy-config test
does not carry its own copy of `78f1e3700da15aa1`, it imports the authority.

The suite also pins the C-8 constraints: `torch_compile` false, CPU mode required, lazy load, single model
cache (`docs/EVALUATION.md` §7.2).

### 3.6 `test_step7_api_contract_conformance.py`

*"STEP 7 section D -- the API contract, tested against the schema that implements it."* This is the
schema-level counterpart to `test_api_contract_doc.py`: where the doc-guard validates the *examples* in the
markdown, this module tests the *contract* against the implementing schema directly.

### 3.7 What the doc-guards cannot do

The modules say so themselves, and the honesty is worth reproducing because it bounds what a green run
means:

> *"Prose can still be wrong in ways these tests cannot see -- a mis-stated latency, a wrong reason for a
> decision. What is asserted is that every value a frontend must send or read is the real value, which is
> the class of error that produces a 422 and a wasted debugging session."*
> — `tests/unit/test_frontend_guide_doc.py`

> *"These tests validate the EXAMPLES, not the prose. They cannot detect a wrong sentence (e.g. a
> mis-stated latency budget)."*
> — `tests/unit/test_api_contract_doc.py`

So a doc-guard guarantees **example-level and value-level** conformance. It does not guarantee that a
sentence is true. That distinction is the reason `test_runbook_doc.py` exists separately — it targets a
different class of claim (numbers, names, and status honesty) that the example-validators cannot reach.

---

## 4. The evidence engine: purity, determinism and the reproducibility property

### 4.1 Why this suite exists at all

The evidence engine is the component that turns a `SpecialistResult` into a *proof*: an ordered,
identified, digestible `EvidenceCollection`. Everything the release claims about "evidence" downstream —
the trace, the audit trail, the ability to recompute a verdict from a recorded run — rests on one
property of this component: **the same inputs must produce the same output, byte for byte.**

If that property fails, then two runs of the same request can produce two different evidence sets, and
no verdict can be recomputed from a record. The evidence engine is therefore the load-bearing component
behind §9's "verdict-independent evidence" claim, and its suite is the guard on that load-bearing
property.

The suite is `tests/unit/test_evidence_engine.py` — **73 test functions** in 877 lines
(`tests/unit/test_evidence_engine.py`).

### 4.2 The contract, in the module's own words

The module docstring states exactly two failure modes it is written against, and a third honesty rule:

> *"The engine's whole value is that it is *reproducible* and *lossless*. So the tests are mostly about
> the two ways that can silently break:*
>
> *determinism — the same claims must produce the same ordered, identified collection even when
> specialists finish in a different order, or when the uuids differ between processes.*
> *no loss — no specialist's evidence may vanish without being counted, and agreement between
> specialists must be recorded rather than collapsed with one name thrown away.*
>
> *Plus the honesty rule on confidence: an uncalibrated engine must pass the raw score through and SAY
> it is uncalibrated, never manufacture a fitted mapping."*
> — `tests/unit/test_evidence_engine.py`

Three properties, then: **determinism**, **no loss**, and **confidence honesty**. The suite is
organised around them.

### 4.3 The determinism family

The core purity claim is a single test whose docstring is the whole argument:

```python
def test_aggregate_is_deterministic_across_repeat_calls() -> None:
    """Same input -> byte-identical output. The core purity claim."""
```

It asserts three things across two calls with the same input — identical `ids()`, identical
`evidence_digest()`, and identical `model_dump()` lists. "Byte-identical output" is not a figure of
speech: the digest is the serialised form, so a difference anywhere in the collection changes the digest.

The determinism family, read from the file:

| Test | What it pins |
|---|---|
| `test_aggregate_is_deterministic_across_repeat_calls` | the core purity claim (line 132) |
| `test_order_is_independent_of_input_order` | forward vs. backward input order give the same digest |
| `test_equal_scores_order_deterministically_by_coordinates` | *"Ties must not fall back to insertion order — that is input-order dependence wearing a disguise"* |
| `test_higher_score_sorts_first_within_a_type` | the ordering rule inside an evidence type |
| `test_unscored_items_sort_after_scored_items` | a total order even when some items carry no score |
| `test_types_are_grouped_together` | type grouping is stable |
| `test_ids_are_sequential_and_zero_padded` | the identifier format |
| `test_ids_restart_from_one_for_each_aggregation` | ids are a function of the collection, not global state |
| `test_source_results_are_not_mutated` | purity — computing evidence does not write back |
| `test_result_objects_are_not_mutated` | purity, at the result level |

The two mutation tests are the *purity* half of "purity and determinism": the engine is a function, not
a transformer of its inputs. A caller's `SpecialistResult` is the same object after `aggregate` as
before.

### 4.4 The no-loss family

Losslessness is the harder property to test, because the ways evidence can vanish are subtle. The
deduplication tests are where it is pinned:

| Test | What it pins |
|---|---|
| `test_payload_only_differences_are_deduplicated_as_one_claim` | two items differing only in payload are one claim |
| `test_identical_items_from_one_specialist_collapse` | intra-specialist dedup |
| `test_deduplication_ignores_random_uuid` | dedup is on content, not on the process-local uuid |
| `test_deduplication_ignores_payload_differences_and_merges_them` | payloads merge rather than one being dropped |
| `test_different_coordinates_are_not_deduplicated` | dedup does not over-collapse |
| `test_different_coordinate_systems_are_not_deduplicated` | a normalised box and a pixel box are different claims |
| `test_same_claim_from_two_specialists_is_kept_and_marked_corroborated` | **agreement is recorded, not collapsed** — the second specialist's name is kept as corroboration |
| `test_unrelated_items_carry_no_corroboration_marker` | the marker is meaningful, not always-on |
| `test_multiple_results_aggregate_without_loss` | the headline no-loss assertion |
| `test_aggregation_does_not_double_count_a_specialists_own_evidence` | lossless ≠ duplicating |

The corroboration tests are the direct expression of the docstring's *"agreement between specialists
must be recorded rather than collapsed with one name thrown away."* This is a design decision encoded
as a test: when two specialists agree, the collection records **both** names and marks the claim
corroborated, rather than keeping one and discarding the other.

The cap tests pin that truncation is *counted*, not silent:

| Test | What it pins |
|---|---|
| `test_zero_max_items_is_rejected` | an invalid cap is refused |
| `test_cap_is_respected_and_the_drop_is_counted` | items past the cap are dropped **and counted** |
| `test_no_truncation_flag_when_under_the_cap` | the truncation flag is truthful |
| `test_default_cap_matches_the_config_value` | `DEFAULT_MAX_ITEMS` tracks the config |
| `test_sources_are_reported_before_the_cap_is_applied` | *which* specialists contributed is recorded before truncation |

That last test is a specific honesty property: if the cap drops a specialist's only item, the collection
still records that the specialist contributed. Losslessness at the *source* level survives truncation at
the *item* level.

The empty/degenerate cases are pinned too: `test_empty_input_yields_an_empty_collection`,
`test_a_result_with_no_evidence_contributes_nothing`, `test_accepts_a_single_result_without_wrapping`,
and the two argument-error tests `test_both_results_and_evidence_is_rejected` /
`test_neither_results_nor_evidence_is_rejected`.

### 4.5 The confidence honesty rule

This is where the project's style guide (`DOCS_STYLE_GUIDE.md` §3) meets a test. The rule the engine
must obey: an uncalibrated engine **passes the raw score through and says it is uncalibrated**; it never
manufactures a fitted mapping. The tests that pin this:

| Test | What it pins |
|---|---|
| `test_no_calibration_passes_raw_through_and_says_so` | the default is identity + an explicit "uncalibrated" label |
| `test_uncalibrated_is_never_labelled_as_fitted` | the label cannot be silently upgraded |
| `test_identity_temperature_is_reported_as_uncalibrated` | `T = 1.0` is *not* calibration |
| `test_engine_without_calibration_does_not_claim_calibration` | the engine-level claim |
| `test_raw_outside_unit_range_is_clamped_not_rejected` | a robustness decision, pinned |
| `test_non_finite_raw_is_refused` | NaN/Inf are refused, not clamped |

This matters because the project's calibration genuinely made the metric **worse** (ECE
`0.013755 → 0.014929`, `DOCS_STYLE_GUIDE.md` §3) and is retained only because it is in the frozen
config. The engine must therefore never present calibration as an improvement, and these tests are what
enforce that at the code level. The `METHOD_TEMPERATURE` / `METHOD_UNCALIBRATED` constants are the two
labels, and `test_uncalibrated_is_never_labelled_as_fitted` is the guard that the first is never used
for the second.

The fitted path is tested separately and is deterministic:
`test_temperature_scaling_is_applied_when_fitted`,
`test_temperature_above_one_softens_and_below_one_sharpens`,
`test_temperature_scaling_is_monotonic`, `test_temperature_scaling_is_deterministic`,
`test_endpoint_scores_do_not_produce_nan`, `test_invalid_temperatures_are_rejected`, and
`test_specialist_degradation_survives_calibration` / `test_specialist_components_are_preserved_through_calibration`.

### 4.6 Calibration artifact I/O

The engine reads a calibration artifact from disk, and the tests pin the failure modes explicitly rather
than letting them degrade silently:

| Test | What it pins |
|---|---|
| `test_artifact_round_trips_through_json` | write→read is lossless |
| `test_nested_artifact_form_is_accepted` | both artifact shapes are accepted |
| `test_missing_artifact_degrades_instead_of_failing` | a missing artifact is a *degradation* by default |
| `test_missing_artifact_raises_when_required` | …but raises when the caller says it is required |
| `test_malformed_artifact_is_not_silently_swallowed` | a corrupt artifact is an error, not a shrug |
| `test_artifact_without_a_temperature_is_rejected` | the required field is required |
| `test_load_calibration_respects_the_master_switch` | the config switch is honoured |
| `test_load_calibration_reads_the_configured_file` | the configured path is the one read |
| `test_load_calibration_without_a_file_configured_is_none` | no config ⇒ `None`, not a guess |
| `test_engine_from_config_uses_the_configured_cap` / `..._falls_back_to_the_default_cap` | the cap wiring |

The pairing of `degrades_instead_of_failing` with `raises_when_required` is the pattern this repository
uses throughout: a soft default for a convenience path, and a hard error for the path where silence
would be a lie.

### 4.7 The scratch-root workaround, and why it is documented in the test file

The suite does **not** use pytest's built-in `tmp_path`. The module says why, and the reason is a
sandbox property, not a preference:

> *"Scratch root, repo-local. The built-in `tmp_path` fixture cannot finalize under this machine's
> sandbox, so artifact tests use this instead. It is the same directory `--basetemp=.pytest_tmp` points
> pytest at, so nothing is written outside the workspace."*
> — `tests/unit/test_evidence_engine.py`

```python
SCRATCH_ROOT = Path(__file__).resolve().parents[2] / ".pytest_tmp"
```

And the module docstring carries the operational requirement:

> *"The `--basetemp=.pytest_tmp` flag is mandatory (see the project handoff): the default pytest temp
> root triggers a sandbox denial on this machine."*

The `scratch` fixture is deliberately **not** cleaned up on teardown, and the docstring says why: *"the
sandbox's safe-delete guard rejects the recursive delete, and a test that writes into the repo's own
ignored scratch directory is harmless. Each test gets a unique path so no state leaks between them."*
This is the same guard that causes §7's four `test_safe_delete_shim` failures — documented here as a
design constraint rather than hidden as a quirk.

### 4.8 The public surface

`test_module_public_surface_is_declared` pins the exported names, and
`test_serialises_through_the_schema_serialiser` pins that the collection serialises through the *schema*
serialiser (not a bespoke one), so the wire form and the evidence form cannot drift. The shorthand and
helpers are pinned by `test_aggregate_evidence_shorthand_matches_the_engine`,
`test_collection_helpers_filter_correctly`, `test_collection_summary_reports_the_observable_facts`,
`test_digest_changes_when_a_claim_changes`, `test_digest_is_stable_across_regenerated_uuids`,
`test_collection_is_iterable_and_sized`.

`test_digest_is_stable_across_regenerated_uuids` is the one that makes §9 possible: because the digest
ignores process-local uuids, a recorded run's evidence digest can be recomputed in a *different process*
and still match. Without it, "recompute the verdict from the record" would be a claim you could not test.

### 4.9 What the evidence-engine suite does NOT establish

- It does **not** establish that the specialists produce correct evidence — only that the *aggregator*
  is deterministic and lossless given whatever they produce.
- It does **not** establish end-to-end reproducibility of a live run; that is §8's job.
- It does **not** establish that the calibration is *good* — the tests pin that calibration is
  *honestly labelled*, not that it improves anything (it does not; ECE got worse).

---

## 5. The frontend live-wiring suite

### 5.1 What it proves, and why it reads source text

`tests/unit/test_frontend_live_wiring.py` is the regression net for the frontend↔orchestrator boundary.
Its docstring states two things it proves, **both of which were broken before the change**:

> *"A. **CORS.** The deployment allowed exactly one origin (`https://satquery.pages.dev`), so the
> frontend could not be driven from a local development server at all: every request from
> `http://localhost:8080` was refused with `Disallowed CORS origin`, and a developer had to deploy to
> Cloudflare to test a one-line JavaScript change. The fix adds explicit localhost origins while keeping
> production listed and keeping `*` rejected.*
>
> *B. **The task vocabulary.** The page's router and the server's `Task` enum must agree. They are two
> independently written vocabularies in two languages, and nothing previously checked that a value the
> frontend would send is a value the server accepts. This module reads BOTH and compares them."*
> — `tests/unit/test_frontend_live_wiring.py`

And it pins a third, subtler defect — one invisible to a validity check:

> *"It also pins the `chang` word-boundary defect: `\bchang\b` cannot match 'changed', so the page's own
> default question ('What changed here?') routed to `vqa` instead of the change path. That is invisible
> to any test that only checks 'is this a valid enum value' — both branches produce a valid value. The
> test therefore asserts the ROUTE, not just the validity."*

That last sentence is the philosophical centre of the whole suite: **assert the route, not the
validity.** A test that asks "is `vqa` a valid task?" passes for both the correct and the defective
router, because both emit valid tasks. Only a test that asks "does *this question* route to *this
task*?" can see the defect.

**Why source-text parsing.** The module reads the frontend source rather than importing it, and says so:

> *"the frontend is plain ES5 with no build step and no module exports, so there is no importable
> symbol. Parsing the source is the only way to assert the shipped bytes. The parsing is anchored on
> named tokens rather than line numbers, so it does not silently pass if the file is restructured."*

The anchors are module-level path constants — `REPO_ROOT`, `DEPLOY_RENDER`, `FRONTEND_JS`, `MISSION_JS`,
`LIVE_JS`, `CORE_JS`, `MISSION_HTML` — and the production origin is a single constant,
`PRODUCTION_ORIGIN = "https://satquery.pages.dev"`. The CORS half imports the orchestrator with a clean
environment via `_render_module()`, because `_allowed_origins` reads `os.environ` at **call** time while
`_DEV_ORIGINS` is built at **import** time — a distinction the module's helper comment records
explicitly.

### 5.2 The measured result

**106 passed** (`release/CURRENT_RELEASE_STATE.md:121`; `README.md` §The test suites;
`docs/EVALUATION.md` §7.1). The command is:

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_frontend_live_wiring.py -q     # 106 passed
```

This is a **re-run-this-session** figure, and it is the number this chapter quotes.

### 5.3 The fifteen test classes

The classes were read from the file. Each name is a claim:

| # | Class | What it pins |
|---|---|---|
| 1 | `TestTheCorsAllowlistKeepsProductionAndAddsDevelopment` | production origin always allowed, not duplicated, dev origins added, `*` rejected even when hidden in a list |
| 2 | `TestTheCorsDecisionMatchesTheAllowlist` | the response-leg decision matches the allowlist (echo, no-headers, lookalike-host rejection) |
| 3 | `TestTheFrontendSpeaksTheServersTaskVocabulary` | every mapped value is a real server task; the mapping covers every task the router can emit; specialist names are human labels, not tasks |
| 4 | `TestTheDefaultQuestionReachesTheChangePath` | the page's own default question routes to the change path; the change stem matches its inflections; the temporal slot is required for a change question |
| 5 | `TestALocationQuestionIsNotAChangeQuestion` | a "where" question reads as grounding, with one or two assets |
| 6 | `TestTheArchitecturePolicyDoesNotReadALocationAsAChange` | the architecture sample routes to grounding; a place word in a "where" question is not a change |
| 7 | `TestTheTaskRespectsThePairRequirement` | paired tasks substitute correctly when given one asset; single-asset tasks are never substituted |
| 8 | `TestTheLiveClientUsesTheRealIngestionPath` | the client targets the orchestrator's routes; uploads send raw bytes with a derived content type; the infer body carries asset ids under `assets`; the client never sends a filesystem path |
| 9 | `TestThePageLoadsTheLiveClient` | `mission.html` includes live JS before mission JS; the declared API base is an absolute origin, not the relative fallback, not the frontend origin; no fixture image is wired into the analysis path |
| 10 | `TestTheLivePathKeepsTheHonestFailureContract` | a failure is attributed to the step that failed; the preview is kept for the no-file case |
| 11 | `TestTheCaptionStopsCallingARealUploadIllustrative` | a successful run rewrites the leading word and the alt text; the failure path does not claim an analysis happened; a new file clears the previous run's claim |
| 12 | `TestTheFrontendSendsOnlyTheAssetsTheTaskRequires` | a single-asset task with a pair uploaded uses only `t1`; the old wiring would have sent both; change/optical-SAR send the pair when present |
| 13 | `TestDescriptiveQueriesRouteToCaption` | descriptive queries reach caption; a description is not mistaken for change; caption is single-asset |
| 14 | `TestTheOpticalSarInputValidation` | a real optical+GeoTIFF SAR pair is ok; two plain photos warn; a missing SAR image is an error; the warning names the optical+radar expectation |
| 15 | `TestTheServerErrorTranslation` | invalid request gets asset guidance; 422 gets malformed guidance; recoverable `false` gets restart guidance; the server's message and detail are preserved; a null error is handled; a recoverable transport code is not called unrecoverable |

Two of these deserve a note because they encode *counter-intuitive* decisions:

**`TestTheFrontendSendsOnlyTheAssetsTheTaskRequires` (class 12).** The class contains
`test_the_old_wiring_would_have_sent_both_files_for_vqa` and
`test_the_fix_sends_one_file_where_the_old_sent_two`. This is a *differential* test: it asserts not just
that the new code is right, but that the *old* code was wrong in the specific way claimed. That is the
falsification discipline (§1.3) applied inside a suite — the fix is only meaningful if the defect it
fixes is real, and the test proves the defect by asserting the old behaviour differs.

**`TestTheCaptionStopsCallingARealUploadIllustrative` (class 11).** This is a *copy-honesty* test. The
page used to caption a real analysis "illustrative"; the suite asserts the leading word and the alt text
are rewritten on success, and — crucially — that the **failure path does not claim an analysis
happened** (`test_the_failure_path_does_not_claim_an_analysis_happened`). A UI that says "analysis
complete" on a failure is a claim defect, and it is tested as one.

### 5.4 The historical count: 94 → 100 → 106

A reviewer may see a different number. `docs/FINAL_DELIVERY_REPORT.md §7` records **94 passed** for this
file, while the current figure is **106**. The difference is the suite growing after the delivery report
was written: the regression tests added for the router fix took this file from **100 → 106** tests
(`DELIVERY_REPORT_2026-09-25.md`, regression-tests note). This chapter quotes **106** and cites both
figures so neither looks like a contradiction (`docs/REPRODUCIBILITY.md` §5.2).

### 5.5 What the frontend live-wiring suite does NOT establish

- It does **not** run the browser. It asserts the *shipped bytes* of the frontend against the
  *orchestrator module* in-process. Whether the page actually loads and runs is §8's job (live
  validation).
- It does **not** prove the router's *quality* — only that specific questions route to the specific
  tasks asserted. The router's residual misroutes (`"What is the new runway?"` → `change`) are real and
  recorded (`docs/LIMITATIONS.md`; `LIVE_VALIDATION_POSTFIX.md`), not hidden by a green suite.
- It does **not** test the inference service. It tests the boundary the frontend speaks to, which is the
  orchestrator.

---

## 6. The doc/frontend suite

### 6.1 The five files, and the measured result

The "doc/frontend suite" is a named grouping of **five files** that together report **183 passed**
(`docs/FINAL_DELIVERY_REPORT.md §7`; `docs/EVALUATION.md` §7.2; `docs/REPRODUCIBILITY.md` §5.3). The
five are:

| File | What it pins |
|---|---|
| `tests/unit/test_frontend_guide_doc.py` | both documents agree on tasks, coordinate systems, evidence types, geometry shapes; no login screen; no secrets in the browser; the confidence-property trap; serialized analyses |
| `tests/unit/test_frontend_live_wiring.py` | the frontend↔orchestrator boundary — see §5 |
| `tests/unit/test_api_contract_doc.py` | every JSON block parses and validates; no contract example fabricates an artifact URI; documented task/coordinate-system values are the real ones; every error code documented; the frozen config hash is recorded; the calibration caveat is recorded |
| `tests/unit/test_runbook_doc.py` | the runbook names only public serving entrypoints; quoted artifact sizes match the real files; the capability list matches the registry; env vars are the architecture ones; undecided items declared undecided; **no claim that a deployment happened**; verified distinguished from designed |
| `tests/unit/test_deploy_config.py` | the deploy manifest declares the non-registry marker; the validator reports no errors; the **config-hash regression guard**; the C-8 constraints (`torch_compile` false, cpu mode required, lazy load, single model cache) |

These are **documentation conformance tests**: they fail if a doc drifts from the code it describes.
They are the mechanism that keeps this release's documentation honest — the reason a doc claim can be
trusted is that a test would go red if the claim stopped matching the code.

### 6.2 Why four of the five are doc-guards, and one is not

Four of the five (`test_frontend_guide_doc`, `test_api_contract_doc`, `test_runbook_doc`,
`test_deploy_config`) are doc-guards, analysed individually in §3. The fifth
(`test_frontend_live_wiring`) is not a doc-guard — it is a code-behaviour suite that happens to be
grouped here because it reads frontend source text. The grouping is by *audience* (frontend +
documentation) rather than by mechanism, and that is why the count 183 covers both.

### 6.3 The doc-guard guarantee, restated

The doc-guards guarantee **example-level and value-level** conformance, not prose truth (§3.7). The two
module docstrings state the limit themselves:

> *"These tests validate the EXAMPLES, not the prose. They cannot detect a wrong sentence (e.g. a
> mis-stated latency budget)."* — `tests/unit/test_api_contract_doc.py`

> *"Prose can still be wrong in ways these tests cannot see -- a mis-stated latency, a wrong reason for a
> decision."* — `tests/unit/test_frontend_guide_doc.py`

`test_runbook_doc.py` reaches a different class of claim — numbers, names, and **status honesty** — and
its most important assertion is that the document itself *labels its unimplemented sections as
unimplemented* (§3.4). That is the guard against the exact overclaim the style guide forbids.

### 6.4 The config-hash regression guard

`test_deploy_config.py` carries the anti-drift move that makes it more than a validator: it does **not**
re-declare the frozen config hash. It imports it:

> *"We import it rather than re-declare it, so the two cannot drift."*
> — `tests/unit/test_deploy_config.py`

The authority is `tests.test_config`, and the value is the frozen hash `78f1e3700da15aa1`
(`DOCS_STYLE_GUIDE.md` §3). A second copy of the hash anywhere in the tree would be a latent
contradiction; the import makes drift structurally impossible.

The same file proves the manifest is **INERT** — *"it is never merged into the config registry and
cannot move `Config.hash`"* — and that negative cases use in-memory dicts via `validate_documents` *"so
the real config files are never mutated."* It also drives the validator CLI **in a child interpreter**
to prove the module's own `sys.path` bootstrap makes `core` importable *"WITHOUT pytest's `pythonpath =
.` shortcut"* (`test_validator_cli_exits_zero_as_a_standalone_script`).

### 6.5 What the doc/frontend suite does NOT establish

- It does **not** establish that any deployment exists. `test_runbook_doc.py` explicitly asserts the
  opposite direction: that the document does **not** claim a deployment happened (§3.4).
- It does **not** validate prose (§6.3).
- It does **not** check that the *code* is correct — only that the docs match the code. If the code and
  the docs are wrong together, a doc-guard is green.

---

## 7. The full `tests/unit` result, and the 5–6 environmental failures

This is the section where the release's honesty discipline is most visible. Running the entire unit tree
produces failures. They are reported, classified, and *not* reclassified as passes.

### 7.1 The result

Running `python -m pytest tests/unit` in the authoring sandbox produces **5–6 failures**. Every one is
**environmental or ordering**-related, not a regression in shipped code. The classification, exactly as
`docs/FINAL_DELIVERY_REPORT.md §7` records it:

| # | Failure | Attribution | Regression? |
|---|---|---|---|
| 1–4 | `test_safe_delete_shim` — **4 failures** | the sandbox's bulk-**delete guard** (Windows verbatim-path behaviour) | **No** |
| 5 | one **ordering flake** in the router route test | test **ordering** (passes in isolation) | **No** |
| 6 | one **stale adapter test** (`optical_sar` absent when CROMA unshipped) | stale test — **CROMA is now shipped** | **No** |

### 7.2 The evidence that they are not regressions

The evidence is the **re-run**. Re-running the affected files **together** gives **137 passed**
(`docs/FINAL_DELIVERY_REPORT.md §7`; `docs/EVALUATION.md` §7.3–7.4). The logic is stated plainly:

> *"if the failures were real regressions in the code under test, re-running those files together would
> still fail. They do not — which is what distinguishes an environmental/ordering failure from a
> regression."*

This is the falsification discipline (§1.3) applied to a *test result* rather than to a *guard*: the
claim "these are not regressions" is itself tested, by a re-run designed to fail if it were false.

### 7.3 The four `test_safe_delete_shim` failures, in detail

`tests/unit/test_safe_delete_shim.py` (338 lines, `pytestmark = pytest.mark.unit`) guards two defects in
the WorkBuddy Windows safe-delete shim (`cli/vendor/shim/sitecustomize.py`) that produced **phantom test
failures** in this repository. The module's docstring records both with measurements.

**Defect 1 — a verbatim temp path was not recognised as a temp path.**

> *"`_path_for_compare` returned `normcase(realpath(abspath(path)))`, which preserves the Windows verbatim
> prefix `\\?\`. `os.path.relpath` cannot relate `\\?\c:\...` to `c:\...`, so `_is_under_root` returned
> False and the OS-temp exemption in `_should_bypass_safe_delete` did not apply."*

Measured, before the fix:

```
plain    temp subdir  -> bypass True
verbatim temp subdir  -> bypass False      <-- the bug
non-temp subdir       -> bypass False
```

That is why routine pytest `garbage-*` collection — which walks
`\\?\C:\Users\...\Temp\pytest-of-anish\garbage-*` — reached the bulk guard, tripped `confirmRequired` at
**69 entries against a threshold of 50**, and **latched a rejection that then blocked every delete in
the conversation**.

**Defect 2 — a successful delete was reported as a failure.**

> *"`_platform_trash` raised whenever `SHFileOperationW` returned non-zero."*

Measured on the authoring host with `FOF_ALLOWUNDO|FOF_NOCONFIRMATION|FOF_NOERRORUI|FOF_SILENT`, calling
shell32 directly (the shim not involved):

```
series                                   non-temp rc      temp rc
4 regions x 3 interleaved rounds         2, 2, 2, ...     0, 0, 0
6 processes x 20 deletes                  2 (120/120)      0 (120/120)
```

The target is removed in **every** case and the user's Recycle Bin is populated and active (1,300+
`$I`/`$R` entries), so the return code is **not** a reliable "could not delete" signal. The module is
explicit that the code is **intermittent** and that the precise Windows-internal trigger is not known:

> *"across pytest invocations the same non-temp delete sometimes returned 0, and in one run a *temp*
> delete returned non-zero. The precise Windows-internal trigger is **NOT established** and is not
> claimed here."*

This is a place where this chapter writes **`UNKNOWN — not established from the available evidence`**,
following the module's own honesty.

**What the guard asserts** (the `WHAT IS ASSERTED` block of the docstring):

- A verbatim-prefixed path inside the OS temp root compares equal to its plain form and is exempted.
- A verbatim-prefixed path *outside* the temp root is still guarded — *"the fix normalises the prefix, it
  does not widen the exemption."*
- Deleting through a verbatim temp path succeeds end to end.
- A non-zero shell code on a delete whose target is genuinely gone does not raise (stubbed shell, stubbed
  `lexists` — deterministic, deletes nothing).
- A non-zero shell code on a delete whose target **survives** still raises — *"Fail-closed is preserved;
  only the false positive is gone."*
- Against the real shell, a non-temp delete does not raise (behavioural confirmation, not the guard).

**Why the guard is stub-based.** The module states the reason, and it is a direct consequence of defect
2's intermittency:

> *"the guard for this defect stubs the shell: a test that depends on the real return code would
> sometimes pass against the broken shim. The stub tests fail on the original shim on every run."*

That is the difference between a test that *sometimes* catches a defect and a test that *always* does.

**The rule for extending the file** is stated in the docstring and is worth reproducing, because it is
the exact failure this file exists to prevent:

> *"**Do not perform a real delete outside the OS temp root.** … An earlier draft of the defect-2 test
> did exactly that, tripped `confirmRequired` at the threshold and latched a rejection that blocked every
> subsequent delete in the session — the very failure this file exists to prevent."*

The module also **skips cleanly** when the shim is absent or disabled, via
`pytest.importorskip("sitecustomize", ...)` plus `skipif` markers (`needs_shim_helpers`, `windows_only`).
So on a normal user environment these tests skip; in the authoring sandbox they reach the guard and 4 of
the 8 fail. The constants involved are `VERBATIM = "\\\\?\\"` and a `NON_TEMP` probe path that is *"never
created"* — the predicates tested are pure.

### 7.4 The ordering flake

One router route test fails only when the full suite runs in a particular order, and **passes in
isolation**. That is the definition of an ordering flake: the failure is a function of collection order,
not of the code. `docs/REPRODUCIBILITY.md` §5.4 records it in the same row-set as the shim failures.

### 7.5 The stale adapter test

One test asserts `optical_sar` is **absent when CROMA is unshipped**. CROMA is now shipped
(`CURRENT_RELEASE_STATE.md` §1 capability contract: `optical_sar` `available: true`), so the assertion's
premise has changed. This is a **stale test**, not a code defect: the test encodes an assumption about
the environment that the environment has outgrown.

### 7.6 A different environment state: the 2,237-passed run

`docs/PHASE12_115_METRIC_COMPUTED.md` §8 records, for **2026-09-22**, a full unit suite result of
**2,237 passed, 0 failed (622.68 s)**. That is a real, measured result in a *different* environment state
(the safe-delete shim's guard state and the CROMA shipping status differ). It is recorded rather than
suppressed, *"because the two results are both true and the difference is exactly the
environmental/ordering story"* (`docs/EVALUATION.md` §7.3).

The same document records a correction that is itself an honesty lesson: an earlier revision claimed
`2,179 passed` computed as `2,171 + 8`, *"presented as a check but the total was never measured — it was
inferred from a stale baseline."* The measured figure was **2,237**. This is the style guide's rule
(`DOCS_STYLE_GUIDE.md` §1.3) in the wild: an inferred number presented as a measurement was corrected to
the measured one.

### 7.7 The UNKNOWN: the collected count

The exact **collected** test count for the current-session full-suite run is
**`UNKNOWN — not established from the available evidence`**. The record gives the failure classification
and the 137-passed re-run, but not a collected total. Per-suite counts are known (106, 183, 137, 73, 51);
the single collected total is not.

### 7.8 Why this is reported rather than hidden

The style guide's first rule is `DO NOT FABRICATE`, and its status rule is *"a mixed result is never 'all
work perfectly'"* (`DOCS_STYLE_GUIDE.md` §1.7). A full-suite run with 5–6 failures is a mixed result.
Presenting only the 106 and 183 would be exactly the "all work perfectly" overclaim the guide forbids.
The correct presentation is the three-row table (§7.1) plus the re-run evidence (§7.2) plus the explicit
UNKNOWN (§7.7) — which is what this section is.

---

## 8. The live-validation harness and its integrity discipline

### 8.1 What live validation is, and why it is separate from the suites

Every suite in §2–§7 runs in-process. `docs/API_CONTRACT.md` §8 states the boundary plainly: *"no test in
this repository dials a network address."* A green `tests/integration` run is therefore **not** evidence
about a deployment. **Live validation** is the separate activity that drives the *deployed* stack — the
real Cloudflare Pages frontend, the real Render orchestrator, the real tunnel, the real Codespace — with
a browser, and records what happened.

The record lives at `.workbuddy-ai/scratch/live_validation/` — `LIVE_VALIDATION_POSTFIX.md`, the raw
harness outputs (`run_output.txt`, `run_final2.txt`, `run_final3.txt`), the recomputed verdicts
(`results_final.json`, `results_pass3.json`), and eight per-case screenshots (`A1_vqa.png` … `B2_newairport.png`).

### 8.2 The three passes, summarised

| Pass | HEAD | Harness | Result |
|---|---|---|---|
| 1 | `ff46eba42b18` + `d413d3672311` | v1 (`fill_input`) | 8/8 |
| 2 | `2d7ae53b482d` | v2 asserting | 8/8 |
| 3 | `2d7ae53b482d` | v2 asserting (pre-discriminator-fix) | 8/8 (recomputed) |

**24 live runs, 24 correct dispatches, 0 mock nodes, no run id repeated across passes.** The trace fill
was **94.4444 %** on every case, and every case carries a real `run_*` id, the Hugging Face link in the
DOM, and the `capabilities → assets → infer` call sequence all addressed to
`<backend-host>` (two `assets` calls for the pair tasks).

The eight cases are Phase A (six regression cases: `vqa`, `caption`, `grounding`, `change`,
`change_vqa`, `optical_sar`) and Phase B (the two defect cases: `"Where are the built-up areas in this
image?"` and `"Where is the new airport?"`, both expected to reach `grounding` with a **single** asset —
the exact condition under which the old router collapsed to `vqa` and answered "River").

### 8.3 The CDP synthetic-key-event trap

This is the integrity lesson that shaped the harness, and it is the most transferable finding in this
chapter.

**The shape of the false pass.** Pass 1's harness drove the query box with `fill_input()`, which types
with **real CDP key events**. A re-run attempt failed on case 1 with `run_id=0002`, `mock_nodes=9`,
`answer="No answer yet"`, and only a `capabilities` call — the **mock path**. The root cause, measured
directly:

> *"`fill_input` types with **real CDP key events**, and Chrome **drops synthesized key events when the
> browser window does not hold OS focus**; it has **no assertion**, so it clicked Run with the page's
> default query still in the box. Measured directly: with Chrome backgrounded, `press_key("Z")` left
> `#qtext.value` unchanged, while `type_text("Q")` (CDP `Input.insertText`, not focus-gated) inserted
> fine."*
> — `LIVE_VALIDATION_POSTFIX.md`

The measurement is recorded in `diag_focus.harness`, which prints the value of `#qtext` after
`press_key("Z")` and after `type_text("Q")` — showing the first is a no-op and the second works.

**The generalisation.** A harness that types into a form and then reads a *result* cannot tell the
difference between:

- "the query was entered, and the run used it", and
- "the query was never entered, and the run used the page's default".

Both produce a result. The result of the second is a *plausible* answer to a *different* question, and a
harness without an assertion records it as the verdict for the first. That is a **silent false pass**,
and it is the reason the fix is not "use a different typing helper" but "**assert the form state before
dispatch**".

### 8.4 The fix: deterministic query entry plus pre-dispatch assertions

The harness was rebuilt as `run_all_postfix2.harness`, whose header states the change:

> *"v2 sets the query deterministically (js value + input event, falling back to `Input.insertText`) and
> ASSERTS the form state before clicking, recording `q_ok` / `obs_ok` / `t0_ok` and a computed verdict
> per case."*

The three assertions, and the failure each prevents:

| Assertion | Checks | Failure it prevents |
|---|---|---|
| `q_ok` | `#qtext.value` holds the intended query | the silent-drop false pass |
| `obs_ok` | `#obsTail == 'ready'` **and** one file on `#fileInput` | an upload that did not land |
| `t0_ok` | both frames present, for pair tasks | a pair task run on one asset |
| `no_mock_nodes` | `mock_nodes == 0` | the preview path being recorded as live |

The rule, stated in the session handoff (`HANDOFF_NEXT_AGENT.md` §5.2) and reproduced in
`docs/architecture/09-frontend.md` §10.3:

> *"**Do NOT use `fill_input()` or `press_key()` to enter the query.** They type with real CDP key events,
> which Chrome **silently drops when the browser window does not hold OS focus** … Use `js()` to set
> `#qtext.value` (plus `input`/`change` events) and/or `type_text()` (CDP `Input.insertText`, not
> focus-gated). **Always assert the form state before clicking Run** … or a no-op will be recorded as a
> pass."*

The `set_query()` function in the harness implements exactly this: it focuses the box, clears it, tries
`type_text(q)` (the non-focus-gated CDP path), reads the value back, and **falls back** to a direct
`js` set with `input`/`change` events if the read-back does not match. It returns the value actually read
back, and `q_ok = (qgot == c["q"])`. The assertion is on the *read-back*, not on the *attempt*.

`obs_ok`'s second half is a real DOM fact: `#obsTail` reads `none` in the markup (`mission.html:67`) and
the live driver flips it to `ready` on upload, so `obs_ok` proves the upload landed rather than merely
that a click happened.

### 8.5 The three discriminators that proved the earlier 8/8 was clean

The response to a silent-false-pass risk is not to re-run and hope; it is to check whether the *earlier*
result was contaminated, using evidence the harness recorded. The report does exactly that:

> *"I then checked whether the earlier 8/8 run was infected by the same silent failure. It was not:*
>
> * *its recorded intents are **query-specific** — A1 reads `taskvqa…temporalnone`, whereas the default
>   query "What changed here?" would read `taskchange…temporalrequired` (exactly what the failed run
>   showed);*
> * *its answers **embed the query text** — e.g. `[grounding] Located 6 candidate region(s) for 'Where
>   are the built-up areas in this image?'`;*
> * *A6 required two files (`optical 4/12 + SAR 2/2` channels), which only the uploaded pair supplies.*
>
> *So the 8/8 result is a valid measurement."*
> — `DELIVERY_REPORT_2026-09-25.md` §3

The three discriminators generalise into a reusable checklist:

| Discriminator | What it proves |
|---|---|
| the recorded **intent** is query-specific | the query reached the router |
| the **answer** embeds the query text | the server received the intended query |
| a case **requires an artefact** only the setup supplies | the setup really happened |

This is why the harness records the full per-case payload — `intent`, `answer`, `files_t1`, `files_t0`,
`obs_tail`, `mock_nodes`, `api_calls` — rather than just a pass/fail bit. The bit can be wrong; the raw
record can be re-examined.

### 8.6 The two discriminator bugs (and why they were false *failures*, not false passes)

Two harness bugs were found and fixed, both causing **false failures** — the safe direction:

1. **The answer `[task]` tag exists only for region tasks.** `vqa`/`caption` answers are bare
   ("Grassland", prose), so a tag-based verdict heuristic reported them as failures. The fix computes the
   dispatched task as `answer_tag` when present, else the intent's reading.
2. **The intent panel is a concatenated string.** `task([a-z_]+)` must be matched **non-greedily** up to
   `modality`; a greedy match swallows the whole string. The fix is
   `re.search(r"task([a-z_]+?)modality", intent)`.

The harness header states the asymmetry: *"Two further harness bugs were found and fixed, both causing
**false failures**."* A false failure costs a re-run; a false pass costs the integrity of the result. The
harness was built so that its bugs err toward false failure, and §9 makes even the false-failure case
recoverable without a re-run.

### 8.7 Liveness traps discovered during the runs

Two operational traps are recorded because they look exactly like stalls:

- **`browser-use` block-buffers stdout even when redirected to a file**, so the output file stays at
  **0 bytes until the process exits** — indistinguishable from a stall. The harness now forces
  line-buffering (`sys.stdout.reconfigure(line_buffering=True)`) and prints `CASE_START <id>` per case,
  so the file grows case by case.
- **`grep` block-buffers when piped**, so piping the harness through `grep` swallows all output if the
  pipeline is killed. Redirect to a file instead.

The harness also records `ALL_DONE` as a final marker, so a truncated run is detectable from the output
file alone.

### 8.8 What the live validation does NOT establish

- It is **not** the independent audit. `LIVE_VALIDATION_POSTFIX.md` opens by saying so: *"This is a
  *re-validation*, not the independent audit: the prior independent audit is
  `../2026-09-25-00-37-41/live_validation/INDEPENDENT_LIVE_VALIDATION.md`."*
- It does **not** establish model *quality*. The two verdicts are kept separate: **Deployment/integration:
  PASS** and **Model quality: MIXED** (caption and grounding meaningful; change/change_vqa plausible; VQA
  weak-but-related — A1 answers "Grassland"; optical-SAR returns a bare class index `class_18`, not a
  human label).
- It does **not** establish load behaviour, latency ceilings, or cold-start times. It is eight cases, run
  three times.
- It does **not** clear B-07: transient tunnel-agent gaps remain **OPEN**, and the patch is **prepared,
  NOT deployed** (`CURRENT_RELEASE_STATE.md` §6).

---

## 9. Verdict-independent evidence: recomputing from raw records

### 9.1 The property

The live-validation harness has a `verdict` field. The project's design decision is that **the verdict
must not be the primary record.** The recorded evidence — `q_ok`, `obs_ok`, `t0_ok`, `run_id`,
`mock_nodes`, `intent`, `answer`, `api_calls` — is recorded *independently of* the verdict computation,
so a harness bug in the verdict logic can never silently turn a real failure into a pass, and can be
corrected without a re-run.

### 9.2 The recomputation script

`.workbuddy-ai/scratch/recompute_verdicts.py` implements the property. Its docstring states the case that
motivated it:

> *"The v2 harness recorded, per case: `q_ok`, `obs_ok`, `t0_ok`, `run_id`, `mock_nodes`, `intent`,
> `answer`, `api_calls` -- all independent of the verdict computation. Its verdict field was wrong for
> vqa/caption because the tag heuristic assumed every answer starts with `"[task]"`; vqa/caption answers
> are bare. This recomputes the verdict from the intent's `task<name>` slot (primary) plus the answer tag
> (cross-check only when present)."*

The script is read-only over the recorded run and writes `results_final.json`. The core of it:

```python
mi = re.search(r"task([a-z_]+?)modality", intent)
intent_task = mi.group(1) if mi else ""
tag = answer[1:answer.find("]")] if (answer.startswith("[") and "]" in answer) else ""
expect = d["expect"]

# The DISPATCHED task is what the server actually ran. The answer carries a "[task]"
# prefix for the region tasks (grounding/change/change_vqa/optical_sar); vqa and
# caption answers are bare, so fall back to the intent's reading there.
# NOTE: A5 is a legitimate case where the two differ -- the router READS "change" but
# dispatches "change_vqa" (the documented quantifier upgrade). That is not a failure.
dispatched = tag if tag else intent_task

checks = {
    "query_in_box": bool(d.get("q_ok")),
    "asset_attached": bool(d.get("obs_ok")),
    "pair_attached": bool(d.get("t0_ok")),
    "live_run": str(d.get("run_id", "")).startswith("run_"),
    "no_mock_nodes": d.get("mock_nodes") == 0,
    "task_dispatched": dispatched == expect,
}
verdict = "PASS" if all(checks.values()) else "FAIL"
```

Every input to the verdict is a *recorded observation*; the verdict is a pure function of them. The
script also records `read_vs_dispatched_differs` — so the one legitimate read≠dispatch case (A5, the
documented quantifier upgrade) is **flagged, not failed**.

### 9.3 Pass 3 is the proof of the property

Pass 3 was launched with the harness build that still carried the two discriminator bugs (§8.6), so its
raw output says **`SUMMARY 0/8`**. The verdicts were recomputed from the recorded evidence by
`recompute_verdicts.py` and are **8/8 PASS** (`results_pass3.json`). The record states the significance:

> *"That is the intended safety property — **the recorded evidence (run id, `mock_nodes`, intent, answer,
> assertions) is independent of the verdict computation**, so a harness bug can never silently turn a
> real failure into a pass, and never costs a re-run to correct."*
> — `LIVE_VALIDATION_POSTFIX.md`

This is the strongest form of the claim: the property was not merely designed, it was **exercised**. A
run with a broken verdict computation was corrected *post hoc* from its own raw record, without touching
the deployment or re-running the browser.

### 9.4 The generalisation

The pattern is: **record observations, compute verdicts separately, keep the observations.** It is the
same shape as the evidence engine's digest (§4.8): the *record* is content-addressed and process-independent,
and the *interpretation* is a function over it. Both make the same guarantee — that a conclusion can be
re-derived from the evidence rather than trusted as an assertion.

---

## 10. How to run the tests

### 10.1 The interpreter

Use the repository virtual environment, not a bare `python`. On Windows:

```bash
.venv/Scripts/python.exe -m pytest <target> -q
```

`docs/REPRODUCIBILITY.md` §10.2 records this as the invocation that reproduces the published results. On
Windows, pytest must be given `-p no:cacheprovider` because the sandbox refuses `.pytest_cache` writes
(`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §0).

### 10.2 Run targeted files, not the whole tree

**A full `tests/unit` run trips the sandbox's bulk-delete guard** (the 4× `test_safe_delete_shim`
failures of §7.3). The workaround is to run targeted suites:

```bash
.venv/Scripts/python.exe -m pytest tests/unit/test_frontend_live_wiring.py -q     # 106 passed
.venv/Scripts/python.exe -m pytest tests/unit/test_evidence_engine.py --basetemp=.pytest_tmp -q
.venv/Scripts/python.exe -m pytest tests/unit/test_deploy_config.py -p no:cacheprovider -q
.venv/Scripts/python.exe -m pytest tests/unit/test_api_contract_doc.py tests/unit/test_frontend_guide_doc.py -p no:cacheprovider -q
```

The runbook notes that **multi-suite invocations in one command have been refused by the environment
before; single suites are reliable** (`docs/BACKEND_DEPLOYMENT_RUNBOOK.md` §2.6). The `--basetemp`
flag is **mandatory** for the evidence-engine suite (§4.7): the default pytest temp root triggers a
sandbox denial on this machine, and the module's `scratch` fixture writes to a repo-local `.pytest_tmp`
instead.

### 10.3 `pytest.ini`

The configuration is four lines plus the marker registrations:

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -q --tb=short
```

(`pytest.ini:1-4`). `pythonpath = .` is what makes `from core.config import ...` resolve from the repo
root without an installed package — and `test_deploy_config.py` deliberately proves the module *also*
bootstraps `sys.path` itself, *"WITHOUT pytest's `pythonpath = .` shortcut"* (§6.4). The markers are
registered at `pytest.ini:23-27` (§0.1), and `filterwarnings` silences three warning classes including
`rasterio.errors.NotGeoreferencedWarning` (`pytest.ini:5-8`).

### 10.4 The output-buffering trap

**Do not pipe pytest through `grep`** in the authoring sandbox — output is block-buffered and a killed
pipeline swallows it. Redirect to a file instead (`docs/REPRODUCIBILITY.md` §5.6). The same trap applies
to the live-validation harness (§8.7).

### 10.5 The expected results

| Suite | Command | Expected | Status |
|---|---|---|---|
| Frontend live-wiring | `pytest tests/unit/test_frontend_live_wiring.py -q` | **106 passed** | `VERIFIED` |
| Doc/frontend suite | `pytest` on the 5 doc/frontend files | **183 passed** | `VERIFIED` |
| Evidence engine | `pytest tests/unit/test_evidence_engine.py --basetemp=.pytest_tmp` | **73 passed** | `VERIFIED` |
| Gateway policy | `pytest tests/unit/test_gateway_policy.py` | **51 passed** | `VERIFIED` |
| Full unit suite | `pytest tests/unit` | **5–6 environmental/ordering failures**, rest pass; a re-run passes **137** | `MEASURED` |

(`release/repo/README.md` §The test suites; `docs/PHASE19_FINAL_HARDENING.md` §6;
`docs/REPRODUCIBILITY.md` §5.1.) The precise full-suite **collected** count is
`UNKNOWN — not established from the available evidence` (§7.7).

---

## 11. What is NOT tested

A green suite is not evidence about a property nobody wrote an assertion for (`DOCS_STYLE_GUIDE.md` §1.6;
§0.3). This section lists the properties the release does **not** have test evidence for, so that no
reader mistakes coverage for completeness.

| Property | State | Why / what exists instead |
|---|---|---|
| **Continuous integration** | **does not exist** | there is **no `.github/` directory** and no CI workflow in the repository. Every test result in this chapter was produced by a **manual** run in the authoring sandbox. Nothing runs the suites automatically on push. |
| **End-to-end system benchmark** | **NOT RUN (none exists)** | `CURRENT_RELEASE_STATE.md` §4: *"System-level end-to-end benchmark — **NOT RUN (none exists)**."* No system-level accuracy is claimed anywhere in the release. |
| **Load / soak / concurrency** | **NOT RUN** | no test exercises sustained load, concurrent users, or a long-running process. The per-IP rate limiter is a **fairness** control, not a security or capacity control (`docs/DEPLOYMENT_ARCHITECTURE.md` §5.2). |
| **Adversarial / fuzz testing** | **NOT RUN** | no fuzzing or adversarial-input suite. Input validation is tested by example (§2.2.9), not by search. |
| **Cross-dataset generalization** | **NOT RUN** | the measured metrics are per-dataset (LEVIR-CD-256, VRSBench, held-out test sets); no test measures transfer to an unseen dataset. |
| **Browser / device matrix** | **NOT RUN** | live validation ran in **headed Chromium** only. No Firefox/Safari/WebKit, no mobile, no accessibility audit. |
| **Deployment verification in CI** | **NOT RUN** | the runbook guard asserts the document does **not** claim a deployment happened (§3.4). Deployment reachability is established only by the manual live validation (§8). |
| **Router test split** | **NOT RUN** | `CURRENT_RELEASE_STATE.md` §4: router accuracy `0.965116` is **validation, ungated, n = 86**; *"the test split was NOT RUN"* (`DOCS_STYLE_GUIDE.md` §3). |
| **Calibration as an improvement** | **not established (and measured the other way)** | ECE went `0.013755 → 0.014929` — **worse**. The engine suite pins that calibration is *honestly labelled* (§4.5), not that it helps. |
| **Optical-SAR / change-VQA acceptance** | **OPEN** | the rulings are OPEN (`CURRENT_RELEASE_STATE.md` §4; `DOCS_STYLE_GUIDE.md` §3). A suite passing is not a model acceptance. |
| **VLM adapter acceptance** | **REJECTED** | metrics `USABLE_VERIFIED` (exact_match 0.963) but status **ACCEPTANCE-REJECTED**. USABLE ≠ ACCEPTED (`DOCS_STYLE_GUIDE.md` §3). |
| **Tunnel reliability (B-07)** | **OPEN** | transient tunnel-agent gaps; patch **prepared, NOT deployed**. No test can cover a defect that is not fixed in production. |
| **The inference service's internals** | **UNKNOWN** | the tunnel-agent source is not in the monorepo; `docs/architecture/02-deployment-topology.md` §3.3/§9.4 documents the topology as a **measured** fact, but the agent's internals are `UNKNOWN — not established from the available evidence`. |
| **Credential write permission (HF)** | **UNKNOWN** | `CURRENT_RELEASE_STATE.md` §7: HF token identity `thundercode`, role `fineGrained`; *"write permission not yet proven."* |

Two of these deserve emphasis:

- **No CI is the largest gap.** Every count in this chapter is a manual measurement. A reader should not
  read "106 passed" as "the suite passes on every commit" — it means "it passed when it was run, in the
  recorded environment." There is no automation that would catch a regression on push.
- **No end-to-end benchmark is the second.** The individual task metrics are artifact-backed and
  measured, but there is no measurement of the *system* — router + specialist + envelope + frontend — on
  a held-out end-to-end corpus. The live validation (§8) proves the *pipeline runs and dispatches
  correctly*; it does not measure system accuracy.

---

## 12. Open items and evidence index

### 12.1 Open / blocked / not-run items for this chapter's topic

Following the style guide's requirement that every doc end with an explicit list
(`DOCS_STYLE_GUIDE.md` §4):

| Item | State | Note |
|---|---|---|
| B-07 — transient tunnel-agent gaps | **OPEN** | patch **prepared, NOT deployed**; root shape measured (in `auto` mode a tunnel timeout falls through to the forward path, burning `wake_timeout_s=120` on a 302 ≈ 249 s) |
| B-02 — `codespace_name` trailing `\n` | **OPEN (cosmetic)** | confirmed still live; the wake path strips it |
| No CI | **does not exist** | no `.github/` directory; all results are manual |
| System-level end-to-end benchmark | **NOT RUN (none exists)** | no system-level accuracy claimed |
| Router test split | **NOT RUN** | val-only, n = 86 |
| Full-suite collected test count | **UNKNOWN** | per-suite counts known; the single collected total is not |
| `test_safe_delete_shim` failures | **environmental** | sandbox delete-guard; not regressions; precise Windows trigger **UNKNOWN** |
| Ordering flake | **environmental** | passes in isolation |
| Stale adapter test | **stale** | asserts CROMA unshipped; CROMA is shipped |
| Live validation | **MEASURED** | 3 passes × 8 cases, 8/8 each, 24 runs, 0 mock nodes, trace fill 94.4444 % |
| Optical-SAR / change-VQA rulings | **OPEN** | 0.931 / macro-F1 0.434161; 0.697626 / 0.378373 |
| VLM adapter acceptance | **REJECTED** | metrics usable; acceptance rejected |
| Calibration | **not an improvement** | ECE worse (0.013755 → 0.014929); retained only because it is in the frozen config |

### 12.2 Where the evidence lives

| What | Where |
|---|---|
| Test configuration and markers | `pytest.ini` |
| Frontend live-wiring suite | `tests/unit/test_frontend_live_wiring.py` — **106 passed** |
| Evidence-engine suite | `tests/unit/test_evidence_engine.py` — **73 passed** |
| Safe-delete shim guard | `tests/unit/test_safe_delete_shim.py` — 338 lines; skips cleanly off-sandbox |
| Doc-guards | `tests/unit/test_api_contract_doc.py`, `test_frontend_guide_doc.py`, `test_runbook_doc.py`, `test_deploy_config.py`, `test_step7_api_contract_conformance.py` |
| Integration suite scope | `tests/integration/test_step7_backend_chain.py`, `test_step7_r02_serving.py`; `docs/ITEM5_INTEGRATION_SUITE_SCOPE.md` |
| The suite results, reported in full | `docs/EVALUATION.md` §7.1–7.5; `docs/REPRODUCIBILITY.md` §5; `release/repo/README.md` §The test suites |
| The full-suite failure classification | `docs/FINAL_DELIVERY_REPORT.md` §7; `docs/PHASE12_115_METRIC_COMPUTED.md` §8 (the 2,237-passed run) |
| Live-validation record | `.workbuddy-ai/scratch/live_validation/LIVE_VALIDATION_POSTFIX.md` |
| Live-validation raw output | `.workbuddy-ai/scratch/live_validation/run_output.txt`, `run_final2.txt`, `run_final3.txt` |
| Live-validation screenshots | `.workbuddy-ai/scratch/live_validation/A1_vqa.png` … `B2_newairport.png` (8) |
| Live-validation harness (v1) | `.workbuddy-ai/scratch/run_all_postfix.harness` |
| Live-validation harness (v2, asserting) | `.workbuddy-ai/scratch/run_all_postfix2.harness` |
| CDP focus diagnostic | `.workbuddy-ai/scratch/diag_focus.harness` |
| Verdict recomputation | `.workbuddy-ai/scratch/recompute_verdicts.py` |
| Prior independent audit | `../2026-09-25-00-37-41/live_validation/INDEPENDENT_LIVE_VALIDATION.md` |
| Release-state inventory | `release/CURRENT_RELEASE_STATE.md` |

### 12.3 Cross-references

- The trust model, secret custody, CORS allowlist, size limits and rate-limiting-as-fairness that the
  frontend suite asserts against: [`SECURITY.md`](SECURITY.md).
- The API contract the doc-guards validate: [`API_CONTRACT.md`](architecture/08-api-contract.md).
- The deployment topology the live validation drives: [`DEPLOYMENT.md`](DEPLOYMENT.md),
  [`architecture/02-deployment-topology.md`](architecture/02-deployment-topology.md).
- The measured metrics this chapter does **not** re-derive: [`EVALUATION.md`](EVALUATION.md).
- The reproduction commands and their expected outputs: [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §5.
- The router defect the live validation re-validates: [`architecture/04-router.md`](architecture/04-router.md),
  [`architecture/09-frontend.md`](architecture/09-frontend.md) §10.
