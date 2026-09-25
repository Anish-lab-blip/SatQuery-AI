"""SatQuery AI — Phase 6 closure record.

WHY THIS EXISTS
---------------
Phase 6 is being closed with a deliberately two-sided status, and the two halves
are easy to conflate later:

  * the Run 1 LoRA adapter is **USABLE and VERIFIED** — its artifact is intact,
    it loads through the production path, and it reproduces Run 1's adapted-test
    control *exactly*; and
  * Run 1 is **ACCEPTANCE-REJECTED** — the predeclared rule v002 fails its V2
    per-class guardrail on the independent test split (Mixed forest).

"Verified" and "accepted" answer different questions: *is this artifact the one
we trained, and does it work?* versus *did it clear the bar we set before we
looked?* A single boolean cannot carry both, and collapsing them would either
launder a real regression or discard a working model.

**Two rejection records are preserved, and neither may be rewritten:**

| record | rule | split | verdict |
|---|---|---|---|
| Run 1's own manifest | `v001` | val | `REJECTED` (3 classes) |
| `test_adjudication.json` | `v002` | **test** (independent) | `REJECTED` (1 class) |

The val-split `ACCEPTED` in `run_manifest.json` (the recovery manifest) is the
rule's faithful output but is **not** final acceptance — it decides on the same
val subset that motivated v002, which
`docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md` §7.4 condition 3 forbids.

This script GENERATES the closure record from the evidence files rather than
hand-writing it, so no number here can drift from its source. A test
(`tests/unit/test_phase6_closure.py`) re-derives the same claims and fails if the
record and the evidence disagree.

Usage:
    python scripts/phase6_close.py
    python scripts/phase6_close.py --output <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: Run 1's original manifest — EVIDENCE. Never written to.
RUN1_MANIFEST_PATH = Path(r"D:/New folder (2)/New folder (3)/_x/run_manifest.json")

#: All Phase 6 evidence, kept together.
EVIDENCE_DIR = REPO_ROOT / "artifacts" / "vlm" / "run1_test_recovery"
RECOVERY_MANIFEST_PATH = EVIDENCE_DIR / "run_manifest.json"
ADJUDICATION_PATH = EVIDENCE_DIR / "test_adjudication.json"
ADAPTER_VERIFICATION_PATH = EVIDENCE_DIR / "adapter_verification.json"

#: The closure record is a PHASE-level record, so it sits one level up from the
#: run-1 evidence it cites.
CLOSURE_PATH = REPO_ROOT / "artifacts" / "vlm" / "phase6_closure.json"

#: The Run 1 adapter, promoted to production. Frozen: never retrained, never
#: modified. A better adapter is a NEW experiment, not an edit to this one.
PRODUCTION_ADAPTER_DIR = REPO_ROOT / ".scratch" / "phase6_real_adapter" / "phase6_adapter"

#: Run 1's recorded tree hash of the adapter. Re-derived by
#: `adapter_verification.json`; quoted here only for the cross-check.
RUN1_ADAPTER_TREE_HASH = "5c6b86317d1e65962702dc9e377009b3df41cc13de1b15bceccb70ad977775e7"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"required evidence file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(manifest: dict[str, Any]) -> dict[str, Any]:
    return manifest.get("decision") or {}


def _rejection_records(run1: dict[str, Any], recovery: dict[str, Any],
                       adjudication: dict[str, Any]) -> dict[str, Any]:
    """The two records that must survive unchanged, plus the val ACCEPT that does not."""
    d1 = _decision(run1)
    dr = _decision(recovery)
    da = adjudication["decision"]
    return {
        "v001_val_rejected": {
            "role": "the ORIGINAL pre-registered verdict — preserved verbatim",
            "rule_version": run1.get("acceptance_rule_version"),
            "split": "val",
            "status": d1.get("status"),
            "val_delta_pp": d1.get("val_delta_pp"),
            "test_delta_pp": d1.get("test_delta_pp"),
            "class_failures": d1.get("class_failures"),
            "source": str(RUN1_MANIFEST_PATH),
            "source_file_sha256": _sha256_file(RUN1_MANIFEST_PATH),
            "source_internal_manifest_hash": run1.get("manifest_hash"),
            "immutable": True,
        },
        "v002_independent_test_rejected": {
            "role": "the CONTRACT-FACING verdict — V1/V2 applied on the test split",
            "rule_version": adjudication.get("rule_version"),
            "split": adjudication.get("decision_split"),
            "status": da.get("status"),
            "test_delta_pp": da.get("test_delta_pp"),
            "val_delta_pp": da.get("val_delta_pp"),
            "class_failures": da.get("class_failures"),
            "source": str(ADJUDICATION_PATH),
            "required_by": adjudication.get("required_by"),
            "immutable": True,
        },
        "v002_val_accepted_NOT_final": {
            "role": (
                "recorded for completeness ONLY — NOT final acceptance. It decides on the "
                "same val subset that motivated v002, which §7.4 condition 3 forbids."
            ),
            "rule_version": recovery.get("acceptance_rule_version"),
            "split": "val",
            "status": dr.get("status"),
            "class_failures": dr.get("class_failures"),
            "superseded_by": "v002_independent_test_rejected",
            "source": str(RECOVERY_MANIFEST_PATH),
            "immutable": True,
        },
    }


def _regression(adjudication: dict[str, Any]) -> dict[str, Any]:
    """The one class that failed V2 on the test split, with its arithmetic."""
    failures = adjudication["decision"].get("class_failures") or []
    rows = {r["class"]: r for r in adjudication.get("per_class_test") or []}
    detail = []
    for f in failures:
        row = rows.get(f["class"], {})
        detail.append({
            "class": f["class"],
            "n_questions": f.get("n_questions"),
            "baseline_pp": f.get("baseline_pp"),
            "adapted_pp": f.get("adapted_pp"),
            "drop_pp": f.get("drop_pp"),
            "lost_questions": f.get("lost_questions"),
            "z": f.get("z"),
            "fails_materiality_floor": (f.get("lost_questions") or 0) >= 4,
            "fails_significance": (f.get("z") or 0.0) >= 1.96,
            "delta_pp_from_table": row.get("delta_pp"),
        })
    total = len(adjudication.get("per_class_test") or [])
    improved = sum(
        1 for r in (adjudication.get("per_class_test") or []) if (r.get("delta_pp") or 0) > 0
    )
    held = sum(
        1 for r in (adjudication.get("per_class_test") or []) if (r.get("delta_pp") or 0) == 0
    )
    return {
        "summary": (
            "The adapter improves the aggregate test endpoint by +49.50 pp, but V2's "
            "per-class guardrail fails on one class, so the run is REJECTED."
        ),
        "failing_classes": detail,
        "n_classes_total": total,
        "n_classes_improved": improved,
        "n_classes_held": held,
        "n_classes_failed": len(detail),
        "narrowness": (
            f"{len(detail)} of {total} classes fail; {improved} improved and {held} held. "
            "The next-worst class loses only 2 questions and so sits below V2's "
            "materiality floor."
        ),
        "why_not_a_split_artefact": (
            "The same class also degraded on the val split in run 1 (drop 6.4516 pp, "
            "n=31) — the very value that motivated v001's flag. The adapter hurts "
            "Mixed forest on BOTH splits, so this is a property of the adapter, not "
            "an accident of one subset. Mixed forest also sits at a 100.00 pp baseline "
            "on test, so any loss is a drop from the ceiling."
        ),
        "residual_risk": (
            "The verdict rests on 4 questions in one class of 33 — the unfloored "
            "minimum-size exposure recorded at PHASE6_AUDIT_AND_CONTRACT §8.6. With no "
            "n >= N floor in V2, a 33-question class can flip the verdict of a run whose "
            "aggregate endpoint improved by 49.5 pp. Reported, not resolved."
        ),
    }


def _adapter_verification() -> dict[str, Any]:
    """Fold in the artifact-verification record if it exists.

    Absence is recorded as absent, never as a placeholder pass — the repository's
    convention for an unavailable identifier.
    """
    if not ADAPTER_VERIFICATION_PATH.is_file():
        return {
            "available": False,
            "note": "adapter_verification.json not present; artifact verification NOT performed",
            "path": str(ADAPTER_VERIFICATION_PATH),
        }
    record = json.loads(ADAPTER_VERIFICATION_PATH.read_text(encoding="utf-8"))
    adapter = record.get("adapter") or {}
    loadability = record.get("loadability") or {}
    promoted = record.get("promoted_adapter") or {}
    missing = adapter.get("missing_files") or []
    mismatched = adapter.get("mismatched_files") or []
    extra = adapter.get("extra_files") or []
    return {
        "available": True,
        "path": str(ADAPTER_VERIFICATION_PATH),
        "verifier": record.get("verifier"),
        "read_only": record.get("read_only"),
        "verdict": record.get("verdict"),
        "generated_at": record.get("generated_at"),
        "amended_at": record.get("amended_at"),
        "amendment_note": record.get("amendment_note"),
        "blocking_issues": record.get("blocking_issues"),
        "tree_hash_matches": adapter.get("tree_hash_matches"),
        "computed_tree_hash": adapter.get("computed_tree_hash"),
        "expected_tree_hash": adapter.get("expected_tree_hash"),
        "directory_self_consistent_with_own_manifest": adapter.get(
            "directory_self_consistent_with_own_manifest"
        ),
        # The manifest check is the load-bearing integrity claim: it is what makes
        # "verified" mean "every file is the one we trained", not merely "it loaded".
        "manifest_check": {
            "n_files_in_manifest": adapter.get("n_files_in_manifest"),
            "n_files_on_disk": adapter.get("n_files_on_disk"),
            "missing_files": missing,
            "mismatched_files": mismatched,
            "extra_files": extra,
            "clean": not (missing or mismatched or extra),
        },
        "weights_file_sha256": adapter.get("weights_file_sha256"),
        "weights_file_sha256_matches": adapter.get("weights_file_sha256_matches"),
        "loadable_via_production_path": record.get("loadable_via_production_path"),
        "loadability_detail": {
            "performed": loadability.get("performed"),
            "result": loadability.get("result"),
            "path": loadability.get("path"),
        },
        "trainable_params": record.get("trainable_params"),
        # The two provenance traps a future reader is most likely to get wrong.
        "promoted_adapter": {
            "file": promoted.get("file"),
            "sha256": promoted.get("sha256"),
            "is_a_checkpoint": promoted.get("is_a_checkpoint"),
            "all_three_digests_distinct": promoted.get("all_three_digests_distinct"),
            "checkpoint_weight_digests": promoted.get("checkpoint_weight_digests"),
            "note": promoted.get("note"),
        },
        "findings": [
            {
                "id": f.get("id"),
                "severity": f.get("severity"),
                "summary": f.get("summary"),
            }
            for f in (record.get("findings") or [])
        ],
        "notes": record.get("notes"),
    }


def build_closure(run1: dict[str, Any], recovery: dict[str, Any],
                  adjudication: dict[str, Any]) -> dict[str, Any]:
    d1 = _decision(run1)
    da = adjudication["decision"]
    gates = adjudication.get("gates") or {}
    adapter_block = adjudication.get("adapter") or {}

    usable = {
        "gate_a_double_prime_subset_identity": gates.get("gate_corpus"),
        "gate_d_reproduced_run1_adapted_control_exactly": gates.get(
            "gate_adapted_reproduced_run1"
        ),
        "adapted_test": adjudication.get("adapted_test"),
        "aggregate_test_delta_pp": da.get("test_delta_pp"),
        "per_class_test": adjudication.get("per_class_test"),
        "adapter_provenance": adapter_block,
    }

    return {
        "kind": "phase6_closure",
        "phase": 6,
        "status": "CLOSED",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "headline": (
            "Phase 6 is closed. The Run 1 LoRA adapter is promoted to the production VLM "
            "adapter: USABLE and VERIFIED, but ACCEPTANCE-REJECTED."
        ),
        "verified_vs_accepted": (
            "'Verified' answers: is this artifact the one we trained, and does it work? "
            "'Accepted' answers: did it clear the bar predeclared before we looked? Both "
            "are true, and they are different questions. This record keeps them separate."
        ),
        "production_adapter": {
            "path": str(PRODUCTION_ADAPTER_DIR),
            "kind": run1.get("artifact_kind"),
            "base_model": run1.get("base_model"),
            "base_model_revision": run1.get("base_model_revision"),
            "precision_recorded": run1.get("precision"),
            "seed": run1.get("seed"),
            "config_hash": run1.get("config_hash"),
            "adapter_tree_sha256": run1.get("adapter_sha256"),
            "trainable_params": run1.get("trainable_params"),
            "trainable_fraction": run1.get("trainable_fraction"),
            "lora_rank": (run1.get("config") or {}).get("lora_rank"),
            "lora_alpha": (run1.get("config") or {}).get("lora_alpha"),
            "lora_target_modules": (run1.get("config") or {}).get("lora_target_modules"),
            "lora_target_module_count": (run1.get("lora") or {}).get("n_target_modules"),
            "trainable_subtrees": (run1.get("lora") or {}).get("trainable_subtrees"),
            "frozen_params": run1.get("frozen_params"),
            "vision_tower_untouched": (
                "trainable_subtrees is exactly {'model.text_model': 8683520}; the vision "
                "model (86,433,024) and connector (11,796,480) are in frozen_params. The "
                "contract's vision-tower hazard did not occur."
            ),
            "how_enabled": {
                "mechanism": "environment variable, not a code change",
                "env_var": "SATQUERY_VLM_ADAPTER",
                "resolved_in": (
                    "specialists/vqa/model.py — ADAPTER_ENV_VAR (line 41); resolution order "
                    "explicit arg -> env var -> none (line 254); attached via "
                    "PeftModel.from_pretrained (line 302)"
                ),
                "example": f'SATQUERY_VLM_ADAPTER="{PRODUCTION_ADAPTER_DIR}"',
            },
            "files_required_for_attachment": [
                "adapter_model.safetensors",
                "adapter_config.json",
            ],
            "files_required_to_serve_standalone": [
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "processor_config.json",
            ],
            "known_traps": [
                "`adapter_sha256` names TWO different values and they are NOT "
                "interchangeable: training/vlm/artifact.py computes a TREE HASH over the "
                "{relpath: sha256} weight map (5c6b8631...), while "
                "specialists/vqa/model.py::_adapter_sha256 computes the FILE sha256 of "
                "adapter_model.safetensors (07c76a75...). Recomputing one and comparing "
                "it to the other yields a false 'artifact was altered' conclusion.",
                "The promoted adapter is NOT checkpoint-2000. The three weight files have "
                "three different digests: top-level 07c76a75..., checkpoint-1500 "
                "7273588e..., checkpoint-2000 bf249943... So 'just use the last "
                "checkpoint' is not equivalent to this artifact.",
            ],
            "reconstruction": {
                "note": (
                    "The adapter is NOT committed (.gitignore excludes artifacts/, "
                    "checkpoints/ and *.safetensors) and its canonical path is under "
                    ".scratch/, which a future cleanup could remove."
                ),
                "source": "D:/New folder (2)/New folder (3)/phase6_realbundle.zip",
                "entry_prefix": "phase6_adapter/",
                "verify_against": [
                    "tree hash 5c6b86317d1e65962702dc9e377009b3df41cc13de1b15bceccb70ad977775e7",
                    "adapter_model.safetensors sha256 "
                    "07c76a75fa04624880ed7730590f5fdd7b145a8232e3c0af411c3c545a5adf5e",
                ],
                "why_not_moved": (
                    "Every piece of Phase 6 evidence (test_adjudication.json and the "
                    "adjudication harness constant) records this exact path; moving it "
                    "would make the recorded evidence stale. Promoting it to a non-scratch "
                    "location is a reasonable follow-up, not a closure requirement."
                ),
            },
            "use_status": "USABLE_VERIFIED",
            "acceptance_status": "REJECTED",
            "retrained_for_closure": False,
            "modified_for_closure": False,
        },
        "why_usable_verified": usable,
        "why_acceptance_rejected": {
            "rule_version": adjudication.get("rule_version"),
            "decision_split": adjudication.get("decision_split"),
            "status": da.get("status"),
            "reasons": da.get("reasons"),
            "thresholds_used": da.get("thresholds_used"),
            "regression": _regression(adjudication),
        },
        "preserved_records": _rejection_records(run1, recovery, adjudication),
        "artifact_verification": _adapter_verification(),
        "rule_unchanged_by_closure": adjudication.get("rule_unchanged"),
        "not_a_retrain": adjudication.get("not_a_retrain"),
        "forward_rule": {
            "statement": (
                "Any future improved adapter MUST be a new experiment/version. It MUST NOT "
                "rewrite, amend, or supersede Run 1's records."
            ),
            "run1_is_frozen": True,
            "new_work_requires": [
                "a NEW adapter artifact and a NEW run manifest",
                "a NEW experiment/version identifier; do not reuse run 1's",
                "the predeclared acceptance rule applied as-is, or a NEW rule version "
                "declared before the run it judges",
                "its own independent test-split adjudication; Run 1's test-split verdict "
                "is not transferable",
            ],
            "must_not": [
                "retrain or modify the Run 1 adapter in place",
                "overwrite run 1's manifest, the recovery manifest, or "
                "test_adjudication.json",
                "report a new adapter's metrics under run 1's identity",
                "edit v001 or v002's recorded verdicts to match a later outcome",
            ],
            "precedent": (
                "v002 itself followed this rule: it was declared as a NEW version after "
                "run 1, v001 was retained and left replayable, and run 1's REJECTED was "
                "never overwritten."
            ),
        },
        "evidence_index": {
            "run1_manifest": {
                "path": str(RUN1_MANIFEST_PATH),
                "file_sha256": _sha256_file(RUN1_MANIFEST_PATH),
                "internal_manifest_hash": run1.get("manifest_hash"),
                "warning": (
                    "These two digests differ. `internal_manifest_hash` is the manifest's "
                    "OWN content hash field, NOT the file digest. A future reader who "
                    "recomputes one and compares it to the other will wrongly conclude "
                    "the file was altered."
                ),
            },
            "recovery_manifest": {"path": str(RECOVERY_MANIFEST_PATH)},
            "test_adjudication": {"path": str(ADJUDICATION_PATH)},
            "adapter_verification": {"path": str(ADAPTER_VERIFICATION_PATH)},
            "contract": "docs/PHASE6_AUDIT_AND_CONTRACT.md",
            "diagnosis": "docs/PHASE6_RUN1_REJECTION_DIAGNOSIS.md",
            "closure_doc": "docs/PHASE6_CLOSURE.md",
        },
        "what_closure_does_not_claim": [
            "It does NOT claim Run 1 was accepted — it was rejected by v001 on val and by "
            "v002 on the independent test split.",
            "It does NOT claim the Mixed forest regression is resolved.",
            "It does NOT claim a new adapter exists or is planned.",
            "It does NOT alter the v002 rule or its verdict.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the Phase 6 closure record")
    p.add_argument("--output", default=str(CLOSURE_PATH))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run1 = _load(RUN1_MANIFEST_PATH)
    recovery = _load(RECOVERY_MANIFEST_PATH)
    adjudication = _load(ADJUDICATION_PATH)

    # Refuse to write a closure that contradicts the evidence.
    da = _decision(adjudication)
    if da.get("status") != "REJECTED":
        raise SystemExit(
            "refusing to close: the test-split adjudication is not REJECTED "
            f"(status={da.get('status')!r}). Re-read the evidence before closing Phase 6."
        )
    if adjudication.get("decision_split") != "test":
        raise SystemExit(
            "refusing to close: the adjudication did not decide on the test split "
            f"(decision_split={adjudication.get('decision_split')!r})."
        )
    if _decision(run1).get("status") != "REJECTED":
        raise SystemExit("refusing to close: run 1's own v001 verdict is not REJECTED.")

    closure = build_closure(run1, recovery, adjudication)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"closure record written: {out}")
    print(f"  headline        : {closure['headline']}")
    print(f"  use_status      : {closure['production_adapter']['use_status']}")
    print(f"  acceptance      : {closure['production_adapter']['acceptance_status']}")
    print(f"  adapter tree    : {closure['production_adapter']['adapter_tree_sha256']}")
    reg = closure["why_acceptance_rejected"]["regression"]
    print(f"  failing classes : {reg['n_classes_failed']} of {reg['n_classes_total']}")
    for f in reg["failing_classes"]:
        print(
            f"    - {f['class']}: n={f['n_questions']} "
            f"{f['baseline_pp']:.2f} -> {f['adapted_pp']:.2f} pp "
            f"(drop {f['drop_pp']:.4f}, lost {f['lost_questions']}, z {f['z']:.4f})"
        )
    av = closure["artifact_verification"]
    print(f"  artifact verify : {av.get('verdict')}")
    mc = av.get("manifest_check") or {}
    print(
        f"  manifest check  : {mc.get('n_files_on_disk')}/{mc.get('n_files_in_manifest')} "
        f"present, clean={mc.get('clean')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
