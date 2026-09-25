"""Verify every metric quoted in the release README against its source artifact.

Read-only. Each claim is compared at the precision at which the README states it.
Writes nothing; prints a report suitable for pasting into FINAL_RELEASE_VERIFICATION.md.

Key names were discovered by walking the artifacts, NOT assumed: several live under nested
paths (e.g. change metrics are `metrics.pooled.iou`, grounding is
`results.head_threshold.mean_best_iou`, VLM is `why_usable_verified.adapted_test.*`).

WHERE THE ARTIFACTS COME FROM
-----------------------------
This tool reads the project's `artifacts/` directory, which is NOT part of the public release
(the release ships the six trained weights on the Hugging Face Hub, not the full artifacts tree).
Point it at a checkout that has the artifacts:

    SATQUERY_ARTIFACTS_ROOT=/path/to/satquery-ai python verify_readme_metrics.py

Without that variable it defaults to `../..` relative to this file, which is the layout the tool
was authored in.
"""
import json
import os
import sys

ROOT = os.environ.get(
    "SATQUERY_ARTIFACTS_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)

# (label, relative path, dotted key, value as printed in the README)
CLAIMS = [
    ("change pooled IoU", "artifacts/change/eval_test/eval_result.json",
     "metrics.pooled.iou", "0.8122"),
    ("change macro IoU", "artifacts/change/eval_test/eval_result.json",
     "metrics.macro.miou", "0.8457"),
    ("change pooled F1", "artifacts/change/eval_test/eval_result.json",
     "metrics.pooled.f1", "0.8964"),
    ("grounding canonical head_threshold mean_best_IoU",
     "artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json",
     "results.head_threshold.mean_best_iou", "0.2838"),
    ("grounding canonical head_threshold recall@0.5",
     "artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json",
     "results.head_threshold.recall.0.50", "0.2198"),
    ("grounding matched6 head_threshold mean_best_IoU",
     "artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json",
     "results.head_threshold.mean_best_iou", "0.2566"),
    ("grounding matched6 head_threshold recall@0.5",
     "artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json",
     "results.head_threshold.recall.0.50", "0.1938"),
    ("grounding head_argmax mean_best_IoU (canonical)",
     "artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json",
     "results.head_argmax.mean_best_iou", "0.1215"),
    ("grounding zero-shot baseline IoU (canonical)",
     "artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json",
     "results.zero_shot_matched.mean_best_iou", "0.0972"),
    ("optical-SAR fusion accuracy",
     "artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json",
     "accuracy", "0.931"),
    ("optical-SAR fusion macro_F1",
     "artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json",
     "macro_f1", "0.434161"),
    ("change_vqa test accuracy", "artifacts/change_vqa/run/PROMOTION.json",
     "verification.test_accuracy", "0.697626"),
    ("change_vqa test macro_F1", "artifacts/change_vqa/run/PROMOTION.json",
     "verification.test_macro_f1", "0.378373"),
    ("change_vqa test2 accuracy", "artifacts/change_vqa/run/PROMOTION.json",
     "verification.test2_accuracy", "0.651469"),
    ("change_vqa test2 macro_F1", "artifacts/change_vqa/run/PROMOTION.json",
     "verification.test2_macro_f1", "0.372309"),
    ("router overall ungated accuracy", "artifacts/router/threshold_sweep_val.json",
     "overall_ungated_accuracy", "0.965116"),
    ("calibration ECE before scaling", "artifacts/calibration_v001.json",
     "metrics.ece_before", "0.013755"),
    ("calibration ECE after scaling", "artifacts/calibration_v001.json",
     "metrics.ece_after", "0.014929"),
    ("VLM adapter exact_match", "artifacts/vlm/phase6_closure.json",
     "why_usable_verified.adapted_test.exact_match", "0.963"),
    ("VLM adapter F1", "artifacts/vlm/phase6_closure.json",
     "why_usable_verified.adapted_test.f1", "0.96432"),
]


def get(obj, dotted):
    """Resolve a dotted path, preferring the LONGEST matching key at each step.

    Needed because some artifact keys themselves contain dots (e.g. the recall
    dict is keyed "0.10"/"0.25"/"0.50"), so a naive split(".") walk would break
    `results.head_threshold.recall.0.50` into ...recall -> 0 -> 50 and fail.
    """
    return _get(obj, dotted.split("."))


def _get(cur, parts):
    if not parts:
        return cur, True
    if isinstance(cur, dict):
        for i in range(len(parts), 0, -1):          # longest key first
            key = ".".join(parts[:i])
            if key in cur:
                return _get(cur[key], parts[i:])
        return None, False
    if isinstance(cur, list):
        return _get(cur[int(parts[0])], parts[1:])
    return None, False


def decimals(s):
    return len(s.split(".")[1]) if "." in s else 0


fails = 0
print(f"{'STATUS':8} {'claim':48} {'artifact':>14} {'readme':>12}  source")
print("-" * 118)
for label, rel, key, claimed in CLAIMS:
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        print(f"{'NOFILE':8} {label:48} {'':>14} {claimed:>12}  {rel}")
        fails += 1
        continue
    data = json.load(open(path, encoding="utf-8"))
    val, found = get(data, key)
    if not found:
        print(f"{'NOKEY':8} {label:48} {'':>14} {claimed:>12}  {rel}#{key}")
        fails += 1
        continue
    ok = round(float(val), decimals(claimed)) == float(claimed)
    if not ok:
        fails += 1
    print(f"{'MATCH' if ok else 'DIFFER':8} {label:48} {val:>14} {claimed:>12}  {rel}#{key}")

# --- artifact-declared statuses that the README also asserts ---
print()
print("=== status assertions ===")
d = json.load(open(os.path.join(ROOT, "artifacts/vlm/phase6_closure.json"), encoding="utf-8"))
hl = d.get("headline", "")
print(f"  VLM headline contains ACCEPTANCE-REJECTED : {'ACCEPTANCE-REJECTED' in hl}")
print(f"  VLM status                               : {d.get('status')}")
r = json.load(open(os.path.join(ROOT, "artifacts/router/threshold_sweep_val.json"), encoding="utf-8"))
print(f"  router corpus_limited                    : {r.get('corpus_limited')}")
print(f"  router n_val                             : {r.get('n_val')}")
c = json.load(open(os.path.join(ROOT, "artifacts/calibration_v001.json"), encoding="utf-8"))
print(f"  calibration temperature (temperature_scaling.temperature) : "
      f"{c.get('temperature_scaling', {}).get('temperature')}")
print(f"  calibration ece_improvement              : "
      f"{c.get('metrics', {}).get('ece_improvement')}  (negative => calibration did NOT help)")

print()
print(f"RESULT: {'ALL CLAIMS VERIFIED' if fails == 0 else str(fails) + ' CLAIM(S) FAILED'}")
sys.exit(0 if fails == 0 else 1)
