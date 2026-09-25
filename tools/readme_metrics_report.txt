STATUS   claim                                                  artifact       readme  source
----------------------------------------------------------------------------------------------------------------------
MATCH    change pooled IoU                                        0.8122       0.8122  artifacts/change/eval_test/eval_result.json#metrics.pooled.iou
MATCH    change macro IoU                                         0.8457       0.8457  artifacts/change/eval_test/eval_result.json#metrics.macro.miou
MATCH    change pooled F1                                         0.8964       0.8964  artifacts/change/eval_test/eval_result.json#metrics.pooled.f1
MATCH    grounding canonical head_threshold mean_best_IoU         0.2838       0.2838  artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json#results.head_threshold.mean_best_iou
MATCH    grounding canonical head_threshold recall@0.5            0.2198       0.2198  artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json#results.head_threshold.recall.0.50
MATCH    grounding matched6 head_threshold mean_best_IoU          0.2566       0.2566  artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json#results.head_threshold.mean_best_iou
MATCH    grounding matched6 head_threshold recall@0.5             0.1938       0.1938  artifacts/grounding/remoteclip_grounding_v001/eval_result_matched6.json#results.head_threshold.recall.0.50
MATCH    grounding head_argmax mean_best_IoU (canonical)          0.1215       0.1215  artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json#results.head_argmax.mean_best_iou
MATCH    grounding zero-shot baseline IoU (canonical)             0.0972       0.0972  artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json#results.zero_shot_matched.mean_best_iou
MATCH    optical-SAR fusion accuracy                               0.931        0.931  artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json#accuracy
MATCH    optical-SAR fusion macro_F1                            0.434161     0.434161  artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json#macro_f1
MATCH    change_vqa test accuracy                            0.697626367     0.697626  artifacts/change_vqa/run/PROMOTION.json#verification.test_accuracy
MATCH    change_vqa test macro_F1                            0.378373275     0.378373  artifacts/change_vqa/run/PROMOTION.json#verification.test_macro_f1
MATCH    change_vqa test2 accuracy                           0.651469262     0.651469  artifacts/change_vqa/run/PROMOTION.json#verification.test2_accuracy
MATCH    change_vqa test2 macro_F1                           0.372308516     0.372309  artifacts/change_vqa/run/PROMOTION.json#verification.test2_macro_f1
MATCH    router overall ungated accuracy                        0.965116     0.965116  artifacts/router/threshold_sweep_val.json#overall_ungated_accuracy
MATCH    calibration ECE before scaling                         0.013755     0.013755  artifacts/calibration_v001.json#metrics.ece_before
MATCH    calibration ECE after scaling                          0.014929     0.014929  artifacts/calibration_v001.json#metrics.ece_after
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
