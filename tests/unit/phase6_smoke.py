"""Phase 6 real CPU smoke test -- standalone, NOT collected by pytest.

Run:
    .venv/Scripts/python.exe tests/unit/phase6_smoke.py

It genuinely exercises the pipeline end to end on the **real** cached SmolVLM and
the **real** BigEarthNet corpus:

    1. load the cached model + the pinned processor (F5-2 pin)
    2. build the corpus on a real slice; check T2 block scene ids + split==official
    3. render a real patch to RGB and run it through the processor
    4. baseline evaluation on a handful of real samples
    5. attach the scoped LoRA; record the full LoRAInjectionReport
    6. a real forward + backward step; loss printed; LoRA gradients checked
    7. adapted evaluation on the same samples
    8. save the adapter, reload it onto a fresh base, run one inference

Prints wall-clock seconds per stage. Writes nothing outside `tests/`.
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# `pytest.ini` puts the repo root on sys.path; a standalone run must do it too.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ADAPTER_DIR = REPO / "tests" / "_smoke_adapter"
CORPUS_LIMIT = 200
N_EVAL = 4
N_STEP_SAMPLES = 2


def _p(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    timings: dict[str, float] = {}
    failed_stage = "none"
    t_all = time.perf_counter()

    try:
        # -- config --------------------------------------------------------
        from training.vlm.config import VLMTrainingConfig

        cfg = VLMTrainingConfig.from_registry()
        _p(
            f"[config] base_model={cfg.base_model} revision={cfg.revision} "
            f"max_seq_length={cfg.max_seq_length} precision={cfg.precision} "
            f"effective_batch_size={cfg.effective_batch_size} "
            f"target_modules={list(cfg.lora_target_modules)}"
        )

        # -- stage 1: load model + pinned processor ------------------------
        failed_stage = "1_load"
        from specialists.vqa.model import SmolVLM

        t0 = time.perf_counter()
        vlm = SmolVLM(
            checkpoint=cfg.base_model,
            revision=cfg.revision,
            processor_longest_edge=cfg.processor_longest_edge,
            device="cpu",
            do_image_splitting=cfg.do_image_splitting,
        )
        timings["1_load_model_and_processor"] = time.perf_counter() - t0
        processor = vlm.processor
        model = vlm.model
        _p(
            f"[stage1] loaded in {timings['1_load_model_and_processor']:.1f}s "
            f"params={vlm.load_info.parameters} device={vlm.load_info.device} "
            f"loader={vlm.load_info.loader_class}"
        )

        # -- stage 2: build the corpus on a real slice ---------------------
        failed_stage = "2_build_corpus"
        from training.vlm.dataset import build_corpus

        t0 = time.perf_counter()
        corpus = build_corpus(cfg, limit=CORPUS_LIMIT)
        timings["2_build_corpus"] = time.perf_counter() - t0
        _p(
            f"[stage2] corpus built in {timings['2_build_corpus']:.1f}s "
            f"patches={len(corpus.patches)} samples={len(corpus.samples)} "
            f"splits={corpus.split_counts()} "
            f"blocks={corpus.split_info.get('blocks_by_split')}"
        )
        _p(f"[stage2] scene ids sample: {sorted({p.scene_id for p in corpus.patches})[:6]}")
        mismatched = [p.patch_id for p in corpus.patches if p.split != p.official_split]
        _p(f"[stage2] patches with split != official_split: {len(mismatched)}")

        # -- stage 3: render a real patch -> processor ---------------------
        failed_stage = "3_render"
        from training.vlm.collate import Collator, make_item, render_sample
        from training.vlm.formatting import build_training_messages

        t0 = time.perf_counter()
        real_patch = corpus.patches[0]
        image = render_sample(real_patch)
        messages = build_training_messages("Is water present in this image?")
        prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        processed = processor(images=image, text=prompt, return_tensors="pt")
        timings["3_render_and_process"] = time.perf_counter() - t0
        _p(
            f"[stage3] patch={real_patch.patch_id} image.size={image.size} "
            f"mode={image.mode}; processor keys={sorted(processed)} "
            f"pixel_values={tuple(processed['pixel_values'].shape)} "
            f"input_ids={tuple(processed['input_ids'].shape)} "
            f"in {timings['3_render_and_process']:.1f}s"
        )

        eval_samples = corpus.samples[:N_EVAL]
        _p(f"[stage3] eval samples: {[s.sample_id for s in eval_samples]}")

        # -- stage 4: baseline evaluation ----------------------------------
        failed_stage = "4_baseline_eval"
        from training.vlm.evaluate import evaluate_split

        t0 = time.perf_counter()
        base_metrics = evaluate_split(model, processor, eval_samples, device="cpu")
        timings["4_baseline_eval"] = time.perf_counter() - t0
        _p(
            f"[stage4] baseline MetricSet={json.dumps(base_metrics.to_dict())} "
            f"in {timings['4_baseline_eval']:.1f}s "
            f"({timings['4_baseline_eval'] / len(eval_samples):.1f}s/sample)"
        )

        # -- stage 5: attach the scoped LoRA -------------------------------
        failed_stage = "5_attach_lora"
        from training.vlm.lora import attach_lora, save_adapter

        t0 = time.perf_counter()
        peft_model, report = attach_lora(model, cfg)
        timings["5_attach_lora"] = time.perf_counter() - t0
        _p(f"[stage5] LoRA attached in {timings['5_attach_lora']:.1f}s")
        _p("[stage5] report=" + json.dumps(report.to_dict(), indent=2))

        # -- stage 6: forward + backward -----------------------------------
        failed_stage = "6_forward_backward"
        import torch

        t0 = time.perf_counter()
        items = [
            make_item(
                processor,
                question=sample.question,
                answer=sample.answer,
                image=render_sample(sample),
                max_seq_length=cfg.max_seq_length,
            )
            for sample in eval_samples[:N_STEP_SAMPLES]
        ]
        collator = Collator(processor, cfg.max_seq_length)
        batch = collator(items)
        _p(
            f"[stage6] batch keys={sorted(batch)} "
            f"input_ids={tuple(batch['input_ids'].shape)} "
            f"pixel_values={tuple(batch['pixel_values'].shape)} "
            f"labels={tuple(batch['labels'].shape)}"
        )
        peft_model.train()
        peft_model.zero_grad()
        outputs = peft_model(**batch)
        loss = outputs.loss
        loss.backward()
        lora_params = [
            (name, param)
            for name, param in peft_model.named_parameters()
            if "lora" in name.lower() and param.requires_grad
        ]
        with_grad = [
            name
            for name, param in lora_params
            if param.grad is not None and torch.is_tensor(param.grad)
        ]
        nonzero_grad = [
            name
            for name, param in lora_params
            if param.grad is not None and float(param.grad.abs().sum()) > 0.0
        ]
        timings["6_forward_backward"] = time.perf_counter() - t0
        _p(
            f"[stage6] loss={float(loss):.6f} in "
            f"{timings['6_forward_backward']:.1f}s; "
            f"lora_trainable={len(lora_params)} with_grad={len(with_grad)} "
            f"non_zero_grad={len(nonzero_grad)}"
        )

        # -- stage 7: adapted evaluation -----------------------------------
        failed_stage = "7_adapted_eval"
        t0 = time.perf_counter()
        peft_model.eval()
        adapted_metrics = evaluate_split(peft_model, processor, eval_samples, device="cpu")
        timings["7_adapted_eval"] = time.perf_counter() - t0
        _p(
            f"[stage7] adapted MetricSet={json.dumps(adapted_metrics.to_dict())} "
            f"in {timings['7_adapted_eval']:.1f}s "
            f"({timings['7_adapted_eval'] / len(eval_samples):.1f}s/sample)"
        )
        _p(
            f"[stage7] delta exact_match = "
            f"{(adapted_metrics.exact_match - base_metrics.exact_match) * 100:+.2f} pp"
        )

        # -- stage 8: save, reload, one inference --------------------------
        failed_stage = "8_save_reload"
        from training.vlm.lora import load_adapter

        t0 = time.perf_counter()
        save_adapter(peft_model, ADAPTER_DIR)
        save_seconds = time.perf_counter() - t0
        files = sorted(p.name for p in ADAPTER_DIR.iterdir())
        _p(f"[stage8] adapter saved in {save_seconds:.1f}s files={files}")

        # Free the adapted model before loading a fresh base.
        del peft_model, model, vlm
        gc.collect()

        t0 = time.perf_counter()
        fresh = SmolVLM(
            checkpoint=cfg.base_model,
            revision=cfg.revision,
            processor_longest_edge=cfg.processor_longest_edge,
            device="cpu",
            do_image_splitting=cfg.do_image_splitting,
        )
        reloaded = load_adapter(fresh.model, ADAPTER_DIR)
        reloaded_metrics = evaluate_split(
            reloaded, fresh.processor, eval_samples[:1], device="cpu"
        )
        timings["8_save_reload_inference"] = time.perf_counter() - t0
        _p(
            f"[stage8] reloaded {type(reloaded).__name__} and ran one inference in "
            f"{timings['8_save_reload_inference']:.1f}s; "
            f"MetricSet={json.dumps(reloaded_metrics.to_dict())}"
        )

    except Exception:  # noqa: BLE001 - the smoke test reports, it does not hide
        _p(f"[FAILED] stage={failed_stage}")
        _p(traceback.format_exc())
        _p(f"[total] {time.perf_counter() - t_all:.1f}s")
        _p("SMOKE TEST FAILED")
        return 1

    _p("--- stage timings (seconds) ---")
    for name, seconds in timings.items():
        _p(f"  {name}: {seconds:.1f}")
    _p(f"[total] {time.perf_counter() - t_all:.1f}s")
    _p("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
