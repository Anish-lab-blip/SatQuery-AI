"""Generate models/manifest.json and models/checksums.sha256 from the real artifact files.

RULE: nothing in the manifest is typed by hand. Every byte count and every sha256 is computed
here by reading the file. Where a value cannot be determined from disk it is emitted as null,
never guessed.

Read-only with respect to the artifacts. Writes only the two generated files.
"""
import hashlib
import json
import os
import sys
import datetime

SRC = r"C:/Users/anish/satquery-ai"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repo", "models")
OUT_DIR = os.path.abspath(OUT_DIR)

CONFIG_HASH = "78f1e3700da15aa1"  # verified by running get_config().hash

# The six released artifacts. Paths are relative to SRC.
ARTIFACTS = [
    {
        "id": "change_head",
        "task": "change",
        "kind": "trained_head",
        "path": "artifacts/change/levir_change_v001/head.pt",
        "hf_path": "change/head.pt",
        "backbone": None,
        "architecture": "STANet-style Siamese change detector (ResNet-18 + PAM)",
        "source_metric_artifact": "artifacts/change/eval_test/eval_result.json",
    },
    {
        "id": "change_vqa_head",
        "task": "change_vqa",
        "kind": "trained_head",
        "path": "artifacts/change_vqa/run/head.pt",
        "hf_path": "change_vqa/head.pt",
        "backbone": "STANet change detector (frozen, backing the head's change features)",
        "architecture": "change_vqa_head_v1",
        "source_metric_artifact": "artifacts/change_vqa/run/PROMOTION.json",
    },
    {
        "id": "optical_sar_fusion_head",
        "task": "optical_sar",
        "kind": "trained_head",
        "path": "artifacts/optical_sar/fusion_head_production_v001/head.pt",
        "hf_path": "optical_sar/head.pt",
        "backbone": "antofuller/CROMA (CROMA_base.pt, revision 0dd28e3d633b)",
        "architecture": "CROMA-base fusion head (input_dim 2318 -> hidden 512 -> 19 classes)",
        "source_metric_artifact": "artifacts/optical_sar/fusion_head_production_v001/pre_registered_115_metric.json",
    },
    {
        "id": "grounding_head",
        "task": "grounding",
        "kind": "trained_head",
        "path": "artifacts/grounding/remoteclip_grounding_v001/head.pt",
        "hf_path": "grounding/head.pt",
        "backbone": "chendelong/RemoteCLIP (RemoteCLIP-ViT-B-32.pt, revision bf1d8a3ccf2d)",
        "architecture": "RemoteCLIP ViT-B/32 grounding head (feature_dim 2048, hidden 512)",
        "source_metric_artifact": "artifacts/grounding/remoteclip_grounding_v001/eval_result_canonical.json",
    },
    {
        "id": "router_adapter",
        "task": "router",
        "kind": "trained_adapter",
        "path": "artifacts/router/router_adapter_v001/adapter.pt",
        "hf_path": "router/adapter.pt",
        "backbone": "sentence-transformers/all-MiniLM-L6-v2 (revision 1110a243fdf4)",
        "architecture": "task/modality adapter over frozen MiniLM embeddings (~50,822 params)",
        "source_metric_artifact": "artifacts/router/threshold_sweep_val.json",
    },
    {
        "id": "vlm_lora_adapter",
        "task": "vlm",
        "kind": "lora_adapter",
        "path": ".scratch/phase6_real_adapter/phase6_adapter/adapter_model.safetensors",
        "hf_path": "vlm/adapter_model.safetensors",
        "backbone": "HuggingFaceTB/SmolVLM-500M-Instruct (revision a7da5b986cb5)",
        "architecture": "PEFT LoRA (r=16, alpha=32, dropout=0.05) on text_model projections",
        "source_metric_artifact": "artifacts/vlm/phase6_closure.json",
        "acceptance": "ACCEPTANCE-REJECTED (metrics usable; not promoted)",
    },
]


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    entries = []
    missing = []
    for a in ARTIFACTS:
        full = os.path.join(SRC, a["path"])
        e = dict(a)
        e["config_hash"] = CONFIG_HASH
        if not os.path.exists(full):
            e["bytes"] = None
            e["sha256"] = None
            e["status"] = "MISSING_ON_DISK"
            missing.append(a["path"])
        else:
            e["bytes"] = os.path.getsize(full)
            e["sha256"] = sha256_of(full)
            e["status"] = "PRESENT"
        # pull the artifact's own declared parameter count where it records one
        e["parameters"] = None
        src_metric = os.path.join(SRC, a.get("source_metric_artifact") or "")
        if os.path.exists(src_metric):
            d = read_json(src_metric)
            if isinstance(d, dict):
                art = d.get("artifact")
                if isinstance(art, dict):
                    e["parameters"] = art.get("parameters")
        entries.append(e)

    manifest = {
        "schema": "satquery_model_manifest_v1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "generator": "release/tools/generate_model_manifest.py",
        "note": (
            "Generated by reading the files. No byte count or hash is typed by hand. "
            "Backbones are NOT redistributed; they are fetched from the Hugging Face Hub, "
            "pinned by revision."
        ),
        "config_hash": CONFIG_HASH,
        "release_repo": "thundercode/SatQuery",
        "artifact_count": len(entries),
        "artifacts": entries,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    mpath = os.path.join(OUT_DIR, "manifest.json")
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    cpath = os.path.join(OUT_DIR, "checksums.sha256")
    with open(cpath, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# sha256 of the six released artifacts, keyed by their path in this repository.\n")
        fh.write("# Verify with:  sha256sum -c checksums.sha256\n")
        for e in entries:
            if e["sha256"]:
                # sha256sum format: "<hash>  <name>"
                fh.write(f"{e['sha256']}  {e['hf_path']}\n")

    print(f"wrote {mpath}")
    print(f"wrote {cpath}")
    print()
    for e in entries:
        b = f"{e['bytes']:,}" if e["bytes"] is not None else "-"
        h = (e["sha256"] or "-")[:16]
        print(f"  {e['status']:16} {e['id']:24} {b:>14}  {h}…  {e['path']}")
    if missing:
        print()
        print("MISSING FILES (manifest records null, never a guess):")
        for m in missing:
            print("  " + m)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
