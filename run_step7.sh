CROMA="C:/Users/anish/.cache/huggingface/hub/models--antofuller--CROMA/snapshots/0dd28e3d633bd6715856ae9890e8c49360040598/CROMA_base.pt"
MAN=artifacts/phase12_selection/selection_manifest_seed10.jsonl
OUT=artifacts/optical_sar/fusion_features
CHUNK=400                      # measured-safe; see §5 before raising this

run_split () {                 # $1 = split, $2 = total rows in that split
  S=$1; N=$2
  for OFF in $(seq 0 $CHUNK $((N - 1))); do
    echo "=== $S offset=$OFF limit=$CHUNK ==="
    .venv/Scripts/python.exe scripts/extract_fusion_features.py \
      --selection-manifest $MAN --reben-root data/bigearthnet_v2/reben \
      --split $S --offset $OFF --limit $CHUNK --batch-size 8 --arm A \
      --out-cache $OUT/$S.npz --croma-checkpoint "$CROMA" || {
        echo "!!! FAILED at $S offset=$OFF — re-run that exact command to resume"; break; }
  done
}

run_split train 20000
run_split val    4000
run_split test   4000