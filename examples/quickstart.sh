#!/usr/bin/env bash
# Quickstart: vanilla fine-tune Whisper-base on Spanish, then eval on FLEURS-test.
# ~30-60 min on a single consumer GPU (e.g. RTX 4090) on first run; subsequent
# runs skip the model + dataset download.
#
#   1. Install:  pip install -r requirements.txt
#   2. Run:      bash examples/quickstart.sh
set -e

OUT=${OUT:-runs/spanish-base-quickstart}

# Fine-tune (no W&B; add --wandb-project to enable)
python src/train.py \
  --lang spanish \
  --fleurs-code es_419 \
  --whisper-lang spanish \
  --model openai/whisper-base \
  --output-dir "$OUT" \
  --max-epochs 3 \
  --eval-steps 100 \
  --patience 3 \
  --seed 42

# Eval the best ckpt on FLEURS-test
python src/eval.py \
  --ckpt-dir "$OUT/best" \
  --fleurs-code es_419 \
  --whisper-lang spanish \
  --results-path "$OUT/test_results.json"

echo
echo "Done. Final metrics:"
cat "$OUT/test_results.json"
