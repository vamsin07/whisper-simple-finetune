# whisper-vanilla-finetune

Plain supervised fine-tuning of OpenAI's [Whisper](https://github.com/openai/whisper)
on the [FLEURS](https://huggingface.co/datasets/google/fleurs) multilingual
speech-recognition benchmark. No vocab swap, no new BPE, no auxiliary text
objectives — just standard Whisper fine-tuning on one language at a time, with
HuggingFace `datasets`, a clean argparse CLI, and optional W&B logging.

Built as a starting point for new contributors to the multilingual ASR project.
The same Python entry points run on a laptop GPU, a campus server, or a
Nautilus Kubernetes cluster — pick how you want to deploy from the sections
below.

## What's in here

```
.
├── src/
│   ├── train.py             # fine-tune on FLEURS-train, save best by val WER
│   └── eval.py              # load a saved ckpt, eval on FLEURS-test
├── configs/
│   ├── spanish.yaml         # example training config (reference; not auto-loaded)
│   ├── sweep.yaml           # W&B sweep config
│   └── README.md
├── nautilus/                # OPTIONAL — Kubernetes deployment for Nautilus PRP
│   ├── Dockerfile
│   ├── train-job.yaml
│   ├── sweep-agent-job.yaml
│   └── README.md
├── examples/
│   └── quickstart.sh        # one-command demo: train + eval Spanish
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

## Install

```bash
git clone https://github.com/vamsin07/whisper-vanilla-finetune.git
cd whisper-vanilla-finetune
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.5. A CUDA-capable GPU is strongly
recommended (training on CPU works but is glacially slow).

## Quickstart (local GPU, ~30-60 min on a consumer card)

The one-liner that does everything:
```bash
bash examples/quickstart.sh
```

Or explicitly — fine-tune Whisper-base on Spanish, then eval on FLEURS-test:
```bash
# Train
python src/train.py \
  --lang spanish \
  --fleurs-code es_419 \
  --whisper-lang spanish \
  --model openai/whisper-base \
  --output-dir runs/spanish-base \
  --max-epochs 3 \
  --seed 42

# Eval the best ckpt on FLEURS-test
python src/eval.py \
  --ckpt-dir runs/spanish-base/best \
  --fleurs-code es_419 \
  --whisper-lang spanish \
  --results-path runs/spanish-base/test_results.json

cat runs/spanish-base/test_results.json
```

What gets written:
- `runs/spanish-base/best/checkpoint.pt`        ← best-by-val-WER weights
- `runs/spanish-base/best/training_config.json` ← every flag + hyperparam used
- `runs/spanish-base/test_results.json`         ← final FLEURS-test WER/CER

## Reproducing the "paper-scale" recipe

The default `openai/whisper-base` is for fast iteration. For numbers comparable
to a typical research paper, switch to `openai/whisper-large-v3` and use the
recipe knobs the project's existing code uses:

```bash
python src/train.py \
  --lang italian --fleurs-code it_it --whisper-lang italian \
  --model openai/whisper-large-v3 \
  --output-dir runs/italian-large \
  --learning-rate 6.48e-6 \
  --encoder-lr-ratio 0.3 \
  --weight-decay 4.35e-5 \
  --grad-accum-steps 16 \
  --warmup-steps 150 \
  --scheduler cosine \
  --max-epochs 6 \
  --patience 3 \
  --seed 42 \
  --wandb-project whisper-vanilla-italian
```

Expected runtime: ~2-4 hours on one A100 / H100, ~6-8 hours on a 3090.
Whisper-large-v3 needs ~24GB VRAM at `batch_size=2, grad_accum=16`.

## W&B hyperparameter sweep

```bash
export WANDB_API_KEY=...     # https://wandb.ai/settings → API keys

# 1) Register the sweep (returns a sweep ID like 'vamsin07/proj/abc123')
wandb sweep configs/sweep.yaml

# 2) Launch agents (one per GPU you have)
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu wandb agent <user>/<project>/<sweep-id> &
done
wait
```

The sweep searches `learning_rate`, `scheduler`, `encoder_lr_ratio`, and
`seed`. Open the sweep page on wandb.ai to see live results — each agent run
shows up as a colored line on the loss / WER plots.

## Running on Nautilus

See [`nautilus/README.md`](nautilus/README.md) for the full setup. Summary:

1. **One-time per user**: create a shared HF cache PVC + a personal output
   PVC, register `wandb-credentials` and (optional) `hf-credentials` secrets,
   build + push the image from `nautilus/Dockerfile` to
   `gitlab-registry.nautilus.optiputer.net/<your-group>/whisper-vanilla`.
2. **Per fine-tune**: edit the `EDIT ME` placeholders in `nautilus/train-job.yaml`
   (image path, PVC names, `--lang` args), then `kubectl apply -f nautilus/train-job.yaml`.
3. **For sweeps**: register the sweep on your laptop with `wandb sweep
   configs/sweep.yaml`, then submit N copies of `nautilus/sweep-agent-job.yaml`,
   one per parallel worker.

The Python code in `src/` is identical between local and Nautilus — only the
container + Job YAMLs differ. Adding new launch targets (Slurm, AWS Batch, …)
is mostly a matter of writing one more wrapper.

## How the recipe works

- **Data**: FLEURS-train via `datasets.load_dataset("google/fleurs", <code>, split="train")`
  — audio resampled to 16kHz, transcriptions tokenized with Whisper's tokenizer
  using the appropriate language prefix (`set_prefix_tokens(language=..., task="transcribe")`).
- **Model**: stock `WhisperForConditionalGeneration` from HuggingFace — no
  vocab swap, no new BPE. Gradient checkpointing is on for memory efficiency.
- **Optimizer**: AdamW with separate LR groups for encoder and decoder
  (`encoder_lr = encoder_lr_ratio × decoder_lr`, default ratio 0.3). Cosine
  LR schedule with 150-step warmup.
- **Loss**: standard next-token cross-entropy on the transcription tokens.
- **Eval**: every `--eval-steps` steps, greedy-decode on FLEURS-validation,
  compute WER + CER + eval-loss. Save the best-WER checkpoint to `best/`.
- **Stopping**: early-stop if val WER hasn't improved for `--patience` eval
  rounds, OR hard cap at `--max-epochs` × steps-per-epoch.
- **Inference (eval.py)**: greedy decode with `no_repeat_ngram_size=3` and
  `repetition_penalty=1.2` to suppress the rare repetition collapses that
  greedy + autoregressive can fall into. Reports both raw WER/CER and a
  normalized version (lowercase + drop non-letter/digit chars) for
  cross-language comparability.

## License

MIT. See [`LICENSE`](LICENSE).
