"""Vanilla fine-tuning for Whisper on FLEURS.

Plain supervised fine-tuning — no vocab swap, no new BPE, no text-MTL.
Trains on FLEURS-train for one language, evaluates on FLEURS-validation
every N steps, saves the best-by-WER checkpoint, and stops early when
validation WER stops improving.

Designed to be portable:
  - All paths configurable via CLI flags or env vars (HF_HOME, WANDB_API_KEY, ...)
  - FLEURS pulled from HuggingFace datasets — no local copy required
  - Model size selectable via --model (default: openai/whisper-base for fast iteration)

Example (Whisper-base on Spanish, single GPU, ~30 min on a consumer GPU):
  python src/train.py \\
    --lang spanish --fleurs-code es_419 \\
    --model openai/whisper-base \\
    --output-dir runs/spanish-base \\
    --max-epochs 3

Example (Whisper-large-v3 on Italian, single A100 / H100, ~2-4 h):
  python src/train.py \\
    --lang italian --fleurs-code it_it \\
    --model openai/whisper-large-v3 \\
    --output-dir runs/italian-large \\
    --max-epochs 6 --learning-rate 6.48e-6
"""
import argparse, json, logging, math, os, random
from pathlib import Path

# Defaults are reasonable; users can override via env. HF_HOME controls where
# the dataset + model cache live (important on Nautilus where you'll point this
# at a shared PVC).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from jiwer import wer as compute_wer, cer as compute_cer
from transformers import (
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperFeatureExtractor,
)
from datasets import load_dataset

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AudioASRDataset(Dataset):
    """Wraps a HuggingFace audio dataset (FLEURS) for Whisper fine-tuning."""

    def __init__(self, ds, feature_extractor, tokenizer, text_column,
                 max_audio_seconds=30.0, max_label_length=448):
        self.ds = ds
        self.feat = feature_extractor
        self.tok = tokenizer
        self.text_column = text_column
        self.max_audio_samples = int(max_audio_seconds * 16000)
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio[: self.max_audio_samples]
        feats = self.feat(audio, sampling_rate=16000, return_tensors="pt").input_features.squeeze(0)
        text = item[self.text_column]
        labels = self.tok(text, max_length=self.max_label_length, truncation=True,
                          padding=False, return_tensors=None)["input_ids"]
        return {"input_features": feats, "labels": torch.tensor(labels, dtype=torch.long), "text": text}


def asr_collate(batch):
    """Stack features + right-pad labels with -100 (ignored by cross-entropy)."""
    feats = torch.stack([b["input_features"] for b in batch])
    max_len = max(b["labels"].size(0) for b in batch)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :b["labels"].size(0)] = b["labels"]
    texts = [b["text"] for b in batch]
    return {"input_features": feats, "labels": labels, "texts": texts}


@torch.no_grad()
def evaluate(model, tokenizer, loader, device, language):
    """Greedy decode on a held-out set, compute WER/CER/loss."""
    model.eval()
    base = model.module if hasattr(model, "module") else model
    refs, hyps, losses = [], [], []
    for batch in loader:
        feats = batch["input_features"].to(device, dtype=torch.float16)
        labels = batch["labels"].to(device)
        with autocast():
            out = base(input_features=feats, labels=labels)
            losses.append(out.loss.item())
            gen = base.generate(feats, max_new_tokens=256, language=language,
                                task="transcribe", num_beams=1)
        for g, t in zip(gen, batch["texts"]):
            hyps.append(tokenizer.decode(g, skip_special_tokens=True).strip())
            refs.append(t)
    model.train()
    return {
        "wer": compute_wer(refs, hyps) if any(refs) else -1,
        "cer": compute_cer(refs, hyps) if any(refs) else -1,
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "n": len(refs),
    }


def make_scheduler(optimizer, kind, warmup_steps, total_steps, decay_rate=0.95):
    """Linear-warmup + {cosine, linear, exponential} decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if kind == "linear":
            return max(0.0, 1.0 - progress)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "exponential":
            return decay_rate ** progress
        raise ValueError(f"unknown scheduler kind: {kind}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Required: what language and where to write
    p.add_argument("--lang", required=True, help="Human-readable language name (e.g., 'spanish')")
    p.add_argument("--fleurs-code", required=True,
                   help="FLEURS dataset code (e.g., 'es_419', 'it_it', 'ja_jp')")
    p.add_argument("--whisper-lang", default=None,
                   help="Whisper language token (e.g., 'spanish'). Defaults to --lang.")
    p.add_argument("--output-dir", required=True,
                   help="Where to write training_config.json + best/ + latest/ checkpoints")

    # Model + optimization
    p.add_argument("--model", default="openai/whisper-base",
                   help="HF model id (default: openai/whisper-base for fast iteration; "
                        "use openai/whisper-large-v3 for the paper-scale recipe)")
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="Decoder LR (encoder LR = encoder-lr-ratio × this; default 1e-5)")
    p.add_argument("--encoder-lr-ratio", type=float, default=0.3,
                   help="encoder_lr = encoder_lr_ratio × decoder_lr (default 0.3, matches the paper recipe)")
    p.add_argument("--weight-decay", type=float, default=4e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--warmup-steps", type=int, default=150)
    p.add_argument("--scheduler", default="cosine", choices=["cosine", "linear", "exponential"])
    p.add_argument("--scheduler-decay-rate", type=float, default=0.95,
                   help="Only used when --scheduler=exponential")

    # Step controls + early stopping
    p.add_argument("--max-epochs", type=int, default=6, help="cap on epochs over FLEURS-train")
    p.add_argument("--max-steps", type=int, default=None,
                   help="hard step cap (overrides --max-epochs if set)")
    p.add_argument("--eval-steps", type=int, default=100, help="eval every N optimizer steps")
    p.add_argument("--patience", type=int, default=3,
                   help="stop training if val WER doesn't improve for this many eval rounds")
    p.add_argument("--checkpoint-every", type=int, default=1000,
                   help="save a 'latest/' checkpoint every N steps for preemption recovery; 0 disables")
    p.add_argument("--seed", type=int, default=42)

    # Tracking
    p.add_argument("--wandb-project", default=None,
                   help="If set, log to this W&B project. Requires WANDB_API_KEY in env.")
    p.add_argument("--wandb-run-name", default=None,
                   help="Override the auto-generated run name")
    p.add_argument("--wandb-tags", nargs="+", default=None)

    args = p.parse_args()
    args.whisper_lang = args.whisper_lang or args.lang

    # Seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best"; best_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== vanilla Whisper fine-tune: {args.lang} ({args.fleurs_code}) ===")
    logger.info(f"  model={args.model}  seed={args.seed}")
    logger.info(f"  decoder_lr={args.learning_rate:.2e}  encoder_lr={args.learning_rate * args.encoder_lr_ratio:.2e}")
    logger.info(f"  batch_size={args.batch_size}  grad_accum={args.grad_accum_steps}  scheduler={args.scheduler}")
    logger.info(f"  output={out_dir}")

    # Model + tokenizer
    logger.info(f"Loading {args.model}...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = WhisperTokenizer.from_pretrained(args.model)
    feat = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer.set_prefix_tokens(language=args.whisper_lang, task="transcribe")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("No CUDA available; training on CPU will be very slow.")
    model.to(device)
    model.gradient_checkpointing_enable()

    # FLEURS data (via HuggingFace datasets — auto-downloads to HF_HOME on first run)
    logger.info(f"Loading FLEURS {args.fleurs_code} train + validation from HuggingFace datasets...")
    fleurs_train = load_dataset("google/fleurs", args.fleurs_code, split="train")
    fleurs_val   = load_dataset("google/fleurs", args.fleurs_code, split="validation")
    logger.info(f"  train: {len(fleurs_train):,} samples   val: {len(fleurs_val):,} samples")

    train_ds = AudioASRDataset(fleurs_train, feat, tokenizer, text_column="transcription")
    eval_ds  = AudioASRDataset(fleurs_val,   feat, tokenizer, text_column="transcription")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=asr_collate, drop_last=True)
    eval_loader  = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=asr_collate)

    # Step budget
    steps_per_epoch = len(train_ds) // (args.batch_size * args.grad_accum_steps)
    if args.max_steps is None:
        args.max_steps = steps_per_epoch * args.max_epochs
    logger.info(f"  steps_per_epoch={steps_per_epoch}  max_epochs={args.max_epochs}  "
                f"max_steps={args.max_steps}  patience={args.patience}")

    # Encoder/decoder LR split — encoder typically tuned lower than decoder for Whisper FT
    enc_params = list(model.model.encoder.parameters())
    dec_params = list(model.model.decoder.parameters()) + list(model.proj_out.parameters())
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.learning_rate * args.encoder_lr_ratio},
        {"params": dec_params, "lr": args.learning_rate},
    ], weight_decay=args.weight_decay)
    scaler = GradScaler()
    scheduler = make_scheduler(optimizer, args.scheduler, args.warmup_steps,
                               args.max_steps, args.scheduler_decay_rate)

    # W&B (optional)
    run = None
    if args.wandb_project and _HAS_WANDB:
        try:
            run_name = args.wandb_run_name or f"{args.lang}_{Path(args.model).name}_s{args.seed}"
            run = wandb.init(project=args.wandb_project, name=run_name,
                             config=vars(args), tags=args.wandb_tags)
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
    elif args.wandb_project and not _HAS_WANDB:
        logger.warning("--wandb-project set but the `wandb` package is not installed; pip install wandb")

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    train_iter = iter(train_loader)
    best_wer, best_cer, best_loss = float("inf"), float("inf"), float("inf")
    patience_counter = 0
    loss_accum = 0.0; loss_count = 0

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        feats = batch["input_features"].to(device, dtype=torch.float16)
        labels = batch["labels"].to(device)
        with autocast():
            out = model(input_features=feats, labels=labels)
            loss = out.loss
        scaler.scale(loss / args.grad_accum_steps).backward()
        loss_accum += loss.item(); loss_count += 1

        if loss_count % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
            scheduler.step()
            global_step += 1

            if global_step % 10 == 0:
                avg = loss_accum / max(loss_count, 1)
                logger.info(f"Step {global_step}: loss={avg:.4f}")
                if run:
                    wandb.log({"train/loss": avg, "train/step": global_step,
                               "train/lr_decoder": optimizer.param_groups[1]["lr"]},
                              step=global_step)
                loss_accum = 0.0; loss_count = 0

            if global_step % args.eval_steps == 0:
                logger.info(f"Evaluating at step {global_step}...")
                m = evaluate(model, tokenizer, eval_loader, device, args.whisper_lang)
                logger.info(f"Step {global_step}: WER={m['wer']*100:.2f}%  "
                            f"CER={m['cer']*100:.2f}%  eval_loss={m['loss']:.4f}  N={m['n']}")
                if run:
                    wandb.log({"eval/wer": m["wer"], "eval/cer": m["cer"], "eval/loss": m["loss"]},
                              step=global_step)
                if m["wer"] < best_wer:
                    best_wer, best_cer, best_loss = m["wer"], m["cer"], m["loss"]
                    patience_counter = 0
                    logger.info(f"  New best WER: {best_wer*100:.2f}% — saving to {best_dir}")
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "step": global_step,
                        "wer": best_wer, "cer": best_cer, "eval_loss": best_loss,
                    }, best_dir / "checkpoint.pt")
                    with open(best_dir / "training_config.json", "w") as f:
                        json.dump({
                            "lang": args.lang, "fleurs_code": args.fleurs_code,
                            "model": args.model, "seed": args.seed,
                            "learning_rate": args.learning_rate,
                            "encoder_lr_ratio": args.encoder_lr_ratio,
                            "step": global_step,
                            "best_wer": best_wer, "best_cer": best_cer,
                        }, f, indent=2)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        logger.info(f"Early stopping at step {global_step} "
                                    f"(no improvement for {args.patience} eval rounds)")
                        break

            if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                latest_dir = out_dir / "latest"; latest_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "step": global_step, "best_wer": best_wer, "best_cer": best_cer,
                }, latest_dir / "checkpoint.pt")
                logger.info(f"  [periodic ckpt] saved latest at step {global_step}")

    logger.info("=" * 60)
    logger.info(f"DONE: {args.lang}  best_WER={best_wer*100:.2f}%  best_CER={best_cer*100:.2f}%")
    logger.info("=" * 60)
    if run:
        wandb.finish()


if __name__ == "__main__":
    main()
