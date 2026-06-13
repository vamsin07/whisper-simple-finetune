"""Eval a fine-tuned Whisper checkpoint on FLEURS-test (or any FLEURS split).

Loads the best/ checkpoint saved by train.py, runs greedy generation on the
held-out test split, and reports raw + normalized WER/CER.

Example:
  python src/eval.py \\
    --ckpt-dir runs/spanish-base/best \\
    --fleurs-code es_419 --whisper-lang spanish \\
    --results-path runs/spanish-base/test_results.json

The normalizer is intentionally simple (lowercase + strip non-letter/digit/space):
it's meant for cross-lang comparability, not Whisper's full English normalizer.
"""
import argparse, json, logging, os, re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import regex as rx
import torch
from jiwer import wer as compute_wer, cer as compute_cer
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor

# Re-use the dataset + collator from train.py to keep the eval-time data pipeline
# identical to training (same audio resampling, same feature extractor, etc.)
from train import AudioASRDataset, asr_collate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def normalize_text(t):
    """Lang-agnostic normalization: lowercase + drop non-letter/digit/space tokens."""
    t = (t or "").lower()
    t = rx.sub(r"[^\p{L}\p{M}\p{N}\s'\-]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def metrics(refs, preds):
    if not preds or not any(p.strip() for p in preds):
        return {"wer": -1, "cer": -1, "wer_norm": -1, "cer_norm": -1, "n": len(preds)}
    nr  = [normalize_text(r) for r in refs]
    np_ = [normalize_text(p) for p in preds]
    return {
        "wer":      compute_wer(refs, preds),
        "cer":      compute_cer(refs, preds),
        "wer_norm": compute_wer(nr, np_),
        "cer_norm": compute_cer(nr, np_),
        "n": len(preds),
    }


@torch.no_grad()
def generate_all(model, tokenizer, loader, device, language,
                 max_new_tokens, no_repeat_ngram_size, repetition_penalty):
    refs, hyps = [], []
    for batch in loader:
        feats = batch["input_features"].to(device, dtype=torch.float16)
        with autocast():
            gen = model.generate(
                feats,
                max_new_tokens=max_new_tokens,
                language=language, task="transcribe",
                num_beams=1,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
            )
        for g, t in zip(gen, batch["texts"]):
            hyps.append(tokenizer.decode(g, skip_special_tokens=True).strip())
            refs.append(t)
    return refs, hyps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True,
                    help="Directory containing checkpoint.pt (typically <output-dir>/best/)")
    ap.add_argument("--fleurs-code", required=True, help="FLEURS code (e.g., 'es_419')")
    ap.add_argument("--whisper-lang", required=True, help="Whisper language token (e.g., 'spanish')")
    ap.add_argument("--model", default=None,
                    help="HF model id used during training. If omitted, read from training_config.json.")
    ap.add_argument("--fleurs-split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    ap.add_argument("--results-path", default=None,
                    help="Where to write JSON metrics + sample preds (default: <ckpt-dir>/test_results.json)")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / "checkpoint.pt"
    if not ckpt_path.exists():
        ap.error(f"checkpoint not found: {ckpt_path}")

    # Resolve base model from training_config.json if available
    train_cfg_path = ckpt_dir / "training_config.json"
    if args.model is None:
        if train_cfg_path.exists():
            args.model = json.load(open(train_cfg_path))["model"]
            log.info(f"Resolved --model from {train_cfg_path}: {args.model}")
        else:
            ap.error("--model not provided and no training_config.json in ckpt-dir")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading {args.model} ...")
    model = WhisperForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    tokenizer = WhisperTokenizer.from_pretrained(args.model)
    feat = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer.set_prefix_tokens(language=args.whisper_lang, task="transcribe")

    log.info(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    log.info(f"  ckpt step={ckpt.get('step','?')} train-time WER={ckpt.get('wer',-1)*100:.2f}%")

    log.info(f"Loading FLEURS {args.fleurs_code} split={args.fleurs_split} ...")
    ds = load_dataset("google/fleurs", args.fleurs_code, split=args.fleurs_split)
    log.info(f"  {len(ds):,} samples")
    eval_ds = AudioASRDataset(ds, feat, tokenizer, text_column="transcription")
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=asr_collate)

    log.info("Generating ...")
    refs, hyps = generate_all(
        model, tokenizer, eval_loader, device, args.whisper_lang,
        args.max_new_tokens, args.no_repeat_ngram_size, args.repetition_penalty,
    )
    m = metrics(refs, hyps)
    log.info(f"FLEURS-{args.fleurs_split} WER={m['wer']*100:.2f}%  CER={m['cer']*100:.2f}%  "
             f"WER_norm={m['wer_norm']*100:.2f}%  CER_norm={m['cer_norm']*100:.2f}%  N={m['n']}")

    results_path = Path(args.results_path) if args.results_path else (ckpt_dir / f"{args.fleurs_split}_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fleurs_code": args.fleurs_code,
        "whisper_lang": args.whisper_lang,
        "fleurs_split": args.fleurs_split,
        "model": args.model,
        "ckpt": str(ckpt_path),
        "metrics": m,
        "sample_predictions": [
            {"ref": r, "hyp": h} for r, h in zip(refs[:10], hyps[:10])
        ],
    }
    json.dump(payload, open(results_path, "w"), indent=2, ensure_ascii=False)
    log.info(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
