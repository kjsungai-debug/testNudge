"""
Convert the fine-tuned tinyBERT classifier in `tinyBERT-checkpoint-68370/`
into a plain TFLite (LiteRT) model that the Android Gating screen can run via
`org.tensorflow.lite.Interpreter`.

The HF checkpoint is a `BertForSequenceClassification` with 4 multi-label
heads (action / recall / safety / unknown). Conversion path:

    HF pytorch_model.bin (BertForSequenceClassification)
        └─ ai_edge_torch.convert
            (input_ids, attention_mask, token_type_ids → logits[1, 4])
          ▼
        tinybert_gating.tflite   ──▶  app/src/main/assets/

Plus a side artifact extracted from the HF tokenizer:

        tinybert_vocab.txt       ──▶  app/src/main/assets/

(line N = token with id N, used by `org.tensorflow.lite.support.text.tokenizers.BertTokenizer`).

Usage (from project root):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1            # PowerShell
    # or:  source .venv/bin/activate        # bash
    pip install --upgrade pip
    pip install ai-edge-torch torch transformers
    python tools/convert_tinybert.py --copy-to-assets
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "tinyBERT-checkpoint-68370"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backup"
DEFAULT_TFLITE_NAME = "tinybert_gating.tflite"
DEFAULT_VOCAB_NAME = "tinybert_vocab.txt"
DEFAULT_MAX_SEQ_LEN = 128


def convert_to_tflite(checkpoint: Path, output_path: Path, max_seq_len: int) -> None:
    import torch
    import torch.nn as nn
    import ai_edge_torch
    from transformers import AutoModelForSequenceClassification

    print(f"[1/2] Loading {checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    model.eval()

    class BertWrapper(nn.Module):
        """Flatten the HF kwargs interface to positional args for ai-edge-torch."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    wrapped = BertWrapper(model)
    sample = (
        torch.zeros(1, max_seq_len, dtype=torch.long),
        torch.ones(1, max_seq_len, dtype=torch.long),
        torch.zeros(1, max_seq_len, dtype=torch.long),
    )
    print(f"     Tracing with seq_len={max_seq_len}")
    edge_model = ai_edge_torch.convert(wrapped, sample)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(output_path))
    size_mb = output_path.stat().st_size / 1_048_576
    print(f"     → {output_path} ({size_mb:.1f} MB)")


def extract_vocab(checkpoint: Path, output_path: Path) -> None:
    from transformers import AutoTokenizer

    print(f"[2/2] Extracting vocab from HF tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    vocab = tokenizer.get_vocab()  # dict[str, int]
    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")
    print(f"     → {output_path} ({len(sorted_vocab)} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--tflite-name", default=DEFAULT_TFLITE_NAME)
    parser.add_argument("--vocab-name", default=DEFAULT_VOCAB_NAME)
    parser.add_argument(
        "--copy-to-assets", action="store_true",
        help="Also copy the tflite + vocab into app/src/main/assets/."
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint folder not found: {args.checkpoint}")

    tflite_path = args.output_dir / args.tflite_name
    vocab_path = args.output_dir / args.vocab_name

    convert_to_tflite(args.checkpoint, tflite_path, args.max_seq_len)
    extract_vocab(args.checkpoint, vocab_path)

    if args.copy_to_assets:
        assets = PROJECT_ROOT / "app" / "src" / "main" / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tflite_path, assets / args.tflite_name)
        shutil.copy2(vocab_path, assets / args.vocab_name)
        print(f"\nCopied to {assets}\\")
        print(f"  {args.tflite_name}")
        print(f"  {args.vocab_name}")

    print(
        "\nDone. Before building the app, ensure both files are present at "
        "app/src/main/assets/."
    )


if __name__ == "__main__":
    main()
