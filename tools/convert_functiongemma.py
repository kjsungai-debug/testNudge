"""
Convert the fine-tuned FunctionGemma 270M (HuggingFace safetensors) checkpoint
in `functiongemma-270m-0504/` to a LiteRT-LM `.litertlm` bundle that the
Android app can load.

Pipeline:

    HF safetensors (functiongemma-270m-0504/)
        └─ ai_edge_torch.generative.examples.gemma3.convert_gemma3_to_tflite
              (model_size=270m, dynamic_int8 quantization)
            ▼
        functiongemma_270m.tflite
              └─ litert-lm-builder
                    (bundle .tflite + tokenizer.model into a Task Bundle)
                ▼
            functiongemma_270m.litertlm  ──▶  app/src/main/assets/

Usage:

    1. Make sure Python >= 3.10 and pip are available.
    2. (Recommended) create a virtualenv:
           python -m venv .venv
           # PowerShell:
           .\.venv\Scripts\Activate.ps1
           # bash:
           source .venv/bin/activate
    3. Install dependencies (large — ~3 GB):
           pip install --upgrade pip
           pip install ai-edge-torch transformers safetensors torch
           pip install litert-lm-builder
    4. Run this script from the project root:
           python tools/convert_functiongemma.py

The resulting `backup/functiongemma_270m.litertlm` should be copied into
`app/src/main/assets/functiongemma_270m.litertlm` before building the app
(or wired up via a Gradle copy task).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "functiongemma-270m-0504"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "tools" / "build"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backup"
DEFAULT_TFLITE_NAME = "functiongemma_270m.tflite"
DEFAULT_LITERTLM_NAME = "functiongemma_270m.litertlm"


def run(cmd: list[str]) -> None:
    """Run a subprocess command with live output, abort on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def step_convert_to_tflite(checkpoint_path: Path, build_dir: Path, quantize: str) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    tflite_path = build_dir / DEFAULT_TFLITE_NAME
    print(f"[1/2] HF safetensors → TFLite ({quantize})")
    run(
        [
            sys.executable,
            "-m",
            "ai_edge_torch.generative.examples.gemma3.convert_gemma3_to_tflite",
            "--model_size=270m",
            f"--checkpoint_path={checkpoint_path}",
            f"--output_path={build_dir}",
            f"--output_name_prefix={tflite_path.stem}",
            f"--quantize={quantize}",
        ]
    )
    if not tflite_path.exists():
        # convert_gemma3_to_tflite may suffix with size/quant, find a likely match
        candidates = sorted(build_dir.glob(f"{tflite_path.stem}*.tflite"))
        if not candidates:
            sys.exit(f"TFLite output not found under {build_dir}")
        tflite_path = candidates[0]
    print(f"  → {tflite_path}  ({tflite_path.stat().st_size / 1_048_576:.1f} MB)")
    return tflite_path


def step_bundle_litertlm(
    tflite_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    bundle_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / bundle_name
    print("[2/2] Bundle .tflite + tokenizer → .litertlm")
    run(
        [
            "litert-lm-builder",
            "system_metadata",
            "--str", "Authors", "FunctionGemma 270M action-recall fine-tune",
            "tflite_model",
            "--path", str(tflite_path),
            "--model_type", "prefill_decode",
            "sp_tokenizer",
            "--path", str(tokenizer_path),
            "output",
            "--path", str(output_path),
        ]
    )
    if not output_path.exists():
        sys.exit(f"litertlm bundle not produced at {output_path}")
    print(f"  → {output_path}  ({output_path.stat().st_size / 1_048_576:.1f} MB)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help="HuggingFace safetensors checkpoint directory."
    )
    parser.add_argument(
        "--build-dir", type=Path, default=DEFAULT_BUILD_DIR,
        help="Where to write intermediate .tflite files."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Where to write the final .litertlm bundle."
    )
    parser.add_argument(
        "--bundle-name", default=DEFAULT_LITERTLM_NAME,
        help="Filename for the produced .litertlm bundle."
    )
    parser.add_argument(
        "--quantize", default="dynamic_int8",
        help="ai-edge-torch quantization scheme (e.g. dynamic_int8, none)."
    )
    parser.add_argument(
        "--copy-to-assets", action="store_true",
        help="Also copy the resulting .litertlm into app/src/main/assets/."
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint folder not found: {args.checkpoint}")

    tokenizer = args.checkpoint / "tokenizer.model"
    if not tokenizer.exists():
        sys.exit(f"tokenizer.model missing in {args.checkpoint}")

    tflite_path = step_convert_to_tflite(
        checkpoint_path=args.checkpoint,
        build_dir=args.build_dir,
        quantize=args.quantize,
    )
    bundle_path = step_bundle_litertlm(
        tflite_path=tflite_path,
        tokenizer_path=tokenizer,
        output_dir=args.output_dir,
        bundle_name=args.bundle_name,
    )

    if args.copy_to_assets:
        assets_dir = PROJECT_ROOT / "app" / "src" / "main" / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        target = assets_dir / args.bundle_name
        shutil.copy2(bundle_path, target)
        print(f"\nCopied to {target}")

    print(
        "\nDone. Before building the app, ensure "
        f"{args.bundle_name} is present at app/src/main/assets/."
    )


if __name__ == "__main__":
    main()
