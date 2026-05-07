# Model conversion tooling

These scripts convert HuggingFace fine-tunes — the FunctionGemma 270M
generative model and the tinyBERT classifier — into the runtime formats the
Android app loads. They run on a desktop / laptop with Python 3.10+ and are
**not** part of the Android Gradle build.

## Source checkpoints

The HuggingFace checkpoint folders are **not committed to the repository**
(too large; gitignored). Before running either conversion, download or copy
the appropriate fine-tune locally. Both scripts default to looking in the
project root, but accept any path via `--checkpoint`:

| Script | Default `--checkpoint` path | Used by |
|---|---|---|
| `convert_functiongemma.py` | `functiongemma-270m-0504/` | Models tab (FunctionGemma) |
| `convert_tinybert.py` | `tinyBERT-checkpoint-68370/` | Gating tab (tinyBERT classifier) |

If your checkpoint lives somewhere else, pass `--checkpoint /path/to/ckpt`.

## convert_functiongemma.py

Converts `functiongemma-270m-0504/` (HF safetensors) to
`backup/functiongemma_270m.litertlm`.

### One-time setup

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install ai-edge-torch transformers safetensors torch
pip install litert-lm-builder
```

(`torch` alone is ~2 GB; the full install is ~3 GB. A virtualenv is
recommended so it doesn't pollute the system Python.)

### Run

```powershell
python tools/convert_functiongemma.py --copy-to-assets
```

This produces:

- `tools/build/functiongemma_270m.tflite` — intermediate TFLite (≈ 270 MB at `dynamic_int8`)
- `backup/functiongemma_270m.litertlm` — final Task Bundle (≈ 280 MB)
- `app/src/main/assets/functiongemma_270m.litertlm` — copy used by the app
  (only when `--copy-to-assets` is passed)

### Pipeline

1. **HF → TFLite** — `ai_edge_torch.generative.examples.gemma3.convert_gemma3_to_tflite`
   with `--model_size=270m`, default `dynamic_int8` quantization.
2. **TFLite + tokenizer → litertlm** — `litert-lm-builder` packages the TFLite
   model together with the original `tokenizer.model` (SentencePiece) into
   the Task Bundle that `Engine` expects.

### Notes

- The chat template embedded in the original HF checkpoint
  (`chat_template.jinja`-style FunctionGemma turn tokens) is **not** carried
  into the bundle — `litert-lm-builder` does not currently expose a chat
  template field. The Android app already formats prompts manually via
  [FunctionGemmaPrompt](../app/src/main/java/com/example/testnudge/llm/FunctionGemmaPrompt.kt),
  so this is fine in practice.
- `dynamic_int8` matches the published FunctionGemma on-device benchmarks. To
  preserve full precision (larger, slower), pass `--quantize none`.
- If the conversion fails on a memory-constrained machine, run on a host
  with at least 16 GB RAM (PyTorch holds the safetensors in float32 during
  tracing).
- The resulting `.litertlm` is large; it stays in `backup/` and
  `app/src/main/assets/`, both of which are gitignored.

## convert_tinybert.py

Converts `tinyBERT-checkpoint-68370/` (HF safetensors) to a plain TFLite
classifier that the Gating tab runs via `org.tensorflow.lite.Interpreter`.

### One-time setup

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install ai-edge-torch torch transformers
```

(`litert-lm-builder` is **not** required — the gating model is a plain
classifier, not a generative LLM, so it doesn't go through the LiteRT-LM
bundle path.)

### Run

```powershell
python tools/convert_tinybert.py --copy-to-assets
```

This produces:

- `backup/tinybert_gating.tflite` — converted model (≈ 19 MB)
- `backup/tinybert_vocab.txt` — extracted WordPiece vocabulary (line N = token id N)
- Both copied into `app/src/main/assets/` when `--copy-to-assets` is passed.

### Pipeline

1. **HF → TFLite** — `ai_edge_torch.convert(BertForSequenceClassification, sample_inputs)`
   produces a TFLite graph with three int inputs (`input_ids`,
   `attention_mask`, `token_type_ids`, all shape `[1, 128]`) and one float
   output (`logits`, shape `[1, 4]`).
2. **Vocab extraction** — the HF tokenizer's vocab dict (≈ 150k entries) is
   sorted by id and written line-by-line to `tinybert_vocab.txt`.

### Notes

- Sequence length is fixed at 128 in the conversion. Inputs longer than 126
  WordPiece tokens (after `[CLS]` / `[SEP]`) are truncated by the Android
  service; the model never sees them.
- The model is multi-label (sigmoid per head), not single-label softmax. The
  Android service applies sigmoid manually and uses `>= 0.5` as the decision
  threshold.
- The tokenizer is treated as **case-sensitive** (multilingual checkpoint).
  If your fine-tune was trained with lowercase normalization, change the
  `setDoLowerCase(false)` call in
  [TinyBertGatingService](../app/src/main/java/com/example/testnudge/llm/TinyBertGatingService.kt).
