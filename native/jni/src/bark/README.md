# Bark speech recognition

Native inference for [Bark 3.0](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0)
`.bark` models (CC-BY-4.0): a compressed fine-tune of NVIDIA's
`parakeet-tdt-0.6b-v3`, i.e. a 24-layer FastConformer encoder with an LSTM/TDT
transducer decoder. English only, 16 kHz audio.

- `bark_format.cpp` reads `.bark` files (`bark-levels-v2`/`v3`). It mirrors
  `bark_loader.py` from the model repository, including the ANS decoding of
  constriction 0.5 (`stack.AnsCoder`, `Categorical(perfect=False)`), so the
  weights come out bit-identical. xz blobs go through `third_party/xz-embedded`.
- Encoder weights (2K+1 values per row) stay packed at 3 codes per byte with a
  per-row lookup table, other matrices as int8 with row scales. A loaded model
  takes about 240 MB.
- `bark_model.cpp` follows transformers' `ParakeetFeatureExtractor` and
  `ParakeetForTDT` (greedy TDT decoding, plus NeMo's limit on symbols per
  frame). Like `bark_loader.transcribe`, 0.25 s of silence is prepended.
- `bark_ops.cpp` has the GEMM (weights are dequantized per 64x192 block and
  multiplied with a 4x4 vector kernel) and a small thread pool.

The tokenizer is not part of `.bark` files; the app ships it as the
`bark_vocab.txt` asset in `voiceinput-shared`, and `BarkModel.kt` turns token
ids into text.
