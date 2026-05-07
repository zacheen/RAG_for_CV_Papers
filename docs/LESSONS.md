# SESSIONS

# LESSONS

## Dependency pinning for torch 2.1.x environments (2026-05-06)

Symptom: `streamlit run app.py` produced
`Retrieval failed: name 'nn' is not defined`. The full traceback in the
terminal showed:

```
[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.1.0+cu118
NameError: name 'nn' is not defined
```

Root cause: the latest `transformers` (>= 4.50) requires `torch >= 2.4`. Our
`py310_torch210_cuda118` conda env has `torch 2.1.0+cu118`. When
`transformers` detects the older torch, it disables PyTorch support, which
makes downstream code referencing `torch.nn` (used by `sentence_transformers`
and the `chromadb` embedding function) fail with `name 'nn' is not defined`.

Secondary issue: after pinning `transformers<4.50`, importing it pulled in
`transformers.modeling_tf_utils`, which fails when the env has Keras 3
without the `tf-keras` backwards-compat shim. The env happened to have
Keras 3.12.1 installed (left over from prior work) but not TensorFlow.

Wrong fix attempted: `pip install tf-keras`. This worked, but `tf-keras`
declares `tensorflow>=2.21` as a hard dependency and pip happily pulled
`tensorflow` 2.21 (350 MB) into the env. The env also already had
`tensorflow_cpu` 2.21, `tensorboard`, and `keras` 3.12.1 left over from
prior unrelated work, which were the real reason transformers triggered
the TF integration layer in the first place.

Correct fix:
- Uninstall everything TF-related: `tensorflow`, `tensorflow_cpu`,
  `tf-keras`, `keras`, `tensorboard`, `tensorboard-data-server`. With no
  TF backend present, transformers' auto-detection skips the TF path
  entirely and the Keras 3 conflict cannot occur.
- Keep `transformers<4.50` and `sentence-transformers<5` pins for the
  torch 2.1.x compatibility.

Going forward this env stays PyTorch-only. The `requirements.txt` comment
warns against re-introducing TF / Keras.

Alternative (not chosen): upgrade torch to >= 2.4 with the cu118 wheels
(`pip install --index-url https://download.pytorch.org/whl/cu118 'torch>=2.4'`).
Avoided because the env name implies torch 2.1.0 is intentional for other
work in this conda environment.
