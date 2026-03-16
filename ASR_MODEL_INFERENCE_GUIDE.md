# Loading & Using Saved Model Weights for Inference and Deployment

---

## Quick Reference — What You Need to Load a Checkpoint

| Format | Contains | What you need to load it |
|---|---|---|
| HuggingFace checkpoint folder | Weights + config + tokenizer | Just `from_pretrained(path)` |
| `state_dict` (`.pt` / `.bin`) | Weights only | Instantiate model architecture first |
| Full PyTorch checkpoint | Weights + optimizer + metadata | Instantiate model architecture first |
| TensorFlow SavedModel | Weights + architecture + graph | Nothing extra |

---

## HuggingFace Transformers

The standard format for Whisper and most modern transformer models. The checkpoint folder contains `config.json`, `pytorch_model.bin` (or sharded `.safetensors`), and tokenizer files — so the architecture is bundled with the weights and you never need to reconstruct it manually.

### Loading

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

ckpt_path = "/path/to/checkpoint"  # local folder or HuggingFace Hub model ID

processor = WhisperProcessor.from_pretrained(ckpt_path)
model = WhisperForConditionalGeneration.from_pretrained(ckpt_path)
model = model.to(device="cuda", dtype=torch.float16)
model.eval()
```

### Running Inference

```python
# audio_array: np.float32, 16kHz mono
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(model.device, dtype=model.dtype)

with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        language="zh",    # or "en", or omit for auto-detect
        task="transcribe",
        num_beams=5,
    )

transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

---

## PyTorch — `state_dict` (weights only)

Use this when you saved only the model parameters, not the full HuggingFace config.

```python
import torch

# Saving
torch.save(model.state_dict(), "model_weights.pt")

# Loading — must instantiate the architecture first
model = MyModel()
model.load_state_dict(torch.load("model_weights.pt", map_location="cuda"))
model.eval()
```

## PyTorch — Full Checkpoint (weights + optimizer + epoch)

Use this format when saving mid-training so you can resume later. For inference-only, you only need `model_state_dict`.

```python
# Saving
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "checkpoint.pt")

# Loading for inference
checkpoint = torch.load("checkpoint.pt", map_location="cuda")
model = MyModel()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Loading to resume training
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]
```

---

## TensorFlow / Keras

```python
from tensorflow import keras

# Full model (architecture + weights)
model.save("my_model")                        # save
model = keras.models.load_model("my_model")  # load

# Weights only
model.save_weights("weights.h5")             # save
model.load_weights("weights.h5")             # load (architecture must exist first)
```

---

## Deployment Patterns

### Local Script / Notebook

Load once and call `model.generate()` as shown above. Always wrap inference in `torch.no_grad()`.

### REST API with FastAPI

Load the model once at startup — never per request.

```python
from fastapi import FastAPI, UploadFile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

app = FastAPI()
ckpt_path = "/path/to/checkpoint"

processor = WhisperProcessor.from_pretrained(ckpt_path)
model = WhisperForConditionalGeneration.from_pretrained(ckpt_path)
model = model.to("cuda", dtype=torch.float16).eval()

@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    audio_bytes = await audio.read()
    audio_array = preprocess_audio(audio_bytes)  # your 16kHz mono conversion
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        ids = model.generate(input_features, language="zh", task="transcribe")
    return {"transcript": processor.batch_decode(ids, skip_special_tokens=True)[0]}
```

---

## Inference Optimisation Checklist

| Technique | How | Effect |
|---|---|---|
| Disable gradients | `torch.no_grad()` or `@torch.inference_mode()` | Reduces memory, slight speedup |
| Half precision | `dtype=torch.float16` or `model.half()` | ~2× memory reduction, faster on GPU |
| Eval mode | `model.eval()` | Disables dropout and batchnorm training behaviour — essential for consistent results |
| Batch inputs | Pass multiple samples at once | Better GPU utilisation |
| ONNX export | See below | Portable, faster on CPU, cross-framework |
| Faster-Whisper | See below | 4–8× throughput improvement for Whisper specifically |

### ONNX Export (CPU / cross-framework deployment)

```python
from optimum.exporters.onnx import main_export

main_export(
    "path/to/checkpoint",
    output="onnx_model/",
    task="automatic-speech-recognition"
)
```

### Faster-Whisper (maximum Whisper throughput)

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) uses CTranslate2 under the hood and is 4–8× faster than the HuggingFace implementation at equivalent quality. Convert your checkpoint once, then use the faster-whisper API:

```bash
pip install faster-whisper
ct2-whisper-converter --model /path/to/checkpoint --output_dir ct2_model --quantization int8
```

```python
from faster_whisper import WhisperModel

model = WhisperModel("ct2_model", device="cuda", compute_type="int8_float16")
segments, info = model.transcribe("audio.wav", language="zh")
for segment in segments:
    print(segment.text)
```

---

## Common Pitfalls

**Forgetting `model.eval()`** — dropout and batch normalisation behave differently during training vs. inference. Always call `model.eval()` before generating predictions or you will get inconsistent, slightly randomised outputs.

**Loading weights on the wrong device** — use `map_location` when loading on a different machine or if the checkpoint was saved on GPU but you're loading on CPU:
```python
torch.load("checkpoint.pt", map_location="cpu")
# or
torch.load("checkpoint.pt", map_location=torch.device("cuda:0"))
```

**dtype mismatch** — if the model is in float16 but input tensors are float32 (or vice versa), you'll get a runtime error. Always cast input features to `model.dtype` before passing them in:
```python
input_features = input_features.to(dtype=model.dtype)
```

**Not using `torch.no_grad()`** — without it, PyTorch builds a computation graph for every forward pass, wasting memory and time during inference.

**Loading a `state_dict` into the wrong architecture** — `load_state_dict` will raise a key mismatch error if the saved weights don't match the model you instantiated. Make sure the architecture (number of layers, hidden size, etc.) exactly matches what was used during training.
