# Hapus & More Colab gateway architecture

Last updated: 4 September 2026

This document explains the temporary AI gateway we built for Google Colab. It
focuses on what happens when the single launcher cell is executed. The web
application itself is documented separately in
[web application architecture](https://github.com/HarshadHindlekar/hapus-and-more/blob/main/docs/web-application-architecture.md).

## 1. What we built

We built a temporary, authenticated bridge that runs the Qwen3-VL vision model
inside a Google Colab T4 runtime and makes it reachable by the Hapus & More web
application.

```text
Google Colab notebook
        │
        ├── launch_gateway.py       orchestration and startup checks
        ├── hapus_more_agent.py     model loader and HTTP gateway
        ├── Google Drive             persistent model files
        ├── Qwen3-VL-4B-Instruct     multimodal model
        └── Cloudflare Quick Tunnel  temporary public HTTPS URL
                         │
                         ▼
              Hapus & More web server
              HAPUS_AI_SERVICE_URL
```

The public helper repository contains only the two Python source files. It does
not contain model weights, user data, credentials, or application secrets.

## 2. The single launcher cell

The cell contains two operations:

```python
!wget -q https://raw.githubusercontent.com/HarshadHindlekar/hapus-more-colab-gateway/main/launch_gateway.py -O /content/launch_gateway.py
exec(open("/content/launch_gateway.py", encoding="utf-8").read(), globals())
```

The first line downloads the latest launcher into the temporary Colab disk.
The second line executes that launcher in the notebook's global scope, so the
successful run leaves `processor`, `model`, `gateway`, `token`, and
`tunnel_url` available for inspection.

The launcher then runs the following sequence.

## 3. What happens, step by step

### Step 1 — Install the runtime packages

The launcher runs:

```text
transformers>=4.57.0
bitsandbytes>=0.46.1
accelerate
qwen-vl-utils
Pillow
```

These packages provide the model classes, 4-bit GPU quantization, device
placement, image/video preparation, and image decoding.

The launcher also checks that Pillow can import its image modules. If the
runtime has an inconsistent Pillow installation, it force-reinstalls Pillow
and asks for a runtime restart if the repair does not resolve the problem.

### Step 2 — Mount Google Drive

The launcher mounts:

```text
/content/drive
```

The persistent model location is expected to be:

```text
/content/drive/MyDrive/Hapus More AI/models
```

Drive is persistent storage. The Colab runtime is temporary and can disappear
when the session expires or is restarted.

### Step 3 — Download and validate the gateway source

The launcher downloads `hapus_more_agent.py` from the public helper repository.
If an existing `/content/hapus_more_agent.py` is present, it is reused only if
it contains the current gateway markers. This prevents an old uploaded file
from silently restoring a previous bug.

The source is loaded as a Python module and must contain `class HapusGateway`.

### Step 4 — Check the GPU

The launcher requires `torch.cuda.is_available()` to be true. The intended
runtime is:

```text
Google Colab → Runtime → Change runtime type → T4 GPU
```

If no GPU is available, startup stops before copying or loading the model.

### Step 5 — Find and validate the model in Drive

The gateway searches below the Drive `models` folder for:

```text
config.json
model.safetensors.index.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
```

The exact shard names come from `model.safetensors.index.json`; the code checks
every filename listed by that index. This matters because finding only the
first shard is not enough to load the model.

The processor files are separate from the large model weights. Drive may omit
them because the loader can retrieve the small processor bundle from the
configured Hugging Face model ID.

### Step 6 — Copy the model to temporary Colab storage

The complete model directory is copied from Drive to:

```text
/content/hapus-model
```

Drive is slower for repeated weight reads. The temporary local copy gives the
model loader faster access during this runtime. If a previous failed attempt
left an incomplete copy, the launcher checks the required files and fills in
the missing files.

### Step 7 — Load the processor and Qwen model

The default model is:

```text
Qwen/Qwen3-VL-4B-Instruct
```

The loader creates a `BitsAndBytesConfig` with 4-bit NF4 quantization and
float16 compute. It then:

1. Loads `AutoProcessor` from the local model directory when
   `preprocessor_config.json` exists.
2. Otherwise loads the processor from `HAPUS_PROCESSOR_ID`, which defaults to
   `Qwen/Qwen3-VL-4B-Instruct`.
3. Loads `AutoModelForMultimodalLM` from the local `/content/hapus-model` copy.
4. Uses `device_map="auto"` and automatic model dtype selection.

The processor handles the chat template, tokenizer, image preprocessing, and
model input tensors. Qwen3-VL handles the multimodal response generation.

### Step 8 — Start the local authenticated HTTP gateway

The launcher generates a fresh random token for the runtime and starts the
Python gateway on:

```text
http://127.0.0.1:8000
```

It immediately checks:

```text
GET http://127.0.0.1:8000/health
```

At this point the model is loaded and the local service is ready, but it is not
yet reachable from the hosted web application.

### Step 9 — Create the public tunnel

The launcher downloads the Cloudflare `cloudflared` binary to:

```text
/content/cloudflared
```

It starts a Quick Tunnel from the public internet to local port `8000`:

```text
public HTTPS URL → Cloudflare → 127.0.0.1:8000
```

The tunnel URL normally looks like:

```text
https://<random-name>.trycloudflare.com
```

Cloudflare may publish the URL before its DNS record is immediately
resolvable. The launcher waits up to 60 seconds and checks:

```text
https://<random-name>.trycloudflare.com/health
```

Only after that check succeeds does the launcher print the connection values.

## 4. What the final output means

The launcher prints:

```text
HAPUS_AI_SERVICE_URL=https://<tunnel>.trycloudflare.com/v1/analyze
HAPUS_AI_SERVICE_TOKEN=<fresh runtime token>
HAPUS_AI_SERVICE_TIMEOUT_MS=45000
```

These values belong in the server environment of the Hapus & More web
application, such as Render or Vercel:

- `HAPUS_AI_SERVICE_URL` tells the web server where to send AI requests.
- `HAPUS_AI_SERVICE_TOKEN` authenticates those requests to the Colab gateway.
- `HAPUS_AI_SERVICE_TIMEOUT_MS` limits how long the web server waits.

The token and URL are temporary. Never put them in browser code, source
control, screenshots, or a public notebook.

## 5. Gateway files and responsibilities

### `launch_gateway.py`

This is the startup orchestrator. It is responsible for:

- package installation and Pillow repair;
- Google Drive mounting;
- downloading and validating the gateway module;
- GPU availability checks;
- model discovery and loading coordination;
- starting the local gateway;
- downloading and starting Cloudflare Tunnel;
- local and public `/health` checks;
- printing the three server environment values.

### `hapus_more_agent.py`

This is the model and API implementation. It is responsible for:

- finding the complete model in Drive;
- copying model files to temporary Colab storage;
- loading the processor and Qwen3-VL model;
- building the evidence-first agriculture prompt;
- validating image URLs and base64 images;
- running image/text generation;
- serving `/health` and `/v1/analyze`;
- requiring the runtime Bearer token;
- returning the answer, model, confidence, disclaimer, and references.

## 6. Request flow after startup

```text
1. User submits a question/photo in the web application.
2. Next.js checks the signed-in user's session and tree access.
3. Next.js builds orchard and agriculture context.
4. Next.js sends POST /v1/analyze through the tunnel.
5. Gateway validates the Bearer token.
6. Gateway downloads or decodes the image and verifies it with Pillow.
7. qwen-vl-utils prepares image inputs.
8. Qwen3-VL generates an evidence-first response.
9. Gateway returns JSON to Next.js.
10. Next.js returns the result to the browser.
```

If the hosted gateway is not configured, unreachable, times out, or returns
invalid data, the web application uses its transparent local intelligence
fallback. The application therefore does not depend on Colab for basic demo
operation.

## 7. Gateway API

### Health check

```text
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "gateway": "colab"
}
```

### Analysis request

```text
POST /v1/analyze
Authorization: Bearer <runtime-token>
Content-Type: application/json
```

Example payload:

```json
{
  "task": "farm_advisor",
  "question": "Describe visible evidence and the next observation to record.",
  "crop": "mango",
  "image_url": "https://example.com/leaf.jpg",
  "context": {
    "orchard": {},
    "agriculture": {}
  }
}
```

The gateway accepts either `image_url` or `image_base64`. Images are limited to
8 MB and are checked as images before being sent to the model.

## 8. Troubleshooting map

| Output or error | Meaning | Action |
| --- | --- | --- |
| `Pillow is still inconsistent` | Colab's image package state is broken | Restart the runtime and run the launcher again |
| `No GPU is available` | The notebook is not attached to a GPU | Select a T4 runtime and reconnect |
| `No complete model folder found` | Configuration, index, or a required shard is missing | Check both safetensors shards and the index in Drive |
| `Can't load image processor ... preprocessor_config.json` | An old gateway tried to load processor files only from Drive | Refresh the launcher; current code falls back to Hugging Face processor metadata |
| `Local gateway health check` failure | Python gateway did not start correctly | Inspect the preceding model or Python error |
| `Public health check` DNS failure | Tunnel URL was not resolvable yet or Cloudflare exited | Allow the 60-second retry; if it fails, inspect the printed Cloudflared log tail |
| Hosted app says AI is unavailable | URL/token expired or server cannot reach Colab | Copy the newest URL and token into the hosting environment |

## 9. Runtime lifecycle

```text
Fresh Colab runtime
      ↓
Run launcher cell
      ↓
Drive mounted + model loaded + gateway exposed
      ↓
Copy URL/token to Render or Vercel
      ↓
Use gateway for the current session
      ↓
Runtime expires or restarts
      ↓
Old model process, URL, and token are gone
      ↓
Run the launcher again and update hosted environment values
```

The gateway is suitable for prototype demonstrations and evaluation. It is not
a production inference service because Colab runtimes and Quick Tunnel URLs
are temporary.

## 10. Related documents

- [Web application architecture](https://github.com/HarshadHindlekar/hapus-and-more/blob/main/docs/web-application-architecture.md) — the
  Next.js product, database, authentication, APIs, and deployment.
- [AI system architecture](https://github.com/HarshadHindlekar/hapus-and-more/blob/main/docs/ai-system-architecture.md) — AI principles,
  retrieval, guardrails, and future managed inference.
- [Colab gateway guide](README.md) — application connection contract.
- [Colab restart runbook](README.md) — operational restart steps.
- [AI journey](https://github.com/HarshadHindlekar/hapus-and-more/blob/main/docs/ai-journey.md) — product and AI development record.


