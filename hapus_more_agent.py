"""Hapus & More Colab runner.

Run this file from a Google Colab notebook after mounting the shared Drive.
It loads the selected open-weight vision-language model into the temporary
Colab runtime, runs one farm-image analysis, and saves the result to Drive.
"""

from __future__ import annotations

import json
import os
import shutil
import base64
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


# In Colab, run these once in a notebook cell before executing this file:
# !pip install -U "transformers>=4.57.0" accelerate bitsandbytes pillow qwen-vl-utils
# from google.colab import drive
# drive.mount('/content/drive')

MODEL_ID = os.environ.get("HAPUS_MODEL_ID", "Qwen/Qwen3-VL-4B-Instruct")
DRIVE_ROOT = Path(os.environ.get("HAPUS_DRIVE_ROOT", "/content/drive/MyDrive/Hapus More AI"))
DRIVE_MODELS_DIR = DRIVE_ROOT / "models"
RUNTIME_MODEL_DIR = Path("/content/hapus-model")
OUTPUT_DIR = DRIVE_ROOT / "outputs"

AGRICULTURE_KNOWLEDGE = [
    {
        "id": "visual-triage",
        "keywords": {"leaf", "spot", "hole", "yellow", "curl", "wilt", "fruit", "disease", "pest", "image"},
        "summary": "A single image shows evidence, but similar symptoms can come from disease, pests, nutrition, weather, water stress, or mechanical damage.",
        "actions": "Describe visible evidence first; request both leaf surfaces, a whole-plant view, crop stage, location, weather, irrigation, and treatment history.",
    },
    {
        "id": "mango-orchard-ipm",
        "keywords": {"mango", "pruning", "sanitation", "surveillance", "pest", "orchard", "branch"},
        "summary": "ICAR guidance highlights orchard sanitation, timely pruning, pest surveillance, removal of affected parts, and integrated approaches that reduce unnecessary pesticide dependence.",
        "actions": "Inspect nearby trees and record whether the symptom is spreading before changing the care plan.",
    },
    {
        "id": "mango-disease-scope",
        "keywords": {"mango", "anthracnose", "blight", "mildew", "dieback", "fruit", "hopper", "mealybug", "borer"},
        "summary": "ICAR-IIHR materials describe anthracnose, blossom or leaf blight, powdery mildew, dieback, fruit fly, hopper, stone weevil, mealybug, and shoot or stem borers as mango assessment areas.",
        "actions": "Use this as a candidate list for the next observation, not as a diagnosis from one image.",
    },
    {
        "id": "soil-test-nutrition",
        "keywords": {"soil", "nutrient", "fertilizer", "nutrition", "yellow", "irrigation"},
        "summary": "ICAR orchard guidance recommends soil-test-based nutrient management and context-aware input decisions.",
        "actions": "Ask for soil-test results, tree age, crop stage, irrigation, and recent fertilizer history before suggesting a nutrient change.",
    },
]


def retrieve_agriculture_knowledge(question: str, crop: str = "mango", limit: int = 3):
    query = set(" ".join((crop, question).lower().split()))
    scored = []
    for entry in AGRICULTURE_KNOWLEDGE:
        score = len(query.intersection(entry["keywords"]))
        scored.append((score, entry))
    return [entry for _, entry in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]


def build_agriculture_prompt(question: str, crop: str = "mango"):
    references = retrieve_agriculture_knowledge(question, crop)
    reference_text = "\n".join(
        f"- {entry['summary']} Next step: {entry['actions']}"
        for entry in references
    )
    return f"""You are Hapus & More's cautious farm-observation assistant.
Analyze the image as evidence, not as a confirmed diagnosis.
Return these headings: Observation, Possible causes, Confidence, Next observations, Safety note.
Separate what is visible from what is only possible. Ask for missing crop, variety, location,
plant age, growth stage, weather, irrigation, soil, and treatment history.
Do not prescribe pesticide or fertilizer products or dosages from one image.
Escalate low-confidence, severe, rapidly spreading, or economically important cases to a farm
manager or agronomist.

Crop: {crop}
Reference guidance:
{reference_text}

User request: {question}"""


def find_drive_model_dir() -> Path:
    """Find a model directory even when a shared Drive shortcut has a custom name."""
    candidates = []
    if DRIVE_MODELS_DIR.exists():
        for config in DRIVE_MODELS_DIR.rglob("config.json"):
            candidate = config.parent
            if (candidate / "model.safetensors.index.json").exists() and list(candidate.glob("*.safetensors")):
                candidates.append(candidate)
    unique_candidates = list(dict.fromkeys(candidates))
    if not unique_candidates:
        raise FileNotFoundError(
            f"No model folder found under {DRIVE_MODELS_DIR}. Add the sister-owned Drive shortcut and mount Drive first."
        )
    return unique_candidates[0]


def copy_model_to_runtime() -> Path:
    """Copy model weights from persistent Drive to faster ephemeral storage."""
    RUNTIME_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if RUNTIME_MODEL_DIR.exists():
        return RUNTIME_MODEL_DIR
    shutil.copytree(find_drive_model_dir(), RUNTIME_MODEL_DIR)
    return RUNTIME_MODEL_DIR


def load_model(model_path: Path):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype="auto",
        quantization_config=quant_config,
    )
    return processor, model


def analyze_image(processor, model, image_path: str, question: str) -> str:
    from qwen_vl_utils import process_vision_info

    content = [{"type": "text", "text": question}]
    if image_path:
        content.insert(0, {"type": "image", "image": image_path})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    processor_inputs = {
        "text": [prompt],
        "padding": True,
        "return_tensors": "pt",
    }
    if image_inputs:
        processor_inputs["images"] = image_inputs
    if video_inputs:
        processor_inputs["videos"] = video_inputs
    inputs = processor(**processor_inputs).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=300)
    trimmed_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def build_gateway_prompt(question: str, context: dict, crop: str = "mango") -> str:
    agriculture = context.get("agriculture", {}) if isinstance(context, dict) else {}
    references = agriculture.get("references", []) if isinstance(agriculture, dict) else []
    reference_text = "\n".join(
        f"- {item.get('title', 'Approved agriculture reference')}: {item.get('summary', '')}"
        for item in references
        if isinstance(item, dict)
    )
    return f"""You are Hapus & More's cautious farm-observation assistant.
Analyze the image as evidence, not as a confirmed diagnosis.
Return concise headings: Observation, Possible causes, Confidence, Next observations, Safety note.
Separate what is visible from what is only possible. Do not prescribe pesticide or fertilizer
products or dosages from one image. Ask for missing crop, variety, location, plant age, growth
stage, weather, irrigation, soil, and treatment history. Escalate low-confidence, severe,
rapidly spreading, or economically important cases to a farm manager or agronomist.

Crop: {crop}
Approved context:
{reference_text}

Application context:
{json.dumps(context, ensure_ascii=False)[:12000]}

User request: {question}"""


def _download_image(image_url: str) -> Path:
    parsed = urlparse(image_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("image_url must use http or https")
    request = Request(image_url, headers={"User-Agent": "HapusMore-Colab-Gateway/1.0"})
    with urlopen(request, timeout=20) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("image is larger than the 8 MB gateway limit")
    from PIL import Image
    with Image.open(BytesIO(data)) as image:
        image.verify()
    handle = tempfile.NamedTemporaryFile(prefix="hapus-image-", suffix=".jpg", delete=False)
    handle.write(data)
    handle.close()
    return Path(handle.name)


def _decode_image(image_base64: str) -> Path:
    encoded = image_base64.split(",", 1)[-1]
    data = base64.b64decode(encoded, validate=True)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("image is larger than the 8 MB gateway limit")
    from PIL import Image
    with Image.open(BytesIO(data)) as image:
        image.verify()
    handle = tempfile.NamedTemporaryFile(prefix="hapus-image-", suffix=".jpg", delete=False)
    handle.write(data)
    handle.close()
    return Path(handle.name)


class _GatewayHandler(BaseHTTPRequestHandler):
    gateway = None

    def _write_json(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Hapus-Client")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._write_json(200, {"status": "ok", "model": MODEL_ID, "gateway": "colab"})
            return
        self._write_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/analyze":
            self._write_json(404, {"error": "Not found"})
            return
        expected_token = self.gateway.token
        provided_token = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if expected_token and provided_token != expected_token:
            self._write_json(401, {"error": "Unauthorized"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 256 * 1024:
                raise ValueError("request is too large")
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            question = str(payload.get("question", "")).strip()
            if not question:
                raise ValueError("question is required")
            context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
            crop = str(payload.get("crop", "mango"))
            prompt = build_gateway_prompt(question, context, crop)
            image_path = None
            if payload.get("image_url"):
                image_path = _download_image(str(payload["image_url"]))
            elif payload.get("image_base64"):
                image_path = _decode_image(str(payload["image_base64"]))
            if image_path:
                answer = analyze_image(self.gateway.processor, self.gateway.model, str(image_path), prompt)
                image_path.unlink(missing_ok=True)
            else:
                answer = analyze_image(self.gateway.processor, self.gateway.model, "", prompt)
            self._write_json(200, {
                "answer": answer,
                "model": MODEL_ID,
                "mode": "colab-qwen3-vl",
                "confidence": "Early",
                "disclaimer": "This prototype supports farm decisions; it does not replace an agronomist or a field inspection.",
                "references": [item.get("source", {}) for item in context.get("agriculture", {}).get("references", []) if isinstance(item, dict)],
            })
        except Exception as error:
            self._write_json(400, {"error": str(error)})

    def log_message(self, format, *args):
        return


class HapusGateway:
    def __init__(self, processor, model, token: str | None = None):
        self.processor = processor
        self.model = model
        self.token = token or os.environ.get("HAPUS_COLAB_GATEWAY_TOKEN", "").strip()
        if not self.token:
            raise ValueError("Set HAPUS_COLAB_GATEWAY_TOKEN before exposing the gateway")

    def serve(self, port: int = 8000):
        handler = type("HapusGatewayHandler", (_GatewayHandler,), {})
        handler.gateway = self
        server = ThreadingHTTPServer(("0.0.0.0", port), handler)
        print(f"Hapus gateway listening on http://127.0.0.1:{port}")
        server.serve_forever()

    def start_background(self, port: int = 8000):
        thread = threading.Thread(target=self.serve, args=(port,), daemon=True)
        thread.start()
        return thread


def save_result(image_path: str, question: str, answer: str, references) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "latest_observation.json"
    result_path.write_text(json.dumps({
        "model": MODEL_ID,
        "image": image_path,
        "question": question,
        "answer": answer,
        "references": references,
        "safety": {
            "diagnosis": "not_confirmed",
            "disclaimer": "Image analysis supports field decisions; it does not replace an agronomist or field inspection.",
        },
    }, indent=2), encoding="utf-8")
    return result_path


if __name__ == "__main__":
    image_path = os.environ.get("HAPUS_IMAGE_PATH")
    if not image_path:
        raise SystemExit("Set HAPUS_IMAGE_PATH to a farm image before running the analysis.")

    question = os.environ.get(
        "HAPUS_QUESTION",
        "Describe visible evidence in this mango image, possible causes, confidence, and the next observation a farmer should record. Do not diagnose.",
    )
    prompt = build_agriculture_prompt(question)
    references = retrieve_agriculture_knowledge(question)
    model_path = copy_model_to_runtime()
    processor, model = load_model(model_path)
    answer = analyze_image(processor, model, image_path, prompt)
    result_path = save_result(image_path, question, answer, references)
    print(answer)
    print(f"Saved result to {result_path}")
