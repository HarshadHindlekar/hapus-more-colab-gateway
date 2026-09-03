"""Launch the Hapus & More AI gateway in one Google Colab cell.

This launcher is intentionally self-contained. It installs the runtime packages,
mounts Drive, downloads the gateway module from the repository, loads the model,
starts the authenticated HTTP gateway, and exposes it through a Cloudflare Quick
Tunnel. It is designed for a fresh free-Colab runtime.

Run from Colab with:

    !wget -q https://raw.githubusercontent.com/HarshadHindlekar/hapus-more-colab-gateway/main/launch_gateway.py -O /content/launch_gateway.py
    exec(open("/content/launch_gateway.py", encoding="utf-8").read())

The repository version must be committed and pushed before using the one-cell
launcher. Set HAPUS_AGENT_URL before executing if a different gateway source is
needed.
"""

from __future__ import annotations

import importlib.util
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen


AGENT_URL = os.environ.get(
    "HAPUS_AGENT_URL",
    "https://raw.githubusercontent.com/HarshadHindlekar/hapus-more-colab-gateway/main/hapus_more_agent.py",
)
AGENT_PATH = Path("/content/hapus_more_agent.py")
CLOUDFLARED_PATH = Path("/content/cloudflared")
CLOUDFLARED_LOG = Path("/content/hapus-tunnel.log")
GATEWAY_PORT = 8000


def _run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.check_call(command)


def _download(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "HapusMore-Colab-Launcher/1.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def _install_dependencies() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "-U",
            "transformers>=4.57.0",
            "bitsandbytes>=0.46.1",
            "accelerate",
            "pillow",
            "qwen-vl-utils",
        ]
    )


def _mount_drive() -> None:
    from google.colab import drive

    drive_root = Path("/content/drive/MyDrive")
    if not drive_root.exists():
        drive.mount("/content/drive")
    else:
        print("Google Drive is already mounted")


def _load_gateway_module():
    if AGENT_PATH.exists() and os.environ.get("HAPUS_FORCE_AGENT_DOWNLOAD") != "1":
        print("Using the existing local gateway file:", AGENT_PATH)
    else:
        _download(AGENT_URL, AGENT_PATH)
        print("Downloaded gateway source from:", AGENT_URL)
    source = AGENT_PATH.read_text(encoding="utf-8")
    if "class HapusGateway" not in source:
        raise RuntimeError(
            "The downloaded gateway file does not contain HapusGateway. "
            f"Check HAPUS_AGENT_URL: {AGENT_URL}"
        )

    spec = importlib.util.spec_from_file_location("hapus_more_agent", AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the downloaded gateway module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("Gateway module loaded:", AGENT_PATH)
    return module


def _load_model(hapus_module):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU is available. In Colab choose Runtime > Change runtime type > T4 GPU, "
            "then reconnect and run this cell again."
        )

    model_path = hapus_module.copy_model_to_runtime()
    processor, model = hapus_module.load_model(model_path)
    print("Model loaded successfully:", model_path)
    return processor, model


def _start_gateway(hapus_module, processor, model):
    token = secrets.token_urlsafe(32)
    os.environ["HAPUS_COLAB_GATEWAY_TOKEN"] = token
    gateway = hapus_module.HapusGateway(processor, model, token=token)
    gateway.start_background(port=GATEWAY_PORT)
    time.sleep(2)

    with urlopen(f"http://127.0.0.1:{GATEWAY_PORT}/health", timeout=10) as response:
        if response.status != 200:
            raise RuntimeError(f"Local gateway health check returned HTTP {response.status}")
    print("Local gateway is healthy")
    return gateway, token


def _start_cloudflare_tunnel() -> str:
    if not CLOUDFLARED_PATH.exists():
        print("Downloading cloudflared")
        _download(
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            CLOUDFLARED_PATH,
        )
        CLOUDFLARED_PATH.chmod(
            CLOUDFLARED_PATH.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        )

    log_handle = CLOUDFLARED_LOG.open("w", encoding="utf-8")
    subprocess.Popen(
        [
            str(CLOUDFLARED_PATH),
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://127.0.0.1:{GATEWAY_PORT}",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    tunnel_pattern = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
    for _ in range(30):
        time.sleep(1)
        log_text = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")
        matches = tunnel_pattern.findall(log_text)
        if matches:
            return matches[-1]

    log_text = CLOUDFLARED_LOG.read_text(encoding="utf-8", errors="ignore")
    raise RuntimeError(
        "Cloudflare Quick Tunnel did not start within 30 seconds. "
        f"See {CLOUDFLARED_LOG}. Last output:\n{log_text[-2000:]}"
    )


def _check_public_health(tunnel_url: str) -> None:
    health_url = f"{tunnel_url}/health"
    last_error = None
    for _ in range(5):
        try:
            with urlopen(health_url, timeout=15) as response:
                if response.status == 200:
                    print("Public gateway is healthy:", health_url)
                    return
        except Exception as error:  # tunnel startup can take a few seconds
            last_error = error
        time.sleep(2)
    raise RuntimeError(f"Public health check failed: {last_error}")


def main() -> None:
    print("=== Hapus & More Colab gateway launcher ===")
    _install_dependencies()
    _mount_drive()
    hapus_module = _load_gateway_module()
    processor, model = _load_model(hapus_module)
    gateway, token = _start_gateway(hapus_module, processor, model)
    tunnel_url = _start_cloudflare_tunnel()
    _check_public_health(tunnel_url)

    # These names remain available in the notebook when this file is executed
    # with exec(...), which makes troubleshooting easier after a successful run.
    globals().update(
        {
            "hapus_module": hapus_module,
            "processor": processor,
            "model": model,
            "gateway": gateway,
            "token": token,
            "tunnel_url": tunnel_url,
        }
    )

    print("\n=== Copy these server-side values to Render and Vercel ===")
    print(f"HAPUS_AI_SERVICE_URL={tunnel_url}/v1/analyze")
    print(f"HAPUS_AI_SERVICE_TOKEN={token}")
    print("HAPUS_AI_SERVICE_TIMEOUT_MS=45000")
    print("\nDo not share the token or this output in screenshots. It expires with this runtime.")


if __name__ == "__main__":
    main()
