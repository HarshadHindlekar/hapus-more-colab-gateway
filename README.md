# Hapus & More Colab gateway

This public helper repository contains the two files needed to start the
temporary Google Colab AI gateway. It contains no model weights, credentials,
or application secrets.

## One-cell start

Run this cell in a fresh Google Colab runtime with a T4 GPU:

```python
!wget -q https://raw.githubusercontent.com/HarshadHindlekar/hapus-more-colab-gateway/main/launch_gateway.py -O /content/launch_gateway.py
exec(open("/content/launch_gateway.py", encoding="utf-8").read(), globals())
```

The launcher installs dependencies, mounts Google Drive, loads the model from
`MyDrive/Hapus More AI/models`, starts the authenticated gateway, opens a
Cloudflare Quick Tunnel, checks `/health`, and prints the server-side values
for Vercel or Render.

The tunnel URL and token change whenever the Colab runtime restarts. This is a
prototype demonstration bridge, not a production endpoint.

## Architecture

See [colab-gateway-architecture.md](colab-gateway-architecture.md) for a
step-by-step explanation of the one-cell launcher, Drive model loading, local
gateway, Cloudflare tunnel, request flow, output values, and troubleshooting.
