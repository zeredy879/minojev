"""Local HTTP decision API for a loaded checkpoint.

Endpoints:

- ``GET /`` or ``GET /health``: model metadata.
- ``POST /score``: body is one request object or ``{"requests": [...], "mode": "fresh|reuse"}``;
  returns the scored records.

The server uses only the standard library, binds to ``127.0.0.1`` by default,
and never decodes output tokens.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .model import DecisionModel, ScoreOptions
from .types import request_from_object, request_to_object


class DecisionService:
    def __init__(self, model: DecisionModel, device: str = "auto") -> None:
        self.model = model
        self.device = device
        self.info = {
            "model": "minojev",
            "checkpoint": model.config.get("backbone", {}).get("source", "local"),
            "objective": model.config.get("objective"),
            "calibration": model.calibration.to_dict(),
            "decode_steps": 0,
        }

    def score_payload(self, payload: Any) -> dict:
        mode = "fresh"
        if isinstance(payload, dict) and "requests" in payload:
            mode = str(payload.get("mode", "fresh"))
            rows = payload["requests"]
            if not isinstance(rows, list) or not rows:
                raise ValueError("requests must be a nonempty list")
        else:
            rows = [payload]
        requests = [request_from_object(row, index) for index, row in enumerate(rows)]
        records = self.model.score(requests, ScoreOptions(mode=mode, device=self.device))
        return {"mode": mode, "records": records, "count": len(records)}


def make_handler(service: DecisionService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "minojev/1.0"

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib handler API
            if self.path in ("/", "/health"):
                self._send(200, {**service.info, "status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib handler API
            if self.path != "/score":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                self._send(200, service.score_payload(payload))
            except (ValueError, KeyError) as error:
                self._send(400, {"error": str(error)})
            except Exception as error:  # pragma: no cover - defensive
                self._send(500, {"error": f"internal error: {error}"})

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            return

    return Handler


def build_server(checkpoint: str, host: str = "127.0.0.1", port: int = 8000, device: str = "auto") -> ThreadingHTTPServer:
    model = DecisionModel.load(checkpoint, device=device)
    service = DecisionService(model, device=device)
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.minojev_info = service.info  # type: ignore[attr-defined]
    return server


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8000, device: str = "auto") -> None:
    server = build_server(checkpoint, host=host, port=port, device=device)
    info = getattr(server, "minojev_info", {})
    print(json.dumps({"serving": checkpoint, "host": host, "port": port, **info}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def score_once(checkpoint: str, payload: dict, device: str = "auto") -> dict:
    """In-process helper used by tests and scripts."""
    model = DecisionModel.load(checkpoint, device=device)
    return DecisionService(model, device=device).score_payload(payload)
