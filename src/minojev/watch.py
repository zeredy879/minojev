"""Live training dashboard.

Serves the monitor page plus JSON endpoints backed by a run directory:

- ``GET /``                 dashboard (auto-refreshes every 3 s)
- ``GET /api/status``       current status.json
- ``GET /api/metrics``      metrics.jsonl (last ``limit`` entries)
- ``GET /api/train_log``    train_log.json when present
- ``GET /api/summary``      summary.json when present

Run it in a second terminal while training:

    minojev watch --run runs/general-head --port 8010
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MONITOR_PAGE = Path(__file__).with_name("monitor_page.html")


class RunStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def read_json(self, name: str):
        path = self.run_dir / name
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def status(self):
        return self.read_json("status.json")

    def summary(self):
        return self.read_json("summary.json")

    def train_log(self):
        return self.read_json("train_log.json") or []

    def metrics(self, limit: int = 2000):
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows[-limit:]


def make_watch_handler(store: RunStore, page: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "minojev-watch/1.0"

        def _json(self, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path in ("/", "/index.html"):
                body = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/status":
                self._json(store.status())
            elif parsed.path == "/api/summary":
                self._json(store.summary())
            elif parsed.path == "/api/train_log":
                self._json(store.train_log())
            elif parsed.path == "/api/metrics":
                limit = int(query.get("limit", ["2000"])[0])
                self._json(store.metrics(limit=limit))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):  # noqa: A002 - stdlib signature
            return

    return Handler


def build_watch_server(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8010) -> ThreadingHTTPServer:
    store = RunStore(run_dir)
    handler = make_watch_handler(store, MONITOR_PAGE)
    return ThreadingHTTPServer((host, port), handler)


def watch(run_dir: str | Path, host: str = "127.0.0.1", port: int = 8010) -> None:
    server = build_watch_server(run_dir, host=host, port=port)
    print(json.dumps({"watching": str(run_dir), "url": f"http://{host}:{port}/"}))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
