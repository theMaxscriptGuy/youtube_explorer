from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from youtube_explorer_backend import run_youtube_explorer_query


HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8000"))
WEB_ROOT = Path(__file__).resolve().parent / "web"


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:
        if self.path != "/api/query":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
        except (ValueError, json.JSONDecodeError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Invalid JSON body."},
            )
            return

        api_key = str(payload.get("apiKey", "")).strip()
        query = str(payload.get("query", "")).strip()

        if not api_key:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "OpenAI API key is required."},
            )
            return

        if not query:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "A query is required."},
            )
            return

        try:
            result = run_youtube_explorer_query(query=query, api_key=api_key)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": str(exc)},
            )
            return

        self._send_json(HTTPStatus.OK, result)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"YouTube Explorer is running at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
