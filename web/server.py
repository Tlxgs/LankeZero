"""Local static server for the web demo.

Python's stdlib http.server does not map .mjs to a JavaScript MIME type, and
module workers / ONNX Runtime loaders are rejected when served as text/plain.
This handler adds the mappings the demo needs, then serves the repository root.
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import os


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".mjs": "text/javascript; charset=utf-8",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
        ".json": "application/json; charset=utf-8",
        ".svg": "image/svg+xml",
    }

    def log_message(self, fmt, *args):
        print("[web] " + (fmt % args))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(f"[web] http://{args.bind}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
