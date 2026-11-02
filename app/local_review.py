"""Dedicated Windows/local persistence launcher; never used by the Space entry point."""

import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from starlette.middleware import Middleware
from invoice_review_store import DraftStore, StoreError, require_local_environment


class LocalBoundary:
    def __init__(self, store=None):
        require_local_environment()
        self.store = store if store is not None else DraftStore()
        self.demo = None
        self.active = False

    def authorize(self):
        self.validate_launch()
        if not self.active:
            raise StoreError("Local persistence storage is not ready.")

    def validate_launch(self):
        require_local_environment()
        demo = self.demo
        if (demo is None or demo.server_name != "127.0.0.1"
                or demo.share is not False or demo.share_url
                or not demo.server.started or demo.server.config.host != "127.0.0.1"
                or str(self.store.directory) not in demo.blocked_paths):
            raise StoreError("Persistence requires the dedicated active loopback-only launcher.")

    def request_allowed(self, scope):
        try:
            require_local_environment()
        except StoreError:
            return False
        headers = {key.decode("latin1").lower(): value.decode("latin1")
                   for key, value in scope.get("headers", [])}
        if any(key == "forwarded" or key.startswith("x-forwarded-")
               or key in {"x-real-ip", "x-original-host"} for key in headers):
            return False
        client = scope.get("client")
        if not client or client[0] != "127.0.0.1":
            return False
        try:
            host = urlsplit("http://" + headers.get("host", ""))
            port = host.port
        except ValueError:
            return False
        if host.hostname != "127.0.0.1" or host.netloc != headers.get("host"):
            return False
        if self.demo is not None and self.demo.server_port and port != self.demo.server_port:
            return False
        origin = headers.get("origin")
        if origin and origin != "http://" + headers["host"]:
            return False
        if headers.get("sec-fetch-site") == "cross-site":
            return False
        return scope.get("scheme") in {"http", "ws"}


class LocalOnlyMiddleware:
    def __init__(self, app, boundary):
        self.app, self.boundary = app, boundary

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        if not self.boundary.request_allowed(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Local access only."})
            return
        await self.app(scope, receive, send)


def launch_local(*, port=7860, store=None, configure=None):
    """Optional server-side fixture hook is for tests, not a browser/API parameter."""
    require_local_environment()
    from app import main

    if configure is not None:
        configure(main)
    boundary = LocalBoundary(store)
    cache = Path(main.gr.utils.get_upload_folder()).resolve()
    storage = boundary.store.directory.resolve()
    if storage.is_relative_to(cache) or cache.is_relative_to(storage):
        raise StoreError("Review storage must be outside Gradio's cache.")
    if os.name != "nt":
        raise StoreError("This dedicated launcher is approved only for local Windows development.")
    demo = main.build_app(persistence=boundary)
    boundary.demo = demo
    try:
        demo.launch(
            server_name="127.0.0.1", server_port=port, share=False, inbrowser=False,
            prevent_thread_lock=True, max_file_size="20mb", strict_cors=True,
            blocked_paths=[str(boundary.store.directory)], allowed_paths=[],
            app_kwargs={"middleware": [Middleware(LocalOnlyMiddleware, boundary=boundary)]},
        )
        # Do not create the database until the actual server configuration has been checked.
        boundary.validate_launch()
        boundary.store.prepare()
        boundary.active = True
        return demo, boundary
    except Exception:
        boundary.active = False
        demo.close()
        raise


if __name__ == "__main__":
    import threading

    demo, boundary = launch_local(port=int(os.environ.get("PORT", "7860")))
    print(f"LOCAL_REVIEW_READY {demo.local_url}", flush=True)
    try:
        threading.Event().wait()
    finally:
        boundary.active = False
        demo.close()
