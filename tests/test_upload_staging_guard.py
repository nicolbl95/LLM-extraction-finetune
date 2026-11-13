"""PR143 §2: an upload that only the server can see is still refused before it is stored.

`checked_path` guards the read side, but it runs on a path Gradio has already written: the vendored
multipart parser opens its scratch file directly inside the served upload folder
(`route_utils.py:811`) and renames it when the request ends. A document refused afterwards stays
fetchable by `/gradio_api/file=` for the cache lifetime, which is exactly what PR140 and PR141
measured. `SyntheticUploadStagingGuard` is the missing pre-write half of the same rule, and it sits
in front of the handler, so a client that posts to the staging route without a browser meets it too.

The integrated examples are untouched because they never use this route: selecting "France · EUR"
is a `run/predict` call whose answer carries the cached path, and extraction submits that path as
JSON. The guard therefore blocks by content digest and passes anything allowlisted straight through.
"""

import asyncio
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from app import main

ROOT = Path(main.__file__).resolve().parents[1]
SPACE = "nicolbl95/llm-extraction-finetune"
GATE = "Cette démonstration n'accepte que ses trois factures d'exemple synthétiques."
DEAD_ROUTE = "Cette démonstration n'a ni micro ni caméra : cette route ne sert à rien ici."
BOUNDARY = "pr143boundary"


@contextmanager
def deployment(**values):
    """Environment exactly as given: a `None` value removes the variable instead of ignoring it."""
    with patch.dict(os.environ, {k: v for k, v in values.items() if v is not None}, clear=False):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
        yield


@pytest.fixture
def on_a_space(tmp_path):
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


@pytest.fixture
def locally(tmp_path):
    with deployment(SPACE_ID=None, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


def multipart(files=(), fields=()):
    """A body shaped like the one `@gradio/client` builds for the file widget."""
    out = bytearray()
    for name, value in fields:
        out += (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n").encode()
    for name, filename, payload in files:
        out += (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        out += payload + b"\r\n"
    out += f"--{BOUNDARY}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={BOUNDARY}"


def raw_multipart(dispositions, payload):
    """The same body with hand-written Content-Disposition lines."""
    out = bytearray()
    for disposition in dispositions:
        out += f"--{BOUNDARY}\r\n{disposition}\r\n\r\n".encode("latin-1")
        out += payload + b"\r\n"
    out += f"--{BOUNDARY}--\r\n".encode("latin-1")
    return bytes(out), f"multipart/form-data; boundary={BOUNDARY}"


class RecordedGradio:
    """Stand-in for the staging handler: it records what it was handed, which it would then write."""

    def __init__(self):
        self.calls = []

    async def __call__(self, scope, receive, send):
        chunks = []
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        self.calls.append({"method": scope["method"], "path": scope["path"],
                           "body": b"".join(chunks)})
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"[]"})


def ask(guard, body_chunks, *, method="POST", path="/gradio_api/upload", content_type=None):
    """Drive the middleware over raw ASGI, the way uvicorn would."""
    headers = [(b"host", b"space.internal")]
    if content_type:
        headers.append((b"content-type", content_type.encode("latin-1")))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
        "path": path, "raw_path": path.encode("latin-1"), "query_string": b"", "root_path": "",
        "scheme": "https", "headers": headers, "client": ("203.0.113.7", 51234),
    }
    messages = [{"type": "http.request", "body": chunk,
                 "more_body": index < len(body_chunks) - 1}
                for index, chunk in enumerate(body_chunks)]

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    seen = {}

    async def send(message):
        if message["type"] == "http.response.start":
            seen["status"] = message["status"]
        elif message["type"] == "http.response.body":
            seen["body"] = seen.get("body", b"") + message.get("body", b"")

    asyncio.run(guard(scope, receive, send))
    detail = json.loads(seen.get("body", b"{}").decode("utf-8")) if seen.get("body") else {}
    return seen.get("status"), detail


def guarded(on_a_space):
    backend = RecordedGradio()
    return main.SyntheticUploadStagingGuard(backend), backend


def test_a_direct_post_of_a_foreign_document_never_reaches_the_writer(on_a_space):
    guard, backend = guarded(on_a_space)
    body, content_type = multipart(files=[("files", "ma-facture.png",
                                           b"\x89PNG\r\n\x1a\n" + b"real invoice bytes")])
    status, detail = ask(guard, [body], content_type=content_type)
    assert status == 403
    assert detail["detail"] == GATE
    assert backend.calls == []
    assert list(on_a_space.iterdir()) == []


def test_the_alias_route_is_covered_too(on_a_space):
    """Workers serve `/upload` and `/gradio_api/upload` from one handler; both are staging."""
    guard, backend = guarded(on_a_space)
    body, content_type = multipart(files=[("files", "facture.pdf", b"%PDF-1.7 not shipped")])
    for path in ("/upload", "/gradio_api/upload"):
        status, _ = ask(guard, [body], path=path, content_type=content_type)
        assert status == 403, path
    assert backend.calls == []


def test_the_recording_route_is_closed_on_a_space(on_a_space):
    """`process_recording` is a second multipart writer, and no component of this app records:
    refusing it is the only way its bytes never reach the container's temporary directory."""
    guard, backend = guarded(on_a_space)
    body, content_type = multipart(files=[("video", "memoire.webm", b"\x1a\x45\xdf\xa3 real audio")])
    for path in ("/gradio_api/process_recording", "/process_recording"):
        status, detail = ask(guard, [body], path=path, content_type=content_type)
        assert status == 403, path
        assert detail["detail"] == DEAD_ROUTE
    assert backend.calls == []
    assert list(on_a_space.iterdir()) == []


def test_the_recording_route_stays_usable_in_local_development(locally):
    guard = main.SyntheticUploadStagingGuard(RecordedGradio())
    body, content_type = multipart(files=[("video", "test.webm", b"\x1a\x45\xdf\xa3 synthetic")])
    status, _ = ask(guard, [body], path="/gradio_api/process_recording", content_type=content_type)
    assert status == 200


@pytest.mark.parametrize("label", ["france", "uk", "us"])
def test_an_allowlisted_document_is_handed_to_the_writer_byte_for_byte(on_a_space, label):
    guard, backend = guarded(on_a_space)
    source = ROOT / f"app/examples/{label}.png"
    body, content_type = multipart(files=[("files", source.name, source.read_bytes())])
    status, _ = ask(guard, [body], content_type=content_type)
    assert status == 200
    assert backend.calls == [{"method": "POST", "path": "/gradio_api/upload", "body": body}]
    assert hashlib.sha256(backend.calls[0]["body"]).hexdigest() == hashlib.sha256(body).hexdigest()


def test_one_changed_byte_is_a_different_document(on_a_space):
    guard, backend = guarded(on_a_space)
    original = (ROOT / "app/examples/france.png").read_bytes()
    for mutated in (original + b"\x00", original[:-1],
                    original[:20] + bytes([original[20] ^ 0xFF]) + original[21:]):
        body, content_type = multipart(files=[("files", "france.png", mutated)])
        status, detail = ask(guard, [body], content_type=content_type)
        assert status == 403
        assert detail["detail"] == GATE
    assert backend.calls == []


def test_a_bundled_name_cannot_smuggle_another_document(on_a_space):
    """The rule is about content: a development layout wearing an allowed name is refused."""
    guard, backend = guarded(on_a_space)
    payload = (ROOT / "tests/fixtures/layout-joanna-binet.png").read_bytes()
    body, content_type = multipart(files=[("files", "france.png", payload)])
    status, _ = ask(guard, [body], content_type=content_type)
    assert status == 403
    assert backend.calls == []


def test_a_chunked_request_is_reassembled_before_the_decision(on_a_space):
    guard, backend = guarded(on_a_space)
    payload = (ROOT / "app/examples/uk.png").read_bytes()
    body, content_type = multipart(files=[("files", "uk.png", payload)])
    status, _ = ask(guard, [body[:900], body[900:]], content_type=content_type)
    assert status == 200
    assert backend.calls[0]["body"] == body


def test_a_body_larger_than_the_demo_accepts_is_turned_away_without_parsing(on_a_space):
    guard, backend = guarded(on_a_space)
    body, content_type = multipart(files=[("files", "huge.png", b"\x00" * 4096)])
    with patch.object(main, "MAX_BYTES", 512):
        status, detail = ask(guard, [body], content_type=content_type)
    assert status == 413
    assert "20 Mo" in detail["detail"]
    assert backend.calls == []


def test_an_unreadable_multipart_body_is_refused_not_forwarded(on_a_space):
    """Fail closed: bytes the guard cannot hash are bytes it cannot authorise."""
    guard, backend = guarded(on_a_space)
    status, detail = ask(guard, [b"------pr143boundary\r\nnot a part at all"],
                         content_type=f"multipart/form-data; boundary={BOUNDARY}")
    assert status == 400
    assert "multipart" in detail["detail"]
    assert backend.calls == []


def test_a_field_only_request_reaches_the_writer(on_a_space):
    """Nothing is stored for a request without a file part; the guard has no reason to decide."""
    guard, backend = guarded(on_a_space)
    body, content_type = multipart(fields=[("upload_id", "abc123")])
    status, _ = ask(guard, [body], content_type=content_type)
    assert status == 200
    assert backend.calls[0]["body"] == body


@pytest.mark.parametrize(("method", "path", "content_type"), [
    ("POST", "/gradio_api/queue/join", "application/json"),
    ("POST", "/gradio_api/run/predict", "application/json"),
    ("GET", "/gradio_api/upload", None),
    ("OPTIONS", "/gradio_api/upload", None),
    ("POST", "/gradio_api/upload", "text/plain"),
])
def test_everything_else_is_left_to_gradio(on_a_space, method, path, content_type):
    guard, backend = guarded(on_a_space)
    status, _ = ask(guard, [b"{}"], method=method, path=path, content_type=content_type)
    assert status == 200
    assert len(backend.calls) == 1


def test_local_development_keeps_free_staging(locally):
    guard, backend = guarded(locally)
    body, content_type = multipart(files=[("files", "ma-facture.png",
                                           b"\x89PNG\r\n\x1a\n" + b"my own invoice")])
    status, _ = ask(guard, [body], content_type=content_type)
    assert status == 200
    assert backend.calls[0]["body"] == body


def test_the_override_reopens_staging_on_a_space(on_a_space):
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD="1"):
        guard, backend = guarded(on_a_space)
        body, content_type = multipart(files=[("files", "ma-facture.png", b"\x89PNG\r\n\x1a\nx")])
        status, _ = ask(guard, [body], content_type=content_type)
    assert status == 200
    assert len(backend.calls) == 1


def test_the_allowlist_is_plain_content_digests_not_seeded_ones():
    """The guard hashes the request body, so it cannot reuse `processing_utils.hash_file`."""
    from gradio import processing_utils

    assert len(main.SYNTHETIC_UPLOAD_DIGESTS) == 3
    for _, path in main.REGIONAL_EXAMPLES:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in main.SYNTHETIC_UPLOAD_DIGESTS
        assert processing_utils.hash_file(path) in main.SYNTHETIC_DOCUMENT_DIGESTS
    for _, path in main.ADDITIONAL_EXAMPLES:
        assert hashlib.sha256(path.read_bytes()).hexdigest() not in main.SYNTHETIC_UPLOAD_DIGESTS


def test_the_part_digests_are_the_ones_gradio_would_store():
    """The guard must hash the same bytes the writer stores, or its answer is about another file."""
    payload = (ROOT / "app/examples/france.png").read_bytes()
    body, content_type = multipart(files=[("files", "france.png", payload)])
    assert main._file_part_digests(body, content_type) == [
        ("france.png", hashlib.sha256(payload).hexdigest())
    ]
    field_body, field_type = multipart(fields=[("upload_id", "abc123")])
    assert main._file_part_digests(field_body, field_type) == [(None, hashlib.sha256(b"abc123").hexdigest())]
    two, two_type = multipart(files=[("files", "a.png", b"one"), ("files", "b.png", b"two")])
    assert main._file_part_digests(two, two_type) == [
        ("a.png", hashlib.sha256(b"one").hexdigest()),
        ("b.png", hashlib.sha256(b"two").hexdigest()),
    ]


def test_the_guard_is_mounted_on_the_launched_application():
    """A refusal that is never wired is a comment; `__main__` is what a Space runs."""
    kwargs = main.launch_app_kwargs()
    middleware = kwargs["middleware"]
    assert [entry.cls for entry in middleware] == [main.SyntheticUploadStagingGuard]
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "app_kwargs=launch_app_kwargs()" in source


def test_the_staging_guard_closes_only_on_a_space(locally):
    guard, backend = guarded(locally)
    body, content_type = multipart(files=[("files", "x.png", b"\x89PNG\r\n\x1a\nnot shipped")])
    with deployment(SPACE_ID=None, INVOICE_FREE_UPLOAD="0"):
        status, detail = ask(guard, [body], content_type=content_type)
    assert status == 403
    assert detail["detail"] == GATE
    assert backend.calls == []


def test_a_file_hiding_behind_another_field_name_is_refused(on_a_space):
    """Gradio only stores the field `files`, but the guard checks every part that declares one."""
    guard, backend = guarded(on_a_space)
    body, content_type = raw_multipart(
        ['Content-Disposition: form-data; name="invoice"; filename="ma-facture.png"'],
        b"\x89PNG\r\n\x1a\nreal invoice bytes",
    )
    status, _ = ask(guard, [body], content_type=content_type)
    assert status == 403
    assert backend.calls == []


@pytest.mark.parametrize("disposition", [
    'form-data; name="files"; filename="france.png"',
    'form-data; name="files"; filename=france.png',
    'form-data; name="files"; filename="france .png"',
    'form-data; name="files"; filename="france.png"; filename*="other.png"',
    'form-data; name="files"',
    "form-data; name=\"files\"; filename*=UTF-8''france.png",
    'form-data; name="files"; FileName="france.png"',
])
def test_the_guard_calls_a_file_what_gradio_calls_a_file(disposition):
    """No part can be written by Gradio without first being hashed by the guard.

    Both ask the same function the same question: does this Content-Disposition declare a
    `filename`? If the answers ever diverged, a stored part could escape the digest check.
    """
    from python_multipart.multipart import parse_options_header

    written_by_gradio = b"filename" in parse_options_header(disposition)[1]
    body, content_type = raw_multipart(
        [f"Content-Disposition: {disposition}".encode("latin-1").decode("latin-1")], b"payload",
    )
    (name, digest), = main._file_part_digests(body, content_type)
    assert (name is not None) is written_by_gradio
    assert digest == hashlib.sha256(b"payload").hexdigest()
