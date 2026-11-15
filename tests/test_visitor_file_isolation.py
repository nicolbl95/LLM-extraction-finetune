"""PR172 Phase 5: a Space that accepts one visitor's invoice must not hand it to the next visitor.

Gradio authorises a file read by *directory* (`route_utils.file_fetch` asks only whether a path is
under the upload folder or the examples cache, never whose it is), and both live in one shared
folder. PR171 measured the consequence on the deployed configuration: a document shown to one
browser stayed fetchable by another for the whole cache lifetime, and the preview image derived
from it was not even on the sweep's list. `VisitorFileIsolation` is the missing ownership layer.

Nothing here opens a port. The middleware is driven over raw ASGI the way uvicorn drives it, in
front of a stand-in for the routes Gradio owns, and every scenario goes through the whole mounted
chain — including the clients that bypass the interface and post to the file and call routes
directly, which is where the brief asks isolation to hold.
"""

import asyncio
import hashlib
import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, unquote
from unittest.mock import patch
import os

import pytest

from app import main

ROOT = Path(main.__file__).resolve().parents[1]
SPACE = "nicolbl95/llm-extraction-finetune"
REFUSAL = main.SESSION_ISOLATION_ERROR
BOUNDARY = "pr172boundary"
FILE = b"\x89PNG\r\n\x1a\n"


class _AgedClock:
    """`main.time` with every reading pushed a fixed distance into the future."""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    def perf_counter(self) -> float:
        return time.perf_counter() + self.offset


@contextmanager
def deployment(**values):
    """Environment exactly as given: a `None` value removes the variable instead of ignoring it."""
    with patch.dict(os.environ, {k: v for k, v in values.items() if v is not None}, clear=False):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
        yield


@pytest.fixture
def space(tmp_path):
    """A hosted demo with upload open, the only configuration this rule governs."""
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD="1", GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


@pytest.fixture
def closed_space(tmp_path):
    with deployment(SPACE_ID=SPACE, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


@pytest.fixture
def locally(tmp_path):
    with deployment(SPACE_ID=None, INVOICE_FREE_UPLOAD=None, GRADIO_TEMP_DIR=str(tmp_path)):
        yield tmp_path


def multipart(files):
    out = bytearray()
    for name, filename, payload in files:
        out += (f"--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
        out += payload + b"\r\n"
    out += f"--{BOUNDARY}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={BOUNDARY}"


def file_part(body: bytes) -> tuple[str, bytes]:
    """The part this test sent, read the way Gradio would: header line, then its bytes."""
    for chunk in body.split(f"--{BOUNDARY}".encode())[1:]:
        header, _, payload = chunk.partition(b"\r\n\r\n")
        declared = re.search(rb'filename="([^"]+)"', header)
        if declared:
            return declared.group(1).decode("utf-8"), payload[: -len(b"\r\n")]
    raise AssertionError("multipart sans partie fichier")


async def _drain(receive) -> bytes:
    body = bytearray()
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        body += message.get("body", b"")
        if not message.get("more_body", False):
            break
    return bytes(body)


async def _reply(send, status: int, content_type: bytes, payload: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type)]})
    await send({"type": "http.response.body", "body": payload})


class SharedFolderGradio:
    """The routes Gradio owns, with the shapes that matter for this rule.

    Two properties make the test mean something: every visitor's file is staged in the one folder
    every visitor can read, and the preview derived from a document is a different file per
    document — otherwise there would be nothing for isolation to separate.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[str] = []
        self.blocks = type("Blocks", (), {"upload_file_set": set()})()
        self.app = type("Server", (), {"blocks": self.blocks})()
        self.answers_with_an_error = False

    async def __call__(self, scope, receive, send) -> None:
        body = await _drain(receive)
        route = self.route(scope)
        self.calls.append(f"{scope['method']} {route}")
        if route in ("/gradio_api/upload", "/upload"):
            await self._stage(body, send)
        elif route.startswith(("/gradio_api/file", "/file")):
            await self._read(route, send)
        elif route.startswith("/gradio_api/call/"):
            await self._answer(body, send)
        else:
            await _reply(send, 200, b"application/json", b"{}")

    @staticmethod
    def route(scope) -> str:
        path = scope["path"]
        root_path = scope.get("root_path") or ""
        return path[len(root_path):] if root_path and path.startswith(root_path) else path

    async def _stage(self, body: bytes, send) -> None:
        """Gradio stores the part under the digest of its bytes, in the served upload folder."""
        filename, payload = file_part(body)
        directory = self.root / hashlib.sha256(payload).hexdigest()
        directory.mkdir(parents=True, exist_ok=True)
        stored = directory / filename
        stored.write_bytes(payload)
        answer = json.dumps([{"path": str(stored), "orig_name": filename,
                              "url": f"/gradio_api/file={quote(str(stored))}",
                              "size": len(payload)}]).encode("utf-8")
        await _reply(send, 200, b"application/json", answer)

    async def _read(self, route: str, send) -> None:
        marker = "/file/" if route.startswith(("/file/", "/gradio_api/file/")) else "/file="
        tail = route.split(marker, 1)[1].split("?", 1)[0].split("#", 1)[0]
        path = Path(unquote(unquote(tail)))
        if path.is_file():
            await _reply(send, 200, b"image/png", path.read_bytes())
        else:
            await _reply(send, 404, b"application/json", b'{"detail":"Not Found"}')

    async def _answer(self, body: bytes, send) -> None:
        """The extraction call previews the page it was handed and answers with the new file."""
        claimed = json.loads(body)["data"][0]["path"]
        rendition = Path(claimed).parent / "preview.webp"
        rendition.write_bytes(b"RIFF derived from that visitor's page")
        if self.answers_with_an_error:
            detail = json.dumps({"detail": "Le document n'a pas pu être lu.",
                                 "file": str(rendition)}).encode("utf-8")
            await _reply(send, 500, b"application/json", detail)
            return
        frames = [json.dumps({"msg": "code", "success": True}),
                  json.dumps({"data": [str(rendition)]}), "[DONE]"]
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")]})
        for index, frame in enumerate(frames):
            await send({"type": "http.response.body", "body": f"data: {frame}\n\n".encode(),
                        "more_body": index < len(frames) - 1})


def mounted(root: Path) -> tuple:
    """The chain exactly as `launch_app_kwargs()` wires it: isolation outermost, guard inside."""
    backend = SharedFolderGradio(root)
    chain = main.VisitorFileIsolation(main.SyntheticUploadStagingGuard(backend))
    return chain, backend


async def exchange(chain, backend, path, *, method="GET", body=b"", content_type=None,
                   cookie=None, interrupt=False, scheme="https") -> dict:
    headers = [(b"host", b"space.internal")]
    if content_type:
        headers.append((b"content-type", content_type.encode("latin-1")))
    if cookie:
        headers.append((b"cookie", cookie.encode("latin-1")))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": method, "path": path, "raw_path": path.encode("latin-1"),
             "query_string": b"", "root_path": "", "scheme": scheme, "headers": headers,
             "client": ("203.0.113.7", 51234), "app": backend.app}
    messages = [{"type": "http.request", "body": body, "more_body": interrupt}]
    if interrupt:
        messages.append({"type": "http.disconnect"})

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    seen: dict = {"body": b"", "headers": {}}

    async def send(message):
        if message["type"] == "http.response.start":
            seen["status"] = message["status"]
            seen["headers"] = {key.decode("latin-1").lower(): value.decode("latin-1")
                               for key, value in message.get("headers", [])}
        elif message["type"] == "http.response.body":
            seen["body"] += message.get("body", b"")

    await chain(scope, receive, send)
    seen["token"] = seen["headers"].get("set-cookie", "").split(";")[0]
    return seen


def ask(chain, backend, path, **kwargs) -> dict:
    return asyncio.run(exchange(chain, backend, path, **kwargs))


def visitor(chain, backend, filename: str, payload: bytes, cookie=None) -> tuple[str, str]:
    """A browser arriving with a document of its own: its session, and where its file was stored.

    A `cookie` is this same browser coming back, which is the only way one session can be shown two
    documents; without one every call is a new visitor.
    """
    body, content_type = multipart([("files", filename, payload)])
    seen = ask(chain, backend, "/gradio_api/upload", method="POST", body=body,
               content_type=content_type, cookie=cookie)
    assert seen["status"] == 200, seen["body"]
    return seen["token"] or cookie, json.loads(seen["body"])[0]["path"]


def read(chain, backend, cookie: str, target: str, *, spelling="/gradio_api/file={}") -> int:
    return ask(chain, backend, spelling.format(quote(target, safe="")), cookie=cookie)["status"]


def call(chain, backend, cookie: str, claimed: str) -> dict:
    body = json.dumps({"data": [{"path": claimed, "orig_name": "document.png"}],
                       "session_hash": "chosen-by-the-browser"}).encode("utf-8")
    return ask(chain, backend, "/gradio_api/call/run_extraction_ui", method="POST", body=body,
               content_type="application/json", cookie=cookie)


# --- Test A: isolation between two visitors ---------------------------------------------------

def test_a_visitor_reads_the_document_they_uploaded(space):
    chain, backend = mounted(space)
    cookie, path = visitor(chain, backend, "visiteuse-a.png", FILE + b"le document de A")
    assert read(chain, backend, cookie, path) == 200
    assert ask(chain, backend, f"/gradio_api/file={path}", cookie=cookie)["status"] == 200


def test_another_visitor_naming_that_exact_path_is_refused(space):
    chain, backend = mounted(space)
    a_cookie, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + b"le document de A")
    b_cookie, _ = visitor(chain, backend, "visiteur-b.png", FILE + b"le document de B")
    assert read(chain, backend, b_cookie, a_path) == 403
    assert read(chain, backend, b_cookie, a_path, spelling="/file={}") == 403
    assert read(chain, backend, b_cookie, a_path, spelling="/gradio_api/file/{}") == 403
    assert read(chain, backend, a_cookie, a_path) == 200


@pytest.mark.parametrize("spelling", [
    "/gradio_api/file={}",
    "/gradio_api/file={}?timestamp=1",
    "/gradio_api/file={}#page",
    "/gradio_api/file/{0}",
    "/gradio_api/file={0}",
    "/file={}",
    "/file/{0}",
])
def test_every_spelling_of_the_read_route_is_governed(space, spelling):
    """The claim is the same file however a client spells it, so one rule cannot be side-stepped."""
    chain, backend = mounted(space)
    a_cookie, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + b"facture de A")
    b_cookie, _ = visitor(chain, backend, "visiteur-b.png", FILE + b"facture de B")
    for cookie, expected in ((b_cookie, 403), (a_cookie, True)):
        for target in (a_path, quote(a_path, safe=""), quote(quote(a_path, safe=""), safe="")):
            seen = ask(chain, backend, spelling.format(target), cookie=cookie)
            if expected is True:
                assert seen["status"] != 403, (spelling, target)
            else:
                assert seen["status"] == 403, (spelling, target)


def test_the_refusal_does_not_carry_the_document(space):
    chain, backend = mounted(space)
    _, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + b"ma facture confidentielle")
    b_cookie, _ = visitor(chain, backend, "visiteur-b.png", FILE + b"la sienne")
    seen = ask(chain, backend, f"/gradio_api/file={quote(a_path, safe='')}", cookie=b_cookie)
    assert seen["status"] == 403
    assert b"confidentielle" not in seen["body"]
    assert json.loads(seen["body"])["detail"] == REFUSAL


def test_a_client_with_no_session_at_all_is_refused(space):
    """A path is never an authorisation on its own, whatever cookie — or none — comes with it."""
    chain, backend = mounted(space)
    _, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + b"document sans cookie")
    assert read(chain, backend, "", a_path) == 403
    assert read(chain, backend, "invoice_sid=guess-my-own-value-too-long", a_path) == 403


# --- Test A, second channel: the extraction call owns the same document ----------------------

def test_a_call_naming_another_visitors_path_is_refused(space):
    """The JSON answer *is* the document, so claiming a path is as much a read as asking for one."""
    chain, backend = mounted(space)
    _, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + "facture privée".encode("utf-8"))
    b_cookie, b_path = visitor(chain, backend, "visiteur-b.png", FILE + "facture à moi".encode("utf-8"))
    assert call(chain, backend, b_cookie, a_path)["status"] == 403
    assert call(chain, backend, b_cookie, b_path)["status"] == 200


def test_naming_another_sessions_hash_does_not_make_it_yours(space):
    """`session_hash` is chosen by the browser, so only a value this server minted can own a file."""
    chain, backend = mounted(space)
    a_cookie, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + b"facture de A")
    b_cookie, _ = visitor(chain, backend, "visiteur-b.png", FILE + b"facture de B")
    stolen = a_cookie.split("=", 1)[1]
    body = json.dumps({"data": [{"path": a_path}], "session_hash": stolen}).encode()
    seen = ask(chain, backend, "/gradio_api/call/run_extraction_ui", method="POST", body=body,
               content_type="application/json", cookie=b_cookie)
    assert seen["status"] == 403


# --- Test B: the files derived from a document ------------------------------------------------

def test_a_derived_preview_follows_the_document_it_came_from(space):
    chain, backend = mounted(space)
    a_cookie, a_path = visitor(chain, backend, "visiteuse-a.png", FILE + b"page de A")
    b_cookie, b_path = visitor(chain, backend, "visiteur-b.png", FILE + b"page de B")
    assert call(chain, backend, a_cookie, a_path)["status"] == 200
    assert call(chain, backend, b_cookie, b_path)["status"] == 200
    a_preview = str(Path(a_path).parent / "preview.webp")
    b_preview = str(Path(b_path).parent / "preview.webp")
    assert read(chain, backend, a_cookie, a_preview) == 200
    assert read(chain, backend, a_cookie, b_preview) == 403
    assert read(chain, backend, b_cookie, b_preview) == 200
    assert read(chain, backend, b_cookie, a_preview) == 403


def test_a_preview_nobody_was_shown_has_no_owner(space):
    """Deny by default: a file on disk is protected because nothing named it, not because it is
    hard to guess — which is the property the brief asks for instead of unguessable names."""
    chain, backend = mounted(space)
    cookie, _ = visitor(chain, backend, "visiteuse-a.png", FILE + b"page de A")
    unattributed = space / "never-served.png"
    unattributed.write_bytes(FILE + "jamais montré".encode("utf-8"))
    assert read(chain, backend, cookie, str(unattributed)) == 403


# --- Test C: bundled examples stay open, and both kinds of file reach the sweep --------------

def cached_example(root: Path, label: str = "france") -> Path:
    """The path Gradio's Examples widget hands the browser for a shipped invoice."""
    from gradio import processing_utils

    source = ROOT / f"app/examples/{label}.png"
    directory = root / processing_utils.hash_file(source)
    directory.mkdir(parents=True, exist_ok=True)
    cached = directory / source.name
    cached.write_bytes(source.read_bytes())
    return cached


def test_a_bundled_example_is_usable_by_every_visitor(space):
    """The rule governs what a visitor brings, not the three invoices the demo ships with."""
    chain, backend = mounted(space)
    cached = str(cached_example(space))
    a_cookie, _ = visitor(chain, backend, "visiteuse-a.png", FILE + b"sa facture")
    b_cookie, _ = visitor(chain, backend, "visiteur-b.png", FILE + b"la sienne")
    assert read(chain, backend, a_cookie, cached) == 200
    assert read(chain, backend, b_cookie, cached) == 200
    assert read(chain, backend, "", cached) == 200
    assert call(chain, backend, a_cookie, cached)["status"] == 200


def test_uploaded_and_derived_paths_reach_the_cache_sweep(space):
    """PR171's second finding: a preview image was never added to the list the sweep reads."""
    chain, backend = mounted(space)
    cookie, path = visitor(chain, backend, "visiteuse-a.png", FILE + "document à nettoyer".encode("utf-8"))
    call(chain, backend, cookie, path)
    registered = {os.path.normcase(p) for p in backend.blocks.upload_file_set}
    assert os.path.normcase(path) in registered
    assert os.path.normcase(str(Path(path).parent / "preview.webp")) in registered


def test_the_bundled_example_is_left_out_of_the_sweep(space):
    """Deleting a shipped example would take it away from every visitor, not just one."""
    chain, backend = mounted(space)
    cached = cached_example(space)
    ask(chain, backend, f"/gradio_api/file={quote(str(cached))}")
    assert str(cached) not in {os.path.normcase(p) for p in backend.blocks.upload_file_set}


def test_the_public_exemption_follows_the_bytes_not_the_file_name(space):
    """Calling a document `france.png` does not borrow the openness of the example called that.

    `_bundled_example` asks only whether the file sits in a directory Gradio names by
    `processing_utils.hash_file` of a shipped example's contents, so a visitor's own bytes — even a
    copy with one byte added — are staged somewhere else and keep needing their owner. A
    byte-identical copy does land in the example's own directory, and it is the example: the demo
    published those bytes itself, so no visitor's invoice is disclosed by reading them.
    """
    chain, backend = mounted(space)
    example = cached_example(space)
    owner, lookalike = visitor(chain, backend, example.name, example.read_bytes() + b"une visite de plus")
    stranger = visitor(chain, backend, "curieux.png", FILE + b"rien a voir")[0]
    assert Path(lookalike).name == example.name
    assert read(chain, backend, owner, lookalike) == 200
    assert read(chain, backend, stranger, lookalike) == 403
    assert main._bundled_example(main._canonical(str(example)))
    assert not main._bundled_example(main._canonical(lookalike))


def test_a_stale_session_no_longer_reads_its_document(space):
    """A browser that has not come back inside the retention window loses its claim on the file."""
    chain, backend = mounted(space)
    cookie, path = visitor(chain, backend, "visiteuse-a.png", FILE + "facture périmée".encode("utf-8"))
    # The clock is moved rather than the threshold set to zero: Windows reads `monotonic()` in
    # 16 ms steps, so two visits landing in one step leave `now - seen` at exactly 0.0 and a
    # threshold of 0.0 expires nothing — the refusal would go untested rather than fail.
    with patch.object(main, "time", _AgedClock(main.SESSION_RETENTION_SECONDS + 1.0)):
        visitor(chain, backend, "visiteur-b.png", FILE + b"une autre visite")
    assert read(chain, backend, cookie, path) == 403


def test_a_session_holds_a_bounded_number_of_documents(space):
    """The bound is on one browser's own files: the oldest it was shown is the one it loses."""
    chain, backend = mounted(space)
    cookie, first = visitor(chain, backend, "une.png", FILE + b"le premier document")
    with patch.object(main, "MAX_PATHS_PER_SESSION", 1):
        _, second = visitor(chain, backend, "deux.png", FILE + b"le second document", cookie=cookie)
        assert read(chain, backend, cookie, first) == 403
        assert read(chain, backend, cookie, second) == 200


def test_the_book_holds_a_bounded_number_of_visitors(space):
    chain, backend = mounted(space)
    a_cookie, a_path = visitor(chain, backend, "une.png", FILE + "première visite".encode("utf-8"))
    with patch.object(main, "MAX_SESSIONS", 1):
        b_cookie, b_path = visitor(chain, backend, "deux.png", FILE + b"seconde visite")
        assert read(chain, backend, a_cookie, a_path) == 403
        assert read(chain, backend, b_cookie, b_path) == 200


# --- Test D: failures and interruptions leave nothing unprotected ----------------------------

def test_an_error_answer_attributes_nothing(space):
    """A path named by a failure is not a file the visitor was shown, so nobody may read it."""
    chain, backend = mounted(space)
    cookie, path = visitor(chain, backend, "visiteuse-a.png", FILE + b"lecture impossible")
    backend.answers_with_an_error = True
    assert call(chain, backend, cookie, path)["status"] == 500
    assert read(chain, backend, cookie, str(Path(path).parent / "preview.webp")) == 403


def test_an_interrupted_upload_stores_nothing_anyone_can_read(closed_space):
    """Where upload is closed, a body that stops halfway is never written at all.

    Gradio's parser renames its scratch file when the request ends, so the only way an interrupted
    upload leaves nothing readable is a rule that decides before the first byte is stored. That is
    the mounted chain's job: isolation stamps the session, the guard then refuses, and the writer
    never runs. What the open configuration owes an interrupted request is different — a file that
    was never shown gains no owner, which `test_a_preview_nobody_was_shown_has_no_owner` covers.
    """
    chain, backend = mounted(closed_space)
    payload = FILE + "téléversement interrompu".encode("utf-8")
    body, content_type = multipart([("files", "interrompu.png", payload)])
    seen = ask(chain, backend, "/gradio_api/upload", method="POST", body=body,
               content_type=content_type, interrupt=True)
    assert "status" not in seen, seen
    assert backend.calls == []
    staged = closed_space / hashlib.sha256(payload).hexdigest() / "interrompu.png"
    assert not staged.exists()
    assert read(chain, backend, "invoice_sid=" + "a" * 43, str(staged)) == 404


def test_an_oversized_claim_is_turned_away_before_the_app_reads_it(space):
    chain, backend = mounted(space)
    cookie, _ = visitor(chain, backend, "visiteuse-a.png", FILE + b"facture")
    with patch.object(main, "MAX_BYTES", 256):
        body = json.dumps({"data": [{"path": "x" * 4096}]}).encode()
        seen = ask(chain, backend, "/gradio_api/call/run_extraction_ui", method="POST", body=body,
                   content_type="application/json", cookie=cookie)
    assert seen["status"] == 413
    assert "20 Mo" in seen["body"].decode("utf-8")


# --- Test E: the protections already deployed stay exactly where they were -------------------

def test_the_governed_area_is_exactly_the_two_served_folders(space):
    roots = set(main._served_roots())
    assert os.path.normcase(os.path.abspath(str(space))) in roots
    outside = ROOT / "README.md"
    assert not any(os.path.normcase(str(outside)).startswith(root) for root in roots)


def test_a_file_the_rule_does_not_govern_is_left_to_the_application(space):
    """Isolation can refuse more files and never fewer: outside the served folders it says nothing,
    so the format, size, pixel and allowlist rules in `checked_path` stay the only gate."""
    chain, backend = mounted(space)
    cookie, _ = visitor(chain, backend, "visiteuse-a.png", FILE + b"facture")
    for target in (str(ROOT / "README.md"), str(ROOT / "app" / "main.py")):
        spelling = f"/gradio_api/file={quote(target, safe='')}"
        seen = ask(chain, backend, spelling, cookie=cookie)
        assert seen["status"] == 200, target
        assert backend.calls[-1] == f"GET {spelling}", target


def test_a_closed_space_serves_exactly_what_it_served_before(closed_space):
    """With upload closed nothing is staged, so the middleware must not even mint a session."""
    chain, backend = mounted(closed_space)
    document = closed_space / "une-copie.png"
    document.write_bytes(FILE + b"exemple public")
    spelling = f"/gradio_api/file={quote(str(document))}"
    seen = ask(chain, backend, spelling)
    assert seen["status"] == 200
    assert seen["token"] == ""
    assert backend.calls == [f"GET {spelling}"]


def test_local_development_isolates_two_browsers_too(locally):
    """The same rule runs on the laptop, which is where the reactivation is tested first."""
    chain, backend = mounted(locally)
    a_cookie, a_path = visitor(chain, backend, "a.png", FILE + b"ma facture locale")
    b_cookie, _ = visitor(chain, backend, "b.png", FILE + b"celle de l'autre onglet")
    assert read(chain, backend, a_cookie, a_path) == 200
    assert read(chain, backend, b_cookie, a_path) == 403


# --- Test F: a crowd of simultaneous visitors ------------------------------------------------

async def nine_uploads(chain, backend):
    tasks = []
    for index in range(9):
        body, content_type = multipart([("files", f"doc{index}.png",
                                         FILE + f"facture {index}".encode("utf-8"))])
        tasks.append(exchange(chain, backend, "/gradio_api/upload", method="POST", body=body,
                              content_type=content_type))
    return await asyncio.gather(*tasks)


def test_nine_simultaneous_visitors_share_one_folder_without_sharing_files(space):
    chain, backend = mounted(space)
    arrivals = asyncio.run(nine_uploads(chain, backend))
    cookies = [seen["token"] for seen in arrivals]
    paths = [json.loads(seen["body"])[0]["path"] for seen in arrivals]
    assert len(set(cookies)) == 9, "each visitor must get a session of their own"
    assert len(set(paths)) == 9, "each document must be staged apart"
    matrix = [[read(chain, backend, cookie, target) for target in paths] for cookie in cookies]
    for index, row in enumerate(matrix):
        assert [status == 200 for status in row] == [
            other == index for other in range(9)], f"visitor {index}"


def test_the_queue_bounds_are_still_the_deployed_ones():
    """Opening upload must not quietly raise how many pipelines a ZeroGPU container runs at once."""
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "default_concurrency_limit=1, max_size=8" in source
    assert 'max_file_size="20mb"' in source


# --- Test G: the wiring and the tokens themselves --------------------------------------------

def test_both_halves_are_mounted_and_isolation_runs_first():
    """Staging only reopens once a session exists to own what is written, so order is the rule."""
    middleware = main.launch_app_kwargs()["middleware"]
    assert [entry.cls for entry in middleware] == [main.VisitorFileIsolation,
                                                   main.SyntheticUploadStagingGuard]
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "app_kwargs=launch_app_kwargs()" in source
    assert 'os.environ["GRADIO_NUM_WORKERS"] = "0"' in source


def test_the_staging_guard_stays_closed_when_isolation_is_not_mounted(space):
    """A build that drops the middleware keeps documents shut away instead of storing them openly."""
    backend = SharedFolderGradio(space)
    guard = main.SyntheticUploadStagingGuard(backend)
    body, content_type = multipart([("files", "ma-facture.png", FILE + "isolée ou refusée".encode("utf-8"))])
    seen = ask(guard, backend, "/gradio_api/upload", method="POST", body=body,
               content_type=content_type)
    assert seen["status"] == 403
    assert backend.calls == []


def test_the_session_id_appears_only_in_a_cookie(space):
    chain, backend = mounted(space)
    cookie, path = visitor(chain, backend, "visiteuse-a.png", FILE + "facture à protéger".encode("utf-8"))
    token = cookie.split("=", 1)[1]
    assert re.fullmatch(r"[A-Za-z0-9_-]{32,}", token)
    served = ask(chain, backend, f"/gradio_api/file={quote(path, safe='')}", cookie=cookie)
    extracted = call(chain, backend, cookie, path)
    for body in (served["body"], extracted["body"]):
        assert token.encode() not in body
        assert b"invoice_sid" not in body


def test_the_cookie_is_http_only_same_site_and_secure_only_over_https(space):
    chain, backend = mounted(space)
    body, content_type = multipart([("files", "a.png", FILE + b"attributs du cookie")])
    seen = ask(chain, backend, "/gradio_api/upload", method="POST", body=body,
               content_type=content_type)
    for required in ("HttpOnly", "SameSite=Lax", "Path=/", "Secure"):
        assert required in seen["headers"]["set-cookie"], required
    plain = ask(chain, backend, "/gradio_api/upload", method="POST", body=body,
                content_type=content_type, scheme="http")
    assert "Secure" not in plain["headers"]["set-cookie"]
