"""Local-only review persistence. No document bytes or runtime tokens are stored."""

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
from uuid import uuid4

import invoice_review_edit as edit

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / ".local-review"
SCHEMA_VERSION = 1
TRUE_VALUES = {"1", "true", "yes", "on"}


class StoreError(ValueError):
    """Persistence was refused; the existing state must be preserved."""


class RevisionConflict(StoreError):
    """Another session saved or deleted this draft."""


def persistence_requested(environ=None):
    env = os.environ if environ is None else environ
    return str(env.get("INVOICE_REVIEW_PERSIST", "")).strip().lower() in TRUE_VALUES


def require_local_environment(environ=None):
    env = os.environ if environ is None else environ
    if any(key.startswith("SPACE_") or key == "HF_SPACE_ID" or
           key == "SYSTEM" and str(value).lower() == "spaces" for key, value in env.items()):
        raise StoreError("Persistence is forbidden in Hugging Face Spaces.")
    if not all(str(env.get(key, "")).strip().lower() in TRUE_VALUES for key in (
        "INVOICE_REVIEW_UI", "INVOICE_REVIEW_EDIT", "INVOICE_REVIEW_PERSIST",
    )):
        raise StoreError("All three local review flags must be enabled.")
    if env.get("GRADIO_SHARE", "").lower() in TRUE_VALUES:
        raise StoreError("Gradio sharing is forbidden.")
    if env.get("GRADIO_SERVER_NAME", "127.0.0.1") != "127.0.0.1":
        raise StoreError("Only 127.0.0.1 is authorized.")
    if env.get("GRADIO_ROOT_PATH"):
        raise StoreError("Proxy/root-path configurations are not supported.")


def _encode(value):
    # Every value is tagged, including containers: no reserved-key collisions.
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float and math.isfinite(value):
        return ["float", value.hex()]
    if isinstance(value, Decimal) and value.is_finite():
        return ["decimal", str(value)]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["list", [_encode(item) for item in value]]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return ["dict", [[key, _encode(value[key])] for key in sorted(value)]]
    raise StoreError("Unsupported or non-finite value in saved review.")


def _decode(node):
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise StoreError("Malformed typed record.")
    tag = node[0]
    if tag == "null" and len(node) == 1:
        return None
    if len(node) != 2:
        raise StoreError("Malformed typed record.")
    value = node[1]
    if tag == "bool" and type(value) is bool:
        return value
    if tag == "str" and isinstance(value, str):
        return value
    if tag == "int" and isinstance(value, str):
        return int(value)
    if tag == "float" and isinstance(value, str):
        result = float.fromhex(value)
        if math.isfinite(result):
            return result
    if tag == "decimal" and isinstance(value, str):
        result = Decimal(value)
        if result.is_finite():
            return result
    if tag == "list" and isinstance(value, list):
        return [_decode(item) for item in value]
    if tag == "dict" and isinstance(value, list):
        result = {}
        for pair in value:
            if (not isinstance(pair, list) or len(pair) != 2
                    or not isinstance(pair[0], str) or pair[0] in result):
                raise StoreError("Duplicate or malformed saved key.")
            result[pair[0]] = _decode(pair[1])
        return result
    raise StoreError("Unknown or invalid saved value type.")


def dumps(value):
    return json.dumps(_encode(value), ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def loads(text):
    try:
        result = _decode(json.loads(text))
        if dumps(result) != text:
            raise StoreError("Non-canonical saved record.")
        return result
    except (ValueError, TypeError, ArithmeticError, RecursionError) as exc:
        raise StoreError("Invalid saved record.") from exc


def digest(value):
    return hashlib.sha256(dumps(value).encode("ascii")).hexdigest()


def document_identity(path):
    path = Path(path)
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".pdf": "application/pdf"}.get(path.suffix.lower())
    if media is None or not path.is_file():
        raise StoreError("Attach a supported source document.")
    size = path.stat().st_size
    if not 0 < size <= 20 * 1024 * 1024:
        raise StoreError("Source document exceeds the supported size.")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return {"sha256": hasher.hexdigest(), "size": size, "media_type": media}


def compatibility():
    files = ("src/invoice_review_edit.py", "data/schema.py", "src/invoice_validation.py",
             "src/invoice_grounding.py", "src/ocr_config.py", "src/invoice_review.py",
             "experiments/review_assessment.py")
    return {
        "state_version": edit.STATE_VERSION,
        "codec_version": 1,
        "policy_sha256": hashlib.sha256(
            (ROOT / "experiments" / "review_assessment_policy.json").read_bytes(),
        ).hexdigest(),
        "implementation": {
            name: hashlib.sha256((ROOT / Path(name)).read_bytes()).hexdigest()
            for name in files
        },
    }


def snapshot_for(state, document, provenance=None):
    edit.validate_draft(state)
    return {
        "schema_version": SCHEMA_VERSION,
        "document": deepcopy(document),
        "strict": deepcopy(state["original_strict"]),
        "review": deepcopy(state["original_review"]),
        "model": deepcopy(state["original_model"]),
        "provenance": deepcopy(provenance or {}),
        "anchors": [row for row in state["row_order"]
                    if state["rows"][row]["origin"] != "human"],
        "compatibility": compatibility(),
    }


def validate_snapshot(snapshot):
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != SCHEMA_VERSION:
        raise StoreError("Unsupported snapshot schema.")
    source = snapshot.get("document")
    if (not isinstance(source, dict) or set(source) != {"sha256", "size", "media_type"}
            or not isinstance(source["sha256"], str)
            or len(source["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in source["sha256"])
            or type(source["size"]) is not int or not 0 < source["size"] <= 20 * 1024 * 1024
            or source["media_type"] not in {"image/png", "image/jpeg", "application/pdf"}):
        raise StoreError("Malformed source-document fingerprint.")


def draft_data(state):
    edit.validate_draft(state)
    return deepcopy({
        "schema_version": SCHEMA_VERSION,
        "rows": state["rows"], "row_order": state["row_order"],
        "values": state["draft"], "origins": state["draft_origins"],
        "decisions": state["decisions"],
        "compatibility": compatibility(),
        "history": state.get("confirmation_history", []),
        "confirmations": {
            key: {"value_hash": digest(state["draft"][key]),
                  "context_hash": digest({
                      "values": state["draft"], "decisions": state["decisions"],
                      "origins": state["draft_origins"], "rows": state["rows"],
                  })}
            for key, confirmed in state["confirmed"].items() if confirmed
        },
    })


def restore_state(snapshot, data, *, document_verified=False):
    validate_snapshot(snapshot)
    if not isinstance(data, dict):
        raise StoreError("Malformed draft.")
    if snapshot.get("schema_version") != SCHEMA_VERSION or data.get("schema_version") != SCHEMA_VERSION:
        raise StoreError("Unsupported snapshot or draft schema.")
    state = edit.restore_edit_state(
        snapshot["strict"], snapshot["review"], snapshot["model"],
        snapshot["anchors"], data["rows"], data["row_order"],
        data["values"], data["origins"], data["decisions"],
    )
    warnings = []
    historical = data["confirmations"]
    if not isinstance(data.get("history", []), list):
        raise StoreError("Malformed confirmation history.")
    if not isinstance(historical, dict) or not set(historical) <= set(state["fields"]):
        raise StoreError("Invalid historical confirmations.")
    compatible = data["compatibility"] == compatibility()
    context_hash = digest({
        "values": state["draft"], "decisions": state["decisions"],
        "origins": state["draft_origins"], "rows": state["rows"],
    })
    for key, record in historical.items():
        if record != {"value_hash": digest(state["draft"][key]), "context_hash": context_hash}:
            raise StoreError("Historical confirmation does not match the saved values.")
        if compatible and document_verified:
            try:
                state = edit.confirm_field(state, key)
            except edit.EditStateError:
                warnings.append(key)
        else:
            warnings.append(key)
    state["document_verified"] = document_verified
    state["confirmation_history"] = deepcopy(data.get("history", []))
    if warnings and historical:
        entry = {"compatibility": data["compatibility"], "confirmations": historical}
        if entry not in state["confirmation_history"]:
            state["confirmation_history"].append(entry)
    if not compatible:
        warnings.insert(0, "Validation/policy changed: renew all historical confirmations.")
    return state, warnings


def _verify_windows_acl(path):
    """Read the effective owner/DACL via Win32, avoiding a subprocess on every operation."""
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    pointer = ctypes.c_void_p
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.GetTokenInformation.argtypes = [
        wintypes.HANDLE, ctypes.c_int, pointer, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(pointer), ctypes.POINTER(pointer), ctypes.POINTER(pointer),
        ctypes.POINTER(pointer), ctypes.POINTER(pointer),
    ]
    advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi.EqualSid.argtypes = [pointer, pointer]
    advapi.GetAclInformation.argtypes = [pointer, pointer, wintypes.DWORD, ctypes.c_int]
    advapi.GetAce.argtypes = [pointer, wintypes.DWORD, ctypes.POINTER(pointer)]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [pointer]
    kernel.LocalFree.restype = pointer

    class AclSize(ctypes.Structure):
        _fields_ = [("count", wintypes.DWORD), ("used", wintypes.DWORD), ("free", wintypes.DWORD)]

    token = wintypes.HANDLE()
    descriptor, owner, dacl = pointer(), pointer(), pointer()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise StoreError("Cannot query the Windows account.")
    try:
        needed = wintypes.DWORD()
        advapi.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
            raise StoreError("Cannot query the Windows user SID.")
        user = pointer.from_buffer(buffer).value
        error = advapi.GetNamedSecurityInfoW(
            str(path), 1, 0x00000005, ctypes.byref(owner), None,
            ctypes.byref(dacl), None, ctypes.byref(descriptor),
        )
        if error or not dacl or not advapi.EqualSid(user, owner):
            raise StoreError("Storage has no private DACL or has the wrong owner.")
        size = AclSize()
        if not advapi.GetAclInformation(dacl, ctypes.byref(size), ctypes.sizeof(size), 2) or not size.count:
            raise StoreError("Cannot verify storage DACL.")
        for index in range(size.count):
            ace = pointer()
            if not advapi.GetAce(dacl, index, ctypes.byref(ace)) or ace.value is None:
                raise StoreError("Cannot read storage access rule.")
            # ACCESS_ALLOWED_ACE consists of a 4-byte header, 4-byte mask, then the SID.
            if ctypes.c_ubyte.from_address(ace.value).value != 0 or not advapi.EqualSid(
                user, ace.value + 8,
            ):
                raise StoreError("Storage is not accessible exclusively by its owner.")
    finally:
        if descriptor:
            kernel.LocalFree(descriptor)
        kernel.CloseHandle(token)


def secure_permissions(path, *, create=False):
    """Use a protected user-only DACL on Windows, mode/ownership checks on POSIX."""
    path = Path(path)
    if path.is_symlink() or path.is_junction():
        raise StoreError("Linked storage is forbidden.")
    if os.name == "nt":
        if create:
            quoted = str(path).replace("'", "''")
            script = (
                "$ErrorActionPreference='Stop';"
                "$sid=[System.Security.Principal.WindowsIdentity]::GetCurrent().User;"
                f"$p='{quoted}';"
                "$acl=New-Object System.Security.AccessControl.DirectorySecurity;"
                "$acl.SetOwner($sid);$acl.SetAccessRuleProtection($true,$false);"
                "$rule=New-Object System.Security.AccessControl.FileSystemAccessRule("
                "$sid,'FullControl','ContainerInherit,ObjectInherit','None','Allow');"
                "$acl.AddAccessRule($rule);Set-Acl -LiteralPath $p -AclObject $acl;"
            )
            try:
                subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                    check=True, capture_output=True, timeout=30,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise StoreError("Cannot establish owner-only storage permissions.") from exc
        _verify_windows_acl(path)
    else:
        if create:
            path.chmod(0o700 if path.is_dir() else 0o600)
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise StoreError("Storage must be owned by the current user only.")


class DraftStore:
    """Paths are supplied by the trusted launcher, never by browser callbacks."""

    def __init__(self, directory=STORAGE):
        self.directory = Path(directory).absolute()
        self.path = self.directory / "drafts.sqlite3"

    def prepare(self):
        if not self.directory.is_relative_to(ROOT):
            raise StoreError("Storage must remain inside this local project.")
        if self.directory.resolve() != self.directory:
            raise StoreError("Linked storage ancestors are forbidden.")
        created = not self.directory.exists()
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        secure_permissions(self.directory, create=created)
        for child in self.directory.iterdir():
            secure_permissions(child)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, SCHEMA_VERSION):
                raise StoreError("Unsupported database schema version.")
            if version == 0:
                if db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1",
                ).fetchone():
                    raise StoreError("Refusing to initialize an unrecognized existing database.")
                db.execute(
                    "CREATE TABLE snapshots (id TEXT PRIMARY KEY, hash TEXT NOT NULL, "
                    "data TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE drafts (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL "
                    "REFERENCES snapshots(id), snapshot_hash TEXT NOT NULL, revision INTEGER NOT NULL, "
                    "created TEXT NOT NULL, modified TEXT NOT NULL, data TEXT NOT NULL, "
                    "hash TEXT NOT NULL)"
                )
                db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        secure_permissions(self.path)

    @contextmanager
    def connection(self):
        try:
            secure_permissions(self.directory)
            if self.path.exists():
                secure_permissions(self.path)
            else:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            db = sqlite3.connect(self.path, timeout=5)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA synchronous=FULL")
            try:
                with db:
                    yield db
            finally:
                db.close()
        except (sqlite3.Error, OSError) as exc:
            raise StoreError("Local draft storage is unavailable; no save was confirmed.") from exc

    def list_drafts(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, revision, modified FROM drafts ORDER BY modified DESC",
            )]

    def save(self, snapshot, state, *, draft_id=None, revision=None):
        data = draft_data(state)
        # Reconstructing checks rows, identifiers and derived metadata before writing.
        restore_state(snapshot, data)
        if (dumps(snapshot["strict"]) != dumps(state["original_strict"])
                or dumps(snapshot["review"]) != dumps(state["original_review"])
                or dumps(snapshot["model"]) != dumps(state["original_model"])):
            raise StoreError("Extraction snapshot mismatch.")
        snapshot_hash = digest(snapshot)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if draft_id is None:
                if revision is not None:
                    raise StoreError("Invalid new-draft revision.")
                draft_id, snapshot_id = str(uuid4()), str(uuid4())
                db.execute("INSERT INTO snapshots VALUES (?,?,?)",
                           (snapshot_id, snapshot_hash, dumps(snapshot)))
                db.execute("INSERT INTO drafts VALUES (?,?,?,?,?,?,?,?)", (
                    draft_id, snapshot_id, snapshot_hash, 1, now, now, dumps(data), digest(data),
                ))
                revision = 1
            else:
                row = db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
                if row is None or row["revision"] != revision:
                    raise RevisionConflict("Draft changed or was deleted. Reopen it; stale save refused.")
                if row["snapshot_hash"] != snapshot_hash:
                    raise StoreError("Cannot move a draft to a different extraction snapshot.")
                snapshot_row = db.execute(
                    "SELECT data, hash FROM snapshots WHERE id=?", (row["snapshot_id"],),
                ).fetchone()
                if snapshot_row["hash"] != snapshot_hash or digest(loads(snapshot_row["data"])) != snapshot_hash:
                    raise StoreError("Stored snapshot integrity check failed.")
                revision += 1
                db.execute(
                    "UPDATE drafts SET revision=?,modified=?,data=?,hash=? WHERE id=?",
                    (revision, now, dumps(data), digest(data), draft_id),
                )
        return {"draft_id": draft_id, "revision": revision, "saved_hash": digest(data)}

    def open(self, draft_id, *, document=None):
        with self.connection() as db:
            row = db.execute(
                "SELECT d.*,s.data AS snapshot_data,s.hash AS stored_snapshot_hash FROM drafts d "
                "JOIN snapshots s ON s.id=d.snapshot_id WHERE d.id=?", (draft_id,),
            ).fetchone()
        if row is None:
            raise StoreError("Saved draft no longer exists.")
        snapshot, data = loads(row["snapshot_data"]), loads(row["data"])
        if (digest(snapshot) != row["snapshot_hash"]
                or row["stored_snapshot_hash"] != row["snapshot_hash"]
                or digest(data) != row["hash"]):
            raise StoreError("Saved draft/snapshot integrity check failed.")
        try:
            if document is not None and document != snapshot["document"]:
                raise StoreError("Source document mismatch: its bytes do not match the saved SHA-256.")
            state, warnings = restore_state(snapshot, data, document_verified=document is not None)
        except (KeyError, TypeError, AttributeError, edit.EditStateError) as exc:
            raise StoreError("Malformed saved review state.") from exc
        return {
            "human_edit": state, "snapshot": snapshot, "history": data["confirmations"],
            "warnings": warnings, "draft_id": row["id"], "revision": row["revision"],
            "saved_hash": digest(draft_data(state)),
        }

    def delete(self, draft_id, revision, *, confirmed=False):
        if confirmed is not True:
            raise StoreError("Explicit deletion confirmation is required.")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,)).fetchone()
            if row is None or row["revision"] != revision:
                raise RevisionConflict("Draft changed or was deleted; deletion refused.")
            db.execute("DELETE FROM drafts WHERE id=?", (draft_id,))
            db.execute(
                "DELETE FROM snapshots WHERE id=? AND NOT EXISTS "
                "(SELECT 1 FROM drafts WHERE snapshot_id=?)",
                (row["snapshot_id"], row["snapshot_id"]),
            )
