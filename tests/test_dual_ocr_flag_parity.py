"""PR101 flag parity: with EXPERIMENTAL_DUAL_OCR unset the application must not reach the second engine.

The experimental path is only reachable through the flag, and the flag is read at call time. These
tests stub the OCR stage and a recording arbitration stub, so they assert what the application calls
rather than what a real engine produces: that is the only way to prove the disabled branch is still
the old branch, and it costs no OCR.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "app")]

from PIL import Image

import app.main as application
import dual_ocr_experiment
from invoice_evidence import DocumentEvidence, EvidenceText


class Stub:
    """Stand-in for the experimental module: records every call and never runs an engine."""

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def enabled(self):
        return dual_ocr_experiment.enabled()

    def experiment(self, image, evidence, *, strategy_name=None):
        self.calls.append(strategy_name)
        return (self.result or evidence), {"strategy": strategy_name or "A", "available": True}


def instrument(monkeypatch, stub):
    calls = {"recover": 0}

    def fake_recover(image, reader, *, budget=None):
        calls["recover"] += 1
        return DocumentEvidence("Facture", None, [], [], source_kind="consensus")

    # app.main binds these names at import, so the patch targets are its own bindings.
    monkeypatch.setattr(application, "recover_image", fake_recover)
    monkeypatch.setattr(application, "dual_ocr_experiment", stub)
    monkeypatch.setattr(application, "require_ocr_languages", lambda: {
        "requested": ["fra", "eng"], "effective": ["fra", "eng"], "missing": [],
        "tessdata_prefix": "x"})
    monkeypatch.setattr(application, "check_pixels", lambda *pixels: None)
    return calls


def test_disabled_flag_never_reaches_the_arbitration_layer(monkeypatch):
    monkeypatch.delenv(dual_ocr_experiment.FLAG_NAME, raising=False)
    stub = Stub()
    calls = instrument(monkeypatch, stub)
    application.ocr_image(Image.new("RGB", (40, 40)))
    assert calls["recover"] == 1
    assert stub.calls == []


def test_enabled_flag_arms_the_arbitration_layer_once_per_document(monkeypatch):
    monkeypatch.setenv(dual_ocr_experiment.FLAG_NAME, "1")
    stub = Stub()
    calls = instrument(monkeypatch, stub)
    application.ocr_image(Image.new("RGB", (40, 40)))
    assert calls["recover"] == 1
    assert len(stub.calls) == 1


def test_the_flag_is_read_at_call_time_not_at_import(monkeypatch):
    monkeypatch.delenv(dual_ocr_experiment.FLAG_NAME, raising=False)
    stub = Stub()
    calls = instrument(monkeypatch, stub)
    image = Image.new("RGB", (40, 40))
    application.ocr_image(image)
    assert stub.calls == []
    monkeypatch.setenv(dual_ocr_experiment.FLAG_NAME, "1")
    application.ocr_image(image)
    assert len(stub.calls) == 1
    monkeypatch.setenv(dual_ocr_experiment.FLAG_NAME, "0")
    application.ocr_image(image)
    assert len(stub.calls) == 1, "turning the flag back off must stop the experiment again"


def test_the_strategy_environment_selects_which_arm_runs(monkeypatch):
    monkeypatch.setenv(dual_ocr_experiment.FLAG_NAME, "1")
    monkeypatch.setenv(dual_ocr_experiment.STRATEGY_ENV, "B")
    stub = Stub()
    instrument(monkeypatch, stub)
    application.ocr_image(Image.new("RGB", (40, 40)))
    assert stub.calls == [None], "the app calls once and the layer resolves the strategy"
    assert dual_ocr_experiment.strategy() == "B"


def test_an_experiment_record_reaches_the_diagnostics(monkeypatch):
    monkeypatch.setenv(dual_ocr_experiment.FLAG_NAME, "1")
    evidence = DocumentEvidence("Facture", None, [], [], source_kind="consensus")
    evidence.dual_ocr = {"strategy": "A", "available": True, "cell_provenance": {"both": 3}}
    stub = Stub(result=evidence)
    instrument(monkeypatch, stub)
    application.ocr_image(Image.new("RGB", (40, 40)))
    diagnostics = {}
    payload = application.extract_invoice(EvidenceText(evidence), diagnostics=diagnostics)
    assert payload["line_items"] == []
    assert diagnostics["dual_ocr"] == evidence.dual_ocr
