import os
from importlib.util import find_spec
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("example,total_ht,total_ttc", [
    ("france.png", 5716.77, 6860.12), ("uk.png", 6527.52, 7743.58), ("us.png", 8972.80, 8972.80),
])
def test_real_local_model_document_to_grounded_json(monkeypatch, example, total_ht, total_ttc):
    adapter = os.environ.get("INVOICE_SLOW_ADAPTER_PATH")
    if not adapter:
        pytest.skip("Set INVOICE_SLOW_ADAPTER_PATH to explicit local model weights")
    missing = [name for name in ("torch", "transformers", "peft") if find_spec(name) is None]
    if missing:
        pytest.skip("Real inference dependencies unavailable: " + ", ".join(missing))
    from app import main
    from invoice_grounding import read_model
    assert Path(adapter).is_dir()
    monkeypatch.setenv("INVOICE_ADAPTER_PATH", adapter)
    monkeypatch.setenv("INVOICE_ADAPTER_REPO", "")
    main.resolve_adapter_path.cache_clear()
    main.get_extractor.cache_clear()
    predictions = []
    real_predict = main.predict_invoice

    def observed_predict(row, messages):
        result = real_predict(row, messages)
        predictions.append(result)
        return result

    monkeypatch.setattr(main, "predict_invoice", observed_predict)
    try:
        text = main.document_text(str(main.ROOT / "app/examples" / example))
        warnings = []
        output = main.extract_invoice(text, warnings=warnings)
        assert predictions, "OCR-only fallback must not masquerade as a real-model success"
        assert predictions[0]["completion_tokens"] > 0
        assert predictions[0]["raw"].strip()
        assert predictions[0]["finish_reason"] == "stop"
        assert read_model(predictions[0]["raw"]).get("line_items")
        assert output["line_items"]
        assert all(item["quantity"] is not None for item in output["line_items"])
        assert output["total_ht"] == total_ht
        assert output["total_ttc"] == total_ttc
        first_raw = predictions[0]["raw"]
        predictions.clear()
        repeated = main.extract_invoice(text, warnings=[])
        assert predictions
        assert predictions[0]["raw"] == first_raw
        assert repeated == output
    finally:
        main.resolve_adapter_path.cache_clear()
        main.get_extractor.cache_clear()
