import contextlib
import copy
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import evaluate_finetuned as evaluation


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = evaluation.load_test_rows(ROOT / "data" / "test.jsonl")
        cls.gold = cls.rows[0]["output"]

    def test_prompt_extends_training_without_gold_or_metadata(self):
        namespace = {}
        notebook = json.loads((ROOT / "notebooks" / "fine_tune_unsloth.ipynb").read_text())
        config = next(cell for cell in notebook["cells"] if cell["id"] == "f3275e63")
        with contextlib.redirect_stdout(io.StringIO()):
            exec("".join(config["source"]), namespace)
        self.assertTrue(evaluation.SYSTEM_PROMPT.startswith(namespace["SYSTEM_PROMPT"] + "\n"))
        self.assertIn("Ne jamais confondre les en-têtes de colonnes", evaluation.SYSTEM_PROMPT)
        self.assertIn("Copier littéralement le texte exact des descriptions", evaluation.SYSTEM_PROMPT)
        self.assertIn("Sous-total", evaluation.SYSTEM_PROMPT)
        self.assertIn('"description": "Maintenance préventive"', evaluation.SYSTEM_PROMPT)
        self.assertIn("fictif", evaluation.SYSTEM_PROMPT)
        self.assertEqual(evaluation.INSTRUCTION, namespace["INSTRUCTION"])
        messages = evaluation.build_messages(self.rows[0])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["content"], namespace["INSTRUCTION"] + "\n\n" + self.rows[0]["input"])
        self.assertNotIn("assistant", [message["role"] for message in messages])

    def test_strict_and_recovered_json_are_distinct(self):
        raw = json.dumps(self.gold)
        result = evaluation.score(raw, self.gold)
        self.assertTrue(result["exact"])
        self.assertTrue(all(result["fields"].values()))
        for wrapped in (f"```json\n{raw}\n```", f"Here is the invoice: {raw}"):
            result = evaluation.score(wrapped, self.gold)
            self.assertFalse(result["strict_json"])
            self.assertFalse(result["exact"])
            self.assertTrue(result["recovered_exact"])
            self.assertIsNotNone(result["parse_error"])

    def test_invalid_outputs_fail_without_crashing(self):
        for raw in ("", "not JSON", "[]", "null", '{"total_ht": NaN}', '{"a":1,"a":2}', '{"a":'):
            result = evaluation.score(raw, self.gold)
            self.assertFalse(result["exact"])
        result = evaluation.score('{"total_ht":1e999}', self.gold)
        self.assertFalse(result["valid"])
        self.assertIsNotNone(result["schema_error"])
        for key, value in (("total_ht", True), ("total_ttc", "incorrect"), ("line_items", [None]),
                           ("line_items", {}), ("line_items", ["bad"]), ("line_items", None)):
            payload = {**self.gold, key: value}
            result = evaluation.score(json.dumps(payload), self.gold)
            self.assertFalse(result["fields"][key])
            self.assertFalse(result["exact"])
        result = evaluation.score(json.dumps({**self.gold, "total_ht": 10**1000}), self.gold)
        self.assertFalse(result["fields"]["total_ht"])

    def test_cent_tolerance_and_item_order(self):
        gold = {**self.gold, "total_ht": 1.0}
        self.assertTrue(evaluation.score(json.dumps({**gold, "total_ht": 1.01}), gold)["fields"]["total_ht"])
        self.assertFalse(evaluation.score(json.dumps({**gold, "total_ht": 1.011}), gold)["fields"]["total_ht"])
        row = next(row for row in self.rows if len(row["output"]["line_items"]) > 1)
        payload = copy.deepcopy(row["output"])
        payload["line_items"].reverse()
        self.assertFalse(evaluation.score(json.dumps(payload), row["output"])["fields"]["line_items"])

    def test_mocked_full_evaluation_group_counts_and_throughput(self):
        gold_by_input = {row["input"]: row["output"] for row in self.rows}

        def predict(row):
            self.assertEqual(set(row), {"id", "input"})
            return {"raw": json.dumps(gold_by_input[row["input"]]), "completion_tokens": 10,
                    "prompt_tokens": 20, "finish_reason": "stop"}

        sync = MagicMock()
        results = evaluation.evaluate_rows(self.rows, predict, sync)
        report = evaluation.build_report(results, "unit-test-only")
        self.assertEqual(sync.call_count, 120)
        self.assertEqual(report["num_examples"], 60)
        self.assertEqual(report["exact_match_rate"], 1)
        self.assertEqual(report["completion_tokens_total"], 600)
        self.assertGreater(report["tokens_per_second"], 0)
        self.assertEqual({key: value["num_examples"] for key, value in report["per_locale"].items()},
                         dict.fromkeys(("fr-FR", "en-GB", "en-US"), 20))
        self.assertEqual(set(report["per_currency"]), {"EUR", "GBP", "USD"})
        self.assertEqual(set(report["per_template"]), set(map(str, range(1, 10))))
        results[0]["finish_reason"] = "length"
        self.assertEqual(evaluation.build_report(results, "test")["truncation_rate"], 1 / 60)
        with self.assertRaisesRegex(ValueError, "empty"):
            evaluation.build_report([], "test")

    def test_preflight_needs_no_inference_dependencies(self):
        with patch.object(sys, "argv", ["evaluate_finetuned.py", "--check-data-only"]), \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                patch.object(evaluation, "LoRAInference", side_effect=AssertionError("Must not load model")):
            evaluation.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "preflight_only")
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["num_examples"], 60)

    def test_bad_test_split_and_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "test.jsonl"
            rows = copy.deepcopy(self.rows)
            rows[0]["currency"] = "BAD"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "currency"):
                evaluation.load_test_rows(path)
            path.write_text("\n".join(json.dumps(row) for row in self.rows))
            (path.parent / "train.jsonl").write_text(json.dumps(self.rows[0]))
            with self.assertRaisesRegex(ValueError, "leakage"):
                evaluation.load_test_rows(path)

    def make_adapter(self, directory):
        adapter = Path(directory) / "lora_adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": "test/base", "revision": None,
        }))
        (adapter / "adapter_model.safetensors").write_bytes(b"mock-only-not-model-weights")
        (adapter / "tokenizer_config.json").write_text("{}")
        (adapter.parent / "training_manifest.json").write_text(json.dumps({
            "base_model": "test/base", "base_model_revision": "test-revision",
            "training_args": {"max_length": 4096},
            "dataset_version": "international-v3", "status": "completed",
            "dataset_sha256": {name: evaluation.file_hash(ROOT / "data" / f"{name}.jsonl")
                               for name in ("train", "val")},
        }))
        return adapter

    def test_training_manifest_hashes_prevent_wrong_dataset_evaluation(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            adapter = self.make_adapter(directory)
            test_path = ROOT / "data" / "test.jsonl"
            self.assertEqual(evaluation.check_training_provenance(adapter, test_path), ["train", "val"])
            manifest_path = adapter.parent / "training_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["dataset_sha256"]["train"] = "wrong-hash"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "differs"):
                evaluation.check_training_provenance(adapter, test_path)

    def mocked_dependencies(self):
        torch = types.ModuleType("torch")
        torch.cuda = MagicMock()
        torch.cuda.is_available.return_value = True
        torch.cuda.is_bf16_supported.return_value = True
        torch.bfloat16, torch.float16, torch.float32 = "bf16", "fp16", "fp32"
        torch.inference_mode = contextlib.nullcontext
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = MagicMock()
        transformers.AutoModelForCausalLM.from_pretrained.return_value.peft_config = None
        transformers.AutoModelForCausalLM.from_pretrained.return_value.config._commit_hash = "resolved-test-revision"
        transformers.AutoTokenizer = MagicMock()
        transformers.BitsAndBytesConfig = MagicMock()
        peft = types.ModuleType("peft")
        peft.PeftModel = MagicMock()
        return {"torch": torch, "transformers": transformers, "peft": peft}

    def test_lora_loads_declared_base_revision_and_saved_tokenizer(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            adapter = self.make_adapter(directory)
            modules = self.mocked_dependencies()
            with patch.dict(sys.modules, modules):
                inference = evaluation.LoRAInference(adapter, "cuda", True, 4096, 2048)
            transformers = modules["transformers"]
            args, kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args
            self.assertEqual(args, ("test/base",))
            self.assertEqual(kwargs["revision"], "test-revision")
            self.assertIn("quantization_config", kwargs)
            modules["peft"].PeftModel.from_pretrained.assert_called_once_with(
                transformers.AutoModelForCausalLM.from_pretrained.return_value, str(adapter), is_trainable=False,
            )
            transformers.AutoTokenizer.from_pretrained.assert_called_once_with(str(adapter), trust_remote_code=False)
            self.assertEqual(inference.metadata["base_model"], "test/base")
            with self.assertRaisesRegex(ValueError, "exceeds trained"):
                evaluation.LoRAInference(adapter, "cuda", True, 8192, 2048)
            with self.assertRaisesRegex(ValueError, "requires CUDA"):
                evaluation.LoRAInference(adapter, "cpu", True, 4096, 2048)
            (adapter / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "adapter_model"):
                evaluation.check_adapter(adapter)

    def test_raw_base_is_nf4_without_peft_or_adapter_tokenizer(self):
        modules = self.mocked_dependencies()
        with patch.dict(sys.modules, modules):
            inference = evaluation.BaseInference("cuda", True, 4096, 2048)
        transformers = modules["transformers"]
        args, kwargs = transformers.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(args, ("unsloth/Qwen2.5-3B-Instruct",))
        self.assertIn("quantization_config", kwargs)
        transformers.BitsAndBytesConfig.assert_called_once_with(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bf16", bnb_4bit_use_double_quant=True,
        )
        modules["peft"].PeftModel.from_pretrained.assert_not_called()
        self.assertIs(inference.model, transformers.AutoModelForCausalLM.from_pretrained.return_value)
        transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
            evaluation.BASE_MODEL, trust_remote_code=False,
        )
        self.assertFalse(inference.metadata["adapter_applied"])
        self.assertNotIn("adapter_path", inference.metadata)
        self.assertEqual(inference.metadata["resolved_base_revision"], "resolved-test-revision")
        self.assertIs(evaluation.BaseInference.predict, evaluation.LoRAInference.predict)

    def test_raw_base_rejects_accidental_adapters_and_unavailable_cuda(self):
        modules = self.mocked_dependencies()
        modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value.peft_config = {"unexpected": {}}
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(ValueError, "unexpectedly attached"):
            evaluation.BaseInference("cuda", True, 4096, 2048)
        modules["torch"].cuda.is_available.return_value = False
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            evaluation.BaseInference("cuda", True, 4096, 2048)

    def prediction(self, raw, finish_reason="stop"):
        return {"raw": raw, "finish_reason": finish_reason,
                "completion_tokens": 10, "prompt_tokens": 20}

    def test_correction_uses_original_prompt_error_and_no_gold(self):
        backend = MagicMock()
        initial = self.prediction('{"supplier_name":')
        corrected = self.prediction(json.dumps(self.gold))
        backend.predict.side_effect = [initial, corrected]
        row = {"id": "unit-test", "input": "Original invoice"}
        with self.assertLogs(level="WARNING"):
            result = evaluation.predict_with_correction(backend, row)
        self.assertEqual(backend.predict.call_count, 2)
        initial_messages = backend.predict.call_args_list[0].kwargs["messages"]
        corrected_messages = backend.predict.call_args_list[1].kwargs["messages"]
        self.assertEqual(initial_messages, evaluation.build_messages(row))
        self.assertEqual(corrected_messages[:-2], initial_messages)
        self.assertEqual(corrected_messages[-2]["content"], initial["raw"])
        self.assertIn("Le JSON précédent est invalide", corrected_messages[-1]["content"])
        self.assertEqual(result["raw"], corrected["raw"])
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["completion_tokens"], 20)
        self.assertEqual(result["prompt_tokens"], 40)
        self.assertEqual(result["attempts"], [initial, corrected])
        self.assertFalse(result["arithmetic_warnings"])
        with self.assertLogs(level="WARNING"):
            backend.predict.side_effect = [initial, corrected]
            rows = evaluation.evaluate_rows(self.rows[:1], lambda row: evaluation.predict_with_correction(backend, row))
        report = evaluation.build_report(rows, "unit-test-only")
        self.assertEqual(report["initial_json_validity_rate"], 0)
        self.assertEqual(report["json_validity_rate"], 1)
        self.assertEqual(report["correction_rate"], 1)

    def test_correction_arithmetic_warnings_remain_failures_for_scoring(self):
        invalid = copy.deepcopy(self.gold)
        invalid["line_items"][0]["quantity"] += 5
        raw = json.dumps(invalid)
        backend = MagicMock()
        backend.predict.side_effect = [self.prediction(raw), self.prediction(raw)]
        with self.assertLogs(level="WARNING"):
            result = evaluation.predict_with_correction(backend, {"id": "test", "input": "invoice"})
        self.assertEqual(backend.predict.call_count, 2)
        self.assertTrue(result["arithmetic_warnings"])
        self.assertFalse(evaluation.score(result["raw"], invalid)["valid"])
        self.assertIn(
            "Priorité de correction arithmétique",
            backend.predict.call_args.kwargs["messages"][-1]["content"],
        )

    def test_hallucinated_quantity_retries_with_operands_line_and_invoice_totals(self):
        corrected = {
            "supplier_name": "Test supplier", "invoice_date": "2026-10-21",
            "total_ht": 5716.77, "total_ttc": 6860.12,
            "line_items": [
                {"description": "Maintenance préventive", "quantity": 7.2,
                 "unit_price": 95.07, "total_ht": 684.50},
                {"description": "Service", "quantity": 1,
                 "unit_price": 5032.27, "total_ht": 5032.27},
            ],
        }
        invalid = copy.deepcopy(corrected)
        invalid["line_items"][0]["quantity"] = 142
        with self.assertRaises(evaluation.ValidationError) as validation:
            evaluation.Invoice.model_validate(invalid)
        backend = MagicMock()
        backend.predict.side_effect = [
            self.prediction(json.dumps(invalid)), self.prediction(json.dumps(corrected)),
        ]
        with self.assertLogs(level="WARNING"):
            result = evaluation.predict_with_correction(
                backend, {"id": "test", "input": "Maintenance préventive | 7,2 | 95,07 | 684,50"},
            )
        self.assertEqual(backend.predict.call_count, 2)
        prompt = backend.predict.call_args.kwargs["messages"][-1]["content"]
        self.assertIn(str(validation.exception), prompt)
        self.assertIn("Erreur arithmétique détectée", prompt)
        self.assertIn("142.0 * 95.07", prompt)
        self.assertIn("684.5", prompt)
        self.assertIn("5716.77", prompt)
        self.assertIn("ré-extrais la ligne exacte", prompt)
        self.assertTrue(evaluation.score(result["raw"], corrected)["exact"])
        self.assertEqual(result["arithmetic_warnings"], [])

    def test_invoice_sum_mismatch_automatically_triggers_correction(self):
        invalid = {**self.gold, "total_ht": self.gold["total_ht"] - 10}
        backend = MagicMock()
        backend.predict.side_effect = [
            self.prediction(json.dumps(invalid)), self.prediction(json.dumps(self.gold)),
        ]
        with self.assertLogs(level="WARNING"):
            result = evaluation.predict_with_correction(backend, {"id": "test", "input": "invoice"})
        self.assertEqual(backend.predict.call_count, 2)
        prompt = backend.predict.call_args.kwargs["messages"][-1]["content"]
        self.assertIn("somme des total_ht", prompt)
        self.assertIn("total_ht global", prompt)
        self.assertTrue(evaluation.score(result["raw"], self.gold)["exact"])

    def test_failed_correction_keeps_structural_fallback_with_explicit_error(self):
        invalid = {**self.gold, "total_ht": self.gold["total_ht"] + 1}
        raw = json.dumps(invalid)
        for second in (self.prediction("broken"), self.prediction("cut", "length"),
                       RuntimeError("generation failed")):
            backend = MagicMock()
            backend.predict.side_effect = [self.prediction(raw), second]
            with self.subTest(second=second), self.assertLogs(level="WARNING"):
                result = evaluation.predict_with_correction(backend, {"id": "test", "input": "invoice"})
            self.assertEqual(result["raw"], raw)
            self.assertEqual(result["attempt_count"], 2)
            self.assertTrue(result["validation_errors"] or result["operational_error"])
            self.assertTrue(result["arithmetic_warnings"])
            self.assertEqual(backend.predict.call_count, 2)
            if isinstance(second, dict) and second["finish_reason"] == "length":
                scored = {
                    **result, **evaluation.score(result["raw"], self.gold),
                    **{key: self.rows[0][key] for key in ("format", "locale", "currency", "template_id")},
                    "latency_ms": 1,
                }
                report = evaluation.build_report([scored], "test")
                self.assertEqual(report["truncation_rate"], 0)
                self.assertEqual(report["truncated_attempt_rate"], 1)

    def test_two_invalid_responses_never_trigger_a_third_pass(self):
        backend = MagicMock()
        backend.predict.side_effect = [self.prediction("broken first"), self.prediction("broken second")]
        with self.assertLogs(level="WARNING"):
            result = evaluation.predict_with_correction(backend, {"id": "test", "input": "invoice"})
        self.assertEqual(backend.predict.call_count, 2)
        self.assertEqual(result["raw"], "broken second")
        self.assertFalse(evaluation.score(result["raw"], self.gold)["exact"])
        self.assertTrue(result["validation_errors"])

    def test_no_correction_after_valid_output_or_truncation(self):
        for initial in (self.prediction(json.dumps(self.gold)), self.prediction("cut", "length")):
            backend = MagicMock()
            backend.predict.return_value = initial
            result = evaluation.predict_with_correction(backend, {"id": "test", "input": "invoice"})
            backend.predict.assert_called_once()
            self.assertFalse(result["correction_attempted"])
            self.assertEqual(result["attempt_count"], 1)

    def test_base_only_cli_is_disjoint_from_adapter_and_cpu(self):
        for args in (["--base-only", "--adapter-path", "no-adapter"], ["--base-only", "--device", "cpu"]):
            with patch.object(sys, "argv", ["evaluate_finetuned.py", *args]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluation.main()

    def test_base_only_full_split_cli_persists_real_backend_contract(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory) / "baseline-qwen"
            golds = {row["id"]: row["output"] for row in self.rows}
            argv = ["evaluate_finetuned.py", "--base-only", "--self-correct",
                    "--output-dir", str(output), "--warmup", "0"]
            with patch.object(sys, "argv", argv), patch.object(evaluation, "BaseInference") as backend, \
                    patch.object(evaluation, "LoRAInference") as lora, \
                    contextlib.redirect_stdout(io.StringIO()):
                backend.return_value.metadata = {"model": "mock-base-unit-test", "adapter_applied": False}
                backend.return_value.predict.side_effect = (
                    lambda row, *, messages: self.prediction(json.dumps(golds[row["id"]]))
                )
                evaluation.main()
            backend.assert_called_once_with("cuda", True, 4096, 2048)
            lora.assert_not_called()
            self.assertEqual(backend.return_value.predict.call_count, 60)
            report = json.loads((output / "metrics.json").read_text())
            self.assertTrue(report["full_test"])
            self.assertEqual(report["num_examples"], 60)
            self.assertFalse(report["adapter_applied"])
            self.assertTrue(report["generation"]["self_correction"])
            self.assertEqual(report["test_sha256"], evaluation.file_hash(ROOT / "data/test.jsonl"))
            self.assertEqual(len((output / "predictions.jsonl").read_text().splitlines()), 60)

    def test_lora_generation_slices_prompt_and_checks_budget(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            adapter = self.make_adapter(directory)
            modules = self.mocked_dependencies()
            with patch.dict(sys.modules, modules):
                inference = evaluation.LoRAInference(adapter, "cuda", False, 4096, 2048)
            inputs = MagicMock()
            inputs.to.return_value = inputs
            inputs.__getitem__.return_value.shape = (1, 10)
            inference.tokenizer.apply_chat_template.return_value = inputs
            inference.model.config.max_position_embeddings = 4096
            inference.model.generation_config.eos_token_id = [2]
            inference.tokenizer.pad_token_id = 0
            inference.tokenizer.decode.return_value = '{"test":1}'
            inference.model.generate.return_value = [[0] * 10 + [5, 6, 2]]
            result = inference.predict({"input": "invoice", "id": "test"})
            self.assertEqual(result["completion_tokens"], 3)
            self.assertEqual(result["finish_reason"], "stop")
            inference.tokenizer.decode.assert_called_once_with([5, 6, 2], skip_special_tokens=True)
            self.assertEqual(inference.model.generate.call_args.kwargs["pad_token_id"], 0)
            self.assertIs(inference.model.generate.call_args.kwargs["do_sample"], False)
            self.assertEqual(
                inference.tokenizer.apply_chat_template.call_args.args[0],
                evaluation.build_messages({"input": "invoice", "id": "test"}),
            )
            messages = [{"role": "system", "content": "Live numerical precision prompt"},
                        {"role": "user", "content": "Original invoice"}]
            inference.predict({"input": "invoice", "id": "test"}, messages=messages)
            self.assertEqual(inference.tokenizer.apply_chat_template.call_args.args[0], messages)
            inputs.__getitem__.return_value.shape = (1, 3000)
            with self.assertRaisesRegex(ValueError, "exceeds context"):
                inference.predict({"input": "invoice", "id": "test"})

    def test_gguf_backend_preserved(self):
        llm = MagicMock()
        llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {},
        }
        llm.tokenize.return_value = [1, 2]
        result = evaluation.generate(llm, self.rows[0], 1024)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(llm.create_chat_completion.call_args.kwargs["max_tokens"], 1024)
        self.assertEqual(llm.create_chat_completion.call_args.kwargs["temperature"], 0.0)

    def test_cli_writes_auditable_outputs_without_overwriting(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            adapter = self.make_adapter(directory)
            output = Path(directory) / "evaluation"
            argv = ["evaluate_finetuned.py", "--adapter-path", str(adapter), "--limit", "1",
                    "--output-dir", str(output), "--warmup", "0"]
            with patch.object(sys, "argv", argv), patch.object(evaluation, "LoRAInference") as backend, \
                    contextlib.redirect_stdout(io.StringIO()):
                backend.return_value.metadata = {"model": "mock-unit-test"}
                backend.return_value.predict.return_value = {
                    "raw": "invalid", "completion_tokens": 1, "prompt_tokens": 10, "finish_reason": "length",
                }
                evaluation.main()
                metrics = json.loads((output / "metrics.json").read_text())
                self.assertEqual(metrics["num_examples"], 1)
                self.assertFalse(metrics["full_test"])
                self.assertEqual(metrics["exact_match_rate"], 0)
                predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
                self.assertIsNotNone(predictions[0]["parse_error"])
                self.assertEqual(predictions[0]["raw"], "invalid")
                with self.assertRaisesRegex(FileExistsError, "overwrite"):
                    evaluation.main()


if __name__ == "__main__":
    unittest.main()
