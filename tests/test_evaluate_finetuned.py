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

    def test_prompt_matches_training_without_gold_or_metadata(self):
        namespace = {}
        notebook = json.loads((ROOT / "notebooks" / "fine_tune_unsloth.ipynb").read_text())
        config = next(cell for cell in notebook["cells"] if cell["id"] == "f3275e63")
        with contextlib.redirect_stdout(io.StringIO()):
            exec("".join(config["source"]), namespace)
        self.assertEqual(evaluation.SYSTEM_PROMPT, namespace["SYSTEM_PROMPT"])
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
