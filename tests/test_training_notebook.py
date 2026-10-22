import ast
import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "fine_tune_unsloth.ipynb"


class TrainingNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        cls.code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]

    def config(self):
        namespace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.cells["f3275e63"], namespace)
        namespace["DATA_DIR_OVERRIDE"] = ROOT / "data"
        return namespace

    def load_data(self, namespace):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.cells["fc5f5679"], namespace)

    def test_all_python_cells_compile(self):
        for cell in self.code_cells:
            code = "\n".join(line for line in "".join(cell["source"]).splitlines() if not line.startswith("%"))
            compile(code, f"{NOTEBOOK.name}:{cell['id']}", "exec")
            self.assertEqual(cell.get("outputs", []), [])

    def test_config_and_current_v3_data(self):
        namespace = self.config()
        self.load_data(namespace)
        self.assertEqual(len(namespace["train_rows"]), 480)
        self.assertEqual(len(namespace["val_rows"]), 60)
        self.assertEqual(set(namespace["DATA_HASHES"]), {"train", "val"})
        self.assertEqual(namespace["MAX_SEQ_LENGTH"], 4096)
        self.assertEqual(namespace["BATCH_SIZE"] * namespace["GRAD_ACCUM"], 8)
        self.assertEqual(namespace["ADAPTER_DIR"].parent, namespace["RESULTS_DIR"])
        self.assertEqual(namespace["RESULTS_DIR"].parent.name, "results")
        self.assertFalse(namespace["RESULTS_DIR"].exists())
        self.assertFalse(namespace["USE_WANDB"])
        self.assertIsNone(namespace["HF_REPO_GGUF"])

    def test_rejects_wrong_version_and_overlap(self):
        namespace = self.config()
        self.load_data(namespace)
        train = namespace["train_rows"]
        val = namespace["val_rows"]
        train[0]["dataset_version"] = "v2"
        namespace["load_jsonl"] = lambda path: train if path.stem == "train" else val
        validation = self.cells["fc5f5679"].split("for name, rows, expected_count, expected_region", 1)[1]
        validation = "for name, rows, expected_count, expected_region" + validation
        with self.assertRaisesRegex(ValueError, "international-v3"):
            exec(validation, namespace)
        train[0]["dataset_version"] = "international-v3"
        val[0]["id"] = train[0]["id"]
        with self.assertRaisesRegex(ValueError, "Fuite"):
            exec(validation, namespace)

    def test_validation_length_is_checked_without_metadata_in_prompt(self):
        namespace = self.config()
        self.load_data(namespace)
        module = types.ModuleType("datasets")

        class Dataset:
            @staticmethod
            def from_list(rows):
                return {"text": [row["text"] for row in rows]}

        class Tokenizer:
            def apply_chat_template(self, messages, tokenize=False):
                text = json.dumps(messages)
                if tokenize:
                    return [0] * (4097 if "TOO_LONG_VALIDATION" in text else 100)
                return text

        module.Dataset = Dataset
        namespace["tokenizer"] = Tokenizer()
        namespace["val_rows"][0]["input"] = "TOO_LONG_VALIDATION"
        with patch.dict(sys.modules, {"datasets": module}), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "val: exemples trop longs"):
                exec(self.cells["9328f9a1"], namespace)
        row = namespace["train_rows"][0]
        messages = namespace["to_messages"](row)
        self.assertEqual(messages[1]["content"], namespace["INSTRUCTION"] + "\n\n" + row["input"])
        self.assertEqual(json.loads(messages[2]["content"]), row["output"])

    def test_trainer_limit_and_save_before_evaluation(self):
        tree = ast.parse(self.cells["5418ad6d"])
        config = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "SFTConfig")
        keywords = {keyword.arg: keyword.value for keyword in config.keywords}
        self.assertEqual(ast.unparse(keywords["max_length"]), "MAX_SEQ_LENGTH")
        self.assertEqual(ast.literal_eval(keywords["per_device_eval_batch_size"]), 1)
        source = self.cells["b7d0b02a"]
        self.assertLess(source.index("trainer.train()"), source.index("model.save_pretrained"))
        self.assertLess(source.index("model.save_pretrained"), source.index("trainer.evaluate()"))
        self.assertIn("adapter_model.safetensors", source)
        self.assertIn("training_manifest.json", source)
        self.assertIn("RESULTS_DIR.parent", self.cells["f137448d"])
        self.assertNotIn("push_to_hub", source)


if __name__ == "__main__":
    unittest.main()
