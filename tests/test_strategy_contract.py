"""Metadata-only strategy and multi-model path checks, without model loading."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rift.strategy import (check_strategy_files, load_model_registry,
                           load_strategy_submission, model_binding)
from rift.submission import SubmissionError, validate_manifest, check_submission_files
from scripts.prepare_strategy import prepare

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class StrategyContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.agent = self.root/"agent"
        self.agent.mkdir()
        (self.agent/"__init__.py").write_text("", encoding="utf-8")
        (self.agent/"run.py").write_text("raise RuntimeError('MUST_NOT_EXECUTE')", encoding="utf-8")
        self.descriptor = {"schema_version": "1.0.0", "team_id": "team01",
                           "submission_id": "run01", "version": "1"}
        write_json(self.agent/"strategy.json", self.descriptor)
        self.models_file = self.root/"models.json"
        for name in ("planner-v2", "solver-v3"):
            path = self.root/name
            write_json(path/"config.json", {})
            write_json(path/"tokenizer.json", {})
            (path/"model.safetensors").write_bytes(b"fixture-weight-not-a-real-tensor")
        write_json(self.root/"solver-adapter"/"adapter_config.json", {"peft_type": "LORA", "r": 2})
        (self.root/"solver-adapter"/"adapter_model.safetensors").write_bytes(b"fixture-adapter")
        self.registry = {"schema_version": "1.0.0", "primary_model_id": "solver", "models": [
            {"id": "planner", "role": "experiment-planner", "base_model": "27b",
             "path": "planner-v2", "adaptation": "sft"},
            {"id": "solver", "role": "law-proposer", "base_model": "27b",
             "path": "solver-v3", "adaptation": "rl", "adapter_path": "solver-adapter"},
        ]}
        write_json(self.models_file, self.registry)

    def test_two_paths_capture_all_changed_models_and_resolve_independent_of_cwd(self):
        with patch("pathlib.Path.cwd", return_value=self.root/"unrelated"):
            registry = load_model_registry(self.models_file, check_files=True)
            manifest = load_strategy_submission(self.agent, self.models_file)
        self.assertEqual(manifest["entrypoint"], ["python", "-m", "agent.run"])
        self.assertEqual([item["id"] for item in manifest["models"]], ["planner", "solver"])
        self.assertEqual(manifest["models"][0]["path"], (self.root/"planner-v2").as_posix())
        self.assertEqual(manifest["models"][1]["adapter_path"], (self.root/"solver-adapter").as_posix())
        self.assertEqual(model_binding(registry)["id"], "solver")
        self.assertEqual(model_binding(registry, "planner")["role"], "experiment-planner")
        self.assertEqual(validate_manifest(manifest), manifest)
        # Existing generic file checks include every secondary adapter too.
        self.assertTrue(check_submission_files(manifest, self.agent)[1]["adapter_checked"])

    def test_duplicate_ids_unknown_primary_and_nonwhite_models_rejected(self):
        for mutate in (
            lambda r: r["models"][1].update(id="planner"),
            lambda r: r.update(primary_model_id="missing"),
            lambda r: r["models"][1].update(base_model="external-model"),
            lambda r: r["models"][1].pop("id"),
        ):
            registry = copy.deepcopy(self.registry)
            mutate(registry)
            write_json(self.models_file, registry)
            with self.assertRaises(SubmissionError):
                load_model_registry(self.models_file)

    def test_relative_traversal_and_network_urls_rejected(self):
        for path in ("../weights", "http://example/model", "folder//weights", "C:\\weights"):
            registry = copy.deepcopy(self.registry)
            registry["models"][1]["path"] = path
            write_json(self.models_file, registry)
            with self.assertRaises(SubmissionError):
                load_model_registry(self.models_file)

    def test_secondary_checkpoint_and_adapter_are_not_silently_skipped(self):
        (self.root/"solver-v3"/"model.safetensors").unlink()
        with self.assertRaises(SubmissionError) as error:
            load_strategy_submission(self.agent, self.models_file)
        self.assertEqual(error.exception.path, "$.models[1].path")
        (self.root/"solver-v3"/"model.safetensors").write_bytes(b"fixture")
        (self.root/"solver-adapter"/"adapter_model.safetensors").unlink()
        with self.assertRaises(SubmissionError) as error:
            load_strategy_submission(self.agent, self.models_file)
        self.assertEqual(error.exception.path, "$.models[1].adapter_path")

    def test_declared_index_shards_must_exist_and_remain_within_checkpoint(self):
        index = self.root/"planner-v2"/"model.safetensors.index.json"
        for shard in ("missing.safetensors", "../solver-v3/model.safetensors"):
            write_json(index, {"weight_map": {"weight": shard}})
            with self.assertRaises(SubmissionError):
                load_model_registry(self.models_file, check_files=True)
        write_json(index, {"weight_map": {"weight": "model.safetensors"}})
        self.assertEqual(len(load_model_registry(self.models_file, check_files=True)["models"]), 2)

    def test_unknown_descriptor_fields_cannot_select_outside_code(self):
        for field, value in (("entrypoint", ["python", "../outside.py"]), ("environment", {"PYTHONPATH": ".."})):
            write_json(self.agent/"strategy.json", dict(self.descriptor, **{field: value}))
            with self.assertRaises(SubmissionError):
                load_strategy_submission(self.agent, self.models_file)

    def test_required_folder_name_and_entrypoint_files_checked(self):
        other = self.root/"other"
        other.mkdir()
        with self.assertRaises(SubmissionError):
            check_strategy_files(other)
        (self.agent/"run.py").unlink()
        with self.assertRaises(SubmissionError):
            check_strategy_files(self.agent)

    @unittest.skipIf(os.name == "nt", "Creating symlinks requires Windows privilege")
    def test_strategy_symlink_escape_is_rejected(self):
        outside = self.root/"outside.py"
        outside.write_text("external code", encoding="utf-8")
        (self.agent/"external.py").symlink_to(outside)
        with self.assertRaises(SubmissionError):
            check_strategy_files(self.agent)

    def test_json_duplicate_keys_rejected_in_registry(self):
        self.models_file.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}', encoding="utf-8")
        with self.assertRaises(SubmissionError):
            load_model_registry(self.models_file)

    def test_binding_returns_detached_data_and_rejects_unknown_id(self):
        registry = load_model_registry(self.models_file)
        bound = model_binding(registry)
        bound["path"] = "changed"
        self.assertNotEqual(bound["path"], registry["models"][1]["path"])
        with self.assertRaises(SubmissionError):
            model_binding(registry, "unknown")

    def test_prepare_requires_explicit_overwrite_and_preserves_inputs(self):
        before = self.models_file.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.agent, self.models_file, team_id="newteam", submission_id="newrun")
        self.assertEqual(self.models_file.read_bytes(), before)
        target = self.root/"renamed-models.json"
        result = prepare(self.agent, target, team_id="newteam", submission_id="newrun",
                         models_from=self.models_file, overwrite=True)
        self.assertEqual(result["model_count"], 2)
        self.assertEqual(self.models_file.read_bytes(), before)
        self.assertEqual(load_strategy_submission(self.agent, target)["team_id"], "newteam")

    def test_prepare_rolls_back_if_strategy_layout_is_invalid(self):
        # An invalid symlink-free tree shape fails after metadata writes, which
        # must restore an existing valid registry and strategy descriptor.
        before_models = self.models_file.read_bytes()
        before_strategy = (self.agent/"strategy.json").read_bytes()
        with patch("scripts.prepare_strategy.load_strategy_submission", side_effect=SubmissionError("$", "fixture")):
            with self.assertRaises(SubmissionError):
                prepare(self.agent, self.models_file, team_id="newteam", submission_id="newrun", overwrite=True)
        self.assertEqual(self.models_file.read_bytes(), before_models)
        self.assertEqual((self.agent/"strategy.json").read_bytes(), before_strategy)

    def test_cli_checks_files_without_importing_submitted_code_or_loading_weights(self):
        result = subprocess.run([sys.executable, str(ROOT/"scripts/validate_strategy.py"),
            "--agent-dir", str(self.agent), "--models-file", str(self.models_file), "--check-files"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["model_count"], 2)
        self.assertFalse(report["code_executed"])
        self.assertFalse(report["model_loaded"])


if __name__ == "__main__":
    unittest.main()
