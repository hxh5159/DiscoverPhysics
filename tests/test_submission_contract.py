"""Manifest, artifact identity and public starter boundary regression checks."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rift.submission import (SubmissionError, WHITELIST_MODELS, artifact_digest,
                             check_submission_files, load_manifest, manifest_digest,
                             parse_manifest, validate_manifest)

spec = importlib.util.spec_from_file_location("public_starter_packager", ROOT / "scripts/package_contest_starter.py")
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def manifest():
    return {"schema_version": "1.0.0", "submission_id": "team01-v1", "team_id": "team01",
            "version": "1.0", "entrypoint": ["python", "agent.py"],
            "models": [{"base_model": "Qwen3.6-35B-A3B", "path": "/models/model"}]}


def public_fixture(root):
    for name in packager.PUBLIC_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "docs/contracts/numerics.json":
            path.write_text(json.dumps({"numerics_version": "1.1", "integrator": "yoshida4",
                                       "dt": .005, "sampling": {"rule": "strict_ticks"},
                                       "dp_reference_review": "SECRET_MARKER_MUST_NOT_BE_PACKAGED"}), encoding="utf8")
        elif name.endswith(".schema.json"):
            path.write_text(json.dumps({"$id": "urn:discoverphysics:example:1.0.0", "type": "object"}), encoding="utf8")
        else:
            path.write_bytes(("public:" + name).encode())


class SubmissionTests(unittest.TestCase):
    def assert_path(self, callable_, path):
        with self.assertRaises(SubmissionError) as context:
            callable_()
        self.assertEqual(context.exception.path, path)
        self.assertIn("message", context.exception.as_dict())

    def test_supplied_example_validates_without_running_model(self):
        result = load_manifest(ROOT / "examples/submission.json")
        self.assertEqual(result["entrypoint"], ["python", "agent.py"])
        self.assertIn(result["models"][0]["base_model"], WHITELIST_MODELS)

    def test_all_published_whitelist_identifiers_accepted(self):
        for name in WHITELIST_MODELS:
            value = manifest()
            value["models"][0]["base_model"] = name
            self.assertEqual(validate_manifest(value)["models"][0]["base_model"], name)

    def test_normalization_detaches_mutable_fields(self):
        source = manifest()
        result = validate_manifest(source)
        result["entrypoint"].append("--changed")
        result["models"][0]["path"] = "changed"
        self.assertEqual(source, manifest())

    def test_identity_independent_of_key_order(self):
        a = manifest()
        b = dict(reversed(list(a.items())))
        self.assertEqual(manifest_digest(a), manifest_digest(b))
        b["version"] = "2"
        self.assertNotEqual(manifest_digest(a), manifest_digest(b))

    def test_bom_and_crlf_json_accepted(self):
        encoded = "\ufeff" + json.dumps(manifest(), indent=2).replace("\n", "\r\n")
        self.assertEqual(parse_manifest(encoded), manifest())

    def test_duplicate_nested_json_keys_rejected(self):
        text = json.dumps(manifest()).replace('"base_model":', '"path":"extra","base_model":')
        self.assert_path(lambda: parse_manifest(text), "$")

    def test_nonstandard_json_and_malformed_text_rejected(self):
        for text in ("{", "NaN", "Infinity", "-Infinity"):
            with self.subTest(text=text):
                self.assert_path(lambda: parse_manifest(text), "$")

    def test_missing_fields_and_untrusted_runtime_controls_rejected(self):
        for field in ("entrypoint", "models"):
            value = manifest()
            del value[field]
            self.assert_path(lambda: validate_manifest(value), "$")
        for key in ("host_mounts", "network", "image", "world_config", "environment"):
            value = manifest()
            value[key] = {"privileged": True}
            self.assert_path(lambda: validate_manifest(value), "$")

    def test_command_string_shell_wrapper_and_control_chars_rejected(self):
        for command in ("python agent.py", ["bash", "-lc", "python agent.py"], [], ["python", "a\n.py"]):
            value = manifest()
            value["entrypoint"] = command
            with self.subTest(command=command), self.assertRaises(SubmissionError):
                validate_manifest(value)

    def test_shell_metacharacters_are_preserved_as_arguments_not_evaluated(self):
        value = manifest()
        value["entrypoint"] += ["--label", "$(not-executed); literal"]
        self.assertEqual(validate_manifest(value)["entrypoint"], value["entrypoint"])

    def test_nonwhitelist_model_or_additional_model_permission_rejected(self):
        for change in ({"base_model": "closed-model"}, {"external_api": "https://invalid.example"}):
            value = manifest()
            value["models"][0].update(change)
            with self.assertRaises(SubmissionError):
                validate_manifest(value)

    def test_host_and_traversing_model_paths_rejected(self):
        for path in ("../weights", "weights/../secret", "/etc/shadow", "/models", "C:\\weights", "http://host/model", "weights//model"):
            value = manifest()
            value["models"][0]["path"] = path
            self.assert_path(lambda: validate_manifest(value), "$.models[0].path")

    def test_derived_model_keeps_whitelisted_base_and_valid_digest(self):
        value = manifest()
        value["models"][0].update(adaptation="sft", revision="team-v2", sha256="A"*64)
        self.assertEqual(validate_manifest(value)["models"][0]["sha256"], "a"*64)
        value["models"][0]["adaptation"] = "external_distillation"
        self.assert_path(lambda: validate_manifest(value), "$.models[0].adaptation")

    def test_single_file_digest_match_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.bin").write_bytes(b"invented-weight-bytes")
            value = manifest()
            value["models"][0].update(path="weights.bin", sha256=hashlib.sha256(b"invented-weight-bytes").hexdigest())
            self.assertTrue(check_submission_files(value, root)[0]["sha256_verified"])
            (root / "weights.bin").write_bytes(b"changed")
            self.assert_path(lambda: check_submission_files(value, root), "$.models[0].sha256")

    def test_directory_digest_stable_across_creation_order_and_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a, b = root / "a", root / "b"
            a.mkdir(); b.mkdir()
            for folder, names in ((a, ["z", "a"]), (b, ["a", "z"])):
                for name in names:
                    (folder / name).write_bytes(name.encode())
            self.assertEqual(artifact_digest(a), artifact_digest(b))
            (b / "a").write_bytes(b"different")
            self.assertNotEqual(artifact_digest(a), artifact_digest(b))

    def test_official_mount_not_accessed_by_local_file_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            checked = check_submission_files(manifest(), temporary)
            self.assertEqual(checked, [{"path": "/models/model", "sha256_verified": False,
                                        "status": "official_mount_check_deferred"}])

    def test_missing_relative_artifact_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            value = manifest()
            value["models"][0]["path"] = "missing"
            self.assert_path(lambda: check_submission_files(value, temporary), "$.models[0].path")

    def test_schema_model_identifiers_match_checker(self):
        schema = json.loads((ROOT / "docs/contracts/submission.schema.json").read_text(encoding="utf8"))
        choices = schema["properties"]["models"]["items"]["properties"]["base_model"]["enum"]
        self.assertEqual(choices, list(WHITELIST_MODELS))

    def test_cli_static_check_has_no_execution_or_scientific_claim(self):
        process = subprocess.run([sys.executable, str(ROOT/"scripts/validate_submission.py"),
                                  str(ROOT/"examples/submission.json")], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertTrue(result["ok"])
        self.assertFalse(result["code_execution_checked"])
        self.assertFalse(result["model_provenance_checked"])
        self.assertFalse(result["scientific_correctness_checked"])

    def test_public_package_allowlist_excludes_private_and_research_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_fixture(root)
            for name in ("AGENTS.md", ".project-memory/truth.py", "ScienceAgent/private.py", "docs/research.pdf"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"SECRET_MARKER_MUST_NOT_BE_PACKAGED")
            output = root / "starter.zip"
            result = packager.build_starter(output, root=root)
            with ZipFile(output) as archive:
                names = set(archive.namelist())
                self.assertEqual(names, set(packager.PUBLIC_FILES) | {"START_HERE.txt", "starter-manifest.json"})
                self.assertFalse(any(b"SECRET_MARKER_MUST_NOT_BE_PACKAGED" in archive.read(n) for n in names))
                inventory = json.loads(archive.read("starter-manifest.json"))["files"]
                for item in inventory:
                    self.assertEqual(item["sha256"], hashlib.sha256(archive.read(item["path"])).hexdigest())
                public_numeric = json.loads(archive.read("docs/contracts/numerics.json"))
                self.assertEqual(public_numeric["dt"], .005)
                self.assertNotIn("dp_reference_review", public_numeric)
                schema = json.loads(archive.read("docs/contracts/claims.schema.json"))
                self.assertEqual(schema["$id"], "urn:shore:example:1.0.0")
            self.assertFalse(result["contains_formal_world_simulators"])

    def test_packager_is_reproducible_and_refuses_implicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_fixture(root)
            a, b = root/"a.zip", root/"b.zip"
            packager.build_starter(a, root=root)
            packager.build_starter(b, root=root)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            with self.assertRaises(FileExistsError):
                packager.build_starter(a, root=root)

    def test_actual_public_starter_runs_standalone_prechecks(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            archive_path = folder / "starter.zip"
            packager.build_starter(archive_path, root=ROOT)
            unpacked = folder / "unpacked"
            with ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                self.assertEqual(names, set(packager.PUBLIC_FILES) | {"START_HERE.txt", "starter-manifest.json"})
                self.assertFalse(any(".project-memory" in name or "ScienceAgent" in name or "AGENTS" in name for name in names))
                for name in names:
                    data = archive.read(name).lower()
                    self.assertNotIn(b"discoverphysics", data, name)
                    self.assertNotIn(b"dp_reference_review", data, name)
                    self.assertNotIn(b"contest_world_definitions", data, name)
                archive.extractall(unpacked)
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            commands = [
                ["scripts/validate_submission.py", "examples/submission.json"],
                ["scripts/validate_law.py", "examples/law.py", "--claims", "docs/contracts/claims.example.json"],
                ["scripts/validate_law.py", "examples/law.py", "--case", "examples/law_case.json", "--execute"],
            ]
            for command in commands:
                process = subprocess.run([sys.executable, "-I", "-X", "utf8", *command], cwd=unpacked,
                                         capture_output=True, text=True, env=env)
                self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
                result = json.loads(process.stdout)
                self.assertTrue(result["ok"])
                self.assertFalse(result["scientific_correctness_checked"])


if __name__ == "__main__":
    unittest.main()
