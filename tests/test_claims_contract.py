"""Three-field public format checks; no simulator or scientific truths imported."""
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_claims import (  # noqa: E402
    ClaimsValidationError,
    FIELD_NAMES,
    SCHEMA,
    SCHEMA_ID,
    SCHEMA_VERSION,
    canonical_json,
    load_claims,
    parse_claims,
    validate_claims,
)


EXAMPLE = ROOT / "docs/contracts/claims.example.json"
SCRIPT = ROOT / "scripts/validate_claims.py"
REMOVED_FIELDS = (
    "field_type", "radial_law", "radial_exponent", "length_scale", "period",
    "background", "drift_direction_deg", "background_strength", "parameter_scaling",
)


class ClaimsContractTests(unittest.TestCase):
    def setUp(self):
        self.valid = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def assert_issues(self, operation, *expected):
        with self.assertRaises(ClaimsValidationError) as raised:
            operation()
        actual = [(error.code, error.pointer) for error in raised.exception.errors]
        for issue in expected:
            self.assertIn(issue, actual)
        return raised.exception

    def number_text(self, token):
        return json.dumps(self.valid).replace('"dynamics_order": 1', '"dynamics_order": ' + token)

    def cli(self, *arguments, cwd=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *map(str, arguments)],
            cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        )

    def test_schema_and_example_define_only_three_scientific_fields(self):
        self.assertEqual(FIELD_NAMES, ("dynamics_order", "hidden_matter", "time_dependence"))
        self.assertEqual(tuple(SCHEMA["properties"]), FIELD_NAMES)
        self.assertEqual(tuple(self.valid), FIELD_NAMES)
        self.assertFalse(SCHEMA["additionalProperties"])
        self.assertEqual(SCHEMA_VERSION, "4.0.0")
        self.assertEqual(SCHEMA_ID, "urn:discoverphysics:claims:4.0.0")
        self.assertEqual(SCHEMA["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(SCHEMA["properties"]["dynamics_order"]["enum"], [1, 2])
        self.assertEqual(SCHEMA["properties"]["hidden_matter"]["type"], "boolean")
        self.assertEqual(SCHEMA["properties"]["time_dependence"]["type"], "boolean")
        self.assertNotIn("enum", SCHEMA["properties"]["time_dependence"])
        self.assertEqual(load_claims(EXAMPLE), self.valid)

    def test_all_eight_allowed_combinations_pass_without_scientific_truth_checks(self):
        combinations = list(product((1, 2), (False, True), (False, True)))
        self.assertEqual(len(combinations), 8)
        for order, hidden, temporal in combinations:
            with self.subTest(order=order, hidden=hidden, temporal=temporal):
                data = dict(zip(FIELD_NAMES, (order, hidden, temporal)))
                normalized = validate_claims(data)
                self.assertEqual(normalized, data)
                self.assertEqual(parse_claims(json.dumps(data)), data)
                self.assertIs(type(normalized["hidden_matter"]), bool)
                self.assertIs(type(normalized["time_dependence"]), bool)

    def test_integer_decimal_and_exponent_spellings_are_equivalent(self):
        for token, expected in (("1", 1), ("1.0", 1), ("1e0", 1), ("10e-1", 1), ("2", 2), ("2.0", 2), ("2E+0", 2), ("200e-2", 2)):
            with self.subTest(token=token):
                normalized = parse_claims(self.number_text(token))
                self.assertIs(type(normalized["dynamics_order"]), int)
                self.assertEqual(normalized["dynamics_order"], expected)
        for value in (1, 1.0, 2, 2.0):
            data = dict(self.valid, dynamics_order=value)
            self.assertIs(type(validate_claims(data)["dynamics_order"]), int)

    def test_out_of_enum_integers_are_rejected(self):
        for value in (-2, -1, 0, -0.0, 3, 100, int(sys.float_info.max)):
            with self.subTest(value=value):
                data = dict(self.valid, dynamics_order=value)
                self.assert_issues(lambda: validate_claims(data), ("invalid_enum", "/dynamics_order"))

    def test_noninteger_values_cannot_round_into_the_allowed_integers(self):
        for value in (0.5, 1.5, math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0), math.nextafter(2.0, 1.0), math.nextafter(2.0, 3.0)):
            with self.subTest(value=value):
                self.assert_issues(lambda: validate_claims(dict(self.valid, dynamics_order=value)), ("invalid_type", "/dynamics_order"))
        for token in ("1.00000000000000000001", "1.99999999999999999999", "2.00000000000000000001", "1e-999"):
            with self.subTest(token=token):
                self.assert_issues(lambda: parse_claims(self.number_text(token)), ("invalid_type", "/dynamics_order"))

    def test_dynamics_order_rejects_booleans_strings_null_arrays_and_objects(self):
        for value in (True, False, "1", "2.0", None, [], {}):
            with self.subTest(value=value):
                self.assert_issues(lambda: validate_claims(dict(self.valid, dynamics_order=value)), ("invalid_type", "/dynamics_order"))

    def test_hidden_matter_accepts_only_json_booleans(self):
        for value in (0, 1, 1.0, "true", "false", None, [], {}):
            with self.subTest(value=value):
                self.assert_issues(lambda: validate_claims(dict(self.valid, hidden_matter=value)), ("invalid_type", "/hidden_matter"))

    def test_time_dependence_accepts_only_json_booleans_without_legacy_conversion(self):
        for value in ("none", "periodic", "other", "", "NONE", "true", "false", "null", 0, 1, 0.0, 1.0, None, [], {}):
            with self.subTest(value=value):
                data = dict(self.valid, time_dependence=value)
                self.assert_issues(lambda: validate_claims(data), ("invalid_type", "/time_dependence"))
                self.assert_issues(lambda: parse_claims(json.dumps(data)), ("invalid_type", "/time_dependence"))

    def test_every_field_is_required_and_none_is_nullable(self):
        for name in FIELD_NAMES:
            data = deepcopy(self.valid)
            del data[name]
            self.assert_issues(lambda: validate_claims(data), ("missing_field", "/" + name))
            self.assert_issues(lambda: validate_claims(dict(self.valid, **{name: None})), ("invalid_type", "/" + name))
        error = self.assert_issues(lambda: validate_claims({}), *(("missing_field", "/" + name) for name in FIELD_NAMES))
        self.assertEqual(len(error.errors), 3)

    def test_removed_fields_and_version_metadata_are_rejected(self):
        for name in (*REMOVED_FIELDS, "coupling", "ballast_exponent", "schema_version", "scaling_keys"):
            with self.subTest(field=name):
                data = dict(self.valid, **{name: 0})
                self.assert_issues(lambda: validate_claims(data), ("unknown_field", "/" + name))

    def test_unknown_keys_use_escaped_json_pointers(self):
        data = dict(self.valid, **{"x/y~z": 0})
        self.assert_issues(lambda: validate_claims(data), ("unknown_field", "/x~1y~0z"))

    def test_duplicate_root_keys_are_rejected_instead_of_overwritten(self):
        for key, first, second in (("dynamics_order", "1", "2"), ("hidden_matter", "true", "false"), ("time_dependence", "false", "true")):
            text = json.dumps(self.valid).replace(f'"{key}": {first}', f'"{key}": {first}, "{key}": {second}')
            with self.subTest(field=key):
                self.assert_issues(lambda: parse_claims(text), ("duplicate_key", "/" + key))

    def test_duplicate_keys_in_invalid_nested_input_still_report_full_paths(self):
        text = '{"extra":[{"x/y~z":1,"x/y~z":2}]}'
        self.assert_issues(lambda: parse_claims(text), ("duplicate_key", "/extra/0/x~1y~0z"))

    def test_nonfinite_tokens_and_overflowing_values_are_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999"):
            with self.subTest(token=token):
                self.assert_issues(lambda: parse_claims(self.number_text(token)), ("non_finite_number", "/dynamics_order"))
        for value in (float("nan"), float("inf"), -float("inf"), 10 ** 309):
            with self.subTest(value=value):
                self.assert_issues(lambda: validate_claims(dict(self.valid, dynamics_order=value)), ("non_finite_number", "/dynamics_order"))

    def test_utf8_bom_is_accepted_for_text_bytes_and_files(self):
        text = json.dumps(self.valid)
        self.assertEqual(parse_claims("\ufeff" + text), self.valid)
        self.assertEqual(parse_claims(b"\xef\xbb\xbf" + text.encode("utf-8")), self.valid)
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            path = Path(directory) / "bom.json"
            path.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
            self.assertEqual(load_claims(path), self.valid)

    def test_invalid_json_encoding_input_and_root_types(self):
        for text in ("{} trailing", "{", "[1,]", "// comment\n{}", '{"dynamics_order":01}', "\ufeff\ufeff{}"):
            with self.subTest(text=text):
                self.assert_issues(lambda: parse_claims(text), ("invalid_json", ""))
        self.assert_issues(lambda: parse_claims(b"\xff"), ("invalid_encoding", ""))
        for value in (None, 1, {}, []):
            self.assert_issues(lambda: parse_claims(value), ("invalid_input", ""))
        for value in (None, 1, True, "claims", []):
            self.assert_issues(lambda: validate_claims(value), ("invalid_type", ""))

    def test_canonicalization_has_fixed_order_and_never_mutates_input(self):
        data = dict(reversed(list(self.valid.items())))
        data["dynamics_order"] = 1.0
        before = deepcopy(data)
        normalized = validate_claims(data)
        self.assertIsNot(normalized, data)
        self.assertEqual(data, before)
        canonical = canonical_json(data)
        self.assertEqual(tuple(json.loads(canonical)), FIELD_NAMES)
        self.assertEqual(parse_claims(canonical), normalized)
        self.assertEqual(canonical_json(parse_claims(canonical)), canonical)
        self.assertEqual(canonical_json(data), canonical_json(self.valid))

    def test_removed_scaling_context_is_not_an_api_argument(self):
        for operation in (
            lambda: validate_claims(self.valid, scaling_keys=[]),
            lambda: parse_claims(json.dumps(self.valid), scaling_keys=[]),
            lambda: load_claims(EXAMPLE, scaling_keys=[]),
            lambda: canonical_json(self.valid, scaling_keys=[]),
        ):
            with self.assertRaises(TypeError):
                operation()

    def test_cli_default_is_read_only_from_an_arbitrary_directory(self):
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            source = Path(directory) / "input.json"
            source.write_text("\ufeff" + json.dumps(dict(reversed(list(self.valid.items())))), encoding="utf-8")
            before = source.read_bytes()
            process = self.cli(source, cwd=directory)
            self.assertEqual(process.returncode, 0, process.stderr)
            response = json.loads(process.stdout)
            self.assertTrue(response["ok"])
            self.assertEqual(response["schema_version"], "4.0.0")
            self.assertEqual(tuple(response["claims"]), FIELD_NAMES)
            self.assertEqual(response["claims"], self.valid)
            self.assertEqual(source.read_bytes(), before)

    def test_cli_output_is_only_the_three_normalized_scientific_fields(self):
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            source = Path(directory) / "input.json"
            destination = Path(directory) / "normalized.json"
            source.write_text(self.number_text("1e0"), encoding="utf-8")
            before = source.read_bytes()
            process = self.cli(source, "--output", destination)
            self.assertEqual(process.returncode, 0, process.stderr)
            response = json.loads(process.stdout)
            saved = json.loads(destination.read_text(encoding="utf-8"))
            self.assertEqual(response["claims"], saved)
            self.assertEqual(tuple(saved), FIELD_NAMES)
            self.assertIs(type(saved["dynamics_order"]), int)
            self.assertNotIn("schema_version", saved)
            self.assertEqual(source.read_bytes(), before)

    def test_cli_validation_failure_is_json_and_preserves_previous_output(self):
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            source = Path(directory) / "invalid.json"
            destination = Path(directory) / "last-valid.json"
            source.write_text("{}", encoding="utf-8")
            destination.write_text("previous valid draft", encoding="utf-8")
            process = self.cli(source, "--output", destination)
            self.assertEqual(process.returncode, 2, process.stderr)
            response = json.loads(process.stdout)
            self.assertFalse(response["ok"])
            self.assertEqual(response["schema_version"], "4.0.0")
            self.assertEqual(response["errors"][0]["pointer"], "/dynamics_order")
            self.assertEqual(destination.read_text(encoding="utf-8"), "previous valid draft")
            self.assertEqual(source.read_text(encoding="utf-8"), "{}")

    def test_cli_removed_scaling_option_is_a_machine_readable_usage_error(self):
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            destination = Path(directory) / "last-valid.json"
            destination.write_text("previous valid draft", encoding="utf-8")
            process = self.cli(EXAMPLE, "--scaling-keys", "source.p1", "--output", destination)
            self.assertEqual(process.returncode, 2, process.stderr)
            self.assertEqual(json.loads(process.stdout)["errors"][0]["code"], "usage_error")
            self.assertEqual(destination.read_text(encoding="utf-8"), "previous valid draft")

    def test_cli_missing_input_bad_arguments_and_write_failures_have_json_errors(self):
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            missing = Path(directory) / "missing.json"
            output = Path(directory) / "missing-parent" / "output.json"
            for arguments, expected in (((missing,), "io_error"), ((), "usage_error"), ((EXAMPLE, "--output", output), "io_error")):
                process = self.cli(*arguments)
                self.assertEqual(process.returncode, 2, process.stderr)
                self.assertEqual(json.loads(process.stdout)["errors"][0]["code"], expected)

    def test_cli_can_report_a_lone_surrogate_in_an_unknown_key(self):
        with tempfile.TemporaryDirectory(prefix="claims-contract-") as directory:
            source = Path(directory) / "invalid-key.json"
            source.write_text(json.dumps(dict(self.valid, **{"\ud800": 1})), encoding="utf-8")
            process = self.cli(source)
            self.assertEqual(process.returncode, 2, process.stderr)
            response = json.loads(process.stdout)
            self.assertIn(("unknown_field", "/\ud800"), [(error["code"], error["pointer"]) for error in response["errors"]])


if __name__ == "__main__":
    unittest.main()
