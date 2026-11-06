"""Current score feedback shapes and explicit six-world aggregation metadata."""
import json
from pathlib import Path
import unittest

try:
    from jsonschema import Draft202012Validator
except ImportError:
    Draft202012Validator = None


CONTRACTS = Path(__file__).resolve().parents[1] / "docs" / "contracts"
WORLDS = ["Adam", "Lilith", "Sachiel", "Matriel", "Leliel", "Arael"]
WEIGHTS = dict(zip(WORLDS, [.15, .15, .15, .20, .20, .15]))


def load(name):
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


class EvaluationWeightContractTests(unittest.TestCase):
    def test_current_runtime_scoring_and_tiebreak_share_explicit_weights(self):
        scoring = load("scoring.json")
        runtime = load("runtime.json")
        selection = scoring["evaluation_selection"]
        self.assertEqual(selection["evaluation_worlds"], WORLDS)
        self.assertEqual(runtime["outer_submission"]["world_order"], WORLDS)
        self.assertEqual(selection["evaluation_weights"], WEIGHTS)
        self.assertEqual(runtime["outer_submission"]["evaluation_weights"], WEIGHTS)
        self.assertEqual(scoring["ranking"]["evaluation_weights"], WEIGHTS)
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.)
        # The new groups deliberately cross the original difficulty tiers.
        self.assertEqual(WEIGHTS["Leliel"], .20)
        self.assertEqual(WEIGHTS["Sachiel"], .15)
        self.assertNotIn("tier_weights", scoring["ranking"])
        self.assertNotIn("worlds_per_tier_evaluated", selection)
        self.assertEqual(runtime["resource_guardrails"]["profile"], selection["profile"])
        self.assertEqual(scoring["feedback"]["schema_version"],
                         load("feedback.schema.json")["x-contract-version"])
        # This change must not expand the experimental or model-call allowance.
        self.assertEqual(runtime["session"]["energy"], 10)
        self.assertEqual(runtime["session"]["wall_clock_seconds"], 300)
        self.assertEqual(runtime["session"]["baseline_max_model_turns"], 3)
        self.assertEqual(scoring["prediction"]["mapping_scale"], 10)


@unittest.skipIf(Draft202012Validator is None, "jsonschema is not installed")
class FeedbackSchemaTests(unittest.TestCase):
    def setUp(self):
        schema = load("feedback.schema.json")
        Draft202012Validator.check_schema(schema)
        self.validator = Draft202012Validator(schema)

    def valid(self, worlds):
        return {"total_score": 0, "world_scores": dict.fromkeys(worlds, 0)}

    def test_accepts_exact_current_six_and_complete_legacy_nine(self):
        self.validator.validate(self.valid(WORLDS))
        self.validator.validate(self.valid(load("scoring.json")["evaluation_selection"]["available_worlds"]))

    def test_rejects_old_three_partial_and_mixed_world_sets(self):
        for worlds in (["Tabris", "Ramiel", "Leliel"], WORLDS[:-1],
                       WORLDS + ["Tabris"], WORLDS[:-1] + ["Lilin"]):
            with self.subTest(worlds=worlds):
                self.assertTrue(list(self.validator.iter_errors(self.valid(worlds))))

    def test_does_not_expose_score_components_or_out_of_range_values(self):
        for key, value in (("prediction", 80), ("weighted_saving", 2), ("claims", {})):
            payload = self.valid(WORLDS)
            payload[key] = value
            self.assertTrue(list(self.validator.iter_errors(payload)))
        for bad in (-1, 101, {"total": 50, "law": 15}):
            payload = self.valid(WORLDS)
            payload["world_scores"]["Adam"] = bad
            self.assertTrue(list(self.validator.iter_errors(payload)))


if __name__ == "__main__":
    unittest.main()
