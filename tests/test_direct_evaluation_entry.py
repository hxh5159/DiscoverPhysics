"""Exercise the public delegation CLI without models or private physics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/evaluate_submission.py'


class DirectEvaluationEntryTests(unittest.TestCase):
    def test_missing_runtime_stops_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/'results'
            done = subprocess.run([sys.executable, str(ENTRY), '--organizer-root', str(root/'missing'),
                                   '--project-root', str(root), '--output', str(output)],
                                  capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertEqual(json.loads(done.stderr)['error'], 'organizer_runtime_missing')
            self.assertFalse(output.exists())

    def test_delegates_current_project_manifest_environment_and_actual_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            team, runtime = root/'team with spaces', root/'external-runtime'
            team.mkdir(); runtime.mkdir()
            receipt = root/'receipt.json'
            # An explicitly synthetic external runner, not a simulator or model.
            (runtime/'direct_project_eval.py').write_text(
                'import json,os,sys\nfrom pathlib import Path\n'
                'Path(os.environ["SHORE_TEST_RECEIPT"]).write_text(json.dumps('
                '{"argv":sys.argv[1:],"cwd":str(Path.cwd()),"env":os.environ["SHORE_MODEL_PYTHON"]}))\n'
                'raise SystemExit(7)\n', encoding='utf-8')
            manifest = team/'submission.json'
            manifest.write_text('{}', encoding='utf-8')
            profile = root/'scientific-profile.json'
            output = root/'organizer-results'
            env = dict(os.environ, SHORE_TEST_RECEIPT=str(receipt), SHORE_MODEL_PYTHON='model-python-preserved')
            done = subprocess.run([sys.executable, str(ENTRY), '--organizer-root', str(runtime),
                '--manifest', 'submission.json', '--output', str(output), '--profile', str(profile),
                '--agent-python', '/opt/team-env/bin/python', '--check-only'], cwd=team,
                env=env, capture_output=True, text=True)
            self.assertEqual(done.returncode, 7, done.stderr)
            saved = json.loads(receipt.read_text())
            self.assertEqual(Path(saved['cwd']), team)
            self.assertEqual(saved['env'], 'model-python-preserved')
            self.assertEqual(saved['argv'], ['--project-root', str(team.resolve()), '--manifest', str(manifest.resolve()),
                '--output', str(output.resolve()), '--profile', str(profile.resolve()),
                '--agent-python', '/opt/team-env/bin/python', '--check-only'])
            self.assertFalse(output.exists())

    def test_entry_contains_no_private_runtime_import_or_network_dependency(self):
        source = ENTRY.read_text(encoding='utf-8')
        self.assertNotIn('import contest_', source)
        self.assertNotIn('import requests', source)
        self.assertIn('subprocess.run(command, cwd=project', source)


if __name__ == '__main__':
    unittest.main()
