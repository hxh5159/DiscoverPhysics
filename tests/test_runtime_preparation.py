"""Offline build identities and supplied-model manifest preparation boundaries."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from scripts.build_runtime_images import plan, build
from scripts.make_submission import make_manifest
from rift.model_registry import catalog
from rift.submission import validate_manifest


class RuntimePreparationTests(unittest.TestCase):
    def test_every_supplied_variant_yields_a_consistent_small_manifest(self):
        for item in catalog()['models']:
            with self.subTest(model=item['id']):
                result=make_manifest(item['id'],'team01','run01')
                self.assertEqual(result['models'][0]['base_model'],item['base_model'])
                self.assertEqual(result['models'][0]['path'],result['entrypoint'][-1])
                self.assertEqual(result['models'][0]['path'],catalog()['model_root']+'/'+item['directory'])
                self.assertEqual(result['entrypoint'][result['entrypoint'].index('--max-turns')+1], '3')
                self.assertEqual(validate_manifest(result),result)

    def test_manifest_preserves_explicit_turn_override_and_rejects_invalid_caps(self):
        for maximum in (0, 3, 5):
            manifest = make_manifest('27b', 'team01', 'run01', max_turns=maximum)
            command = manifest['entrypoint']
            self.assertEqual(command[command.index('--max-turns')+1], str(maximum))
        for maximum in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                make_manifest('27b', 'team01', 'run01', max_turns=maximum)

    def test_custom_actual_model_root_and_training_checkpoint(self):
        self.assertEqual(make_manifest('27b','team01','run01',model_root='/inspire/team/models')['models'][0]['path'],
                         '/inspire/team/models/Qwen3.8-27B')
        result=make_manifest('27b','team01','run02',model_path='checkpoints/v2',adaptation='sft',
                             python='/opt/team-env/bin/python')
        self.assertEqual(result['entrypoint'][0],'/opt/team-env/bin/python')
        self.assertEqual(result['models'][0]['path'],'checkpoints/v2')
        self.assertEqual(result['models'][0]['adaptation'],'sft')

    def test_image_plan_is_local_and_does_not_copy_the_repository(self):
        settings=plan('already-imported:local',manifest='submission.json')
        for key in ('agent','candidate'):
            self.assertIn('--pull=false',settings[key])
            self.assertIn('--network=none',settings[key])
        root=Path(__file__).resolve().parents[1]
        recipe=(root/'examples/Dockerfile.agent').read_text()
        self.assertNotIn('COPY . ',recipe)
        self.assertIn('SHORE_MODEL_PYTHON=',recipe)
        self.assertIn('scripts/validate_claims.py',recipe)
        self.assertIn('docs/contracts/claims.schema.json',recipe)

    def test_existing_receipt_prevents_any_container_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'receipt.json'
            output.write_text('keep-existing')
            runner=Mock()
            with self.assertRaises(FileExistsError):build(plan('local:base'),output,runner=runner)
            runner.assert_not_called()
            self.assertEqual(output.read_text(),'keep-existing')

    def test_build_receipt_requires_local_import_checks_and_contains_no_weights(self):
        image='sha256:'+'a'*64
        calls=[]
        def runner(argv,**kwargs):
            calls.append(argv)
            if 'inspect' in argv:return subprocess.CompletedProcess(argv,0,image+'\n','')
            return subprocess.CompletedProcess(argv,0,'ready\n','')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'receipt.json'
            receipt=build(plan('local:base'),output,runner=runner)
            self.assertTrue(receipt['cpu_import_checks_passed'])
            self.assertFalse(receipt['gpu_or_model_loaded'])
            self.assertEqual(receipt['agent_image'],image)
            self.assertEqual(json.loads(output.read_text()),receipt)
            runs=[argv for argv in calls if 'run' in argv]
            self.assertEqual(len(runs),2)
            self.assertTrue(all('none' in argv for argv in runs))
            self.assertTrue(all('--gpus' not in argv for argv in calls))


if __name__=='__main__':unittest.main()
