"""Missing-dependency prefetch tests with tiny injected archives, never model packages."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from scripts.prepare_vllm_dependencies import prepare


MIRROR='http://mirror.example/simple'
OFFICIAL='https://pypi.org/simple'
PAYLOAD=b'tiny dependency archive fixture, not a real model package'


def record(name='fixture',version='1.0.0',host='mirror.example'):
    filename=name+'-'+version+'-py3-none-any.whl'
    return {'metadata':{'name':name,'version':version},'download_info':{
        'url':'https://'+host+'/packages/'+filename,
        'archive_info':{'hashes':{'sha256':hashlib.sha256(PAYLOAD).hexdigest()}}}}


class FakePipeline:
    def __init__(self,reports,*,mirror_download_failure=False,resolution_failure=None,corrupt=False):
        self.reports=iter(reports)
        self.calls=[]
        self.mirror_download_failure=mirror_download_failure
        self.resolution_failure=resolution_failure
        self.corrupt=corrupt
    def __call__(self,python,args,*,index_url,trusted_host,log_dir,fallback_index_url,operation,
                 mirror_only=False):
        self.calls.append({'python':python,'args':list(args),'index':index_url,'operation':operation,
                           'fallback':fallback_index_url,'mirror_only':mirror_only})
        folder=Path(log_dir)/'pip-index-fixture';folder.mkdir(parents=True)
        if operation=='install':
            if self.resolution_failure and index_url==MIRROR:
                (folder/'1.log').write_text(self.resolution_failure,encoding='utf-8');return 1
            report={'version':'1','install':next(self.reports)}
            Path(args[args.index('--report')+1]).write_text(json.dumps(report),encoding='utf-8')
            (folder/'1.log').write_text('Would install fixture',encoding='utf-8');return 0
        if self.mirror_download_failure and index_url==MIRROR:
            (folder/'1.log').write_text('ERROR: HTTP error 404 at http://mirror.example/packages/missing.whl',encoding='utf-8')
            return 1
        requirements=Path(args[args.index('-r')+1]).read_text(encoding='utf-8')
        target=Path(args[args.index('--dest')+1])
        for line in requirements.splitlines():
            url=line.split(' @ ',1)[1].split(' --hash=',1)[0]
            name=url.rsplit('/',1)[1]
            (target/name).write_bytes(b'corrupted' if self.corrupt else PAYLOAD)
        (folder/'1.log').write_text('Downloaded fixture',encoding='utf-8')
        return 0


class PrepareDependenciesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='prepare-deps-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.constraints=self.root/'constraints.txt'
        self.constraints.write_text('torch==2.11.0+cu128\ntransformers==5.8.0\n',encoding='utf-8')
    def run_prepare(self,runner,*,output=None,python=None,mirror_only=False):
        return prepare(python or sys.executable,index_url=MIRROR,trusted_host='mirror.example',
            constraints=self.constraints,output=output or self.root/'new-output',pip_runner=runner,
            mirror_only=mirror_only)
    def test_only_reported_missing_archives_are_downloaded_and_verified(self):
        runner=FakePipeline([[record('vllm','0.24.0')]])
        result=self.run_prepare(runner)
        self.assertEqual(result['archive_count'],1)
        self.assertFalse(result['packages_installed'])
        resolve,download=runner.calls
        self.assertNotIn('--ignore-installed',resolve['args'])
        self.assertNotIn('--upgrade',resolve['args'])
        self.assertIn('--no-binary=vllm',resolve['args'])
        self.assertIn('--no-build-isolation',resolve['args'])
        self.assertEqual(resolve['args'][resolve['args'].index('-c')+1],str(self.constraints))
        self.assertIn('vllm==0.24.0',resolve['args'])
        self.assertIn('transformers==5.8.0',resolve['args'])
        self.assertEqual(download['operation'],'download')
        self.assertIn('--no-deps',download['args'])
        self.assertIn('--require-hashes',download['args'])
        self.assertNotIn('torch',Path(result['archive_requirements']).read_text())
        self.assertEqual(Path(result['resolved_constraints']).read_text(),'vllm==0.24.0\n')
        self.assertEqual(Path(result['runtime_requirements']).read_text(),'')
    def test_empty_report_reuses_already_satisfied_environment_without_download(self):
        runner=FakePipeline([[]])
        result=self.run_prepare(runner)
        self.assertEqual(len(runner.calls),1)
        self.assertEqual(result['archive_count'],0)
        self.assertTrue(Path(result['wheelhouse']).is_dir())
    def test_mirror_only_omits_official_links_and_preserves_hash_checks(self):
        runner=FakePipeline([[record('vllm','0.24.0')]])
        result=self.run_prepare(runner,mirror_only=True)
        self.assertTrue(result['mirror_only'])
        self.assertTrue(result['all_archive_sha256_verified'])
        self.assertFalse(result['fallback_used'])
        self.assertEqual(result['archive_count'],1)
        self.assertTrue(all(c['mirror_only'] and c['index']==MIRROR and c['fallback']=='' for c in runner.calls))
        self.assertNotIn('--find-links',runner.calls[0]['args'])
        self.assertIn('--require-hashes',runner.calls[1]['args'])
    def test_mirror_only_missing_source_does_not_resolve_officially(self):
        runner=FakePipeline([],resolution_failure='ERROR: HTTP error 404 at http://mirror.example/packages/a.tar.gz')
        with self.assertRaisesRegex(RuntimeError,'mirror-only mode will not use another index'):
            self.run_prepare(runner,mirror_only=True)
        self.assertEqual(len(runner.calls),1)
        failure=json.loads((self.root/'new-output/failure.json').read_text(encoding='utf-8'))
        self.assertTrue(failure['mirror_only'])
    def test_mirror_only_broken_archive_does_not_select_new_official_urls(self):
        runner=FakePipeline([[record()]],mirror_download_failure=True)
        with self.assertRaisesRegex(RuntimeError,'archive download failed'):
            self.run_prepare(runner,mirror_only=True)
        self.assertEqual([c['index'] for c in runner.calls],[MIRROR,MIRROR])
        self.assertFalse((self.root/'new-output/prepared.json').exists())
    def test_broken_mirror_archive_triggers_official_fresh_resolution(self):
        runner=FakePipeline([[record()], [record(version='1.1.0',host='files.pythonhosted.org')]],mirror_download_failure=True)
        result=self.run_prepare(runner)
        self.assertTrue(result['fallback_used'])
        self.assertEqual([c['index'] for c in runner.calls],[MIRROR,MIRROR,OFFICIAL,OFFICIAL])
        self.assertTrue(all(c['fallback']=='' for c in runner.calls))
        self.assertIn('files.pythonhosted.org',Path(result['archive_requirements']).read_text())
        self.assertNotIn('mirror.example',Path(result['archive_requirements']).read_text())
        self.assertIn('attempt-2',result['wheelhouse'])
        self.assertEqual(self.constraints.read_text(),'torch==2.11.0+cu128\ntransformers==5.8.0\n')
    def test_mirror_resolution_404_also_resolves_officially_once(self):
        runner=FakePipeline([[record(host='files.pythonhosted.org')]],
            resolution_failure='ERROR: HTTP error 404 at http://mirror.example/packages/a.whl')
        self.assertTrue(self.run_prepare(runner)['fallback_used'])
        self.assertEqual(len(runner.calls),3)
    def test_resolution_conflict_is_not_retried_against_another_index(self):
        runner=FakePipeline([],resolution_failure='ERROR: ResolutionImpossible at http://mirror.example/simple')
        with self.assertRaisesRegex(RuntimeError,'resolution failed'):self.run_prepare(runner)
        self.assertEqual(len(runner.calls),1)
        self.assertFalse((self.root/'new-output/prepared.json').exists())
    def test_downloaded_sha_mismatch_is_rejected_and_not_published(self):
        runner=FakePipeline([[record()]],corrupt=True)
        with self.assertRaisesRegex(RuntimeError,'SHA256 differs'):self.run_prepare(runner)
        self.assertEqual(len(runner.calls),2)
        self.assertFalse((self.root/'new-output/prepared.json').exists())
    def test_missing_sha_and_unsafe_archive_names_fail_before_download(self):
        for variant in ('missing-sha','bad-name'):
            item=record()
            if variant=='missing-sha':item['download_info']['archive_info']={}
            else:item['download_info']['url']='https://mirror.example/pkg/%2e%2e%2funsafe.whl'
            runner=FakePipeline([[item]])
            with self.subTest(variant=variant),self.assertRaises(ValueError):
                self.run_prepare(runner,output=self.root/variant)
            self.assertEqual(len(runner.calls),1)
    def test_old_output_is_preserved_and_no_subprocess_runs(self):
        target=self.root/'old-output';target.mkdir();(target/'keep').write_text('old')
        runner=FakePipeline([])
        with self.assertRaises(FileExistsError):self.run_prepare(runner,output=target)
        self.assertEqual((target/'keep').read_text(),'old')
        self.assertFalse(runner.calls)
    @unittest.skipUnless(hasattr(os,'symlink'),'symlinks unsupported')
    def test_virtualenv_python_symlink_is_not_resolved_to_base_interpreter(self):
        target=self.root/'venv/bin/python';target.parent.mkdir(parents=True)
        try:target.symlink_to(sys.executable)
        except OSError as exc:self.skipTest(str(exc))
        runner=FakePipeline([[]])
        result=self.run_prepare(runner,python=target)
        self.assertEqual(result['python'],str(target.absolute()))
        self.assertEqual(runner.calls[0]['python'],str(target.absolute()))


if __name__=='__main__':unittest.main()
