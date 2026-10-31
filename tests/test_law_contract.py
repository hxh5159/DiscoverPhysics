"""Public format/worker checks using invented mathematical laws only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rift.law import (LAW_API_VERSION, ValidationError, validate_source,
                      validate_law, load_law)

HEADER='''import numpy as np
LAW_API_VERSION = "1.0.0"
PARAMS = {"k": 0.2}
BODY_PROPS = {}
HIDDEN = []
'''
FUNCTION='''
def acceleration(t, bodies, params):
    return np.zeros((len(bodies), 2), dtype=np.float32)
'''


class LawContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)

    def law_file(self, source=HEADER+FUNCTION, *, raw=None):
        p=self.directory/'law.py'
        p.write_bytes(source.encode('utf-8') if raw is None else raw)
        return p

    def rejected(self,source,path=None):
        with self.assertRaises(ValidationError) as caught:validate_source(source)
        if path is not None:self.assertEqual(caught.exception.path,path)
        self.assertEqual(set(caught.exception.as_dict()),{'code','path','message'})

    def test_valid_scientific_helpers_and_fixed_constants(self):
        source=HEADER+'''from math import pi
TAU = 2 * np.pi
EPS = 0.05
AXES = ((1.0, 0.0), (0.0, 1.0))
def helper(value, scale=1.0):
    return np.asarray([scale*x for x in value])
def acceleration(time, state, settings):
    result = helper([[0.,0.]][0])
    return np.array([result for body in state], dtype=np.float32)
'''
        model=load_law(self.law_file(source))
        self.assertEqual(model.api_version,LAW_API_VERSION)
        self.assertEqual(model.acceleration(0,[{}],model.params).shape,(1,2))

    def test_static_inspection_does_not_run_function_body(self):
        source=HEADER+'\ndef acceleration(t,bodies,params):\n    raise ValueError("must not execute during static inspection")\n'
        self.assertEqual(validate_source(source).params,{'k':.2})
        self.assertEqual(validate_law(self.law_file(source)).api_version,'1.0.0')

    def test_rejects_executable_top_level_without_running_it(self):
        self.rejected(HEADER+'\nwhile True:\n    pass\n'+FUNCTION)
        self.rejected(HEADER+'\nresult = np.ones((10,2))\n'+FUNCTION)
        self.rejected(HEADER+'\nacceleration(0,[],PARAMS)\n'+FUNCTION)

    def test_metadata_required_types_and_parameter_cap(self):
        for replacement in ['PARAMS = []','PARAMS = {"k": True}',
            'PARAMS = {"k": [1.0,2.0]}','PARAMS = {"k": 1e400}',
            'PARAMS = {"k": 1+2j}', 'PARAMS = {"k": "0.2"}',
            'PARAMS = '+repr({f'p{i}':1.0 for i in range(6)})]:
            with self.subTest(replacement=replacement):self.rejected(HEADER.replace('PARAMS = {"k": 0.2}',replacement)+FUNCTION)
        self.rejected((HEADER+FUNCTION).replace('LAW_API_VERSION = "1.0.0"','LAW_API_VERSION = "0.9"'),'$.LAW_API_VERSION')
        self.rejected((HEADER+FUNCTION).replace('PARAMS = {"k": 0.2}\n',''),'$.PARAMS')

    def test_duplicate_metadata_keys_are_not_silently_collapsed(self):
        self.rejected(HEADER.replace('PARAMS = {"k": 0.2}','PARAMS = {"k": 0.2, "k": 0.3}')+FUNCTION)

    def test_optional_candidate_objects_are_not_extra_global_parameters(self):
        source=HEADER.replace('BODY_PROPS = {}','BODY_PROPS = {"N0": {"__source": 3.0}}')
        hidden=[{'id':f'candidate-{i}','pos':[i,1.0],'vel':(0.,0.),'properties':{'source':.3}} for i in range(8)]
        source=source.replace('HIDDEN = []','HIDDEN = '+repr(hidden))
        metadata=validate_source(source+FUNCTION)
        self.assertEqual(len(metadata.params),1)
        self.assertEqual(len(metadata.hidden),8)
        self.assertEqual(metadata.hidden[0]['id'],'candidate-0')
        self.assertEqual(metadata.hidden[0]['vel'],[0.,0.])

    def test_invalid_hidden_metadata(self):
        examples=[{},[{'id':'h','pos':[0,0],'vel':[0,0]}],
            [{'id':'h','pos':[0,0,0],'vel':[0,0],'properties':{}}],
            [{'id':'h','pos':[0,0],'vel':[0,0],'properties':{}},
             {'id':'h','pos':[1,0],'vel':[0,0],'properties':{}}]]
        for hidden in examples:
            with self.subTest(hidden=hidden):self.rejected(HEADER.replace('HIDDEN = []','HIDDEN = '+repr(hidden))+FUNCTION)

    def test_signature_and_side_effect_imports(self):
        for signature in ['(t,bodies)','(t,bodies,params,extra)','(t,bodies,params={})','(t,bodies,params,*args)']:
            self.rejected(HEADER+FUNCTION.replace('(t, bodies, params)',signature),'$.acceleration')
        for code in ['import os','from pathlib import Path','from rift import rollout',
                     'from numpy import save','from scipy.io import savemat',
                     'import numpy.random','from numpy import *']:
            with self.subTest(code=code):self.rejected(HEADER+'\n'+code+'\n'+FUNCTION)

    def test_rejects_dynamic_io_and_introspection(self):
        expressions=['open("private.txt")','np.load("private.npy")','np.random.rand(2)',
                     'getattr(np,"load")("private.npy")','params.__class__',
                     'np.lib.npyio.DataSource()', 'print("debug")']
        for expression in expressions:
            source=HEADER+f'\ndef acceleration(t,bodies,params):\n    return {expression}\n'
            with self.subTest(expression=expression):self.rejected(source)

    def test_global_metadata_is_frozen_after_loading(self):
        source=HEADER+'\ndef acceleration(t,bodies,params):\n    PARAMS["k"] = 9\n    return np.zeros((len(bodies),2))\n'
        model=load_law(self.law_file(source))
        with self.assertRaises(TypeError):model.acceleration(0.,(),model.params)

    def test_raw_file_digest_and_equal_execution_for_lf_crlf_bom(self):
        raw=(HEADER+FUNCTION).encode('utf8')
        for data in [raw,raw.replace(b'\n',b'\r\n'),b'\xef\xbb\xbf'+raw]:
            with self.subTest(sha=hashlib.sha256(data).hexdigest()):
                p=self.law_file(raw=data)
                expected=hashlib.sha256(data).hexdigest()
                self.assertEqual(validate_law(p).source_sha256,expected)
                model=load_law(p)
                self.assertEqual(model.source_sha256,expected)
                self.assertEqual(model.acceleration(0,[{}],model.params).tolist(),[[0.,0.]])

    def test_defaults_do_not_execute_calls_or_hold_mutable_state(self):
        for default in ['np.zeros(2)','[]','([], )']:
            self.rejected(HEADER+f'\ndef helper(value={default}):\n    return value\n'+FUNCTION)

    def cli(self,*args):
        return subprocess.run([sys.executable,str(ROOT/'scripts/validate_law.py'),*map(str,args)],
                              capture_output=True,text=True,encoding='utf8',timeout=20)

    def test_cli_default_is_static_and_checks_existing_claims_schema(self):
        source=HEADER+'\ndef acceleration(t,bodies,params):\n    raise ValueError("not called")\n'
        run=self.cli(self.law_file(source),'--claims',ROOT/'docs/contracts/claims.example.json')
        self.assertEqual(run.returncode,0,run.stderr)
        value=json.loads(run.stdout)
        self.assertFalse(value['execution_checked'])
        self.assertFalse(value['scientific_correctness_checked'])
        self.assertEqual(len(value['claims']),3)

    def test_cli_explicit_worker_executes_only_supplied_public_case(self):
        run=self.cli(ROOT/'examples/law.py','--case',ROOT/'examples/law_case.json','--execute')
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        value=json.loads(run.stdout)
        self.assertTrue(value['execution_checked'])
        self.assertEqual(value['prediction']['probe_count'],2)
        self.assertFalse(value['scientific_correctness_checked'])

    def test_cli_requires_case_and_bounds_execution_time(self):
        run=self.cli(self.law_file(),'--execute')
        self.assertEqual(run.returncode,2)
        self.assertEqual(json.loads(run.stdout)['errors'][0]['path'],'$.arguments.case')
        source=HEADER+'\ndef acceleration(t,bodies,params):\n    while True:\n        pass\n'
        run=self.cli(self.law_file(source),'--case',ROOT/'examples/law_case.json','--execute','--timeout','.2')
        self.assertEqual(run.returncode,2)
        self.assertEqual(json.loads(run.stdout)['errors'][0]['code'],'execution_timeout')


if __name__=='__main__':unittest.main()
