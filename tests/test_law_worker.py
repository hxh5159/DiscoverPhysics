"""Worker/protocol checks using only public toy laws and inputs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rift import case_from_observation,load_law,rollout,ValidationError

TOY='''import numpy as np
LAW_API_VERSION="1.0.0"
PARAMS={}
BODY_PROPS={}
HIDDEN=[]
def acceleration(t,bodies,params):
    return np.zeros((len(bodies),2),dtype=np.float64)
'''
CASE={'schema_version':'1.0.0','t_launch':.7,'readout':[0.,.005,.1],
      'background':[],'probes':[{'id':'probe-A','pos':[1.,2.],'vel':[.5,-.2],'properties':{}}]}


class WorkerTests(unittest.TestCase):
    def _worker(self,source=TOY,case=None,raw=None):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'law.py'
            data=source if isinstance(source,bytes) else source.encode('utf8')
            path.write_bytes(data)
            result=subprocess.run([sys.executable,'-I','-X','utf8',str(ROOT/'rift/worker.py'),'--law',str(path)],
                input=raw if raw is not None else json.dumps(CASE if case is None else case,ensure_ascii=False),
                text=True,encoding='utf8',capture_output=True,timeout=20)
            return result,json.loads(result.stdout),hashlib.sha256(data).hexdigest()

    def test_fixed_worker_and_raw_source_identity(self):
        for source in (TOY.encode('utf8'),TOY.replace('\n','\r\n').encode('utf8'),b'\xef\xbb\xbf'+TOY.encode('utf8')):
            with self.subTest(source_prefix=source[:6]):
                process,payload,digest=self._worker(source)
                self.assertEqual(process.returncode,0)
                self.assertEqual(payload['status'],'ok')
                self.assertEqual(payload['law_sha256'],digest)
                self.assertEqual(payload['result']['probe_ids'],['probe-A'])
                self.assertAlmostEqual(payload['result']['positions'][-1][0][0],1.05,places=12)

    def test_unicode_ids_round_trip(self):
        case=copy.deepcopy(CASE);case['probes'][0]['id']='测试探针'
        _,payload,_=self._worker(case=case)
        self.assertEqual(payload['result']['probe_ids'],['测试探针'])

    def test_invalid_organizer_case_is_not_candidate_failure(self):
        case=copy.deepcopy(CASE);case['readout']=[.003]
        process,payload,_=self._worker(case=case)
        self.assertEqual(process.returncode,2)
        self.assertEqual(payload['status'],'invalid_case')
        self.assertIn('readout',payload['error']['path'])

    def test_internal_labels_are_rejected(self):
        case=copy.deepcopy(CASE);case['group']='private-group'
        _,payload,_=self._worker(case=case)
        self.assertEqual(payload['status'],'invalid_case')

    def test_duplicate_case_keys_rejected(self):
        raw=json.dumps(CASE)[:-1]+',"t_launch":0}'
        _,payload,_=self._worker(raw=raw)
        self.assertEqual(payload['status'],'invalid_case')

    def test_invalid_law_is_distinct_from_execution_error(self):
        _,payload,_=self._worker(TOY.replace('PARAMS={}','PARAMS={"x":[1,2]}'))
        self.assertEqual(payload['status'],'invalid_law')
        _,payload,_=self._worker(TOY.replace('return np.zeros((len(bodies),2),dtype=np.float64)','raise ValueError("candidate calculation failed")'))
        self.assertEqual(payload['status'],'candidate_error')

    def test_observation_conversion_uses_clean_initial_metadata_only(self):
        mission={'background_initial_state':{'ids':['N0'],'positions':[[0.,0.]],'velocities':[[0.,0.]],'properties':[{'p1':1.}]},
                 'source_property_target_id':'N0'}
        record={'t_launch':.7,'readout':[0.,.1],'source_properties':{'p1':2.},
                'probes':[{'pos':[3.,0.],'vel':[0.,1.],'properties':{'p2':1.}}],
                'positions':[[[999.,999.]]],'velocities':[[[888.,888.]]],
                'name':'private-label-not-a-case-input','group':'not-forwarded'}
        before=copy.deepcopy((mission,record));case=case_from_observation(mission,record)
        self.assertEqual(case['background'][0]['properties']['p1'],2.)
        self.assertEqual(case['probes'][0]['pos'],[3.,0.])
        self.assertEqual(set(case),set(CASE))
        self.assertEqual((mission,record),before)
        del mission['source_property_target_id']
        with self.assertRaises(ValidationError):case_from_observation(mission,record)


if __name__=='__main__':unittest.main()
