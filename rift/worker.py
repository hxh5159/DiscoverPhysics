"""Trusted fixed-integrator worker for one self-contained law.py.

This process receives permitted case inputs only; it has no truth/scoring
interface. A separate process is not an OS permission boundary: deployments
must mount only the candidate and public runtime into its execution account.
"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout,redirect_stderr
import json
from pathlib import Path
import sys

# Permit `python -I /public/sdk/rift/worker.py` without importing candidate-CWD
# modules or relying on a team-controlled PYTHONPATH.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


class _BoundedLog:
    def __init__(self, limit=8192):self.limit=limit;self.chunks=[];self.size=0
    def write(self,text):
        text=str(text)
        if self.size<self.limit:
            part=text[:self.limit-self.size];self.chunks.append(part);self.size+=len(part)
        return len(text)
    def flush(self):pass
    def value(self):return ''.join(self.chunks)


def _jsonable(value):
    if hasattr(value,'tolist'):return value.tolist()
    if isinstance(value,dict):return {k:_jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_jsonable(v) for v in value]
    return value


def main(argv=None):
    for stream in (sys.stdin,sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--law',required=True)
    args=parser.parse_args(argv)
    from rift.law import load_law,ValidationError
    from rift.prediction import rollout,validate_case
    raw=sys.stdin.read(8*1024*1024+1)
    try:
        if len(raw)>8*1024*1024:raise ValueError('case input exceeds technical transport limit')
        def unique_pairs(pairs):
            result={}
            for key,value in pairs:
                if key in result:raise ValueError('duplicate case JSON key')
                result[key]=value
            return result
        case=validate_case(json.loads(raw,object_pairs_hook=unique_pairs))
    except Exception as exc:
        error=exc.as_dict() if isinstance(exc,ValidationError) else {'path':'$','message':'invalid case JSON or transport size'}
        print(json.dumps({'status':'invalid_case','error':error},ensure_ascii=False,allow_nan=False))
        return 2
    log=_BoundedLog()
    try:
        with redirect_stdout(log),redirect_stderr(log):law=load_law(args.law)
    except ValidationError as exc:
        print(json.dumps({'status':'invalid_law','error':exc.as_dict()},ensure_ascii=False,allow_nan=False))
        return 3
    except Exception as exc:
        print(json.dumps({'status':'candidate_error','stage':'load','error_type':type(exc).__name__,
                          'message':str(exc)[:1000]},ensure_ascii=False,allow_nan=False))
        return 4
    try:
        with redirect_stdout(log),redirect_stderr(log):result=rollout(law,case)
        payload={'status':'ok','law_api_version':law.api_version,'law_sha256':law.source_sha256,
                 'result':_jsonable(result),'candidate_log':log.value()}
        print(json.dumps(payload,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as exc:
        error=exc.as_dict() if isinstance(exc,ValidationError) else {'message':str(exc)[:1000]}
        print(json.dumps({'status':'candidate_error','stage':'rollout','error_type':type(exc).__name__,
                          'error':error,'candidate_log':log.value()},ensure_ascii=False,allow_nan=False))
        return 4


if __name__=='__main__':raise SystemExit(main())
