"""Public law.py/claims format preflight; no true-world knowledge or scores.

By default the law is parsed but never executed. --execute explicitly runs
the same fixed public prediction worker on the supplied public case, with a
wall-clock timeout. It is a local functionality check, not an OS isolation
boundary or evidence of scientific correctness.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rift.law import ValidationError, validate_law, LAW_API_VERSION
from scripts.validate_claims import load_claims, ClaimsValidationError


class _Parser(argparse.ArgumentParser):
    def error(self,message):raise ValidationError('$.arguments',message,'usage_error')


def _pairs(values):
    result={}
    for key,value in values:
        if key in result:raise ValidationError('$.case',f'Duplicate JSON key: {key!r}.')
        result[key]=value
    return result


def main(argv=None):
    parser=_Parser(description=__doc__)
    parser.add_argument('path',type=Path,help='UTF-8 law.py; optional BOM is accepted')
    parser.add_argument('--claims',type=Path,help='Also validate the existing three-field claims.json format')
    parser.add_argument('--case',type=Path,help='Canonical public/local case JSON, never hidden truth')
    parser.add_argument('--execute',action='store_true',help='Explicitly run a local fixed-integrator check on --case')
    parser.add_argument('--timeout',type=float,default=10.,help='Local worker wall-clock seconds; default10, not a scoring rule')
    try:
        args=parser.parse_args(argv)
        if not 0<args.timeout<=3600:raise ValidationError('$.arguments.timeout','Expected a positive finite local timeout <=3600 seconds.')
        if args.execute and args.case is None:raise ValidationError('$.arguments.case','--execute requires --case.')
        metadata=validate_law(args.path)
        result={'ok':True,'law_api_version':LAW_API_VERSION,'source_sha256':metadata.source_sha256,
                'parameter_count':len(metadata.params),'hidden_object_count':len(metadata.hidden),
                'scientific_correctness_checked':False,'execution_checked':False,
                'scope':'Public syntax/metadata/interface preflight; no scientific score or truth comparison.'}
        if args.claims is not None:
            result['claims']=load_claims(args.claims)
        case=None
        if args.case is not None:
            from rift.prediction import validate_case
            try:
                raw=args.case.read_text(encoding='utf-8-sig')
                case=validate_case(json.loads(raw,object_pairs_hook=_pairs))
            except (OSError,UnicodeError,json.JSONDecodeError):
                raise ValidationError('$.case','Cannot read a valid UTF-8 case JSON.') from None
            result['case_format_checked']=True
        if args.execute:
            command=[sys.executable,'-I','-X','utf8',str(ROOT/'rift/worker.py'),'--law',str(args.path.resolve())]
            try:
                run=subprocess.run(command,input=json.dumps(case,ensure_ascii=False,allow_nan=False),
                    capture_output=True,text=True,encoding='utf-8',timeout=args.timeout,check=False)
            except subprocess.TimeoutExpired:
                raise ValidationError('$.execution','Local candidate worker timed out; no scientific correctness decision was made.','execution_timeout') from None
            try: payload=json.loads(run.stdout)
            except (json.JSONDecodeError,UnicodeError):
                raise ValidationError('$.execution','Worker did not return the documented JSON result.','execution_error') from None
            if run.returncode!=0 or payload.get('status')!='ok':
                error=payload.get('error') or {'path':'$.execution','message':'Local prediction worker failed.'}
                raise ValidationError(error.get('path','$.execution'),error.get('message','Local prediction worker failed.'),'execution_error')
            if payload.get('law_sha256')!=metadata.source_sha256:
                raise ValidationError('$.source','law.py changed between static inspection and execution; retry with a frozen file.','source_changed')
            predicted=payload['result']
            result['execution_checked']=True
            result['prediction']={'probe_ids':predicted['probe_ids'],'times':predicted['times'],
                                  'sample_count':len(predicted['times']),
                                  'probe_count':len(predicted['probe_ids'])}
        status=0
    except ValidationError as exc:
        result={'ok':False,'law_api_version':LAW_API_VERSION,'scientific_correctness_checked':False,'errors':[exc.as_dict()]};status=2
    except ClaimsValidationError as exc:
        result=exc.as_dict();result['scientific_correctness_checked']=False;status=2
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,allow_nan=False).encode('utf8',errors='backslashreplace').decode('utf8'))
    return status


if __name__=='__main__':raise SystemExit(main())
