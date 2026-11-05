"""Create a small, validated manifest for a supplied model; do not copy weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rift.model_registry import catalog, model_spec
from rift.submission import validate_manifest


def make_manifest(model, team_id, submission_id, *, version='1.0.0',
                  model_path=None, model_root=None, adaptation='base', python='python', max_turns=3):
    if type(max_turns) is not int or max_turns < 0:
        raise ValueError('max_turns must be a nonnegative integer')
    item = model_spec(model)
    path = model_path or (Path(model_root or catalog()['model_root'])/item['directory']).as_posix()
    return validate_manifest({'schema_version': '1.0.0', 'team_id': team_id,
        'submission_id': submission_id, 'version': version,
        'entrypoint': [python, '-m', 'shore_agent.agent', '--model', item['id'],
                       '--max-turns', str(max_turns), '--model-path', path],
        'models': [{'base_model': item['base_model'], 'path': path, 'adaptation': adaptation}]})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='27b')
    p.add_argument('--team-id', required=True)
    p.add_argument('--submission-id', required=True)
    p.add_argument('--version', default='1.0.0')
    p.add_argument('--model-root', help='actual Linux root containing the supplied model directories')
    p.add_argument('--model-path', help='actual Linux model path or project-relative checkpoint directory')
    p.add_argument('--adaptation', choices=('base','quantized','sft','rl','other'), default='base')
    p.add_argument('--python', default='python', help='actual experiment Python executable on this instance')
    p.add_argument('--max-turns', type=int, default=3,
                   help='maximum baseline model calls per universe (default 3); zero removes this cap')
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args(argv)
    try:
        manifest = make_manifest(a.model,a.team_id,a.submission_id,version=a.version,
            model_path=a.model_path,model_root=a.model_root,adaptation=a.adaptation,python=a.python,
            max_turns=a.max_turns)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as stream:
            json.dump(manifest,stream,ensure_ascii=False,indent=2)
            stream.write('\n')
        print(json.dumps({'ok':True,'manifest':str(a.output),'models_copied':False}))
        return 0
    except (OSError,ValueError) as exc:
        p.exit(2,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
