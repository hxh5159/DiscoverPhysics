"""Build local agent/candidate images from an offline base, with no model load."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _run(argv, *, runner=subprocess.run):
    done = runner(argv, check=False, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(done.stderr.strip() or done.stdout.strip() or 'Local container command failed')
    return done.stdout.strip()


def plan(base_image, *, docker='docker', env_root='/opt/shore-h200-cu128',
         agent_tag='shore-agent:local', candidate_tag='shore-candidate:local',
         manifest='examples/submission.json', root=ROOT):
    root = Path(root).resolve()
    if not base_image or base_image.startswith('-'):
        raise ValueError('Provide an existing local base image tag or digest')
    if not env_root.startswith('/') or ':' in env_root or '\\' in env_root:
        raise ValueError('env-root must be an absolute Linux image path')
    if Path(manifest).is_absolute() or '..' in Path(manifest).parts or '\\' in manifest:
        raise ValueError('manifest must be a relative file in the participant build context')
    common = [docker, 'build', '--pull=false', '--network=none',
              '--build-arg', 'BASE_IMAGE='+base_image,
              '--build-arg', 'SHORE_ENV_ROOT='+env_root]
    return {'base_image': base_image, 'env_root': env_root, 'manifest': str(root/manifest),
        'agent': common+['--build-arg', 'SUBMISSION_MANIFEST='+manifest,
                         '--file', str(root/'examples/Dockerfile.agent'), '--tag', agent_tag, str(root)],
        'candidate': common+['--file', str(root/'examples/Dockerfile.candidate'), '--tag', candidate_tag, str(root)],
        'agent_tag': agent_tag, 'candidate_tag': candidate_tag}


def build(settings, output, *, runner=subprocess.run, docker='docker'):
    output = Path(output).expanduser().absolute()
    if output.exists():
        raise FileExistsError('Choose a new image receipt path; existing receipts are immutable')
    if not Path(settings['manifest']).is_file():
        raise FileNotFoundError('Submission manifest is missing from the build context')
    base = _run([docker, 'image', 'inspect', '--format', '{{.Id}}', settings['base_image']], runner=runner)
    if not base.startswith('sha256:'):
        raise RuntimeError('The offline base image could not be resolved to a local image ID')
    for key in ('agent', 'candidate'):
        _run(settings[key], runner=runner)
    ids = {key+'_image': _run([docker, 'image', 'inspect', '--format', '{{.Id}}', settings[key+'_tag']], runner=runner)
           for key in ('agent', 'candidate')}
    # Independent candidate import check. No GPU, weights, simulator or network.
    _run([docker, 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
          ids['candidate_image'], '-c', 'import numpy,scipy; print("candidate environment ready")'], runner=runner)
    _run([docker, 'run', '--rm', '--network', 'none', '--entrypoint', 'python',
          ids['agent_image'], '-c', 'import shore_agent.agent,rift,rift_numerics; print("agent imports ready")'], runner=runner)
    receipt = {'schema_version': '1.0.0', 'base_image': base, **ids,
               'environment_root': settings['env_root'], 'cpu_import_checks_passed': True,
               'gpu_or_model_loaded': False, 'network_used': False,
               'purpose': 'image identities for trusted organizer profile configuration'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-image', required=True, help='already imported local CPU-built image')
    p.add_argument('--env-root', default='/opt/shore-h200-cu128')
    p.add_argument('--agent-tag', default='shore-agent:local')
    p.add_argument('--candidate-tag', default='shore-candidate:local')
    p.add_argument('--manifest', default='examples/submission.json', help='relative to the participant bundle root')
    p.add_argument('--docker', default='docker')
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    settings = plan(a.base_image, docker=a.docker, env_root=a.env_root,
                    agent_tag=a.agent_tag, candidate_tag=a.candidate_tag, manifest=a.manifest)
    if a.dry_run:
        print(json.dumps({'status': 'plan_only', **settings, 'model_or_gpu_calls': 0}, ensure_ascii=False, indent=2))
        return 0
    if not a.output:
        p.error('--output is required for an actual local image build')
    try:
        print(json.dumps(build(settings, a.output, docker=a.docker), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    raise SystemExit(main())
