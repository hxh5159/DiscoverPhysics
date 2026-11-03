"""Resolve missing vLLM dependencies and download verified archives before build.

Uses the already installed target interpreter, including its existing CUDA
PyTorch. Nothing is installed here. A broken mirror artifact causes one fresh
official-index resolution, not a retry of the same broken direct URL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

try:
    from .pip_index_install import install, mirror_download_failed
except ImportError:
    from pip_index_install import install, mirror_download_failed


REQUIREMENTS = ('vllm==0.24.0', 'transformers==5.8.0')
SOURCE_LINKS = 'https://pypi.org/simple/vllm/'
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]*\Z')
_SHA256 = re.compile(r'[a-fA-F0-9]{64}\Z')


def _save(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def _sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def _failed_mirror(log_dir, index):
    logs=sorted(Path(log_dir).glob('pip-index-*/*.log'))
    for log in logs:
        with log.open('rb') as stream:
            stream.seek(0,os.SEEK_END)
            stream.seek(max(0,stream.tell()-192*1024))
            tail=stream.read(192*1024).decode('utf-8',errors='replace')
        if mirror_download_failed(tail,index):
            return True
    return False


def _archives(report):
    if not isinstance(report,dict) or str(report.get('version'))!='1' or not isinstance(report.get('install'),list):
        raise ValueError('pip returned an unsupported installation report')
    records=[]
    names=set()
    filenames=set()
    for item in report['install']:
        metadata=item.get('metadata',{}) if isinstance(item,dict) else {}
        name,version=metadata.get('name'),metadata.get('version')
        if not isinstance(name,str) or not _NAME.fullmatch(name):
            raise ValueError('A dependency report name is invalid')
        if (not isinstance(version,str) or not version or
                any(c.isspace() for c in version) or any(c in version for c in '\r\n\0')):
            raise ValueError('A dependency report version is invalid')
        canonical=re.sub(r'[-_.]+','-',name).lower()
        if canonical in names:
            raise ValueError('The installation report contains a duplicate dependency')
        names.add(canonical)
        download=item.get('download_info',{})
        if not isinstance(download,dict):
            raise ValueError('A dependency lacks its download identity')
        url=download.get('url')
        if not isinstance(url,str) or any(c.isspace() for c in url):
            raise ValueError('A dependency download URL is invalid')
        parsed=urlsplit(url)
        if parsed.scheme not in ('http','https','file') or parsed.username or parsed.password:
            raise ValueError('Only ordinary archive URLs without credentials are supported')
        filename=unquote(parsed.path.rsplit('/',1)[-1])
        if (not filename or filename in ('.','..') or any(c in filename for c in '/\\:\r\n\0')
                or not filename.lower().endswith(('.whl','.tar.gz','.tar.bz2','.tar.xz','.zip'))):
            raise ValueError('A dependency URL does not identify a supported archive filename')
        if filename.casefold() in filenames:
            raise ValueError('Dependency archive filenames collide')
        filenames.add(filename.casefold())
        archive=download.get('archive_info',{})
        if not isinstance(archive,dict):
            raise ValueError('A dependency lacks its archive SHA256')
        digest=archive.get('hashes',{}).get('sha256') if isinstance(archive.get('hashes',{}),dict) else None
        if digest is None and isinstance(archive.get('hash'),str) and archive['hash'].startswith('sha256='):
            digest=archive['hash'].split('=',1)[1]
        if not isinstance(digest,str) or not _SHA256.fullmatch(digest):
            raise ValueError('Every selected dependency archive must have a SHA256 digest')
        records.append({'name':canonical,'version':version,'url':url,'filename':filename,'sha256':digest.lower()})
    return records


def prepare(python, *, index_url, trusted_host, constraints, output,
            fallback_index_url='https://pypi.org/simple', pip_runner=None):
    """Return prepared paths; ``pip_runner`` has the pip_index_install.install API."""
    # Preserve a venv's lexical bin/python path; resolving its symlink would
    # accidentally select the base interpreter and inspect another environment.
    interpreter=os.path.abspath(os.path.expanduser(os.fspath(python)))
    constraints=Path(constraints).expanduser().absolute()
    if not Path(interpreter).is_file() or not constraints.is_file():
        raise ValueError('The actual Python interpreter and constraints file must exist')
    root=Path(output).expanduser().absolute()
    if root.exists() or root.is_symlink():
        raise FileExistsError('Choose a new dependency output directory; old evidence is retained')
    root.mkdir(parents=True,exist_ok=False)
    execute=pip_runner or install
    indexes=[index_url.rstrip('/')]
    if fallback_index_url and fallback_index_url.rstrip('/')!=indexes[0]:
        indexes.append(fallback_index_url.rstrip('/'))
    failures=[]
    for number,index in enumerate(indexes,1):
        attempt=root/('attempt-'+str(number))
        attempt.mkdir()
        report_file=attempt/'resolution.json'
        wheelhouse=attempt/'wheelhouse'
        wheelhouse.mkdir()
        resolve_args=['-c',str(constraints),'--no-binary=vllm','--no-build-isolation',
            '--find-links',SOURCE_LINKS,*REQUIREMENTS,'--dry-run','--report',str(report_file)]
        print('[SHORE dependencies] Resolving missing packages using '+index,flush=True)
        status=execute(interpreter,resolve_args,index_url=index,trusted_host=trusted_host,
            log_dir=attempt/'resolve-logs',fallback_index_url='',operation='install')
        if status!=0:
            failures.append({'stage':'resolution','attempt':number,'exit_code':status})
            if number<len(indexes) and _failed_mirror(attempt/'resolve-logs',index):
                print('[SHORE dependencies] Mirror resolution/download failed; resolving afresh with official fallback.',flush=True)
                continue
            _save(root/'failure.json',{'failures':failures})
            raise RuntimeError('Dependency resolution failed; inspect '+str(attempt/'resolve-logs'))
        if not report_file.is_file():
            raise RuntimeError('pip succeeded without its required dependency report')
        records=_archives(json.loads(report_file.read_text(encoding='utf-8')))
        locked_urls=attempt/'archives.requirements.txt'
        locked_urls.write_text(''.join(r['name']+' @ '+r['url']+' --hash=sha256:'+r['sha256']+'\n'
                                      for r in records),encoding='utf-8')
        resolved=attempt/'resolved-constraints.txt'
        resolved.write_text(''.join(r['name']+'=='+r['version']+'\n' for r in records),encoding='utf-8')
        runtime_requirements=attempt/'runtime-requirements.txt'
        runtime_requirements.write_text(''.join(r['name']+'=='+r['version']+'\n'
            for r in records if r['name']!='vllm'),encoding='utf-8')
        if records:
            download_args=['--no-deps','--no-build-isolation','--require-hashes',
                '-r',str(locked_urls),'--dest',str(wheelhouse)]
            status=execute(interpreter,download_args,index_url=index,trusted_host=trusted_host,
                log_dir=attempt/'download-logs',fallback_index_url='',operation='download')
            if status!=0:
                failures.append({'stage':'archive_download','attempt':number,'exit_code':status})
                if number<len(indexes) and _failed_mirror(attempt/'download-logs',index):
                    print('[SHORE dependencies] Mirror archives failed; discarding those URL selections and resolving afresh.',flush=True)
                    continue
                _save(root/'failure.json',{'failures':failures})
                raise RuntimeError('Dependency archive download failed; inspect '+str(attempt/'download-logs'))
        for record in records:
            archive=wheelhouse/record['filename']
            if not archive.is_file() or archive.is_symlink() or _sha(archive)!=record['sha256']:
                raise RuntimeError('Dependency archive is missing or its SHA256 differs: '+record['filename'])
            record['size']=archive.stat().st_size
        result={'schema_version':'shore-prepared-vllm-dependencies-1.0','status':'prepared',
            'python':interpreter,'requirements':list(REQUIREMENTS),'resolution_index':index,
            'constraints':str(constraints),'constraints_sha256':_sha(constraints),
            'wheelhouse':str(wheelhouse),'resolved_constraints':str(resolved),
            'runtime_requirements':str(runtime_requirements),
            'resolution_report':str(report_file),'archive_requirements':str(locked_urls),
            'artifacts':records,'archive_count':len(records),'all_archive_sha256_verified':True,
            'existing_packages_not_ignored':True,'fallback_used':number>1,
            'prior_failures':failures,'packages_installed':False,'gpu_or_model_calls':0}
        _save(root/'prepared.json',result)
        print('[SHORE dependencies] Verified '+str(len(records))+' archives; prepared manifest: '+str(root/'prepared.json'),flush=True)
        return result
    raise RuntimeError('No package index completed the dependency preparation')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',required=True)
    parser.add_argument('--index-url',required=True)
    parser.add_argument('--trusted-host',required=True)
    parser.add_argument('--constraints',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--fallback-index-url',default='https://pypi.org/simple')
    args=parser.parse_args(argv)
    try:
        prepare(args.python,index_url=args.index_url,trusted_host=args.trusted_host,
            constraints=args.constraints,output=args.output,fallback_index_url=args.fallback_index_url)
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        print('[SHORE dependencies] '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
