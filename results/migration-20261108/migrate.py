#!/usr/bin/env python3
"""One-shot helper evacuation; only delete files after destination MD5 validation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

SERVER_ROOT = '/home/nguyenlt/Programming/Robot'
ARCHIVE = '/hdd/Robot/migration-20261108'

def digest(p):
    h = hashlib.md5()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def log(s):
    print(time.strftime('[%F %T]'), s, flush=True)

REMOTE_VERIFY = r'''
import sys,json,hashlib,fcntl,os
from pathlib import Path
m=json.load(sys.stdin)
root=Path('/home/nguyenlt/Programming/Robot/data/epic_kitchens')
archive=Path('/hdd/Robot/migration-20261108')/m['node']
rows=[]
for item in m['files']:
    p=(root if item['kind']=='data' else archive/'tools')/item['rel']
    assert p.stat().st_size==item['size'], str(p)+' size mismatch'
    h=hashlib.md5()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==item['md5'], str(p)+' MD5 mismatch'
    print('MD5 OK '+str(p),file=sys.stderr,flush=True)
    if item.get('verified'):rows.append((p.stem,item['size'],item['md5']))
cache=root/'EPIC-KITCHENS/.verified'
with (root/'EPIC-KITCHENS/.verified.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    existing={s.split('\t')[0] for s in cache.read_text().splitlines()}
    with cache.open('a') as f:
        for vid,size,md5 in rows:
            if vid not in existing:f.write(f'{vid}\t{size}\t{md5}\n')
(archive/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(archive/'VERIFIED').write_text('All destination files independently MD5 verified\n')
'''

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('node', choices=['macmini','thinkbook'])
    a=ap.parse_args()
    repo=Path.cwd()
    data=repo/'data/epic_kitchens'
    tools=Path.home()/'.local/share/epic-dl'
    state=Path.home()/f'.local/share/epic-migration-20261108-{a.node}'
    state.mkdir(exist_ok=True)
    # Prevent a second migration process from writing the same files.
    import fcntl
    lock=(state/'lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (state/'DONE').exists():
        log('Already complete'); return
    ssh=['ssh','-o','BatchMode=yes',
         '-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
         '-o','ServerAliveCountMax=3','amdserver']
    def remote(cmd, **kw):
        return subprocess.run(ssh+[cmd],check=True,**kw)
    def status(s):
        (state/'status').write_text(s+'\n'); log(s)
    manifest_path=state/'manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
    else:
        status('Building immutable source inventory')
        cache={s.split('\t')[0]:s.split('\t')[2]
               for s in (data/'EPIC-KITCHENS/.verified').read_text().splitlines()}
        official={r['md5'] for r in csv.DictReader((tools/'data/md5.csv').open())}
        files=[]
        for kind,base in [('data',data),('tools',tools)]:
            for p in sorted(base.rglob('*')):
                if not p.is_file():continue
                rel=str(p.relative_to(base))
                st=p.stat()
                verified=kind=='data' and p.suffix=='.MP4' and p.stem in cache
                if verified:
                    md5=cache[p.stem]
                    assert md5 in official, 'Unknown official MD5: '+p.stem
                else:md5=digest(p)
                files.append(dict(kind=kind,rel=rel,size=st.st_size,
                                  mtime=st.st_mtime_ns,md5=md5,verified=verified))
        m=dict(node=a.node,files=files)
        manifest_path.write_text(json.dumps(m,indent=2)+'\n')
    destination='amdserver'
    # rsync supplies its own destination hostname to the transport command.
    transport=ssh[:-1]
    archive=ARCHIVE+'/'+a.node
    while not (state/'destination-verified').exists():
        try:
            status('Copying data and archiving helper tools')
            remote('mkdir -p '+shlex.quote(archive+'/tools'))
            for base,target,exclude in [(data,SERVER_ROOT+'/data/epic_kitchens/',['--exclude=.verified']),
                                         (tools,archive+'/tools/',[])]:
                subprocess.run(['rsync','-a','--partial','--stats',*exclude,
                                '-e',shlex.join(transport),str(base)+'/',
                                destination+':'+target],check=True)
            # Preserve the source cache in the archive, never overwrite the server cache.
            cache_text=(data/'EPIC-KITCHENS/.verified').read_text()
            remote('cat > '+shlex.quote(archive+'/source.verified'),input=cache_text,text=True)
            status('Independently hashing every destination file')
            # Source .verified is archived separately.
            verify=dict(m, files=[x for x in m['files'] if not
                       (x['kind']=='data' and x['rel']=='EPIC-KITCHENS/.verified')])
            remote('/home/nguyenlt/miniforge3/bin/python3 -c '+shlex.quote(REMOTE_VERIFY),
                   input=json.dumps(verify),text=True)
            (state/'destination-verified').touch()
        except subprocess.CalledProcessError as e:
            status('Transfer/verification failed; sources retained; retry in 120s: '+str(e))
            time.sleep(120)
    if a.node=='macmini' and not (state/'server-downloader-started').exists():
        status('Starting remaining P04 downloads on amdserver')
        remote("sed -i 's/^PARTICIPANTS=.*/PARTICIPANTS=P04/' "
               '~/.local/share/epic-dl/node.env && ~/.local/share/epic-dl/epic-dl start')
        (state/'server-downloader-started').touch()
    status('Removing helper files whose server copies passed MD5')
    # Abort before any deletion if an unexpected writer modified any source.
    for x in m['files']:
        p=(data if x['kind']=='data' else tools)/x['rel']
        if p.exists():
            st=p.stat()
            assert (st.st_size,st.st_mtime_ns)==(x['size'],x['mtime']), str(p)+' changed; retaining source'
    for x in m['files']:
        ((data if x['kind']=='data' else tools)/x['rel']).unlink(missing_ok=True)
    for base in [data,tools]:
        for p in sorted(base.rglob('*'), key=lambda p:len(p.parts), reverse=True):
            if p.is_dir():
                try:p.rmdir()
                except OSError:pass
        try:base.rmdir()
        except OSError:pass
    for name in ['epic-dl','epic-sync']:
        p=Path.home()/'.local/bin'/name
        if p.is_symlink() and str(tools) in os.readlink(p):p.unlink()
    note=repo/'CURRENT_MACHINE.md'
    text=note.read_text()
    import re
    text=re.sub(r'## Current job\n.*?(?=\n## |\Z)',
                '## Current job\n\n- Migration complete: dataset files and downloader tools archived on amdserver;\n'
                '  all destination files passed independent MD5 checks before local deletion.\n'
                '- No active downloads or local dataset copies. Downloads continue only on amdserver.\n'
                '- Audit: `/hdd/Robot/migration-20261108/'+a.node+'/`.\n',text,flags=re.S)
    note.write_text(text.replace('listed in `.git/info/exclude`','listed in `.gitignore`'))
    remote('touch '+shlex.quote(archive+'/SOURCE_CLEANED'))
    (state/'DONE').write_text('Destination MD5 verified; local dataset/tools removed\n')
    status('DONE: destination verified, source cleaned'+('; P04 downloader started on amdserver' if a.node=='macmini' else ''))
    if a.node=='macmini':
        subprocess.run(['bash','-c','cat >> WORKLOG.md'],input=
            '\n## 2026-11-08 - Mac mini evacuation completed\n'
            '- Server copies of all P04 source files and helper tools passed independent MD5 checks.\n'
            '- Resumed remaining P04 downloads on amdserver; removed verified local dataset/tool files.\n'
            '- Audit and manifests: `/hdd/Robot/migration-20261108/macmini/`.\n',text=True,check=True)

if __name__=='__main__':main()
