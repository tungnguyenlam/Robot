#!/usr/bin/env python3
"""Record migration and eventual download completion on the canonical server."""
from pathlib import Path
import subprocess
import time
import fcntl

archive=Path('/hdd/Robot/migration-20261108')
repo=Path('/home/nguyenlt/Programming/Robot')
lock=(archive/'finalizer.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

def note(message):
    print(time.strftime('[%F %T]'),message,flush=True)
    (archive/'status').write_text(message+'\n')

note('Transfers running; source cleanup waits for independent destination MD5 checks.')
while not all((archive/node/'SOURCE_CLEANED').exists() for node in ['macmini','thinkbook']):
    time.sleep(30)

note('Both helper datasets MD5 verified on amdserver; helper copies and tools removed. P04 downloading here.')
notes=repo/'CURRENT_MACHINE.md'
s=notes.read_text().replace('| EPIC-KITCHENS P02, P03 | Verified on thinkbook (84/84), **transfer in progress** | ~97 GB |',
                          '| EPIC-KITCHENS P02, P03 | 84/84 copied here, independently MD5 verified; thinkbook cleaned | ~97 GB |')
s=s.replace('Mac downloader stopped; 3 complete + partial P04_04 migrating here',
            'Source files MD5 verified here; macmini cleaned; remaining downloads running here')
notes.write_text(s)
(archive/'MIGRATION_COMPLETE').write_text(time.strftime('%F %T')+'\n')

cmd=['/home/nguyenlt/miniforge3/bin/python3',
     '/home/nguyenlt/.local/share/epic-dl/epic_fast_downloader.py',
     '--participants','P01,P02,P03,P04,P05','--output-path',
     str(repo/'data/epic_kitchens'),'--status']
while True:
    result=subprocess.run(cmd,text=True,capture_output=True)
    (archive/'download-status').write_text(result.stdout+result.stderr)
    if result.returncode==0:break
    time.sleep(30)

note('DONE: P01–P05 train selection fully verified (156 videos); both helper machines cleaned.')
s=notes.read_text().replace('Source files MD5 verified here; macmini cleaned; remaining downloads running here',
                          '43/43 verified here; macmini cleaned')
notes.write_text(s)
(archive/'DOWNLOADS_COMPLETE').write_text(result.stdout)
