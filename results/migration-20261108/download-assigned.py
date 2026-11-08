#!/usr/bin/env python3
"""Download a disjoint P04 assignment using the existing official-MD5 downloader."""
import argparse
import importlib.util
from pathlib import Path
import time

ap = argparse.ArgumentParser()
ap.add_argument('--output-path', required=True)
ap.add_argument('--only', default='')
ap.add_argument('--exclude', default='')
args = ap.parse_args()
source = Path.home() / '.local/share/epic-dl/epic_fast_downloader.py'
spec = importlib.util.spec_from_file_location('epic_fast', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dl = module.FastDownloader(source.parent / 'data', args.output_path,
                           ['train'], ['P04'], 16, 3)
only = set(filter(None, args.only.split(',')))
exclude = set(filter(None, args.exclude.split(',')))
assert only <= dl.videos.keys(), 'Unknown assigned video'
assert exclude <= dl.videos.keys(), 'Unknown excluded video'
dl.videos = {vid: info for vid, info in dl.videos.items()
             if (not only or vid in only) and vid not in exclude}
assert dl.videos, 'Empty assignment'
print('Assigned videos: ' + ','.join(dl.videos), flush=True)
while dl.download_all() != 0:
    time.sleep(120)
assert dl.status() == 0
