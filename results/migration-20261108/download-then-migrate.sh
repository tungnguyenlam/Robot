#!/usr/bin/env bash
set -euo pipefail
cd /Volumes/HP_P900/Users/tungnguyen/Programming/Robot
# Finish only the existing partial; amdserver owns every other missing P04 video.
/opt/homebrew/bin/python3 -u results/migration-20261108/download-assigned.py --output-path "$PWD/data/epic_kitchens" --only P04_04
# A fresh immutable inventory is built only after this assignment passes MD5.
exec /opt/homebrew/bin/python3 -u results/migration-20261108/migrate.py macmini
