#!/bin/bash
# Sync between this machine and the Ashoka HPC (ssh username@10.1.4.95,
# project dir ~/prospect-theory). NOTE: login prompts a CAPTCHA + password,
# so run this interactively (it is not unattended-friendly).
#
# Usage:
#   ./sync_results.sh push     # local code/scripts -> HPC (drivers, pbs, env, agents)
#   ./sync_results.sh pull     # HPC results -> local (docs/**/raw_data.csv, logs)
#
# Set your HPC username once:
HPC_USER="${HPC_USER:-CHANGE_ME}"
HPC_HOST="${HPC_HOST:-10.1.4.95}"
REMOTE="${HPC_USER}@${HPC_HOST}:~/prospect-theory"
LOCAL="$(cd "$(dirname "$0")" && pwd)"

if [ "$HPC_USER" = "CHANGE_ME" ]; then
  echo "Set your username: HPC_USER=ay_ug22 ./sync_results.sh pull"; exit 1
fi

case "${1:-}" in
  push)
    rsync -avz --relative \
      experiments/ agents/ env/ utils/ *.pbs *.sh references*.bib \
      "$REMOTE/" ;;
  pull)
    # only results + logs back; never overwrite local code
    rsync -avz \
      --include="docs/***" --include="logs/***" \
      --include="*/" --exclude="*" \
      "$REMOTE/" "$LOCAL/" ;;
  *)
    echo "usage: $0 {push|pull}"; exit 1 ;;
esac
