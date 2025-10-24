#!/usr/bin/env bash
set -euo pipefail
paper_dir="$1"
if [ ! -f "papers/$paper_dir/zenodo.json" ]; then
  echo "papers/$paper_dir/zenodo.json not found" >&2; exit 1
fi
cp -f "papers/$paper_dir/zenodo.json" .zenodo.json
echo "Root .zenodo.json set from papers/$paper_dir/zenodo.json"
