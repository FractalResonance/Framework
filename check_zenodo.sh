#!/usr/bin/env bash
# Check all FRC papers on Zenodo

papers=(
  "100.001:15073056"
  "100.002:15079278"
  "100.003:15079820"
  "100.003.566:17437878"
  "100.004:17438174"
  "100.005:17438231"
  "100.006:17438360"
  "100.006.002:17438410"
  "566.001:17437759"
  "567.901:17437757"
)

for entry in "${papers[@]}"; do
  IFS=':' read -r paper_id zenodo_id <<< "$entry"
  echo "=== FRC $paper_id (Zenodo $zenodo_id) ==="
  curl -s "https://zenodo.org/api/records/${zenodo_id}" | jq '{
    title: .metadata.title,
    license: .metadata.license.id,
    created: .created[:10],
    files: [.files[].key],
    keywords: .metadata.keywords[0:3]
  }'
  echo
done
