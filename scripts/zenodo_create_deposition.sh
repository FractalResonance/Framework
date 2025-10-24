#!/usr/bin/env bash
# Create a new Zenodo deposition, upload a file, and publish it.
# Usage: ZENODO_TOKEN=... scripts/zenodo_create_deposition.sh metadata.json path/to/file.pdf
set -euo pipefail
API_URL="${ZENODO_API_URL:-https://zenodo.org/api}"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 metadata.json file.pdf" >&2; exit 1
fi
META=$1
FILE=$2
[[ -f "$META" ]] || { echo "metadata file not found: $META" >&2; exit 1; }
[[ -f "$FILE" ]] || { echo "file not found: $FILE" >&2; exit 1; }
[[ -n "${ZENODO_TOKEN:-}" ]] || { echo "ZENODO_TOKEN not set" >&2; exit 1; }

# Create deposition
echo "Creating deposition…" >&2
DEP=$(curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $ZENODO_TOKEN" \
  -X POST "$API_URL/deposit/depositions" -d @"$META")
ID=$(echo "$DEP" | jq -r '.id')
[[ "$ID" != "null" ]] || { echo "$DEP"; echo "Failed to create deposition" >&2; exit 1; }

# Upload file
echo "Uploading file to deposition $ID…" >&2
BUCKET=$(echo "$DEP" | jq -r '.links.bucket')
FNAME=$(basename "$FILE")
curl -s -H "Authorization: Bearer $ZENODO_TOKEN" \
  -X PUT "$BUCKET/$FNAME" --upload-file "$FILE" >/dev/null

# Publish
echo "Publishing deposition $ID…" >&2
curl -s -H "Authorization: Bearer $ZENODO_TOKEN" -X POST "$API_URL/deposit/depositions/$ID/actions/publish" >/dev/null

echo "Published deposition ID: $ID" >&2
