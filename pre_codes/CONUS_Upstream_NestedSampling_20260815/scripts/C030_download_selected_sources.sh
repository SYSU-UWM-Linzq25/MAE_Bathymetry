#!/usr/bin/env bash
# Download the deduplicated TNMAccess source plan with resumable wget jobs.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DOWNLOAD_MANIFEST.tsv DATA_ROOT [PARALLEL]" >&2
  exit 2
fi

MANIFEST=$(readlink -f "$1")
DATA_ROOT=$(readlink -m "$2")
PARALLEL=${3:-8}

[[ -s "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
command -v wget >/dev/null || { echo "wget is required" >&2; exit 2; }
mkdir -p "$DATA_ROOT"

download_record() {
  local record=$1
  local data_root=$2
  local download_key url relpath dst part legacy_dst legacy_part
  IFS=$'\t' read -r download_key url relpath <<< "$record"
  download_key=${download_key%$'\r'}
  url=${url%$'\r'}
  relpath=${relpath%$'\r'}
  [[ "$download_key" == "download_key" ]] && return 0
  dst="$data_root/$relpath"
  part="${dst}.part"
  legacy_dst="${dst}"$'\r'
  legacy_part="${dst}"$'\r.part'
  mkdir -p "$(dirname "$dst")"
  # Older manifests used CRLF; preserve those completed/partial downloads by
  # removing the accidental carriage return from the filename.
  if [[ ! -s "$dst" && -s "$legacy_dst" ]]; then
    mv -f "$legacy_dst" "$dst"
    echo "[download] migrated CRLF filename $download_key"
  fi
  if [[ ! -e "$part" && -e "$legacy_part" ]]; then
    mv -f "$legacy_part" "$part"
    echo "[download] migrated CRLF partial $download_key"
  fi
  if [[ -s "$dst" ]]; then
    echo "[download] skip existing $download_key"
    return 0
  fi
  echo "[download] start $download_key"
  wget -c --retry-connrefused --waitretry=5 --timeout=90 --tries=0 \
    --no-verbose -O "$part" "$url"
  [[ -s "$part" ]] || { echo "Empty download: $url" >&2; return 1; }
  mv -f "$part" "$dst"
  echo "[download] done $download_key"
}
export -f download_record

tail -n +2 "$MANIFEST" \
  | awk 'NF > 0' \
  | xargs -r -d '\n' -P "$PARALLEL" -I '{}' bash -c 'download_record "$1" "$2"' _ '{}' "$DATA_ROOT"

expected=$(awk 'NR > 1 && NF > 0 {n++} END {print n+0}' "$MANIFEST")
present=$(awk -F '\t' -v root="$DATA_ROOT" 'NR > 1 {rel=$3; sub(/\r$/, "", rel); cmd="test -s \"" root "/" rel "\""; if (system(cmd)==0) n++} END {print n+0}' "$MANIFEST")
echo "[download] expected=$expected present=$present"
[[ "$present" -eq "$expected" ]] || exit 3
