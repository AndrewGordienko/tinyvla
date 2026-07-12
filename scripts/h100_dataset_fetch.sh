#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/tinyvla
BASE=https://github.com/AndrewGordienko/tinyvla/releases/download/smolvla-command0-dataset-v1
DIR="$ROOT/dataset_chunks"
mkdir -p "$DIR"
curl -fL --retry 8 --retry-all-errors --connect-timeout 20 -o "$DIR/chunk_manifest.json" "$BASE/chunk_manifest.json"
python3 - "$DIR" "$BASE" <<'PY'
import hashlib,json,pathlib,subprocess,sys
d=pathlib.Path(sys.argv[1]); base=sys.argv[2]; m=json.loads((d/'chunk_manifest.json').read_text())
for r in m['chunks']:
 p=d/r['filename']; ok=p.exists() and p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
 if not ok: subprocess.run(['curl','-fL','--retry','8','--retry-all-errors','--connect-timeout','20','-C','-','-o',str(p),base+'/'+r['filename']],check=True)
 if p.stat().st_size!=r['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']: raise SystemExit('chunk mismatch '+r['filename'])
with (d/'archive.tar.zst').open('wb') as out:
 for r in m['chunks']: out.write((d/r['filename']).read_bytes())
if hashlib.sha256((d/'archive.tar.zst').read_bytes()).hexdigest()!=m['archive_sha256']: raise SystemExit('archive mismatch')
PY
rm -rf "$ROOT/data/datasets/.command0_extract"; mkdir -p "$ROOT/data/datasets/.command0_extract"
zstd -dc "$DIR/archive.tar.zst" | tar -xf - -C "$ROOT/data/datasets/.command0_extract"
TREE=$(PYTHONPATH="$ROOT" "$ROOT/.venv/bin/python" -c "from tinyvla.runtime import sha256_tree; print(sha256_tree('$ROOT/data/datasets/.command0_extract/command0_multiview_32'))")
test "$TREE" = 686808ab96fed5d3005b8cbf8d0351d7cb66f9e77d61f8827913f367f270fbd7
test ! -e "$ROOT/data/datasets/command0_multiview_32" || mv "$ROOT/data/datasets/command0_multiview_32" "$ROOT/data/datasets/command0_multiview_32.old"
mv "$ROOT/data/datasets/.command0_extract/command0_multiview_32" "$ROOT/data/datasets/command0_multiview_32"
echo "dataset_ready tree=$TREE"
