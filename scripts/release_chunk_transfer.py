"""Resumable small-asset uploader for a pinned GitHub release."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def run(cmd): return subprocess.run(cmd,check=True,capture_output=True,text=True).stdout
def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def assets(repo,tag):
 x=json.loads(run(['gh','release','view',tag,'--repo',repo,'--json','assets'])); return {a['name']:a for a in x['assets']}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--archive',required=True); ap.add_argument('--repo',required=True); ap.add_argument('--tag',required=True); ap.add_argument('--out',required=True); ap.add_argument('--chunk-size',type=int,default=1<<20); ap.add_argument('--retries',type=int,default=5)
 a=ap.parse_args(); archive=Path(a.archive); out=Path(a.out); chunks=out/'chunks'; chunks.mkdir(parents=True,exist_ok=True)
 archive_sha=sha(archive); chunk_rows=[]
 with archive.open('rb') as f:
  seq=0
  while True:
   data=f.read(a.chunk_size)
   if not data: break
   name=f'smolvla-command0-dataset-v1.part-{seq:05d}.bin'; p=chunks/name
   if not p.exists() or p.stat().st_size!=len(data): p.write_bytes(data)
   chunk_rows.append({'sequence':seq,'filename':name,'bytes':len(data),'sha256':sha(p)}); seq+=1
 manifest={'release_tag':a.tag,'archive':archive.name,'archive_sha256':archive_sha,'extracted_tree_sha256':'686808ab96fed5d3005b8cbf8d0351d7cb66f9e77d61f8827913f367f270fbd7','chunk_size':a.chunk_size,'chunks':chunk_rows}
 mp=out/'chunk_manifest.json'; mp.write_text(json.dumps(manifest,indent=2)+'\n')
 for path in (mp,):
  for attempt in range(a.retries):
   existing=assets(a.repo,a.tag); row=existing.get(path.name)
   if row and int(row['size'])==path.stat().st_size: break
   try: subprocess.run(['gh','release','upload',a.tag,str(path),'--repo',a.repo,'--clobber'],check=True,timeout=90); time.sleep(1)
   except Exception:
    if attempt==a.retries-1: raise
    time.sleep(2**attempt)
 def upload_row(row):
  p=chunks/row['filename']
  for attempt in range(a.retries):
   existing=assets(a.repo,a.tag); remote=existing.get(row['filename'])
   if remote and int(remote['size'])==row['bytes']: break
   try: subprocess.run(['gh','release','upload',a.tag,str(p),'--repo',a.repo,'--clobber'],check=True,timeout=90); time.sleep(.5)
   except Exception:
    if attempt==a.retries-1: print(f'chunk failed after retries: {row["filename"]}'); continue
    time.sleep(2**attempt)
  return row['filename']
 with ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(upload_row, chunk_rows))
 print(json.dumps({'chunks':len(chunk_rows),'archive_sha256':archive_sha,'manifest':str(mp)},indent=2))
if __name__=='__main__': main()
