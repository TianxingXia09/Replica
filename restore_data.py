"""Restore bundled compressed CSVs, verifying every original SHA-256 digest."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def main():
    entries=json.loads((ROOT/'DATA_MANIFEST.json').read_text())['files']
    for entry in entries:
        target=(ROOT/entry['path']).resolve()
        if ROOT not in target.parents:
            raise ValueError('Path outside repository')
        if target.exists():
            if digest(target)!=entry['sha256']:
                raise ValueError(f'Existing modified file preserved: {target}')
            continue
        compressed=ROOT/entry['stored_path']
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(dir=target.parent,prefix='.restore-')
        os.close(fd)
        temp=Path(name)
        try:
            with gzip.open(compressed,'rb') as src,temp.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            if digest(temp)!=entry['sha256']:
                raise ValueError(f'Checksum mismatch: {target}')
            os.replace(temp,target)
        finally:
            temp.unlink(missing_ok=True)
    for entry in entries:
        if digest(ROOT/entry['path'])!=entry['sha256']:
            raise ValueError(entry['path'])
    print(f"Verified {len(entries)} original data files.")

if __name__=='__main__':
    main()
