"""Reconstruct the locally trained candidate from the pinned baseline and policy update."""
import argparse
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path

models = Path(__file__).resolve().parent
runtime = models.parent / ".runtime"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("name", nargs="?", default="affogato-001",
                    choices=sorted(p.parent.name for p in models.glob("*/manifest.json")))
candidate = models / parser.parse_args().name
manifest = json.loads((candidate / "manifest.json").read_text())
baseline = gzip.decompress((runtime / "model.bin.gz").read_bytes())
if hashlib.sha256(baseline).hexdigest() != manifest["base_sha256"]:
    raise SystemExit("Baseline digest mismatch. Run ./setup to install the pinned network.")
body = bytearray(baseline.split(b"\n", 1)[1])
patch = (candidate / "policy.bin").read_bytes()
offset = manifest["offset"]
body[offset:offset + len(patch)] = patch
model = manifest["name"].encode() + b"\n" + body
if hashlib.sha256(model).hexdigest() != manifest["model_sha256"]:
    raise SystemExit("Candidate digest mismatch")
destination = runtime / (manifest["name"] + ".bin.gz")
with tempfile.NamedTemporaryFile(dir=runtime, delete=False) as temp:
    try:
        with gzip.GzipFile(fileobj=temp, mode="wb", compresslevel=1, mtime=0) as archive:
            archive.write(model)
        temp.flush()
        os.replace(temp.name, destination)
    finally:
        Path(temp.name).unlink(missing_ok=True)
print(destination)
