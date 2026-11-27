"""Fine-tune the specialist policy head on search visit distributions."""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".runtime/KataGo/python"
sys.path.insert(0, str(UPSTREAM))
from katago.train.data_processing_pytorch import apply_symmetry, apply_symmetry_policy
from katago.train.load_model import load_model


def fit(args):
    files = sorted((ROOT / ".runtime/selfplay/baseline/tdata").glob("*.npz"))
    if not files:
        raise SystemExit("Generate teacher self-play data first.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    _, model, _ = load_model(ROOT / ".runtime/teacher.ckpt", True, "cpu", pos_len=9)
    model.cuda().eval().requires_grad_(False)
    model.policy_head.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.policy_head.parameters(), lr=1e-5, weight_decay=0)
    rows = 0
    for path in rng.permutation(files):
        with np.load(path) as data:
            packed = data["binaryInputNCHWPacked"]
            spatial = np.unpackbits(packed, axis=2)[:, :, :81].reshape(len(packed), -1, 9, 9)
            glob = data["globalInputNC"]
            visits = data["policyTargetsNCMove"][:, :1, :]
            weights = data["globalTargetsNC"][:, 25] * data["globalTargetsNC"][:, 26]
        valid = np.flatnonzero((weights > 0) & (visits[:, 0].sum(axis=1) > 0))
        order = rng.permutation(valid)
        for start in range(0, len(order), 16):
            ids = order[start:start + 16]
            inputs, globals_, targets, weight = [
                torch.as_tensor(x[ids], dtype=torch.float32, device="cuda")
                for x in (spatial, glob, visits, weights)
            ]
            symmetry = int(rng.integers(8))
            logits = model(apply_symmetry(inputs, symmetry), globals_)[0][0][:, 0]
            targets = apply_symmetry_policy(targets, symmetry, 9)[:, 0]
            targets /= targets.sum(dim=1, keepdim=True)
            loss = -(targets * logits.log_softmax(dim=1)).sum(dim=1)
            optimizer.zero_grad()
            (loss * weight).sum().div(weight.sum()).backward()
            torch.nn.utils.clip_grad_norm_(model.policy_head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            rows += len(ids)
    if rows == 0:
        raise ValueError("No weighted policy targets")
    export(model, output)
    print(f"Trained on {rows} weighted positions. Candidate: {output / 'model.bin.gz'}")


def export(model, output):
    checkpoint = output / "model.ckpt"
    torch.save({"model": model.cpu().state_dict(), "config": model.config}, checkpoint)
    subprocess.run([sys.executable, str(UPSTREAM / "export_model_pytorch.py"),
                    "-checkpoint", str(checkpoint), "-export-dir", str(output),
                    "-filename-prefix", "model", "-model-name", output.name], check=True)
    subprocess.run(["gzip", "-n", str(output / "model.bin")], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory for this candidate")
    parser.add_argument("--seed", type=int, default=1907)
    fit(parser.parse_args())
