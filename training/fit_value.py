"""Fit the win/loss output on frozen specialist features with a whole-game holdout."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from fit import ROOT, export, load_model


def collect(model):
    features, outcomes, weights, game_ids = [], [], [], []
    def capture(layer, args):
        features.append(args[0].detach().cpu())

    handle = model.value_head.linear_valuehead.register_forward_pre_hook(capture)
    with torch.inference_mode():
        for path in sorted((ROOT / ".runtime/selfplay/baseline/tdata").glob("*.npz")):
            with np.load(path) as data:
                target = data["globalTargetsNC"]
                weight = target[:, 25] * (1 - target[:, 35])
                rows = np.flatnonzero(weight > 0)
                if not len(rows):
                    continue
                bits = np.unpackbits(data["binaryInputNCHWPacked"][rows], axis=2)[:, :, :81]
                spatial = torch.tensor(bits.reshape(len(rows), -1, 9, 9), dtype=torch.float32)
                glob = torch.from_numpy(data["globalInputNC"][rows])
                for x, g in zip(spatial.split(32), glob.split(32)):
                    model(x.cuda(), g.cuda())
                outcomes.append(torch.from_numpy(target[rows, :3]))
                weights.append(torch.from_numpy(weight[rows]))
                game_ids.append(torch.from_numpy(target[rows, 41:47].astype(np.int64)))
    handle.remove()
    return tuple(torch.cat(items) for items in (features, outcomes, weights, game_ids))


def fit(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    _, model, _ = load_model(ROOT / ".runtime/teacher.ckpt", True, "cpu", pos_len=9)
    model.eval().requires_grad_(False).cuda()
    x, y, w, games = collect(model)
    held = games[:, 0] % 5 == 0
    if not held.any() or held.all():
        raise ValueError("Need both training and held-out games")
    head = model.cpu().value_head.linear_valuehead.requires_grad_(True)
    reference = head(x).detach()
    def logloss(mask):
        losses = -(head(x[mask]).log_softmax(1) * y[mask]).sum(1)
        return (losses * w[mask]).sum() / w[mask].sum()

    before = logloss(held).item()
    optimizer = torch.optim.Adam(head.parameters(), lr=0.001)
    for _ in range(50):
        optimizer.zero_grad()
        drift = (head(x[~held]) - reference[~held]).square().mean()
        (logloss(~held) + 0.05 * drift).backward()
        optimizer.step()
    metrics = dict(train_games=len(games[~held].unique(dim=0)), held_games=len(games[held].unique(dim=0)),
                   held_rows=int(held.sum()), baseline_logloss=before, candidate_logloss=logloss(held).item())
    (output / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    export(model, output)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    fit(parser.parse_args().output.resolve())
