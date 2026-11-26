"""Summarize paired games from one frozen-candidate arena."""
import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path


def report(path, expected):
    games = [json.loads(row) for row in path.read_text().splitlines()]
    wins = losses = draws = 0
    pairs = Counter()
    for a, b in zip(games[::2], games[1::2]):
        assert a["opening"] == b["opening"] and [a["candidate_color"], b["candidate_color"]] == ["B", "W"]
        pair_score = 0
        for g in (a, b):
            result = g["result"]
            if result in ("0", "Draw", "Jigo"):
                draws += 1
            elif result.startswith(("W+", "B+")):
                delta = 1 if result[0] == g["candidate_color"] else -1
                wins += delta == 1
                losses += delta == -1
                pair_score += delta
            else:
                raise ValueError(f"Unrecognized game result {result}")
        pairs[(pair_score > 0) - (pair_score < 0)] += 1
    n = pairs[1] + pairs[-1]
    p = sum(comb(n, k) for k in range(pairs[1], n + 1)) / (2 ** n)
    complete = len(games) == expected * 2
    return dict(complete=complete, completed_pairs=sum(pairs.values()), wins=wins, losses=losses,
                draws=draws, winning_pairs=pairs[1], tied_pairs=pairs[0], losing_pairs=pairs[-1],
                one_sided_sign_test=p, passes_gate=complete and wins > losses and p < 0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-pairs", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.log, args.expected_pairs), indent=2))
