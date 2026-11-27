"""Sample distinct plausible openings from the baseline for a frozen candidate."""
import argparse
import json
import random

from arena import ROOT, command, engine


def symmetry_key(moves):
    coords = [("ABCDEFGHJ".index(m[0]), int(m[1:]) - 1) for m in moves]
    forms = []
    for _ in range(4):
        forms.extend([tuple(coords), tuple((8-x, y) for x, y in coords)])
        coords = [(8-y, x) for x, y in coords]
    return min(forms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    args.komi, args.game_seconds, args.move_seconds, args.visits = 7.5, None, 0.35, None
    rng = random.Random(514932)
    book = [[]]
    seen = {()}
    with engine(ROOT / ".runtime/model.bin.gz", args, [
        "-override-config", "chosenMoveTemperatureEarly=1,chosenMoveTemperature=1"
    ]) as bot:
        while len(book) < args.count:
            command(bot, "clear_board")
            moves = [command(bot, "genmove " + "BW"[i % 2]) for i in range(rng.randint(3, 9))]
            if any(m.lower() in ("pass", "resign") for m in moves):
                continue
            key = symmetry_key(moves)
            if key not in seen:
                book.append(moves)
                seen.add(key)
    print(json.dumps(book))
