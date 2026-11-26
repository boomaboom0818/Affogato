"""Play paired 9x9 games against the fixed specialist, logging moves and elapsed times."""
import argparse
import contextlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENINGS = [(), ("E5",), ("D5",), ("D4",), ("E5", "C5"), ("D5", "F5"),
            ("D4", "F6"), ("E4", "D6"), ("C5",), ("C4",), ("E3",), ("C3",)]


def command(bot, request):
    bot.stdin.write(request + "\n")
    bot.stdin.flush()
    response = []
    while True:
        line = bot.stdout.readline()
        if not line:
            raise RuntimeError(f"Engine exited during {request}")
        if not line.strip():
            break
        response.append(line.strip())
    if not response or not response[0].startswith("="):
        raise RuntimeError(f"{request}: {response}")
    return response[0][1:].strip()


@contextlib.contextmanager
def engine(model, args):
    cmd = [str(ROOT / "affogato")]
    if args.game_seconds:
        cmd.append("--game-clock")
    else:
        # Reserve at most a quarter second for returning the move.
        cap = args.move_seconds - min(0.25, args.move_seconds / 10)
        cmd += ["-override-config", f"maxTime={cap}"]
    bot = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           text=True, env={**os.environ, "AFFOGATO_MODEL": str(model)})
    try:
        command(bot, "boardsize 9")
        command(bot, "komi " + str(args.komi))
        yield bot
    finally:
        bot.stdin.close()
        bot.wait()


def game(bots, opening, args):
    remaining = dict.fromkeys("BW", args.game_seconds)
    moves = []
    for bot in bots.values():
        command(bot, "clear_board")
        if args.game_seconds:
            command(bot, f"time_settings {args.game_seconds} 0 0")
    for turn in range(400):
        color, other = ("B", "W") if turn % 2 == 0 else ("W", "B")
        elapsed = 0.0
        if turn < len(opening):
            move = opening[turn]
            command(bots[color], f"play {color} {move}")
        else:
            if args.game_seconds:
                command(bots[color], f"time_left {color} {remaining[color]:.4f} 0")
            start = time.monotonic()
            move = command(bots[color], f"genmove {color}")
            elapsed = time.monotonic() - start
            budget = remaining[color] if args.game_seconds else args.move_seconds
            if elapsed > budget:
                moves.append((color, move, elapsed))
                return other + "+Time", moves
            if args.game_seconds:
                remaining[color] -= elapsed
        moves.append((color, move, elapsed))
        if move.lower() == "resign":
            return other + "+Resign", moves
        command(bots[other], f"play {color} {move}")
        if len(moves) >= 2 and all(m[1].lower() == "pass" for m in moves[-2:]):
            return command(bots["B"], "final_score"), moves
    raise RuntimeError("Exceeded 400 moves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--pairs", type=int, default=1, choices=range(1, len(OPENINGS) + 1))
    parser.add_argument("--komi", type=float, default=7.5)
    clock = parser.add_mutually_exclusive_group()
    clock.add_argument("--move-seconds", type=float, default=5)
    clock.add_argument("--game-seconds", type=int)
    args = parser.parse_args()
    if args.move_seconds <= 0 or (args.game_seconds is not None and args.game_seconds <= 0):
        parser.error("Time budget must be positive")
    for opening in OPENINGS[:args.pairs]:
        for color in "BW":
            # Fresh processes prevent evaluation caches carrying over between paired games.
            with engine(args.candidate.resolve(), args) as challenger, engine(ROOT / ".runtime/model.bin.gz", args) as baseline:
                bots = {"B": challenger, "W": baseline} if color == "B" else {"B": baseline, "W": challenger}
                result, moves = game(bots, opening, args)
                print(json.dumps(dict(candidate=str(args.candidate), candidate_color=color, result=result,
                                      opening=opening, moves=moves, komi=args.komi, game_seconds=args.game_seconds,
                                      move_seconds=None if args.game_seconds else args.move_seconds)), flush=True)


if __name__ == "__main__":
    main()
