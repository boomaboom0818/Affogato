# Affogato 9×9

Affogato runs the open-source KataGo search engine with its specialist 9×9 neural network on the local GPU. Give the tournament controller the absolute path to the `affogato` executable. It speaks Go Text Protocol (GTP) over standard input and output and plays either color.

## Install and play

On x86-64 Linux with the GPU vendor's OpenCL driver, Bash, curl, unzip and sha256sum:

```sh
./setup
./affogato
```

For an interactive game, enter one GTP command per line. This example plays Black at E5 and asks Affogato to play White:

```text
boardsize 9
clear_board
komi 7.5
play B E5
genmove W
showboard
quit
```

Setup pins and verifies both downloads. After setup, games run offline. Setup installs a tested OpenCL profile for the GTX 1660 Ti. The stock tuner selected a 32×32 transform workgroup that failed on this card with CL_OUT_OF_RESOURCES. The included profile uses 8×8. KataGo selects cached profiles by device, board and network dimensions under .runtime/opencltuning. On a new GPU, generate a move before the match to allow time for tuning.

## Match settings

Both modes use Chinese scoring and disable pondering. Set komi to the event value via GTP.

- **Five seconds per move:** Launch `./affogato`. Search is capped at 4.75 seconds, reserving 0.25 seconds for response overhead. For a different move limit, pass an explicit cap, e.g. `./affogato -override-config maxTime=9.75` for a ten-second event.
- **One clock per player:** Launch `./affogato --game-clock`. Select absolute time in the tournament controller and enter the event duration. For 30 minutes each with no overtime, the controller sends `time_settings 1800 0 0` at game start. This profile lets KataGo budget the remaining time across moves.

The controller owns the countdown. It charges the moving player and sends that player's actual remaining seconds before each `genmove`. For example, with 1,740 seconds left for White:

```text
time_left W 1740 0
genmove W
```

For a different network, set `AFFOGATO_MODEL` to its absolute path. To repeat the thread benchmark on the target GPU:

```sh
.runtime/katago benchmark -config affogato.cfg -model .runtime/model.bin.gz \
  -override-config "homeDataDir=$PWD/.runtime" -boardsize 9 -t 2,4,8,16,32 -n 5 -v 300 -i 5
```

## Engine selection

The specialist b18 network was fine-tuned on 9×9 positions including variations the general networks misevaluated. Its authors used it to build deep 9×9 opening books. The public books use specific scoring and komi assumptions. Affogato searches the current match position directly. [Specialist network](https://katagotraining.org/extra_networks/) · [Opening book project](https://katagobooks.org/)

The main training run also publishes newer transformer networks. Its published ratings do not establish their ordering on this GPU under a particular 9×9 clock. A tournament comparison should use the event's time control and rules and replay each opening with colors reversed. [Training networks](https://katagotraining.org/networks/) · [KataGo engine and backend notes](https://github.com/lightvector/KataGo)

On the local GTX 1660 Ti, the five-position benchmark recommended 16 threads for the five-second profile and measured approximately 600 search visits per second. A second measurement at 16 threads returned 593 visits per second. A separate accelerated GTP self-play check completed 73 moves, two passes and final scoring (White +4.5). These runs verify operation and throughput. Tournament win rate still requires head-to-head games under the event rules.

## Affogato's trained network

`affogato-001` is the first locally trained candidate. It starts from the specialist checkpoint's averaged weights and fine-tunes its policy head to predict search visit distributions in 9×9 self-play. The value head and shared layers are frozen. The first run used 12 games and 307 weighted positions. Additional strength needs to be demonstrated in independent matches.

To play this exact candidate on a fresh checkout, run the following after `./setup`. Python 3 reconstructs the trained weights from the original network and the included policy update, verifying both model digests.

```sh
python3 models/restore.py
AFFOGATO_MODEL="$PWD/.runtime/affogato-001.bin.gz" ./affogato
```

To produce a new candidate on an NVIDIA GPU, install uv, then run:

```sh
training/setup
training/selfplay 128
.runtime/venv/bin/python training/fit.py .runtime/affogato-002
python3 training/arena.py .runtime/affogato-002/model.bin.gz --pairs 12 > .runtime/matches-002.jsonl
```

Each self-play invocation adds fresh examples under .runtime/selfplay. The trainer takes one shuffled pass over all saved teacher data with random board symmetries. It starts from the pinned specialist checkpoint each time. The output directory must be new.

The arena uses the original specialist as opponent, Chinese scoring, 7.5 komi, and a five-second deadline. It plays each opening twice with reversed colors, records all moves and elapsed times, and scores clock overruns as losses. Use `--komi` to set the event value or `--game-seconds 1800` for an absolute clock. Keep a separate set of unseen openings for any final strength claim after tuning on these development matches.

For a larger experiment, the following systemd user service caps total wall time at 24 hours and keeps logs under the new run directory. Finish other GPU matches first. The system must stay awake.

```sh
systemd-run --user --unit=affogato-experiment --property=RuntimeMaxSec=24h \
  --working-directory="$PWD" "$PWD/training/run-experiment" .runtime/experiment-002
```

The run collects five hours of teacher games with up to 1,536 visits per search, then fits one candidate and freezes it before generating evaluation openings. It evaluates 64 distinct openings including the empty board, with colors reversed and five seconds per move. Read run.log for the stage and games.jsonl for finished games. The report counts wins, losses and draws. Its predeclared gate requires all 64 pairs, more candidate wins than losses and a one-sided exact sign test below 0.05 across non-tied pairs. It estimates performance on the sampled opening distribution. The 30-minute clock requires a separate evaluation.

The second candidate, `affogato-002`, completed fitting on 18,051 weighted positions. Run `python3 models/restore.py affogato-002` to reconstruct the frozen network and set `AFFOGATO_MODEL="$PWD/.runtime/affogato-002.bin.gz"` when launching it. Its competition-strength evaluation uses the five-second clock.

For a playing-strength comparison, use `training/arena.py` with `--visits 1024`. Both models get the same maximum search visits per move. Elapsed times are recorded and the board result determines the winner. Keep this score separate from clocked matches.
