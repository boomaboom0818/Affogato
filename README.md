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
