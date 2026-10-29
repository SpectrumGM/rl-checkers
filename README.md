# AlphaZero-Style Checkers

A checkers engine in the style of DeepMind's AlphaZero

**Play it in the browser:** https://rl-checkers.onrender.com/

## Results

The final model searches **50 positions per move**. Every match below is 30 games from
randomized openings (the first 4 plies are random), with each opening played once as White
and once as Black. Score counts a win as 1 and a draw as 0.5; brackets are 95% intervals.

| Opponent | W–D–L | Score |
|---|---|---|
| Random moves | 26–4–0 | **0.93** [0.87, 0.99] |
| Minimax depth 1 | 26–4–0 | **0.93** [0.87, 0.99] |
| Minimax depth 2 | 1–27–2 | **0.48** [0.43, 0.54] |
| Minimax depth 3 | 3–20–7 | **0.43** [0.33, 0.53] |
| Minimax depth 4 | 0–19–11 | 0.32 [0.23, 0.40] |
| Minimax depth 5 | 0–13–17 | 0.22 [0.13, 0.31] |

A longer run of 100 games each confirms it: **0.48** [0.45, 0.51] vs minimax depth 2 (4–88–8,
even) and **0.36** [0.31, 0.42] vs depth 3 (4–65–31, clearly weaker). Before training, the
same search with random weights scored 0.23 vs random play and 0.00 vs depth 3.

Fitting Elo ratings to all matches (minimax depth 1 anchored at 1000) places the engine at
the level of a **2–3 ply minimax search**:

| Player | Elo |
|---|---|
| Minimax depth 5 | 1543 |
| Minimax depth 4 | 1497 |
| Minimax depth 3 | 1414 |
| Minimax depth 2 | 1363 |
| **AlphaZero engine (this repo)** | **1360** |
| Minimax depth 1 | 1000 |
| Random | 916 |

![Elo ladder](figures/elo_ladder.png)


**Training** — three stages, 80 iterations in total:

| Stage | Script | What happens | Checkpoints |
|---|---|---|---|
| 1. Imitation | `train.py` | Minimax depth 3 plays 200 games vs random; the network learns to predict its moves and the game outcome | `alpha_supervised.pth` |
| 2. Self-play | `train.py`, then `train_v2.py` | Each iteration: 30–40 self-play games with MCTS, then train on (position, MCTS visit distribution, final result) | `alpha_iter_*.pth`, `alpha_v2_iter_*.pth` |
| 3. Sparring | `train_v3.py` | Same, but half the games are against minimax depth 3 | `alpha_v3_iter_*.pth` |

The three scripts are the three stages run one after another (each loads the previous
stage's `alpha_checkers.pth`), kept separate as they were developed. `models/` contains the
final model and every 5th checkpoint; the full training output is in `training_log*.txt`.



## Layout

```
game/checkers.py        rules: legal moves, mandatory captures, multi-jumps, kings, draw rule
game/display.py         Pygame board with a live neural evaluation bar
agents/                 random and minimax (alpha-beta) opponents
training/model.py       policy/value CNN, board encoding, move encoding
training/mcts.py        Monte Carlo Tree Search and the MCTS agent
training/train*.py      stage 1-2, stage 2 continued, stage 3
benchmark.py            evaluation: Elo ladder and learning curve
web/                    browser demo: FastAPI + one HTML page, NumPy inference (no PyTorch needed)
models/                 final model and every 5th checkpoint
figures/                benchmark plots and CSVs
```
