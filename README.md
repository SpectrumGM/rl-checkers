# AlphaZero, but for Checkers

A checkers AI that learns to play through self-play reinforcement learning process, inspired by DeepMind's AlphaZero project. A CNN with policy and value heads is guided by Monte Carlo Tree Search. Starting from absolutely zero knowledge, the agent was trained in three stages: supervised pre-training on minimax games, self-play, and minimax sparring — all on a single CPU.

The result: **90% win rate vs random, 100% vs minimax depth 1, and unbeatable (all draws) against minimax depth 3 from both sides.** Includes a Pygame GUI with real-time neural eval bar.
```bash
python play.py alpha 3      # watch AlphaZero vs Minimax d3
python play.py play_alpha    # play against it yourself
python play.py alpha 1       # watch it destroy weak minimax
```
