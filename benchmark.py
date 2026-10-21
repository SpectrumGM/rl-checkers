"""Benchmarks for rl-checkers.

1) curve  : every saved checkpoint plays minimax depth 3 -> how strength grew during training
2) ladder : final model + minimax depths 1-5 + random play matches -> Elo ratings

All games start from a random opening (first 4 plies random) and every opening is
played with both colors, so no two games are identical replays.

    python benchmark.py curve  --games 20 --sims 50
    python benchmark.py ladder --games 20 --sims 50
"""
import argparse
import csv
import math
import os
import random
import re
from multiprocessing import Pool

import numpy as np
import torch

from game.checkers import CheckersGame
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from training.mcts import MCTSAgent

OPENING_PLIES = 4


# ---------- games ----------

def random_opening(seed):
    rng = random.Random(seed)
    g = CheckersGame()
    for _ in range(OPENING_PLIES):
        moves = g.get_legal_moves()
        if not moves or g.get_winner() is not None:
            break
        g.make_move(rng.choice(moves))
    return g


def play(game, white, black):
    while game.get_winner() is None:
        agent = white if game.current_player == 1 else black
        move = agent.choose_move(game)
        if move is None:
            break
        game.make_move(move)
    return game.get_winner() or 0


def make_agent(spec, sims):
    kind, arg = spec
    if kind == "net":
        return MCTSAgent(arg, num_simulations=sims)
    if kind == "minimax":
        return MinimaxAgent(depth=arg)
    return RandomAgent()


def run_match(job):
    """job = (spec_a, spec_b, n_openings, sims, seed). Returns (w, d, l) for A."""
    spec_a, spec_b, n, sims, seed = job
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    a, b = make_agent(spec_a, sims), make_agent(spec_b, sims)
    w = d = l = 0
    for i in range(n):
        for color in (1, -1):
            g = random_opening(seed * 1000 + i)
            if g.get_winner() is not None:
                continue
            white, black = (a, b) if color == 1 else (b, a)
            res = play(g, white, black)
            if res == 0:
                d += 1
            elif res == color:
                w += 1
            else:
                l += 1
    return w, d, l


def score_ci(w, d, l):
    n = w + d + l
    s = (w + 0.5 * d) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n
    h = 1.96 * math.sqrt(var / n)
    return s, max(0.0, s - h), min(1.0, s + h)


# ---------- 1) learning curve ----------

def checkpoints(models_dir):
    """Checkpoints in training order: supervised, stage 2a, stage 2b, stage 3."""
    files = os.listdir(models_dir)
    out = []
    if "alpha_supervised.pth" in files:
        out.append(("supervised", "alpha_supervised.pth"))
    for prefix, label in (("alpha_iter_", "self-play"), ("alpha_v2_iter_", "self-play 2"),
                          ("alpha_v3_iter_", "sparring")):
        its = sorted(int(m.group(1)) for f in files if (m := re.fullmatch(prefix + r"(\d+)\.pth", f)))
        out += [(f"{label} {i}", f"{prefix}{i}.pth") for i in its]
    return out


def curve(args):
    cps = checkpoints(args.models)
    opp = ("minimax", args.depth)
    jobs = [(("none", None), opp, args.games, args.sims, 1)]  # untrained baseline
    jobs += [(("net", os.path.join(args.models, f)), opp, args.games, args.sims, 1) for _, f in cps]
    # untrained network: MCTSAgent with no weights
    jobs[0] = (("net", None), opp, args.games, args.sims, 1)
    labels = ["untrained"] + [lab for lab, _ in cps]
    with Pool(args.workers) as pool:
        results = pool.map(run_match, jobs)

    rows = []
    for step, (lab, (w, d, l)) in enumerate(zip(labels, results)):
        s, lo, hi = score_ci(w, d, l)
        rows.append({"step": step, "checkpoint": lab, "wins": w, "draws": d, "losses": l,
                     "score": round(s, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)})
        print(f"{lab:16s} W-D-L {w:3d}-{d:3d}-{l:3d}  score {s:.2f} [{lo:.2f}, {hi:.2f}]", flush=True)

    os.makedirs("figures", exist_ok=True)
    with open(f"figures/learning_curve_d{args.depth}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    plot_curve(rows, args.depth, args.games * 2)


def plot_curve(rows, depth, n_games):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["step"] for r in rows]
    s = [r["score"] for r in rows]
    lo = [r["ci_low"] for r in rows]
    hi = [r["ci_high"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(x, lo, hi, color="#4C78A8", alpha=0.18, linewidth=0, label="95% CI")
    ax.plot(x, s, "o-", color="#4C78A8", lw=2, ms=4, label="score vs minimax")
    ax.axhline(0.5, color="#888", lw=1, ls="--")
    ax.text(x[-1], 0.51, "even with minimax", ha="right", va="bottom", color="#666", fontsize=9)

    # stage boundaries, labels staggered so they never overlap
    stage_of = lambda lab: lab.rsplit(" ", 1)[0] if lab[-1].isdigit() else lab
    prev, k = None, 0
    for r in rows:
        st = stage_of(r["checkpoint"])
        if st != prev and r["step"] > 0:
            ax.axvline(r["step"] - 0.5, color="#ccc", lw=1)
            ax.text(r["step"] - 0.4, 0.97 - 0.05 * (k % 2), st, fontsize=8, color="#555", va="top")
            k += 1
        prev = st
    ax.set_xticks(x)
    ax.set_xticklabels([r["checkpoint"].split()[-1] if r["checkpoint"][-1].isdigit() else {"untrained": "none", "supervised": "sup"}.get(r["checkpoint"], r["checkpoint"])
                        for r in rows], fontsize=7, rotation=90)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (win = 1, draw = 0.5)")
    ax.set_title(f"Strength during training: score vs minimax depth {depth} ({n_games} games per point)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"figures/learning_curve_d{depth}.png", dpi=150)
    print(f"saved figures/learning_curve_d{depth}.png and .csv")


# ---------- 2) Elo ladder ----------

def fit_elo(players, matches, anchor, anchor_elo=1000.0, iters=5000):
    """Bradley-Terry (draw = half win) fitted by gradient ascent. matches: (i, j, w, d, l)."""
    r = np.zeros(len(players))
    k = math.log(10) / 400
    for _ in range(iters):
        grad = np.zeros_like(r)
        for i, j, w, d, l in matches:
            # +1 virtual draw per match keeps a 20-0 sweep from pushing ratings to infinity
            n = w + d + l + 1
            p = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
            g = (w + 0.5 * d + 0.5 - n * p) * k
            grad[i] += g
            grad[j] -= g
        r += 2000 * grad / max(1, sum(m[2] + m[3] + m[4] for m in matches))
    return r - r[anchor] + anchor_elo


def ladder(args):
    players = [("final model (MCTS)", ("net", args.model)),
               ("random", ("random", None))] + \
              [(f"minimax d{d}", ("minimax", d)) for d in range(1, args.max_depth + 1)]
    idx = {name: i for i, (name, _) in enumerate(players)}
    pairs = [("final model (MCTS)", n) for n, _ in players[1:]]
    pairs += [("random", "minimax d1")]
    pairs += [(f"minimax d{d}", f"minimax d{d + 1}") for d in range(1, args.max_depth)]

    spec = dict(players)
    jobs = [(spec[a], spec[b], args.games, args.sims, 7 + k) for k, (a, b) in enumerate(pairs)]
    with Pool(args.workers) as pool:
        results = pool.map(run_match, jobs)

    matches = []
    print("\nmatches (W-D-L for the first player):")
    for (a, b), (w, d, l) in zip(pairs, results):
        s, lo, hi = score_ci(w, d, l)
        print(f"  {a:20s} vs {b:12s} {w:3d}-{d:3d}-{l:3d}  score {s:.2f} [{lo:.2f}, {hi:.2f}]")
        matches.append((idx[a], idx[b], w, d, l))

    elo = fit_elo([p for p, _ in players], matches, anchor=idx["minimax d1"])
    order = np.argsort(-elo)
    print("\nElo (minimax d1 = 1000):")
    rows = []
    for i in order:
        print(f"  {players[i][0]:20s} {elo[i]:7.0f}")
        rows.append({"player": players[i][0], "elo": round(float(elo[i]))})
    os.makedirs("figures", exist_ok=True)
    with open("figures/elo_ladder.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["player", "elo"])
        wr.writeheader()
        wr.writerows(rows)
    print("saved figures/elo_ladder.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["curve", "ladder"])
    ap.add_argument("--models", default="models")
    ap.add_argument("--model", default="models/alpha_checkers.pth")
    ap.add_argument("--games", type=int, default=20, help="openings per match; each played with both colors")
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--depth", type=int, default=3, help="minimax depth for the curve")
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    curve(args) if args.mode == "curve" else ladder(args)
