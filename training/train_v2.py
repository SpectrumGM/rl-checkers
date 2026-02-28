"""
Stage 2 extended: больше self-play на основе supervised модели.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import random
from tqdm import tqdm
from training.model import AlphaCheckersNet, board_to_tensor, encode_move
from training.mcts import MCTS
from game.checkers import CheckersGame


def self_play_game(model, num_sims=50):
    mcts = MCTS(model, num_sims)
    game = CheckersGame()
    states, policies, players = [], [], []

    while game.get_winner() is None:
        legal = game.get_legal_moves()
        if not legal:
            break

        move, move_probs = mcts.get_policy(game)
        if move is None:
            break

        state = board_to_tensor(game.board, game.current_player)
        states.append(state)
        players.append(game.current_player)

        policy_vec = np.zeros(200, dtype=np.float32)
        for m in legal:
            idx = encode_move(m)
            key = str(m)
            if key in move_probs:
                policy_vec[idx] = move_probs[key]
        total = policy_vec.sum()
        if total > 0:
            policy_vec /= total
        policies.append(policy_vec)

        # Temperature sampling
        probs = []
        for m in legal:
            idx = encode_move(m)
            probs.append(policy_vec[idx])
        probs = np.array(probs)
        if probs.sum() > 0:
            probs /= probs.sum()
            move = legal[np.random.choice(len(legal), p=probs)]
        else:
            move = random.choice(legal)

        game.make_move(move)

    winner = game.get_winner() or 0
    values = []
    for p in players:
        if winner == 0: values.append(0.0)
        elif winner == p: values.append(1.0)
        else: values.append(-1.0)
    return states, policies, values, winner


def train(iterations=30, games_per_iter=40, epochs=15,
          num_sims=50, batch_size=64, lr=0.0005):
    model = AlphaCheckersNet()
    model.load_state_dict(torch.load("models/alpha_checkers.pth", weights_only=True))
    print("Loaded model from previous training!")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs("models", exist_ok=True)

    for iteration in range(iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration+1}/{iterations}")
        print(f"{'='*50}")

        model.eval()
        all_s, all_p, all_v = [], [], []
        wins = {1: 0, -1: 0, 0: 0}

        for g in tqdm(range(games_per_iter), desc="Self-play"):
            s, p, v, w = self_play_game(model, num_sims)
            all_s.extend(s)
            all_p.extend(p)
            all_v.extend(v)
            wins[w] = wins.get(w, 0) + 1

        print(f"W={wins[1]} B={wins[-1]} D={wins[0]} | Pos={len(all_s)}")

        model.train()
        X = torch.FloatTensor(np.array(all_s))
        yp = torch.FloatTensor(np.array(all_p))
        yv = torch.FloatTensor(np.array(all_v)).unsqueeze(1)
        dataset = torch.utils.data.TensorDataset(X, yp, yv)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            tp, tv = 0, 0
            for bx, bp, bv in loader:
                pp, pv = model(bx)
                p_loss = -torch.sum(bp * pp) / bp.size(0)
                v_loss = F.mse_loss(pv, bv)
                loss = p_loss + v_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tp += p_loss.item()
                tv += v_loss.item()
            n = len(loader)

        print(f"p_loss={tp/n:.4f} v_loss={tv/n:.4f}")
        torch.save(model.state_dict(), f"models/alpha_v2_iter_{iteration+1}.pth")

    torch.save(model.state_dict(), "models/alpha_checkers.pth")
    print("\nFinal model saved!")


if __name__ == '__main__':
    train(iterations=30, games_per_iter=40, epochs=15, num_sims=50, batch_size=64, lr=0.0005)
