"""
Stage 1: Supervised pre-training на minimax играх.
Stage 2: AlphaZero self-play для усиления.
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
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent


# ===================== STAGE 1: SUPERVISED =====================

def generate_supervised_data(n_games=200):
    """Minimax vs Random — записываем позиции и ходы."""
    print(f"Generating {n_games} supervised games...")
    minimax = MinimaxAgent(depth=3)
    random_ag = RandomAgent()
    states, policy_targets, value_targets = [], [], []
    wins = {1: 0, -1: 0, 0: 0}

    for i in tqdm(range(n_games)):
        game = CheckersGame()
        game_data = []

        if i % 2 == 0:
            white_ag, black_ag = minimax, random_ag
        else:
            white_ag, black_ag = random_ag, minimax

        while game.get_winner() is None:
            player = game.current_player
            legal = game.get_legal_moves()
            if not legal:
                break

            if player == 1:
                move = white_ag.choose_move(game)
            else:
                move = black_ag.choose_move(game)
            if move is None:
                break

            state = board_to_tensor(game.board, player)
            policy_vec = np.zeros(200, dtype=np.float32)
            idx = encode_move(move)
            policy_vec[idx] = 1.0

            game_data.append((state, policy_vec, player))
            game.make_move(move)

        winner = game.get_winner() or 0
        wins[winner] = wins.get(winner, 0) + 1

        for state, pvec, player in game_data:
            states.append(state)
            policy_targets.append(pvec)
            if winner == 0:
                value_targets.append(0.0)
            elif winner == player:
                value_targets.append(1.0)
            else:
                value_targets.append(-1.0)

    print(f"Results: W={wins[1]} B={wins[-1]} D={wins[0]}")
    return states, policy_targets, value_targets


def supervised_train(model, states, policy_targets, value_targets,
                     epochs=30, batch_size=64, lr=0.002):
    """Обучаем нейронку копировать minimax."""
    print(f"\nSupervised training on {len(states)} positions...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X = torch.FloatTensor(np.array(states))
    y_p = torch.FloatTensor(np.array(policy_targets))
    y_v = torch.FloatTensor(np.array(value_targets)).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X, y_p, y_v)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_p, total_v = 0, 0
        for bx, bp, bv in loader:
            pred_p, pred_v = model(bx)
            p_loss = -torch.sum(bp * pred_p) / bp.size(0)
            v_loss = F.mse_loss(pred_v, bv)
            loss = p_loss + v_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_p += p_loss.item()
            total_v += v_loss.item()
        n = len(loader)
        if (epoch+1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | p_loss={total_p/n:.4f} v_loss={total_v/n:.4f}")
    return model


# ===================== STAGE 2: SELF-PLAY =====================

def self_play_game(model, num_sims=30):
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

        # Temperature: sample proportional to visit counts
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

        # Sample move from policy (temperature=1)
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


def selfplay_train(model, iterations=15, games_per_iter=30, epochs=10,
                   num_sims=30, batch_size=32, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs("models", exist_ok=True)

    for iteration in range(iterations):
        print(f"\n{'='*50}")
        print(f"SELF-PLAY ITERATION {iteration+1}/{iterations}")
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

        print(f"Games: W={wins[1]} B={wins[-1]} D={wins[0]} | Pos: {len(all_s)}")
        val_arr = np.array(all_v)
        print(f"Values: +1={int(np.sum(val_arr==1))}, -1={int(np.sum(val_arr==-1))}, 0={int(np.sum(val_arr==0))}")

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

        print(f"Training: p_loss={tp/n:.4f} v_loss={tv/n:.4f}")
        torch.save(model.state_dict(), f"models/alpha_iter_{iteration+1}.pth")

    torch.save(model.state_dict(), "models/alpha_checkers.pth")
    print(f"\nFinal model saved!")
    return model


if __name__ == '__main__':
    print("="*50)
    print("STAGE 1: SUPERVISED PRE-TRAINING")
    print("="*50)
    model = AlphaCheckersNet()
    states, ptargets, vtargets = generate_supervised_data(200)
    model = supervised_train(model, states, ptargets, vtargets, epochs=30)
    torch.save(model.state_dict(), "models/alpha_supervised.pth")
    print("Supervised model saved!")

    print("\n" + "="*50)
    print("STAGE 2: SELF-PLAY REINFORCEMENT")
    print("="*50)
    selfplay_train(model, iterations=15, games_per_iter=30, num_sims=30)
