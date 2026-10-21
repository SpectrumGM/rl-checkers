"""Inference without PyTorch, for the web demo.

The network from training/model.py is re-implemented in NumPy so the server only needs
numpy + fastapi (fits a free hosting tier). BatchNorm is folded into the preceding
convolution, which is exact in eval mode. MCTS follows training/mcts.py one to one.

Weights come from web/model.npz, created by web/export_weights.py.
"""
import math
from pathlib import Path

import numpy as np

from game.checkers import CheckersGame
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent

NUM_ACTIONS = 200


# ---------- same input / move encoding as training/model.py ----------

def board_to_tensor(board, player):
    """4 x 8 x 8: own men, own kings, opponent men, opponent kings. Black sees a flipped board."""
    b = np.asarray(board)
    if player == 1:
        planes = [b == 1, b == 2, b == -1, b == -2]
    else:
        f = b[::-1]
        planes = [f == -1, f == -2, f == 1, f == 2]
    return np.stack(planes).astype(np.float32)


def encode_move(move_chain):
    fr, fc = move_chain[0][0], move_chain[0][1]
    tr, tc = move_chain[-1][2], move_chain[-1][3]
    direction = (1 if tr > fr else 0) * 2 + (1 if tc > fc else 0)
    return ((fr * 8 + fc) * 4 + direction) % NUM_ACTIONS


# ---------- network ----------

def _conv3x3(x, w, b):
    """x: C x 8 x 8, w: O x C x 3 x 3, padding 1."""
    c = x.shape[0]
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((c, 3, 3, 8, 8), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            cols[:, i, j] = p[:, i:i + 8, j:j + 8]
    out = w.reshape(w.shape[0], -1) @ cols.reshape(-1, 64)
    return (out + b[:, None]).reshape(-1, 8, 8)


def _conv1x1(x, w, b):
    out = w.reshape(w.shape[0], -1) @ x.reshape(x.shape[0], -1)
    return (out + b[:, None]).reshape(-1, 8, 8)


relu = lambda z: np.maximum(z, 0)


class NumpyNet:
    def __init__(self, path):
        self.p = dict(np.load(path))

    def __call__(self, x):
        """x: 4 x 8 x 8 -> (policy probabilities over 200 actions, value in [-1, 1])."""
        p = self.p
        h = relu(_conv3x3(x, p["conv1.w"], p["conv1.b"]))
        h = relu(_conv3x3(h, p["conv2.w"], p["conv2.b"]))
        h = relu(_conv3x3(h, p["conv3.w"], p["conv3.b"]))

        ph = relu(_conv1x1(h, p["policy_conv.w"], p["policy_conv.b"])).reshape(-1)
        logits = p["policy_fc.w"] @ ph + p["policy_fc.b"]
        logits -= logits.max()
        policy = np.exp(logits)
        policy /= policy.sum()

        vh = relu(_conv1x1(h, p["value_conv.w"], p["value_conv.b"])).reshape(-1)
        vh = relu(p["value_fc1.w"] @ vh + p["value_fc1.b"])
        value = float(np.tanh(p["value_fc2.w"] @ vh + p["value_fc2.b"])[0])
        return policy, value

    def evaluate(self, game):
        """Value of the position for White, in [-1, 1]."""
        _, v = self(board_to_tensor(game.board, 1))
        return v


# ---------- MCTS, same algorithm as training/mcts.py ----------

class Node:
    __slots__ = ("game", "parent", "move", "prior", "children", "visits", "value_sum", "expanded")

    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game, self.parent, self.move, self.prior = game, parent, move, prior
        self.children, self.visits, self.value_sum, self.expanded = [], 0, 0.0, False

    def ucb(self, c_puct=1.4):
        if self.visits == 0:
            return float("inf")
        return self.value_sum / self.visits + c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)


class MCTSPlayer:
    def __init__(self, net, num_simulations=100):
        self.net, self.n = net, num_simulations

    def _expand(self, node):
        if node.expanded:
            return
        g = node.game
        moves = g.get_legal_moves(g.current_player)
        node.expanded = True
        if not moves:
            return
        policy, _ = self.net(board_to_tensor(g.board, g.current_player))
        pri = np.array([policy[encode_move(m)] for m in moves])
        pri = pri / pri.sum() if pri.sum() > 0 else np.full(len(moves), 1 / len(moves))
        for m, pr in zip(moves, pri):
            child = g.clone()
            child.make_move(m)
            node.children.append(Node(child, node, m, float(pr)))

    def choose_move(self, game):
        root = Node(game.clone())
        self._expand(root)
        if not root.children:
            return None
        for _ in range(self.n):
            node, g = root, game.clone()
            while node.expanded and node.children:
                node = max(node.children, key=lambda c: c.ucb())
                g.make_move(node.move)
            w = g.get_winner()
            if w is not None:
                value = 0.0 if w == 0 else (1.0 if w == g.current_player else -1.0)
            else:
                node.game = g.clone()
                self._expand(node)
                value = self.net(board_to_tensor(g.board, g.current_player))[1]
            value = -value
            while node is not None:
                node.visits += 1
                node.value_sum += value
                value = -value
                node = node.parent
        return max(root.children, key=lambda c: c.visits).move


# ---------- opponents offered on the site ----------

def make_opponent(name, net):
    if name == "alphazero":
        return MCTSPlayer(net, num_simulations=50)
    if name == "network":
        return NetOnlyPlayer(net)
    if name == "random":
        return RandomAgent()
    if name.startswith("minimax"):
        return MinimaxAgent(depth=int(name[-1]))
    raise ValueError(name)


class NetOnlyPlayer:
    """Network alone, no search: plays the move with the highest policy probability."""

    def __init__(self, net):
        self.net = net

    def choose_move(self, game):
        moves = game.get_legal_moves()
        if not moves:
            return None
        policy, _ = self.net(board_to_tensor(game.board, game.current_player))
        return max(moves, key=lambda m: policy[encode_move(m)])


def game_from_state(s):
    g = CheckersGame()
    g.board = np.array(s["board"], dtype=int)
    g.current_player = int(s["current_player"])
    g.move_count = int(s.get("move_count", 0))
    g.no_capture_count = int(s.get("no_capture_count", 0))
    return g


def state_of(g):
    return {"board": g.board.tolist(), "current_player": g.current_player,
            "move_count": g.move_count, "no_capture_count": g.no_capture_count}


ROOT = Path(__file__).resolve().parent
