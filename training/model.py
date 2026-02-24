"""
AlphaZero-style нейронка для шашек.
Два выхода: policy (какой ход) + value (кто выиграет).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AlphaCheckersNet(nn.Module):
    """
    Вход: доска 8x8 (4 канала: свои фишки, свои дамки, чужие фишки, чужие дамки)
    Выход 1 (policy): вероятность каждого хода
    Выход 2 (value): оценка позиции от -1 до +1
    """
    def __init__(self, num_actions=200):
        super().__init__()
        self.num_actions = num_actions

        # Общий ствол (shared backbone)
        self.conv1 = nn.Conv2d(4, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Policy head — какой ход сделать
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_actions)

        # Value head — кто выиграет
        self.value_conv = nn.Conv2d(128, 16, 1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Общий ствол
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


def board_to_tensor(board, player):
    """
    Конвертирует доску в 4-канальный тензор.
    Канал 0: свои фишки
    Канал 1: свои дамки
    Канал 2: чужие фишки
    Канал 3: чужие дамки
    """
    state = np.zeros((4, 8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if player == 1:
                if piece == 1:   state[0][r][c] = 1
                elif piece == 2: state[1][r][c] = 1
                elif piece == -1: state[2][r][c] = 1
                elif piece == -2: state[3][r][c] = 1
            else:
                # Переворачиваем перспективу для чёрных
                if piece == -1:  state[0][7-r][c] = 1
                elif piece == -2: state[1][7-r][c] = 1
                elif piece == 1:  state[2][7-r][c] = 1
                elif piece == 2:  state[3][7-r][c] = 1
    return state


def encode_move(move_chain):
    """
    Кодирует ход в индекс 0-199.
    Простое кодирование: from_square * 4 + direction.
    """
    fr, fc = move_chain[0][0], move_chain[0][1]
    tr, tc = move_chain[-1][2], move_chain[-1][3]
    from_sq = fr * 8 + fc
    dr = 1 if tr > fr else 0
    dc = 1 if tc > fc else 0
    direction = dr * 2 + dc
    idx = (from_sq * 4 + direction) % 200
    return idx


def decode_move(idx, legal_moves):
    """Находит ближайший легальный ход к индексу."""
    if not legal_moves:
        return None
    # Кодируем все легальные ходы и ищем совпадение
    for move in legal_moves:
        if encode_move(move) == idx:
            return move
    # Если точного совпадения нет — вернуть None
    return None


class AlphaCheckersAgent:
    """Агент без MCTS — просто нейронка."""
    def __init__(self, model_path=None):
        self.model = AlphaCheckersNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def choose_move(self, game):
        moves = game.get_legal_moves()
        if not moves:
            return None

        state = board_to_tensor(game.board, game.current_player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            policy, value = self.model(state_tensor)

        policy = torch.exp(policy).squeeze().numpy()

        # Маскируем нелегальные ходы
        legal_probs = []
        for move in moves:
            idx = encode_move(move)
            legal_probs.append(policy[idx])

        # Выбираем ход с наибольшей вероятностью
        legal_probs = np.array(legal_probs)
        if legal_probs.sum() > 0:
            legal_probs /= legal_probs.sum()
        else:
            legal_probs = np.ones(len(moves)) / len(moves)

        best_idx = np.argmax(legal_probs)
        return moves[best_idx]