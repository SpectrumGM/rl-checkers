"""Monte Carlo Tree Search для AlphaZero.
Он использует нейронку чтобы умно искать лучший ход"""
import numpy as np
import torch
import math
from training.model import AlphaCheckersNet, board_to_tensor, encode_move


class MCTSNode:
    """Один узел дерева поиска."""
    def __init__(self, game, parent=None, move=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.move = move
        self.prior = prior  # P(s,a) от нейронки
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def value(self):
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb_score(self, c_puct=1.4):
        """Upper Confidence Bound — баланс exploration vs exploitation."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return exploitation + exploration

    def best_child(self):
        """Выбирает ребёнка с лучшим UCB."""
        return max(self.children, key=lambda c: c.ucb_score())

    def best_move_child(self):
        """Выбирает ребёнка с наибольшим числом визитов (для финального хода)."""
        return max(self.children, key=lambda c: c.visits)


class MCTS:
    """Monte Carlo Tree Search с нейронкой."""
    def __init__(self, model, num_simulations=100):
        self.model = model
        self.model.eval()
        self.num_simulations = num_simulations

    def get_policy(self, game):
        """
        Запускает MCTS и возвращает распределение ходов.
        Возвращает: (best_move, move_probs_dict)
        """
        root = MCTSNode(game.clone())
        self._expand(root)

        for _ in range(self.num_simulations):
            node = root
            scratch_game = game.clone()

            # 1. SELECT — идём вниз по дереву
            while node.is_expanded and node.children:
                node = node.best_child()
                scratch_game.make_move(node.move)

            # 2. EVALUATE — проверяем конец игры или оцениваем нейронкой
            winner = scratch_game.get_winner()
            if winner is not None:
                if winner == 0:
                    value = 0.0
                else:
                    # value с точки зрения текущего игрока
                    value = 1.0 if winner == scratch_game.current_player else -1.0
            else:
                # 3. EXPAND — раскрываем узел
                node.game = scratch_game.clone()
                self._expand(node)
                # Оценка нейронкой
                value = self._evaluate(scratch_game)

            # 4. BACKPROPAGATE — обновляем статистику вверх
            self._backpropagate(node, -value)

        # Выбираем ход
        if not root.children:
            moves = game.get_legal_moves()
            return moves[0] if moves else None, {}

        # Собираем статистику визитов
        move_probs = {}
        total_visits = sum(c.visits for c in root.children)
        for child in root.children:
            move_key = str(child.move)
            move_probs[move_key] = child.visits / total_visits if total_visits > 0 else 0

        best_child = root.best_move_child()
        return best_child.move, move_probs

    def _expand(self, node):
        """Раскрывает узел — создаёт детей для каждого легального хода."""
        if node.is_expanded:
            return
        game = node.game
        moves = game.get_legal_moves(game.current_player)
        if not moves:
            node.is_expanded = True
            return

        # Получаем policy от нейронки
        state = board_to_tensor(game.board, game.current_player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
        policy = torch.exp(policy_logits).squeeze().numpy()

        # Создаём детей с prior от нейронки
        priors = []
        for move in moves:
            idx = encode_move(move)
            priors.append(policy[idx])

        # Нормализуем
        prior_sum = sum(priors)
        if prior_sum > 0:
            priors = [p / prior_sum for p in priors]
        else:
            priors = [1.0 / len(moves)] * len(moves)

        for move, prior in zip(moves, priors):
            child_game = game.clone()
            child_game.make_move(move)
            child = MCTSNode(child_game, parent=node, move=move, prior=prior)
            node.children.append(child)

        node.is_expanded = True

    def _evaluate(self, game):
        """Оценивает позицию нейронкой."""
        state = board_to_tensor(game.board, game.current_player)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(state_tensor)
        return value.item()

    def _backpropagate(self, node, value):
        """Обновляет статистику от листа до корня."""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value  # Переключаем перспективу
            node = node.parent


class MCTSAgent:
    """Агент который использует MCTS для выбора хода."""
    def __init__(self, model_path=None, num_simulations=100):
        self.model = AlphaCheckersNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.mcts = MCTS(self.model, num_simulations)

    def choose_move(self, game):
        move, _ = self.mcts.get_policy(game)
        return move
