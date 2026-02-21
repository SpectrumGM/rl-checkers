import numpy as np


class MinimaxAgent:
    def __init__(self, depth=4):
        self.depth = depth

    def evaluate(self, game):
        b = game.board
        wm = int(np.sum(b == 1))
        bm = int(np.sum(b == -1))
        wk = int(np.sum(b == 2))
        bk = int(np.sum(b == -2))
        material = (wm + 2.5 * wk) - (bm + 2.5 * bk)
        # Контроль центра
        center = 0
        for r in [3, 4]:
            for c in [3, 4]:
                if b[r][c] in (1, 2): center += 0.3
                elif b[r][c] in (-1, -2): center -= 0.3
        # Продвижение
        advance = 0
        for r in range(8):
            for c in range(8):
                if b[r][c] == 1: advance += (7 - r) * 0.05
                elif b[r][c] == -1: advance -= r * 0.05
        return material + center + advance

    def minimax(self, game, depth, alpha, beta, maximizing):
        winner = game.get_winner()
        if winner is not None:
            if winner == 1: return 100
            elif winner == -1: return -100
            else: return 0
        if depth == 0:
            return self.evaluate(game)

        moves = game.get_legal_moves()
        if not moves:
            return -100 if maximizing else 100

        if maximizing:
            max_eval = -999
            for move in moves:
                child = game.clone()
                child.make_move(move)
                val = self.minimax(child, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = 999
            for move in moves:
                child = game.clone()
                child.make_move(move)
                val = self.minimax(child, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_eval

    def choose_move(self, game):
        moves = game.get_legal_moves()
        if not moves:
            return None
        maximizing = game.current_player == 1
        best_move = moves[0]
        best_val = -999 if maximizing else 999

        for move in moves:
            child = game.clone()
            child.make_move(move)
            val = self.minimax(child, self.depth - 1, -999, 999, not maximizing)
            if maximizing and val > best_val:
                best_val = val
                best_move = move
            elif not maximizing and val < best_val:
                best_val = val
                best_move = move
        return best_move