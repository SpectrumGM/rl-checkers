import random


class RandomAgent:
    def choose_move(self, game):
        moves = game.get_legal_moves()
        if not moves:
            return None
        return random.choice(moves)