"""
Тестируем AlphaZero агента.
"""
from game.checkers import CheckersGame
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from training.mcts import MCTSAgent
from tqdm import tqdm


def test(white_agent, black_agent, n_games=50, label=""):
    wins = {1: 0, -1: 0, 0: 0}
    for _ in tqdm(range(n_games), desc=label):
        game = CheckersGame()
        while game.get_winner() is None:
            if game.current_player == 1:
                move = white_agent.choose_move(game)
            else:
                move = black_agent.choose_move(game)
            if move is None:
                break
            game.make_move(move)
        w = game.get_winner()
        wins[w if w is not None else 0] += 1
    print(f"  White wins: {wins[1]} ({wins[1]/n_games*100:.0f}%)")
    print(f"  Black wins: {wins[-1]} ({wins[-1]/n_games*100:.0f}%)")
    print(f"  Draws: {wins[0]} ({wins[0]/n_games*100:.0f}%)\n")


if __name__ == '__main__':
    alpha = MCTSAgent("models/alpha_checkers.pth", num_simulations=50)
    random_ag = RandomAgent()
    minimax = MinimaxAgent(depth=3)

    print("=== AlphaZero (white) vs Random (black) ===")
    test(alpha, random_ag, 30, "vs Random")

    print("=== AlphaZero (white) vs Minimax d3 (black) ===")
    test(alpha, minimax, 10, "vs Minimax")