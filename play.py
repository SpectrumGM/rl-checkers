import sys
import pygame
import torch
from game.checkers import CheckersGame
from game.display import CheckersDisplay
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from training.mcts import MCTSAgent
from training.model import AlphaCheckersNet, board_to_tensor

def get_eval(model, game):
    state = board_to_tensor(game.board, 1)
    tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        _, value = model(tensor)
    return value.item()

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "human"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    game = CheckersGame()
    display = CheckersDisplay(game)

    eval_model = AlphaCheckersNet()
    try:
        eval_model.load_state_dict(torch.load("models/alpha_checkers.pth", weights_only=True))
        eval_model.eval()
        has_eval = True
    except:
        has_eval = False

    white_agent = None
    black_agent = None

    if mode == "human":
        pass
    elif mode == "random":
        black_agent = RandomAgent()
    elif mode == "minimax":
        black_agent = MinimaxAgent(depth=depth)
    elif mode == "alpha":
        white_agent = MCTSAgent("models/alpha_checkers.pth", num_simulations=100)
        black_agent = MinimaxAgent(depth=depth)
    elif mode == "alpha_black":
        white_agent = MinimaxAgent(depth=depth)
        black_agent = MCTSAgent("models/alpha_checkers.pth", num_simulations=100)
    elif mode == "play_alpha":
        black_agent = MCTSAgent("models/alpha_checkers.pth", num_simulations=100)
    elif mode == "alpha_vs_random":
        white_agent = MCTSAgent("models/alpha_checkers.pth", num_simulations=100)
        black_agent = RandomAgent()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    game = CheckersGame()
                    display = CheckersDisplay(game)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.get_winner() is None:
                    current = game.current_player
                    if (current == 1 and white_agent is None) or \
                       (current == -1 and black_agent is None):
                        move = display.handle_click(*event.pos)
                        if move:
                            game.make_move(move)

        if game.get_winner() is None:
            current = game.current_player
            agent = white_agent if current == 1 else black_agent
            if agent:
                move = agent.choose_move(game)
                if move:
                    game.make_move(move)
                    display.selected = None

        if has_eval:
            display.eval_score = get_eval(eval_model, game)

        display.draw()
        pygame.time.delay(200)

    pygame.quit()

if __name__ == '__main__':
    main()
