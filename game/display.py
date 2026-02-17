
import pygame
import sys
import numpy as np

# Цвета — тёплое дерево
DARK_SQUARE = (139, 90, 43)
LIGHT_SQUARE = (222, 184, 135)
BOARD_BORDER = (60, 30, 10)
WHITE_PIECE = (255, 248, 240)
WHITE_PIECE_EDGE = (200, 190, 175)
BLACK_PIECE = (40, 40, 40)
BLACK_PIECE_EDGE = (20, 20, 20)
KING_GOLD = (218, 165, 32)
HIGHLIGHT = (100, 200, 100, 120)
SELECTED = (255, 255, 100, 150)
LAST_MOVE = (80, 160, 220, 100)
BG_COLOR = (30, 30, 30)
TEXT_COLOR = (220, 220, 220)

SQUARE_SIZE = 80
BORDER = 30
BOARD_PX = SQUARE_SIZE * 8
PANEL_W = 300
WIN_W = BORDER * 2 + BOARD_PX + PANEL_W
WIN_H = BORDER * 2 + BOARD_PX


class CheckersDisplay:
    def __init__(self, game):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Checkers AI")
        self.clock = pygame.time.Clock()
        self.game = game
        self.selected = None  # (row, col) выбранной фишки
        self.valid_moves = []  # подсвеченные ходы
        self.last_move = None  # последний ход для подсветки
        self.font = pygame.font.SysFont("Helvetica", 18)
        self.font_big = pygame.font.SysFont("Helvetica", 28, bold=True)
        self.eval_score = 0.0
        self.font_small = pygame.font.SysFont("Helvetica", 14)
        self.message = ""
        self.game_over = False

    def _board_to_pixel(self, row, col):
        x = BORDER + col * SQUARE_SIZE
        y = BORDER + row * SQUARE_SIZE
        return x, y

    def _pixel_to_board(self, mx, my):
        col = (mx - BORDER) // SQUARE_SIZE
        row = (my - BORDER) // SQUARE_SIZE
        if 0 <= row < 8 and 0 <= col < 8:
            return int(row), int(col)
        return None, None

    def _draw_board(self):
        # Фон
        self.screen.fill(BG_COLOR)
        # Рамка доски
        pygame.draw.rect(self.screen, BOARD_BORDER,
                         (BORDER-4, BORDER-4, BOARD_PX+8, BOARD_PX+8), border_radius=4)
        # Клетки
        for row in range(8):
            for col in range(8):
                x, y = self._board_to_pixel(row, col)
                color = DARK_SQUARE if (row + col) % 2 == 1 else LIGHT_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))

        # Координаты
        for i in range(8):
            # Цифры слева
            txt = self.font_small.render(str(i), True, TEXT_COLOR)
            self.screen.blit(txt, (BORDER - 18, BORDER + i * SQUARE_SIZE + SQUARE_SIZE//2 - 7))
            # Буквы снизу
            txt = self.font_small.render(str(i), True, TEXT_COLOR)
            self.screen.blit(txt, (BORDER + i * SQUARE_SIZE + SQUARE_SIZE//2 - 5, BORDER + BOARD_PX + 6))

    def _draw_highlights(self):
        # Подсветка последнего хода
        if self.last_move:
            for (fr, fc, tr, tc) in self.last_move:
                for (r, c) in [(fr, fc), (tr, tc)]:
                    x, y = self._board_to_pixel(r, c)
                    s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    s.fill(LAST_MOVE)
                    self.screen.blit(s, (x, y))

        # Подсветка выбранной фишки
        if self.selected:
            r, c = self.selected
            x, y = self._board_to_pixel(r, c)
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            s.fill(SELECTED)
            self.screen.blit(s, (x, y))

        # Подсветка доступных ходов
        for move_chain in self.valid_moves:
            tr, tc = move_chain[-1][2], move_chain[-1][3]
            x, y = self._board_to_pixel(tr, tc)
            cx = x + SQUARE_SIZE // 2
            cy = y + SQUARE_SIZE // 2
            pygame.draw.circle(self.screen, (100, 200, 100), (cx, cy), 12)
            pygame.draw.circle(self.screen, (60, 160, 60), (cx, cy), 12, 2)

    def _draw_piece(self, row, col, piece):
        x, y = self._board_to_pixel(row, col)
        cx = x + SQUARE_SIZE // 2
        cy = y + SQUARE_SIZE // 2
        radius = SQUARE_SIZE // 2 - 8

        if piece in (1, 2):
            # Тень
            pygame.draw.circle(self.screen, (0, 0, 0, 80), (cx + 3, cy + 3), radius)
            # Фишка
            pygame.draw.circle(self.screen, WHITE_PIECE, (cx, cy), radius)
            pygame.draw.circle(self.screen, WHITE_PIECE_EDGE, (cx, cy), radius, 3)
            # Внутренний круг
            pygame.draw.circle(self.screen, (240, 230, 215), (cx, cy), radius - 6, 2)
            if piece == 2:  # Дамка
                pygame.draw.circle(self.screen, KING_GOLD, (cx, cy), 10)
                pygame.draw.circle(self.screen, (180, 130, 20), (cx, cy), 10, 2)
        elif piece in (-1, -2):
            # Тень
            pygame.draw.circle(self.screen, (0, 0, 0, 80), (cx + 3, cy + 3), radius)
            # Фишка
            pygame.draw.circle(self.screen, BLACK_PIECE, (cx, cy), radius)
            pygame.draw.circle(self.screen, (80, 80, 80), (cx, cy), radius, 3)
            # Внутренний круг
            pygame.draw.circle(self.screen, (60, 60, 60), (cx, cy), radius - 6, 2)
            if piece == -2:  # Дамка
                pygame.draw.circle(self.screen, KING_GOLD, (cx, cy), 10)
                pygame.draw.circle(self.screen, (180, 130, 20), (cx, cy), 10, 2)

    def _draw_pieces(self):
        for row in range(8):
            for col in range(8):
                piece = self.game.board[row][col]
                if piece != 0:
                    self._draw_piece(row, col, piece)

    def _draw_panel(self):
        px = BORDER * 2 + BOARD_PX + 10
        py = BORDER

        # Заголовок
        title = self.font_big.render("CHECKERS", True, TEXT_COLOR)
        self.screen.blit(title, (px, py))

        # Ход
        py += 45
        turn = "White" if self.game.current_player == 1 else "Black"
        color = WHITE_PIECE if self.game.current_player == 1 else (150, 150, 150)
        txt = self.font.render(f"Turn: {turn}", True, color)
        self.screen.blit(txt, (px, py))

        # Счёт
        py += 30
        w = self.game.get_pieces_count(1)
        b = self.game.get_pieces_count(-1)
        txt = self.font.render(f"White: {w}  |  Black: {b}", True, TEXT_COLOR)
        self.screen.blit(txt, (px, py))

        # Ход номер
        py += 30
        txt = self.font.render(f"Move: {self.game.move_count}", True, TEXT_COLOR)
        self.screen.blit(txt, (px, py))

        # Сообщение
        if self.message:
            py += 50
            txt = self.font.render(self.message, True, KING_GOLD)
            self.screen.blit(txt, (px, py))

        # === EVAL BAR ===
        bar_x = px + 190
        bar_y = BORDER + 10
        bar_w = 28
        bar_h = 350
        pygame.draw.rect(self.screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h))
        white_pct = max(0.0, min(1.0, (self.eval_score + 1) / 2))
        white_h = int(bar_h * white_pct)
        black_h = bar_h - white_h
        if white_h > 0:
            pygame.draw.rect(self.screen, (235, 235, 235), (bar_x, bar_y + black_h, bar_w, white_h))
        if black_h > 0:
            pygame.draw.rect(self.screen, (45, 45, 45), (bar_x, bar_y, bar_w, black_h))
        mid_y = bar_y + bar_h // 2
        pygame.draw.line(self.screen, KING_GOLD, (bar_x - 3, mid_y), (bar_x + bar_w + 3, mid_y), 2)
        ev_txt = self.font.render(f"{self.eval_score:+.2f}", True, TEXT_COLOR)
        self.screen.blit(ev_txt, (bar_x - 8, bar_y + bar_h + 8))

        # Инструкции
        py = WIN_H - 120
        instructions = [
            "Click piece to select",
            "Click green dot to move",
            "R = restart",
            "Q = quit"
        ]
        for line in instructions:
            txt = self.font_small.render(line, True, (120, 120, 120))
            self.screen.blit(txt, (px, py))
            py += 20

    def handle_click(self, mx, my):
        if self.game_over:
            return None
        row, col = self._pixel_to_board(mx, my)
        if row is None:
            return None

        # Если уже выбрана фишка — проверяем ход
        if self.selected:
            for move_chain in self.valid_moves:
                target_r, target_c = move_chain[-1][2], move_chain[-1][3]
                if row == target_r and col == target_c:
                    self.last_move = move_chain
                    self.selected = None
                    self.valid_moves = []
                    return move_chain

        # Выбираем новую фишку
        if self.game._is_own_piece(row, col, self.game.current_player):
            self.selected = (row, col)
            all_moves = self.game.get_legal_moves()
            self.valid_moves = [m for m in all_moves if m[0][0] == row and m[0][1] == col]
        else:
            self.selected = None
            self.valid_moves = []
        return None

    def draw(self):
        self._draw_board()
        self._draw_highlights()
        self._draw_pieces()
        self._draw_panel()
        pygame.display.flip()

    def set_message(self, msg):
        self.message = msg

    def quit(self):
        pygame.quit()


def play_human_vs_human():
    """Тест: два человека играют."""
    from game.checkers import CheckersGame
    game = CheckersGame()
    display = CheckersDisplay(game)

    running = True
    while running:
        display.draw()
        display.clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    display.selected = None
                    display.valid_moves = []
                    display.last_move = None
                    display.message = ""
                    display.game_over = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                move = display.handle_click(*event.pos)
                if move:
                    game.make_move(move)
                    winner = game.get_winner()
                    if winner is not None:
                        if winner == 1:
                            display.set_message("White wins!")
                        elif winner == -1:
                            display.set_message("Black wins!")
                        else:
                            display.set_message("Draw!")
                        display.game_over = True

    display.quit()


if __name__ == '__main__':
    play_human_vs_human()