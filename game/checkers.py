import numpy as np


class CheckersGame:
    def __init__(self):
        self.board = self._initial_board()
        self.current_player = 1
        self.move_count = 0
        self.max_moves = 999
        self.no_capture_count = 0
        self.max_no_capture = 40

    def _initial_board(self):
        board = np.zeros((8, 8), dtype=int)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = -1
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    board[row][col] = 1
        return board

    def reset(self):
        self.board = self._initial_board()
        self.current_player = 1
        self.move_count = 0
        self.no_capture_count = 0
        return self.get_state()

    def get_state(self):
        return self.board.copy()

    def get_pieces_count(self, player):
        if player == 1:
            return int(np.sum(self.board == 1) + np.sum(self.board == 2))
        else:
            return int(np.sum(self.board == -1) + np.sum(self.board == -2))

    def _is_own_piece(self, row, col, player):
        if player == 1:
            return self.board[row][col] in (1, 2)
        return self.board[row][col] in (-1, -2)

    def _in_bounds(self, row, col):
        return 0 <= row < 8 and 0 <= col < 8

    def _get_move_directions(self, row, col):
        piece = self.board[row][col]
        if piece == 1:
            return [(-1, -1), (-1, 1)]
        elif piece == -1:
            return [(1, -1), (1, 1)]
        elif piece in (2, -2):
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return []

    def get_simple_moves(self, player):
        moves = []
        for row in range(8):
            for col in range(8):
                if not self._is_own_piece(row, col, player):
                    continue
                for dr, dc in self._get_move_directions(row, col):
                    nr, nc = row + dr, col + dc
                    if self._in_bounds(nr, nc) and self.board[nr][nc] == 0:
                        moves.append((row, col, nr, nc))
        return moves

    def _get_jumps_for_piece(self, row, col, player, board=None):
        if board is None:
            board = self.board
        piece = board[row][col]
        if piece == 1:
            directions = [(-1, -1), (-1, 1)]
        elif piece == -1:
            directions = [(1, -1), (1, 1)]
        elif piece in (2, -2):
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            return []

        jumps = []
        for dr, dc in directions:
            mid_r, mid_c = row + dr, col + dc
            end_r, end_c = row + 2*dr, col + 2*dc
            if not self._in_bounds(end_r, end_c):
                continue
            if board[end_r][end_c] != 0:
                continue
            mid_piece = board[mid_r][mid_c]
            if player == 1 and mid_piece not in (-1, -2):
                continue
            if player == -1 and mid_piece not in (1, 2):
                continue

            new_board = board.copy()
            new_board[end_r][end_c] = piece
            new_board[row][col] = 0
            new_board[mid_r][mid_c] = 0

            became_king = False
            if piece == 1 and end_r == 0:
                new_board[end_r][end_c] = 2
                became_king = True
            elif piece == -1 and end_r == 7:
                new_board[end_r][end_c] = -2
                became_king = True

            first_jump = (row, col, end_r, end_c)
            if became_king:
                jumps.append([first_jump])
                continue

            continuations = self._get_jumps_for_piece(end_r, end_c, player, new_board)
            if continuations:
                for chain in continuations:
                    jumps.append([first_jump] + chain)
            else:
                jumps.append([first_jump])
        return jumps

    def get_jump_moves(self, player):
        all_jumps = []
        for row in range(8):
            for col in range(8):
                if self._is_own_piece(row, col, player):
                    all_jumps.extend(self._get_jumps_for_piece(row, col, player))
        return all_jumps

    def get_legal_moves(self, player=None):
        if player is None:
            player = self.current_player
        jumps = self.get_jump_moves(player)
        if jumps:
            return jumps
        return [[m] for m in self.get_simple_moves(player)]

    def make_move(self, move_chain):
        captured = 0
        for (fr, fc, tr, tc) in move_chain:
            piece = self.board[fr][fc]
            self.board[tr][tc] = piece
            self.board[fr][fc] = 0
            if abs(tr - fr) == 2:
                self.board[(fr+tr)//2][(fc+tc)//2] = 0
                captured += 1
        final_r, final_c = move_chain[-1][2], move_chain[-1][3]
        if self.board[final_r][final_c] == 1 and final_r == 0:
            self.board[final_r][final_c] = 2
        elif self.board[final_r][final_c] == -1 and final_r == 7:
            self.board[final_r][final_c] = -2
        self.move_count += 1
        self.no_capture_count = 0 if captured > 0 else self.no_capture_count + 1
        self.current_player *= -1
        return captured

    def get_winner(self):
        if self.get_pieces_count(1) == 0:
            return -1
        if self.get_pieces_count(-1) == 0:
            return 1
        if not self.get_legal_moves(self.current_player):
            return -self.current_player
        if self.move_count >= self.max_moves or self.no_capture_count >= self.max_no_capture:
            return 0
        return None

    def get_board_features(self):
        b = self.board
        wm, bm = int(np.sum(b==1)), int(np.sum(b==-1))
        wk, bk = int(np.sum(b==2)), int(np.sum(b==-2))
        mat = (wm + 1.5*wk) - (bm + 1.5*bk)
        cen_w = cen_b = 0
        for r in [3,4]:
            for c in [3,4]:
                if b[r][c] in (1,2): cen_w += 1
                elif b[r][c] in (-1,-2): cen_b += 1
        adv_w = adv_b = 0
        for r in range(8):
            for c in range(8):
                if b[r][c] == 1: adv_w += (7-r)
                elif b[r][c] == -1: adv_b += r
        mob = len(self.get_legal_moves(1)) - len(self.get_legal_moves(-1))
        return np.array([wm-bm, wk-bk, mat, cen_w-cen_b, adv_w-adv_b, mob,
                         len(self.get_jump_moves(1))-len(self.get_jump_moves(-1))], dtype=np.float32)

    def clone(self):
        g = CheckersGame()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.move_count = self.move_count
        g.no_capture_count = self.no_capture_count
        return g

    def __str__(self):
        sym = {0:'·', 1:'w', 2:'W', -1:'b', -2:'B'}
        lines = ['  0 1 2 3 4 5 6 7']
        for r in range(8):
            lines.append(f'{r} ' + ' '.join(sym[self.board[r][c]] for c in range(8)))
        lines.append(f'Turn: {"White" if self.current_player==1 else "Black"} | Move: {self.move_count}')
        return '\n'.join(lines)


if __name__ == '__main__':
    game = CheckersGame()
    print(game)
    print(f'\nWhite: {game.get_pieces_count(1)} | Black: {game.get_pieces_count(-1)}')
    moves = game.get_legal_moves()
    print(f'Legal moves: {len(moves)}')
    for i, m in enumerate(moves):
        print(f'  {i}: {m}')
    game.make_move(moves[0])
    print(f'\nAfter move 1:')
    print(game)
    print(f'\nFeatures: {game.get_board_features()}')