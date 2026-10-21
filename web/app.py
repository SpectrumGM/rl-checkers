"""Web demo: play checkers against the trained engine, minimax, or random.

The server is stateless: the browser sends the whole position with every request,
so any number of people can play at once and nothing is stored.

    uvicorn web.app:app --reload        (from the repo root)
"""
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from web.engine import NumpyNet, game_from_state, make_opponent, state_of

HERE = Path(__file__).resolve().parent
NET = NumpyNet(HERE / "model.npz")

Opponent = Literal["alphazero", "network", "minimax1", "minimax2", "minimax3",
                   "minimax4", "minimax5", "random"]

app = FastAPI(title="AlphaZero Checkers", version="1.0")


class State(BaseModel):
    board: list[list[int]]
    current_player: int
    move_count: int = 0
    no_capture_count: int = 0


class MoveRequest(BaseModel):
    state: State
    move: list[list[int]]          # chain of [from_r, from_c, to_r, to_c] steps


class AiRequest(BaseModel):
    state: State
    opponent: Opponent


def describe(g):
    """Everything the page needs to draw a position."""
    winner = g.get_winner()
    return {
        "state": state_of(g),
        "legal_moves": [] if winner is not None else [[list(s) for s in m] for m in g.get_legal_moves()],
        "winner": winner,                       # 1 white, -1 black, 0 draw, None ongoing
        "eval": round(NET.evaluate(g), 3),      # network's opinion, + = good for White
    }


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/new")
def new_game():
    return describe(game_from_state(State(board=_start(), current_player=1).model_dump()))


@app.post("/api/move")
def human_move(req: MoveRequest):
    g = game_from_state(req.state.model_dump())
    if g.get_winner() is not None:
        raise HTTPException(400, "game is over")
    legal = [[list(s) for s in m] for m in g.get_legal_moves()]
    if req.move not in legal:
        raise HTTPException(400, "illegal move")
    g.make_move([tuple(s) for s in req.move])
    return describe(g)


@app.post("/api/ai")
def ai_move(req: AiRequest):
    g = game_from_state(req.state.model_dump())
    if g.get_winner() is not None:
        raise HTTPException(400, "game is over")
    move = make_opponent(req.opponent, NET).choose_move(g)
    g.make_move(move)
    out = describe(g)
    out["ai_move"] = [list(s) for s in move]
    return out


def _start():
    from game.checkers import CheckersGame
    return CheckersGame().board.tolist()
