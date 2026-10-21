"""Convert models/alpha_checkers.pth (PyTorch) into web/model.npz (NumPy) for the web demo.

BatchNorm in eval mode is a fixed affine map, so it is folded into the conv before it:
    scale = gamma / sqrt(running_var + eps)
    w'    = w * scale
    b'    = (b - running_mean) * scale + beta

Run from the repo root:  python web/export_weights.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.model import AlphaCheckersNet  # noqa: E402

SRC = ROOT / "models/alpha_checkers.pth"
DST = ROOT / "web/model.npz"
PAIRS = [("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3"),
         ("policy_conv", "policy_bn"), ("value_conv", "value_bn")]


def fold(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None, None]
    b = (conv.bias - bn.running_mean) * scale + bn.bias
    return w.detach().numpy().astype(np.float32), b.detach().numpy().astype(np.float32)


if __name__ == "__main__":
    net = AlphaCheckersNet()
    net.load_state_dict(torch.load(SRC, weights_only=True))
    net.eval()
    out = {}
    for c, b in PAIRS:
        out[f"{c}.w"], out[f"{c}.b"] = fold(getattr(net, c), getattr(net, b))
    for fc in ("policy_fc", "value_fc1", "value_fc2"):
        layer = getattr(net, fc)
        out[f"{fc}.w"] = layer.weight.detach().numpy().astype(np.float32)
        out[f"{fc}.b"] = layer.bias.detach().numpy().astype(np.float32)
    np.savez_compressed(DST, **out)
    print(f"saved {DST} ({DST.stat().st_size / 1e6:.1f} MB)")

    # check: NumPy forward pass must match PyTorch
    from web.engine import NumpyNet, board_to_tensor  # noqa: E402
    from game.checkers import CheckersGame  # noqa: E402
    import random
    npnet, g, rng = NumpyNet(DST), CheckersGame(), random.Random(0)
    worst = 0.0
    for _ in range(40):
        if g.get_winner() is not None:
            g = CheckersGame()
        x = board_to_tensor(g.board, g.current_player)
        with torch.no_grad():
            lp, v = net(torch.from_numpy(x)[None])
        p_np, v_np = npnet(x)
        worst = max(worst, float(np.abs(np.exp(lp[0].numpy()) - p_np).max()), abs(v.item() - v_np))
        g.make_move(rng.choice(g.get_legal_moves()))
    print(f"max |torch - numpy| over 40 positions: {worst:.2e}")
    assert worst < 1e-4
