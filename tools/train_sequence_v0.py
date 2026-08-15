"""Quick residual sequence-model sanity check for btc_5m_sequences_v0_from_fv.

The model is intentionally small: a causal GRU predicts a residual over the
market-implied logit at every valid timestep.  Loss/metrics use valid_mask only,
so the experiment stays aligned with live-replicable timesteps.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_sequences_v0_from_fv"
DEFAULT_OUT = ROOT / "artifacts" / "sequence_v0_gru"


def _logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    y = y.astype(float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p.astype(float) - y.astype(float)) ** 2))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), p.astype(float)))


def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "brier": _brier(p, y),
        "logloss": _logloss(p, y),
        "auc": _auc(p, y),
    }


def _load_split(path: Path, split: str, device: torch.device) -> dict[str, Any]:
    d = np.load(path / f"{split}.npz", allow_pickle=True)
    x = d["X"].astype(np.float32)
    valid = d["valid_mask"].astype(bool)
    y_ep = d["y"].astype(np.float32)
    y = np.repeat(y_ep[:, None], x.shape[1], axis=1).astype(np.float32)
    p_market = d["p_market"].astype(np.float32)
    p_market[~np.isfinite(p_market)] = 0.5
    p_market = np.clip(p_market, 1e-4, 1 - 1e-4)
    return {
        "X": torch.from_numpy(x).to(device),
        "valid": torch.from_numpy(valid).to(device),
        "y": torch.from_numpy(y).to(device),
        "market_logit": torch.from_numpy(_logit_np(p_market)).to(device),
        "np_y": y,
        "np_valid": valid,
        "np_market_p": p_market,
    }


class ResidualGRU(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


@torch.no_grad()
def evaluate(model: nn.Module, data: dict[str, Any], batch_size: int) -> dict[str, Any]:
    model.eval()
    x = data["X"]
    valid = data["valid"]
    y = data["y"]
    base = data["market_logit"]
    preds = []
    ys = []
    losses = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        mb = valid[i:i + batch_size]
        yb = y[i:i + batch_size]
        bb = base[i:i + batch_size]
        logits = bb + model(xb)
        if mb.any():
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mb], yb[mb])
            losses.append(float(loss.cpu()))
            preds.append(torch.sigmoid(logits[mb]).cpu().numpy())
            ys.append(yb[mb].cpu().numpy())
    p = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    yy = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    out = _metrics(p, yy) if len(yy) else {"n": 0, "brier": float("nan"), "logloss": float("nan"), "auc": float("nan")}
    out["loss"] = float(np.mean(losses)) if losses else float("nan")
    return out


def baseline_metrics(data: dict[str, Any]) -> dict[str, float]:
    valid = data["np_valid"]
    p = data["np_market_p"][valid]
    y = data["np_y"][valid]
    return _metrics(p, y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=18)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    train = _load_split(args.dataset, "train", device)
    val = _load_split(args.dataset, "val", device)
    test = _load_split(args.dataset, "test", device)
    n_features = int(train["X"].shape[-1])
    model = ResidualGRU(n_features, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = TensorDataset(train["X"], train["valid"], train["y"], train["market_logit"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    print(f"device={device} train={train['X'].shape} val={val['X'].shape} test={test['X'].shape} features={n_features}")
    print("baseline market:", json.dumps({
        "train": baseline_metrics(train),
        "val": baseline_metrics(val),
        "test": baseline_metrics(test),
    }, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for xb, mb, yb, bb in loader:
            opt.zero_grad(set_to_none=True)
            logits = bb + model(xb)
            if not mb.any():
                continue
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mb], yb[mb])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            batch_losses.append(float(loss.detach().cpu()))
        val_m = evaluate(model, val, args.batch_size)
        tr_m = evaluate(model, train, args.batch_size)
        history.append({"epoch": epoch, "train": tr_m, "val": val_m, "train_loss": float(np.mean(batch_losses))})
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch:03d} train_logloss={tr_m['logloss']:.5f} val_logloss={val_m['logloss']:.5f} "
                  f"val_brier={val_m['brier']:.5f} val_auc={val_m['auc']:.4f}")
        if val_m["logloss"] + 1e-6 < best_val:
            best_val = val_m["logloss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early_stop epoch={epoch} best_val_logloss={best_val:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final = {
        "dataset": str(args.dataset),
        "device": str(device),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "market_baseline": {
            "train": baseline_metrics(train),
            "val": baseline_metrics(val),
            "test": baseline_metrics(test),
        },
        "gru_residual": {
            "train": evaluate(model, train, args.batch_size),
            "val": evaluate(model, val, args.batch_size),
            "test": evaluate(model, test, args.batch_size),
        },
        "history": history,
        "elapsed_s": round(time.time() - t0, 2),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sequence_v0_gru_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out / "model.pt")
    print("\nFINAL")
    print(json.dumps({k: final[k] for k in ("market_baseline", "gru_residual", "elapsed_s")}, indent=2))
    print(f"report -> {args.out / 'sequence_v0_gru_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
