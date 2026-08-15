"""Train a residual GRU on the 5Hz BTC episode dataset.

The model is causal in episode time: the GRU reads the sequence from market
open toward close and predicts a residual over logit(p_market) at each step.
Loss and headline metrics are masked to a configurable TTC window.
"""
from __future__ import annotations

import argparse
import json
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
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_OUT = ROOT / "artifacts" / "btc_5m_episode_gru_small_ttc15_90"


def _logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32, copy=False), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if y.size == 0:
        return {"n": 0, "brier": float("nan"), "logloss": float("nan"), "auc": float("nan")}
    p = np.clip(p.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)
    y = y.astype(np.float64, copy=False)
    out = {
        "n": int(y.size),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
    }
    if np.unique(y).size < 2:
        out["auc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y.astype(int), p))
    return out


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _ttc_mask(seq_len: int, cadence_s: float, lo: float, hi: float) -> np.ndarray:
    ttc = seq_len * cadence_s - np.arange(seq_len, dtype=np.float32) * cadence_s
    return (ttc > lo) & (ttc <= hi)


def _load_split(path: Path, split: str, *, cadence_s: float, ttc_min: float, ttc_max: float,
                device: torch.device, include_mask_channel: bool) -> dict[str, Any]:
    d = np.load(path / f"{split}.npz", allow_pickle=False)
    x_np = d["X"].astype(np.float32, copy=False)
    valid_np = d["valid_mask"].astype(bool, copy=False)
    ttc_band = _ttc_mask(x_np.shape[1], cadence_s, ttc_min, ttc_max)
    loss_mask_np = valid_np & ttc_band[None, :]
    y_ep_np = d["y"].astype(np.float32, copy=False)
    y_np = np.repeat(y_ep_np[:, None], x_np.shape[1], axis=1).astype(np.float32, copy=False)
    p_market_np = d["p_market"].astype(np.float32, copy=False)
    p_market_np[~np.isfinite(p_market_np)] = 0.5
    p_market_np = np.clip(p_market_np, 1e-5, 1.0 - 1e-5)
    if include_mask_channel:
        x_np = np.concatenate([x_np, valid_np[:, :, None].astype(np.float32)], axis=2)
    return {
        "X": torch.from_numpy(x_np).to(device),
        "loss_mask": torch.from_numpy(loss_mask_np).to(device),
        "y": torch.from_numpy(y_np).to(device),
        "market_logit": torch.from_numpy(_logit_np(p_market_np)).to(device),
        "np_loss_mask": loss_mask_np,
        "np_y": y_np,
        "np_market_p": p_market_np,
        "valid_count": d["valid_count"],
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


def baseline_metrics(data: dict[str, Any]) -> dict[str, float]:
    m = data["np_loss_mask"]
    return _metrics(data["np_market_p"][m], data["np_y"][m])


@torch.no_grad()
def collect_logits(model: nn.Module, data: dict[str, Any], batch_size: int,
                   delta_clamp: float | None) -> dict[str, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    base_out: list[np.ndarray] = []
    delta_out: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    x = data["X"]
    mask = data["loss_mask"]
    y = data["y"]
    base = data["market_logit"]
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        mb = mask[i:i + batch_size]
        if not mb.any():
            continue
        delta = model(xb)
        if delta_clamp is not None and delta_clamp > 0:
            delta = torch.clamp(delta, -float(delta_clamp), float(delta_clamp))
        bb = base[i:i + batch_size]
        logits = bb + delta
        logits_out.append(logits[mb].detach().cpu().numpy())
        base_out.append(bb[mb].detach().cpu().numpy())
        delta_out.append(delta[mb].detach().cpu().numpy())
        ys.append(y[i:i + batch_size][mb].detach().cpu().numpy())
    if not ys:
        empty_f = np.array([], dtype=np.float32)
        return {"logit": empty_f, "base_logit": empty_f, "delta": empty_f, "y": empty_f}
    return {
        "logit": np.concatenate(logits_out).astype(np.float32, copy=False),
        "base_logit": np.concatenate(base_out).astype(np.float32, copy=False),
        "delta": np.concatenate(delta_out).astype(np.float32, copy=False),
        "y": np.concatenate(ys).astype(np.float32, copy=False),
    }


def evaluate(model: nn.Module, data: dict[str, Any], batch_size: int,
             delta_clamp: float | None) -> dict[str, float]:
    z = collect_logits(model, data, batch_size, delta_clamp)
    return _metrics(_sigmoid_np(z["logit"]), z["y"])


def _fit_calibrators(val_logits: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    logit = val_logits["logit"].astype(np.float64, copy=False)
    base = val_logits["base_logit"].astype(np.float64, copy=False)
    delta = val_logits["delta"].astype(np.float64, copy=False)
    y = val_logits["y"].astype(np.int8, copy=False)
    out: dict[str, dict[str, Any]] = {"identity": {"kind": "identity"}}

    # Temperature scaling on the final model logit.
    temps = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))
    losses = [_metrics(_sigmoid_np(logit / t), y)["logloss"] for t in temps]
    best_i = int(np.nanargmin(losses))
    out["temperature"] = {
        "kind": "temperature",
        "temperature": float(temps[best_i]),
        "val_logloss": float(losses[best_i]),
    }

    # Platt scaling: sigmoid(a * model_logit + b).
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    out["platt"] = {
        "kind": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }

    # Isotonic on uncalibrated probabilities.  Useful diagnostic, risky on thin val.
    iso = IsotonicRegression(out_of_bounds="clip")
    p_uncal = _sigmoid_np(logit).astype(np.float64, copy=False)
    iso.fit(p_uncal, y)
    out["isotonic"] = {
        "kind": "isotonic",
        "x_thresholds": iso.X_thresholds_.astype(float).tolist(),
        "y_thresholds": iso.y_thresholds_.astype(float).tolist(),
    }

    # Residual shrinkage keeps the market prior and only scales GRU delta.
    alphas = np.linspace(0.0, 1.5, 151)
    losses = [_metrics(_sigmoid_np(base + a * delta), y)["logloss"] for a in alphas]
    best_i = int(np.nanargmin(losses))
    out["residual_shrink"] = {
        "kind": "residual_shrink",
        "alpha": float(alphas[best_i]),
        "val_logloss": float(losses[best_i]),
    }
    return out


def _apply_calibrator(cal: dict[str, Any], logits: dict[str, np.ndarray]) -> np.ndarray:
    kind = cal["kind"]
    if kind == "identity":
        return _sigmoid_np(logits["logit"])
    if kind == "temperature":
        return _sigmoid_np(logits["logit"] / float(cal["temperature"]))
    if kind == "platt":
        return _sigmoid_np(float(cal["coef"]) * logits["logit"] + float(cal["intercept"]))
    if kind == "isotonic":
        x = np.asarray(cal["x_thresholds"], dtype=np.float64)
        y = np.asarray(cal["y_thresholds"], dtype=np.float64)
        return np.interp(_sigmoid_np(logits["logit"]), x, y).astype(np.float32, copy=False)
    if kind == "residual_shrink":
        return _sigmoid_np(logits["base_logit"] + float(cal["alpha"]) * logits["delta"])
    raise ValueError(f"unknown calibrator kind={kind!r}")


def _evaluate_calibrated(
    split_logits: dict[str, dict[str, np.ndarray]],
    calibrators: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for split, logits in split_logits.items():
        out[split] = {}
        for name, cal in calibrators.items():
            out[split][name] = _metrics(_apply_calibrator(cal, logits), logits["y"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ttc-min", type=float, default=15.0)
    ap.add_argument("--ttc-max", type=float, default=90.0)
    ap.add_argument("--cadence-s", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--delta-clamp", type=float, default=0.0,
                    help="clip GRU residual logit delta to +/- this value; 0 disables")
    ap.add_argument("--delta-l2", type=float, default=0.0,
                    help="extra penalty on squared residual delta at loss timesteps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--include-mask-channel", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"device={device} torch_threads={torch.get_num_threads()} dataset={dataset}", flush=True)
    train = _load_split(dataset, "train", cadence_s=args.cadence_s, ttc_min=args.ttc_min,
                        ttc_max=args.ttc_max, device=device, include_mask_channel=args.include_mask_channel)
    val = _load_split(dataset, "val", cadence_s=args.cadence_s, ttc_min=args.ttc_min,
                      ttc_max=args.ttc_max, device=device, include_mask_channel=args.include_mask_channel)
    test = _load_split(dataset, "test", cadence_s=args.cadence_s, ttc_min=args.ttc_min,
                       ttc_max=args.ttc_max, device=device, include_mask_channel=args.include_mask_channel)

    n_features = int(train["X"].shape[-1])
    model = ResidualGRU(n_features, args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(train["X"], train["loss_mask"], train["y"], train["market_logit"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    market = {"train": baseline_metrics(train), "val": baseline_metrics(val), "test": baseline_metrics(test)}
    print(f"shapes train={tuple(train['X'].shape)} val={tuple(val['X'].shape)} test={tuple(test['X'].shape)}")
    print("market_baseline", json.dumps(market, indent=2), flush=True)

    best_val = float("inf")
    best_state = None
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        losses = []
        seen = 0
        for xb, mb, yb, bb in loader:
            opt.zero_grad(set_to_none=True)
            delta = model(xb)
            if args.delta_clamp > 0:
                delta = torch.clamp(delta, -args.delta_clamp, args.delta_clamp)
            logits = bb + delta
            if not mb.any():
                continue
            loss = nn.functional.binary_cross_entropy_with_logits(logits[mb], yb[mb])
            if args.delta_l2 > 0:
                loss = loss + float(args.delta_l2) * torch.mean(delta[mb] ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            seen += int(mb.sum().detach().cpu())
        val_m = evaluate(model, val, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "train_masked_steps_seen": seen,
            "val": val_m,
            "epoch_s": round(time.time() - ep_t0, 2),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.5f} "
            f"val_logloss={val_m['logloss']:.5f} val_brier={val_m['brier']:.5f} "
            f"val_auc={val_m['auc']:.4f} epoch_s={row['epoch_s']}",
            flush=True,
        )
        if val_m["logloss"] + 1e-6 < best_val:
            best_val = val_m["logloss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early_stop epoch={epoch} best_val_logloss={best_val:.5f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final = {
        "dataset": str(dataset),
        "device": str(device),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "market_baseline": market,
        "gru_residual": {
            "train": evaluate(model, train, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None),
            "val": evaluate(model, val, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None),
            "test": evaluate(model, test, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None),
        },
        "history": history,
        "elapsed_s": round(time.time() - t0, 2),
    }
    split_logits = {
        "train": collect_logits(model, train, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None),
        "val": collect_logits(model, val, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None),
        "test": collect_logits(model, test, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None),
    }
    calibrators = _fit_calibrators(split_logits["val"])
    final["calibrators"] = calibrators
    final["gru_calibrated"] = _evaluate_calibrated(split_logits, calibrators)
    (out_dir / "gru_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), out_dir / "model.pt")
    print("FINAL", json.dumps({
        "market_baseline": final["market_baseline"],
        "gru_residual": final["gru_residual"],
        "gru_calibrated": final["gru_calibrated"],
        "elapsed_s": final["elapsed_s"],
    }, indent=2), flush=True)
    print(f"report -> {out_dir / 'gru_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
