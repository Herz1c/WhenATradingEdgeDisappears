"""Train a causal residual TCN on the 5Hz BTC episode dataset.

The model predicts a per-step residual over logit(p_market).  All convolutions
are left-padded causal convolutions, and normalization is per time step over
channels only, so the architecture is live-replicable.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_OUT = ROOT / "artifacts" / "btc_5m_episode_tcn_medium_ttc15_90"


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def _logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64, copy=False), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    if y.size == 0:
        return float("nan")
    p = np.clip(p.astype(np.float64, copy=False), 0.0, 1.0)
    y = y.astype(np.float64, copy=False)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(y.size)
    acc = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p <= hi) if i == bins - 1 else (p >= lo) & (p < hi)
        if m.any():
            acc += float(m.sum()) / total * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(acc)


def _metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if y.size == 0:
        return {
            "n": 0,
            "p_mean": float("nan"),
            "y_mean": float("nan"),
            "brier": float("nan"),
            "logloss": float("nan"),
            "auc": float("nan"),
            "ece_20": float("nan"),
        }
    p = np.clip(p.astype(np.float64, copy=False), 1e-6, 1.0 - 1e-6)
    y = y.astype(np.float64, copy=False)
    out = {
        "n": int(y.size),
        "p_mean": float(np.mean(p)),
        "y_mean": float(np.mean(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "ece_20": _ece(p, y, bins=20),
    }
    out["auc"] = float("nan") if np.unique(y).size < 2 else float(roc_auc_score(y.astype(int), p))
    return out


def _per_day_metrics(p: np.ndarray, y: np.ndarray, dates: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for day in sorted(set(str(x) for x in dates.tolist())):
        m = dates == day
        row: dict[str, Any] = {"date": day}
        row.update(_metrics(p[m], y[m]))
        out.append(row)
    return out


def _dayblock_score(p: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict[str, float]:
    rows = _per_day_metrics(p, y, dates)
    vals = np.asarray([r["logloss"] for r in rows if np.isfinite(r["logloss"])], dtype=np.float64)
    if vals.size == 0:
        return {"days": 0, "mean_logloss": float("nan"), "std_logloss": float("nan"), "max_logloss": float("nan")}
    return {
        "days": int(vals.size),
        "mean_logloss": float(vals.mean()),
        "std_logloss": float(vals.std(ddof=0)),
        "max_logloss": float(vals.max()),
    }


def _ttc_mask(seq_len: int, cadence_s: float, lo: float, hi: float) -> np.ndarray:
    ttc = seq_len * cadence_s - np.arange(seq_len, dtype=np.float32) * cadence_s
    return (ttc > lo) & (ttc <= hi)


def _load_split(
    path: Path,
    split: str,
    *,
    cadence_s: float,
    ttc_min: float,
    ttc_max: float,
    device: torch.device,
    include_mask_channel: bool,
) -> dict[str, Any]:
    z = np.load(path / f"{split}.npz", allow_pickle=False)
    x_np = z["X"].astype(np.float32, copy=False)
    valid_np = z["valid_mask"].astype(bool, copy=False)
    ttc_band = _ttc_mask(x_np.shape[1], cadence_s, ttc_min, ttc_max)
    loss_mask_np = valid_np & ttc_band[None, :]
    y_ep_np = z["y"].astype(np.float32, copy=False)
    y_np = np.repeat(y_ep_np[:, None], x_np.shape[1], axis=1).astype(np.float32, copy=False)
    p_market_np = z["p_market"].astype(np.float32, copy=False)
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
        "date": z["date"].copy(),
        "market_slug": z["market_slug"].copy(),
        "open_s": z["open_s"].copy(),
        "valid_count": z["valid_count"].copy(),
    }


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.conv = CausalConv1d(channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = y.transpose(1, 2)
        return x + self.residual_scale * y


class ResidualTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: int,
        blocks: int,
        kernel_size: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, channels, kernel_size=1)
        self.input_norm = nn.LayerNorm(channels)
        dilations = [2 ** i for i in range(blocks)]
        self.blocks = nn.ModuleList([
            TCNBlock(channels, kernel_size, dilation, dropout, residual_scale) for dilation in dilations
        ])
        self.head_norm = nn.LayerNorm(channels)
        self.head_drop = nn.Dropout(dropout)
        self.head = nn.Conv1d(channels, 1, kernel_size=1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head.bias)

    @property
    def receptive_steps(self) -> int:
        return 1 + sum(block.conv.left_pad for block in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features)
        h = self.input_proj(x.transpose(1, 2))
        h = self.input_norm(h.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        h = self.head_norm(h.transpose(1, 2))
        h = self.head_drop(F.gelu(h)).transpose(1, 2)
        return self.head(h).squeeze(1)


def baseline_metrics(data: dict[str, Any]) -> dict[str, float]:
    m = data["np_loss_mask"]
    return _metrics(data["np_market_p"][m], data["np_y"][m])


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    data: dict[str, Any],
    batch_size: int,
    delta_clamp: float | None,
    *,
    cadence_s: float,
) -> dict[str, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    base_out: list[np.ndarray] = []
    delta_out: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    slugs: list[np.ndarray] = []
    ep_indices: list[np.ndarray] = []
    step_indices: list[np.ndarray] = []
    x = data["X"]
    mask = data["loss_mask"]
    y = data["y"]
    base = data["market_logit"]
    seq_len = int(x.shape[1])
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
        mb_np = mb.detach().cpu().numpy()
        local_ep, local_step = np.nonzero(mb_np)
        ep_idx = (local_ep + i).astype(np.int32, copy=False)
        step_idx = local_step.astype(np.int16, copy=False)
        logits_out.append(logits[mb].detach().cpu().numpy())
        base_out.append(bb[mb].detach().cpu().numpy())
        delta_out.append(delta[mb].detach().cpu().numpy())
        ys.append(y[i:i + batch_size][mb].detach().cpu().numpy())
        dates.append(data["date"][ep_idx])
        slugs.append(data["market_slug"][ep_idx])
        ep_indices.append(ep_idx)
        step_indices.append(step_idx)
    if not ys:
        empty_f = np.array([], dtype=np.float32)
        empty_i = np.array([], dtype=np.int32)
        empty_s = np.array([], dtype="U1")
        return {
            "logit": empty_f,
            "base_logit": empty_f,
            "delta": empty_f,
            "y": empty_f,
            "date": empty_s,
            "market_slug": empty_s,
            "ep_idx": empty_i,
            "step_idx": empty_i,
            "ttc_s": empty_f,
        }
    step_idx_all = np.concatenate(step_indices).astype(np.int16, copy=False)
    return {
        "logit": np.concatenate(logits_out).astype(np.float32, copy=False),
        "base_logit": np.concatenate(base_out).astype(np.float32, copy=False),
        "delta": np.concatenate(delta_out).astype(np.float32, copy=False),
        "y": np.concatenate(ys).astype(np.float32, copy=False),
        "date": np.concatenate(dates),
        "market_slug": np.concatenate(slugs),
        "ep_idx": np.concatenate(ep_indices).astype(np.int32, copy=False),
        "step_idx": step_idx_all,
        "ttc_s": (seq_len * cadence_s - step_idx_all.astype(np.float32) * cadence_s).astype(np.float32),
    }


def evaluate(
    model: nn.Module,
    data: dict[str, Any],
    batch_size: int,
    delta_clamp: float | None,
    *,
    cadence_s: float,
) -> dict[str, float]:
    z = collect_logits(model, data, batch_size, delta_clamp, cadence_s=cadence_s)
    return _metrics(_sigmoid_np(z["logit"]), z["y"])


def _fit_val_calibrators(val_logits: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    logit = val_logits["logit"].astype(np.float64, copy=False)
    base = val_logits["base_logit"].astype(np.float64, copy=False)
    delta = val_logits["delta"].astype(np.float64, copy=False)
    y = val_logits["y"].astype(np.int8, copy=False)
    out: dict[str, dict[str, Any]] = {"identity": {"kind": "identity"}}

    temps = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))
    losses = [_metrics(_sigmoid_np(logit / t), y)["logloss"] for t in temps]
    best_i = int(np.nanargmin(losses))
    out["val_temperature"] = {
        "kind": "temperature",
        "source": "val",
        "temperature": float(temps[best_i]),
        "val_logloss": float(losses[best_i]),
    }

    lr = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y)
    out["val_platt_l2"] = {
        "kind": "platt",
        "source": "val",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }

    iso = IsotonicRegression(out_of_bounds="clip")
    p_uncal = _sigmoid_np(logit).astype(np.float64, copy=False)
    iso.fit(p_uncal, y)
    out["val_isotonic"] = {
        "kind": "isotonic",
        "source": "val",
        "x_thresholds": iso.X_thresholds_.astype(float).tolist(),
        "y_thresholds": iso.y_thresholds_.astype(float).tolist(),
    }

    alphas = np.linspace(0.0, 1.5, 151)
    losses = [_metrics(_sigmoid_np(base + a * delta), y)["logloss"] for a in alphas]
    best_i = int(np.nanargmin(losses))
    out["val_residual_shrink"] = {
        "kind": "residual_shrink",
        "source": "val",
        "alpha": float(alphas[best_i]),
        "val_logloss": float(losses[best_i]),
    }
    return out


def _fit_train_dayblock_calibrators(
    train_logits: dict[str, np.ndarray],
    *,
    robust_std_weight: float,
) -> dict[str, dict[str, Any]]:
    logit = train_logits["logit"].astype(np.float64, copy=False)
    base = train_logits["base_logit"].astype(np.float64, copy=False)
    delta = train_logits["delta"].astype(np.float64, copy=False)
    y = train_logits["y"].astype(np.int8, copy=False)
    dates = train_logits["date"]
    out: dict[str, dict[str, Any]] = {}

    def choose_grid(name: str, values: np.ndarray, pred_fn) -> dict[str, Any]:
        rows = []
        for v in values:
            p = pred_fn(float(v))
            score = _dayblock_score(p, y, dates)
            objective = score["mean_logloss"] + robust_std_weight * score["std_logloss"]
            rows.append((objective, float(v), score))
        rows.sort(key=lambda x: x[0])
        best_obj, best_value, best_score = rows[0]
        return {
            "source": "train_dayblock",
            "grid": name,
            "value": best_value,
            "objective": float(best_obj),
            "dayblock": best_score,
        }

    temps = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))
    temp = choose_grid("temperature", temps, lambda t: _sigmoid_np(logit / t))
    out["train_dayblock_temperature"] = {
        "kind": "temperature",
        "source": "train_dayblock",
        "temperature": temp["value"],
        "objective": temp["objective"],
        "dayblock": temp["dayblock"],
    }

    alphas = np.linspace(0.0, 1.5, 151)
    shrink = choose_grid("residual_alpha", alphas, lambda a: _sigmoid_np(base + a * delta))
    out["train_dayblock_residual_shrink"] = {
        "kind": "residual_shrink",
        "source": "train_dayblock",
        "alpha": shrink["value"],
        "objective": shrink["objective"],
        "dayblock": shrink["dayblock"],
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
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for split, logits in split_logits.items():
        out[split] = {}
        for name, cal in calibrators.items():
            p = _apply_calibrator(cal, logits)
            out[split][name] = {
                "step": _metrics(p, logits["y"]),
                "dayblock": _dayblock_score(p, logits["y"], logits["date"]),
                "per_day": _per_day_metrics(p, logits["y"], logits["date"]),
            }
    return out


def _save_predictions(out_dir: Path, split: str, logits: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        out_dir / f"predictions_{split}.npz",
        logit=logits["logit"].astype(np.float32, copy=False),
        base_logit=logits["base_logit"].astype(np.float32, copy=False),
        delta=logits["delta"].astype(np.float32, copy=False),
        y=logits["y"].astype(np.float32, copy=False),
        date=logits["date"],
        market_slug=logits["market_slug"],
        ep_idx=logits["ep_idx"].astype(np.int32, copy=False),
        step_idx=logits["step_idx"].astype(np.int16, copy=False),
        ttc_s=logits["ttc_s"].astype(np.float32, copy=False),
    )


def _write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# BTC 5m Episode TCN",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Elapsed: `{report['elapsed_s']}s`",
        f"Best epoch: `{report['best_epoch']}`",
        f"Receptive field: `{report['model']['receptive_steps']} steps`",
        "",
        "## Test TTC 15-90",
        "",
        "| model | n | brier | logloss | auc | ece_20 | day mean logloss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    market = report["market_baseline"]["test"]
    lines.append(
        f"| market | {market['n']} | {market['brier']:.6f} | {market['logloss']:.6f} | "
        f"{market['auc']:.6f} | {market['ece_20']:.6f} |  |"
    )
    for name, payload in report["tcn_calibrated"]["test"].items():
        m = payload["step"]
        d = payload["dayblock"]
        lines.append(
            f"| {name} | {m['n']} | {m['brier']:.6f} | {m['logloss']:.6f} | "
            f"{m['auc']:.6f} | {m['ece_20']:.6f} | {d['mean_logloss']:.6f} |"
        )
    lines.append("")
    lines.append("## Calibrators")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["calibrators"], indent=2))
    lines.append("```")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ttc-min", type=float, default=15.0)
    ap.add_argument("--ttc-max", type=float, default=90.0)
    ap.add_argument("--cadence-s", type=float, default=0.2)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--kernel-size", type=int, default=5)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--residual-scale", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--delta-clamp", type=float, default=0.75)
    ap.add_argument("--delta-l2", type=float, default=0.002)
    ap.add_argument("--dayblock-std-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--torch-threads", type=int, default=8)
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
    model = ResidualTCN(
        n_features,
        args.channels,
        args.blocks,
        args.kernel_size,
        args.dropout,
        args.residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(train["X"], train["loss_mask"], train["y"], train["market_logit"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    market = {"train": baseline_metrics(train), "val": baseline_metrics(val), "test": baseline_metrics(test)}
    print(f"shapes train={tuple(train['X'].shape)} val={tuple(val['X'].shape)} test={tuple(test['X'].shape)}")
    print(
        f"model channels={args.channels} blocks={args.blocks} kernel={args.kernel_size} "
        f"receptive_steps={model.receptive_steps}",
        flush=True,
    )
    print("market_baseline", json.dumps(market, indent=2), flush=True)

    best_val = float("inf")
    best_epoch = 0
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
            loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
            if args.delta_l2 > 0:
                loss = loss + float(args.delta_l2) * torch.mean(delta[mb] ** 2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            seen += int(mb.sum().detach().cpu())
        val_m = evaluate(
            model,
            val,
            args.batch_size,
            args.delta_clamp if args.delta_clamp > 0 else None,
            cadence_s=args.cadence_s,
        )
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
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_logloss={best_val:.5f}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    split_logits = {
        "train": collect_logits(
            model, train, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None,
            cadence_s=args.cadence_s,
        ),
        "val": collect_logits(
            model, val, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None,
            cadence_s=args.cadence_s,
        ),
        "test": collect_logits(
            model, test, args.batch_size, args.delta_clamp if args.delta_clamp > 0 else None,
            cadence_s=args.cadence_s,
        ),
    }
    for split, logits in split_logits.items():
        _save_predictions(out_dir, split, logits)

    calibrators = _fit_val_calibrators(split_logits["val"])
    calibrators.update(_fit_train_dayblock_calibrators(
        split_logits["train"],
        robust_std_weight=args.dayblock_std_weight,
    ))
    calibrated = _evaluate_calibrated(split_logits, calibrators)
    final = {
        "dataset": str(dataset),
        "device": str(device),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": {
            "type": "causal_residual_tcn",
            "n_features": n_features,
            "channels": args.channels,
            "blocks": args.blocks,
            "kernel_size": args.kernel_size,
            "residual_scale": args.residual_scale,
            "receptive_steps": model.receptive_steps,
            "receptive_seconds": round(model.receptive_steps * args.cadence_s, 3),
        },
        "market_baseline": market,
        "tcn_residual": {
            "train": _metrics(_sigmoid_np(split_logits["train"]["logit"]), split_logits["train"]["y"]),
            "val": _metrics(_sigmoid_np(split_logits["val"]["logit"]), split_logits["val"]["y"]),
            "test": _metrics(_sigmoid_np(split_logits["test"]["logit"]), split_logits["test"]["y"]),
        },
        "calibrators": calibrators,
        "tcn_calibrated": calibrated,
        "history": history,
        "best_epoch": best_epoch,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "tcn_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    _write_summary(out_dir, final)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print("FINAL", json.dumps({
        "market_baseline": final["market_baseline"],
        "tcn_residual": final["tcn_residual"],
        "tcn_calibrated_test": {
            name: payload["step"] for name, payload in final["tcn_calibrated"]["test"].items()
        },
        "best_epoch": final["best_epoch"],
        "elapsed_s": final["elapsed_s"],
    }, indent=2), flush=True)
    print(f"report -> {out_dir / 'tcn_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
