"""Train a causal Patch Transformer on the BTC 5m episode dataset.

The model is live-replicable:
- patch tokens are produced with a left-padded causal Conv1d,
- causal attention prevents token-to-future-token reads,
- each 200ms step uses only the most recent patch token at or before that step,
- the head predicts a residual over logit(p_market).
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "datasets" / "btc_5m_episodes_v1_200ms"
DEFAULT_OUT = ROOT / "artifacts" / "btc_5m_episode_patch_transformer_medium_ttc15_90"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64, copy=False), -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32, copy=False)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64, copy=False), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p)).astype(np.float32, copy=False)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
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


def metrics(p: np.ndarray, y: np.ndarray) -> dict[str, float]:
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
        "p_mean": float(p.mean()),
        "y_mean": float(y.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "ece_20": ece(p, y),
    }
    out["auc"] = float("nan") if np.unique(y).size < 2 else float(roc_auc_score(y.astype(int), p))
    return out


def per_day_metrics(p: np.ndarray, y: np.ndarray, dates: np.ndarray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for day in sorted(set(str(x) for x in dates.tolist())):
        m = dates == day
        row: dict[str, Any] = {"date": day}
        row.update(metrics(p[m], y[m]))
        out.append(row)
    return out


def dayblock_score(p: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict[str, float]:
    rows = per_day_metrics(p, y, dates)
    vals = np.asarray([r["logloss"] for r in rows if np.isfinite(r["logloss"])], dtype=np.float64)
    if vals.size == 0:
        return {"days": 0, "mean_logloss": float("nan"), "std_logloss": float("nan"), "max_logloss": float("nan")}
    return {
        "days": int(vals.size),
        "mean_logloss": float(vals.mean()),
        "std_logloss": float(vals.std(ddof=0)),
        "max_logloss": float(vals.max()),
    }


def ttc_mask(seq_len: int, cadence_s: float, lo: float, hi: float) -> np.ndarray:
    ttc = seq_len * cadence_s - np.arange(seq_len, dtype=np.float32) * cadence_s
    return (ttc > lo) & (ttc <= hi)


def maybe_limit(x: np.ndarray, n: int) -> np.ndarray:
    return x[:n] if n > 0 else x


def load_split(
    path: Path,
    split: str,
    *,
    cadence_s: float,
    ttc_min: float,
    ttc_max: float,
    include_mask_channel: bool,
    max_episodes: int,
) -> dict[str, Any]:
    z = np.load(path / f"{split}.npz", allow_pickle=False)
    x_np = maybe_limit(z["X"].astype(np.float32, copy=False), max_episodes)
    valid_np = maybe_limit(z["valid_mask"].astype(bool, copy=False), max_episodes)
    ttc_band = ttc_mask(x_np.shape[1], cadence_s, ttc_min, ttc_max)
    loss_mask_np = valid_np & ttc_band[None, :]
    y_ep_np = maybe_limit(z["y"].astype(np.float32, copy=False), max_episodes)
    y_np = np.repeat(y_ep_np[:, None], x_np.shape[1], axis=1).astype(np.float32, copy=False)
    p_market_np = maybe_limit(z["p_market"].astype(np.float32, copy=False), max_episodes)
    p_market_np[~np.isfinite(p_market_np)] = 0.5
    p_market_np = np.clip(p_market_np, 1e-5, 1.0 - 1e-5)
    if include_mask_channel:
        x_np = np.concatenate([x_np, valid_np[:, :, None].astype(np.float32)], axis=2)
    return {
        "X": torch.from_numpy(x_np),
        "loss_mask": torch.from_numpy(loss_mask_np),
        "y": torch.from_numpy(y_np),
        "market_logit": torch.from_numpy(logit_np(p_market_np)),
        "np_loss_mask": loss_mask_np,
        "np_y": y_np,
        "np_market_p": p_market_np,
        "date": maybe_limit(z["date"].copy(), max_episodes),
        "market_slug": maybe_limit(z["market_slug"].copy(), max_episodes),
        "open_s": maybe_limit(z["open_s"].copy(), max_episodes),
        "valid_count": maybe_limit(z["valid_count"].copy(), max_episodes),
    }


class CausalPatchTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_features: int,
        seq_len: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        layers: int,
        heads: int,
        ff_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.token_count = int(math.floor((seq_len - 1) / patch_stride) + 1)
        self.patch = nn.Conv1d(n_features, d_model, kernel_size=patch_len, stride=patch_stride)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.token_count, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=int(round(d_model * ff_mult)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        step_to_token = torch.div(torch.arange(seq_len), patch_stride, rounding_mode="floor")
        step_to_token = torch.clamp(step_to_token, 0, self.token_count - 1).long()
        self.register_buffer("step_to_token", step_to_token, persistent=False)
        nn.init.normal_(self.pos, mean=0.0, std=0.01)
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def max_staleness_s(self) -> float:
        return (self.patch_stride - 1) * 0.2

    def causal_mask(self, device: torch.device) -> torch.Tensor:
        m = torch.full((self.token_count, self.token_count), float("-inf"), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features).  Left padding makes token k use history
        # ending exactly at step k * patch_stride.
        h = x.transpose(1, 2)
        h = F.pad(h, (self.patch_len - 1, 0))
        tok = self.patch(h).transpose(1, 2)
        tok = self.input_norm(tok) + self.pos[:, :tok.shape[1], :]
        tok = self.encoder(tok, mask=self.causal_mask(tok.device))
        delta_tok = self.head(tok).squeeze(-1)
        return delta_tok[:, self.step_to_token]


def baseline_metrics(data: dict[str, Any]) -> dict[str, float]:
    m = data["np_loss_mask"]
    return metrics(data["np_market_p"][m], data["np_y"][m])


def make_sampler(train: dict[str, Any]) -> WeightedRandomSampler:
    dates = train["date"]
    uniq, counts = np.unique(dates, return_counts=True)
    count_by_day = dict(zip(uniq.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_by_day[d] for d in dates.tolist()], dtype=np.float64)
    weights = weights / weights.mean()
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    data: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
    cadence_s: float,
    delta_clamp: float | None,
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
        xb = x[i:i + batch_size].to(device, non_blocking=True)
        mb = mask[i:i + batch_size].to(device, non_blocking=True)
        if not mb.any():
            continue
        bb = base[i:i + batch_size].to(device, non_blocking=True)
        delta = model(xb)
        if delta_clamp is not None and delta_clamp > 0:
            delta = torch.clamp(delta, -float(delta_clamp), float(delta_clamp))
        logits = bb + delta
        mb_np = mb.detach().cpu().numpy()
        local_ep, local_step = np.nonzero(mb_np)
        ep_idx = (local_ep + i).astype(np.int32, copy=False)
        step_idx = local_step.astype(np.int16, copy=False)
        logits_out.append(logits[mb].detach().cpu().numpy())
        base_out.append(bb[mb].detach().cpu().numpy())
        delta_out.append(delta[mb].detach().cpu().numpy())
        ys.append(y[i:i + batch_size][mask[i:i + batch_size]].numpy())
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


def evaluate(model: nn.Module, data: dict[str, Any], *, batch_size: int, device: torch.device,
             cadence_s: float, delta_clamp: float | None) -> dict[str, float]:
    z = collect_logits(model, data, batch_size=batch_size, device=device, cadence_s=cadence_s,
                       delta_clamp=delta_clamp)
    return metrics(sigmoid_np(z["logit"]), z["y"])


def fit_calibrators(train_logits: dict[str, np.ndarray], val_logits: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    out: dict[str, dict[str, Any]] = {"identity": {"kind": "identity"}}
    val_logit = val_logits["logit"].astype(np.float64, copy=False)
    val_y = val_logits["y"].astype(np.int8, copy=False)
    temps = np.exp(np.linspace(np.log(0.25), np.log(8.0), 121))
    losses = [metrics(sigmoid_np(val_logit / t), val_y)["logloss"] for t in temps]
    best_i = int(np.nanargmin(losses))
    out["val_temperature"] = {"kind": "temperature", "temperature": float(temps[best_i])}

    train_logit = train_logits["logit"].astype(np.float64, copy=False)
    train_y = train_logits["y"].astype(np.int8, copy=False)
    lr = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
    lr.fit(train_logit.reshape(-1, 1), train_y)
    out["train_platt_l2"] = {
        "kind": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
        "C": 0.1,
    }

    base = train_logits["base_logit"].astype(np.float64, copy=False)
    delta = train_logits["delta"].astype(np.float64, copy=False)
    dates = train_logits["date"]
    alphas = np.linspace(-0.5, 1.5, 201)
    scored = []
    for a in alphas:
        p = sigmoid_np(base + a * delta)
        s = dayblock_score(p, train_y, dates)
        scored.append((s["mean_logloss"] + 0.25 * s["std_logloss"], float(a), s))
    scored.sort(key=lambda x: x[0])
    _, alpha, score = scored[0]
    out["train_dayblock_residual_shrink"] = {
        "kind": "residual_shrink",
        "alpha": alpha,
        "dayblock": score,
    }
    return out


def apply_calibrator(cal: dict[str, Any], logits: dict[str, np.ndarray]) -> np.ndarray:
    kind = cal["kind"]
    if kind == "identity":
        return sigmoid_np(logits["logit"])
    if kind == "temperature":
        return sigmoid_np(logits["logit"] / float(cal["temperature"]))
    if kind == "platt":
        return sigmoid_np(float(cal["coef"]) * logits["logit"] + float(cal["intercept"]))
    if kind == "residual_shrink":
        return sigmoid_np(logits["base_logit"] + float(cal["alpha"]) * logits["delta"])
    raise ValueError(f"unknown calibrator kind={kind!r}")


def evaluate_calibrated(
    split_logits: dict[str, dict[str, np.ndarray]],
    calibrators: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for split, logits in split_logits.items():
        out[split] = {}
        for name, cal in calibrators.items():
            p = apply_calibrator(cal, logits)
            out[split][name] = {
                "step": metrics(p, logits["y"]),
                "dayblock": dayblock_score(p, logits["y"], logits["date"]),
                "per_day": per_day_metrics(p, logits["y"], logits["date"]),
            }
    return out


def save_predictions(out_dir: Path, split: str, logits: dict[str, np.ndarray]) -> None:
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


def write_summary(out_dir: Path, report: dict[str, Any]) -> None:
    lines = [
        "# BTC 5m Episode Patch Transformer",
        "",
        f"Dataset: `{report['dataset']}`",
        f"Elapsed: `{report['elapsed_s']}s`",
        f"Best epoch: `{report['best_epoch']}`",
        "",
        "## Test TTC Window",
        "",
        "| model | n | brier | logloss | auc | ece_20 | day mean logloss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    market = report["market_baseline"]["test"]
    lines.append(
        f"| market | {market['n']} | {market['brier']:.6f} | {market['logloss']:.6f} | "
        f"{market['auc']:.6f} | {market['ece_20']:.6f} |  |"
    )
    for name, payload in report["patch_calibrated"]["test"].items():
        m = payload["step"]
        d = payload["dayblock"]
        lines.append(
            f"| {name} | {m['n']} | {m['brier']:.6f} | {m['logloss']:.6f} | "
            f"{m['auc']:.6f} | {m['ece_20']:.6f} | {d['mean_logloss']:.6f} |"
        )
    lines.append("")
    lines.append("## Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["config"], indent=2))
    lines.append("```")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ttc-min", type=float, default=15.0)
    ap.add_argument("--ttc-max", type=float, default=90.0)
    ap.add_argument("--cadence-s", type=float, default=0.2)
    ap.add_argument("--patch-len", type=int, default=30)
    ap.add_argument("--patch-stride", type=int, default=5)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ff-mult", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=2e-4)
    ap.add_argument("--delta-clamp", type=float, default=1.0)
    ap.add_argument("--delta-l2", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--include-mask-channel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--balanced-day-sampler", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-train-episodes", type=int, default=0)
    ap.add_argument("--max-val-episodes", type=int, default=0)
    ap.add_argument("--max-test-episodes", type=int, default=0)
    return ap


def parse_args() -> argparse.Namespace:
    ap = build_arg_parser()
    cfg_probe, _ = ap.parse_known_args()
    if cfg_probe.config is not None:
        cfg_path = cfg_probe.config if cfg_probe.config.is_absolute() else ROOT / cfg_probe.config
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"device={device} torch_threads={torch.get_num_threads()} dataset={dataset}", flush=True)
    train = load_split(dataset, "train", cadence_s=args.cadence_s, ttc_min=args.ttc_min,
                       ttc_max=args.ttc_max, include_mask_channel=args.include_mask_channel,
                       max_episodes=args.max_train_episodes)
    val = load_split(dataset, "val", cadence_s=args.cadence_s, ttc_min=args.ttc_min,
                     ttc_max=args.ttc_max, include_mask_channel=args.include_mask_channel,
                     max_episodes=args.max_val_episodes)
    test = load_split(dataset, "test", cadence_s=args.cadence_s, ttc_min=args.ttc_min,
                      ttc_max=args.ttc_max, include_mask_channel=args.include_mask_channel,
                      max_episodes=args.max_test_episodes)

    n_features = int(train["X"].shape[-1])
    seq_len = int(train["X"].shape[1])
    model = CausalPatchTransformer(
        n_features=n_features,
        seq_len=seq_len,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(train["X"], train["loss_mask"], train["y"], train["market_logit"])
    sampler = make_sampler(train) if args.balanced_day_sampler else None
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    market = {"train": baseline_metrics(train), "val": baseline_metrics(val), "test": baseline_metrics(test)}
    print(f"shapes train={tuple(train['X'].shape)} val={tuple(val['X'].shape)} test={tuple(test['X'].shape)}")
    print(
        f"model tokens={model.token_count} patch_len={args.patch_len} stride={args.patch_stride} "
        f"d_model={args.d_model} layers={args.layers} heads={args.heads}",
        flush=True,
    )
    print("market_baseline", json.dumps(market, indent=2), flush=True)

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    history: list[dict[str, Any]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        losses = []
        seen = 0
        for xb, mb, yb, bb in loader:
            xb = xb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            bb = bb.to(device, non_blocking=True)
            if not mb.any():
                continue
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                delta = model(xb)
                if args.delta_clamp > 0:
                    delta = torch.clamp(delta, -args.delta_clamp, args.delta_clamp)
                logits = bb + delta
                loss = F.binary_cross_entropy_with_logits(logits[mb], yb[mb])
                if args.delta_l2 > 0:
                    loss = loss + float(args.delta_l2) * torch.mean(delta[mb] ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            seen += int(mb.sum().detach().cpu())
        val_m = evaluate(model, val, batch_size=args.batch_size, device=device, cadence_s=args.cadence_s,
                         delta_clamp=args.delta_clamp if args.delta_clamp > 0 else None)
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
        "train": collect_logits(model, train, batch_size=args.batch_size, device=device,
                                cadence_s=args.cadence_s,
                                delta_clamp=args.delta_clamp if args.delta_clamp > 0 else None),
        "val": collect_logits(model, val, batch_size=args.batch_size, device=device,
                              cadence_s=args.cadence_s,
                              delta_clamp=args.delta_clamp if args.delta_clamp > 0 else None),
        "test": collect_logits(model, test, batch_size=args.batch_size, device=device,
                               cadence_s=args.cadence_s,
                               delta_clamp=args.delta_clamp if args.delta_clamp > 0 else None),
    }
    for split, logits in split_logits.items():
        save_predictions(out_dir, split, logits)
    calibrators = fit_calibrators(split_logits["train"], split_logits["val"])
    calibrated = evaluate_calibrated(split_logits, calibrators)
    final = {
        "dataset": str(dataset),
        "device": str(device),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": {
            "type": "causal_patch_transformer",
            "n_features": n_features,
            "seq_len": seq_len,
            "token_count": model.token_count,
            "patch_len": args.patch_len,
            "patch_stride": args.patch_stride,
            "max_step_staleness_s": round((args.patch_stride - 1) * args.cadence_s, 3),
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
        },
        "market_baseline": market,
        "patch_residual": {
            split: metrics(sigmoid_np(logits["logit"]), logits["y"])
            for split, logits in split_logits.items()
        },
        "calibrators": calibrators,
        "patch_calibrated": calibrated,
        "history": history,
        "best_epoch": best_epoch,
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "patch_transformer_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    write_summary(out_dir, final)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print("FINAL", json.dumps({
        "market_baseline": final["market_baseline"],
        "patch_residual": final["patch_residual"],
        "patch_calibrated_test": {
            name: payload["step"] for name, payload in final["patch_calibrated"]["test"].items()
        },
        "best_epoch": final["best_epoch"],
        "elapsed_s": final["elapsed_s"],
    }, indent=2), flush=True)
    print(f"report -> {out_dir / 'patch_transformer_report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
