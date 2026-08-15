"""Run the five published TCN checkpoints on the ten-market public subset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in minimal CI installs
    raise SystemExit("PyTorch is required: pip install -e '.[sequence]'") from exc

from train_episode_tcn import ResidualTCN

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "artifacts/evaluation_repro_v2"
REPORTS = ROOT / "artifacts/tcn_v2_eval"
SEEDS = (7, 11, 23, 42, 101)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def main() -> int:
    torch.set_num_threads(1)
    subset = np.load(
        BUNDLE / "mini_inference/test_subset_10_markets.npz", allow_pickle=False
    )
    x = subset["X"].astype(np.float32)
    valid = subset["valid_mask"].astype(bool)
    x_with_mask = np.concatenate((x, valid[:, :, None].astype(np.float32)), axis=2)
    base_logit = _logit(subset["p_market"])
    original_to_local = {
        int(original): local for local, original in enumerate(subset["original_ep_idx"])
    }

    results = []
    for seed in SEEDS:
        report = json.loads(
            (REPORTS / f"tcn_v2_c64_b7_ttc15_150_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        config = report["config"]
        model_info = report["model"]
        model = ResidualTCN(
            n_features=int(model_info["n_features"]),
            channels=int(config["channels"]),
            blocks=int(config["blocks"]),
            kernel_size=int(config["kernel_size"]),
            dropout=float(config["dropout"]),
            residual_scale=float(config["residual_scale"]),
        )
        state = torch.load(
            BUNDLE / f"checkpoints/seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            delta = model(torch.from_numpy(x_with_mask)).numpy()
        delta = np.clip(delta, -float(config["delta_clamp"]), float(config["delta_clamp"]))

        expected = np.load(
            BUNDLE / f"mini_inference/expected_seed{seed}.npz", allow_pickle=False
        )
        actual_delta = np.asarray(
            [
                delta[original_to_local[int(ep)], int(step)]
                for ep, step in zip(expected["ep_idx"], expected["step_idx"], strict=True)
            ],
            dtype=np.float32,
        )
        actual_base = np.asarray(
            [
                base_logit[original_to_local[int(ep)], int(step)]
                for ep, step in zip(expected["ep_idx"], expected["step_idx"], strict=True)
            ],
            dtype=np.float32,
        )
        actual_logit = actual_base + actual_delta
        errors = {
            "delta": float(np.max(np.abs(actual_delta - expected["delta"]))),
            "base_logit": float(np.max(np.abs(actual_base - expected["base_logit"]))),
            "logit": float(np.max(np.abs(actual_logit - expected["logit"]))),
        }
        # CPU convolution kernels differ slightly across PyTorch builds/threading
        # backends; 1e-4 is far below any reported probability precision.
        if any(value > 1e-4 for value in errors.values()):
            raise SystemExit(f"seed {seed} parity failed: {errors}")
        results.append(
            {
                "seed": seed,
                "compared_loss_band_rows": len(expected["logit"]),
                "max_absolute_error": errors,
            }
        )

    print(
        json.dumps(
            {
                "status": "PASS",
                "markets": len(subset["y"]),
                "checkpoints": len(SEEDS),
                "results": results,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
