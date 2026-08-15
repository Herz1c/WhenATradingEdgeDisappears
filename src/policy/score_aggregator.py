from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


ACTION_SPACE = [
    "quote_both",
    "quote_yes_only",
    "quote_no_only",
    "cancel_yes",
    "cancel_no",
    "cancel_all",
    "taker_emergency_only",
]


@dataclass
class ModelOutputBundle:
    """All upstream model outputs for one decision point."""

    p_bias_up: Optional[float] = None
    bias_confidence: Optional[float] = None
    p_up_fair: Optional[float] = None
    fair_edge_vs_market: Optional[float] = None
    p_fill_maker: Optional[float] = None
    expected_fill_time_s: Optional[float] = None
    expected_markout_1s: Optional[float] = None
    p_adverse_fill: Optional[float] = None
    p_flip_before_close: Optional[float] = None
    mispricing_score: Optional[float] = None
    edge_net_of_spread: Optional[float] = None
    p_mid_up_5s: Optional[float] = None
    expected_move_cents: Optional[float] = None
    regime_state: Optional[str] = None
    p_trend: Optional[float] = None
    p_chop: Optional[float] = None
    p_jump: Optional[float] = None
    t_to_close_s: Optional[float] = None
    delta_to_strike: Optional[float] = None
    current_spread: Optional[float] = None
    current_position: Optional[float] = None


@dataclass
class ActionScore:
    """Score for one action."""

    action: str
    score: float
    confidence: float
    reasoning: dict


class ScoreAggregator:
    """
    Aggregates model outputs into action scores.

    Placeholder weights deliberately produce neutral scores and no trading. Live
    use must provide explicit non-zero weights derived from trained models.
    """

    PLACEHOLDER_WEIGHTS = {
        "w_fair_edge": 0.0,
        "w_flip_risk": 0.0,
        "w_fill_prob": 0.0,
        "w_adverse": 0.0,
        "w_microstructure": 0.0,
        "w_move_size": 0.0,
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights or self.PLACEHOLDER_WEIGHTS)
        self._validate_weights()

    @property
    def placeholder_mode(self) -> bool:
        return self._placeholder_mode

    def _validate_weights(self) -> None:
        missing = set(self.PLACEHOLDER_WEIGHTS) - set(self.weights)
        if missing:
            raise ValueError(f"Missing policy weights: {sorted(missing)}")
        self._placeholder_mode = all(float(self.weights[key]) == 0.0 for key in self.PLACEHOLDER_WEIGHTS)

    def compute_action_scores(self, bundle: ModelOutputBundle) -> list[ActionScore]:
        """Compute scores for all actions in the action space."""
        if self._placeholder_mode:
            return [
                ActionScore(
                    action=action,
                    score=0.0,
                    confidence=0.0,
                    reasoning={"warning": "placeholder_weights_no_trading"},
                )
                for action in ACTION_SPACE
            ]

        scores = [self._score_action(action, bundle) for action in ACTION_SPACE]
        return sorted(scores, key=lambda item: item.score, reverse=True)

    def _score_action(self, action: str, bundle: ModelOutputBundle) -> ActionScore:
        fair_edge = float(bundle.fair_edge_vs_market or bundle.mispricing_score or 0.0)
        fill = float(bundle.p_fill_maker if bundle.p_fill_maker is not None else 0.5)
        adverse = float(bundle.p_adverse_fill or 0.0)
        flip = float(bundle.p_flip_before_close or 0.0)
        micro = float((bundle.p_mid_up_5s or 0.5) - 0.5)
        move = float(bundle.expected_move_cents or 0.0)

        raw = (
            self.weights["w_fair_edge"] * fair_edge
            + self.weights["w_fill_prob"] * fill
            - self.weights["w_adverse"] * adverse
            - self.weights["w_flip_risk"] * flip
            + self.weights["w_microstructure"] * micro
            + self.weights["w_move_size"] * move
        )

        if action == "quote_no_only":
            raw = -raw
        elif action == "quote_both":
            raw -= abs(fair_edge) * 0.25
        elif action in {"cancel_yes", "cancel_no"}:
            raw = max(0.0, adverse + flip - 0.5)
        elif action == "cancel_all":
            raw = max(0.0, adverse + flip - 0.75)
        elif action == "taker_emergency_only":
            raw = max(0.0, flip - 0.7)

        confidence_inputs = [value for value in [bundle.p_fill_maker, bundle.p_adverse_fill, bundle.p_flip_before_close] if value is not None]
        confidence = min(1.0, 0.25 + 0.25 * len(confidence_inputs))
        return ActionScore(
            action=action,
            score=float(raw),
            confidence=float(confidence),
            reasoning={
                "fair_edge": fair_edge,
                "fill": fill,
                "adverse": adverse,
                "flip": flip,
                "micro": micro,
                "move": move,
            },
        )

    def get_recommended_action(self, bundle: ModelOutputBundle, threshold: float = 0.0) -> ActionScore:
        """Return highest scoring action above threshold, otherwise cancel_all."""
        scores = self.compute_action_scores(bundle)
        best = scores[0]
        if best.score < threshold:
            return ActionScore(
                action="cancel_all",
                score=0.0,
                confidence=1.0,
                reasoning={"reason": "no_action_above_threshold"},
            )
        return best

