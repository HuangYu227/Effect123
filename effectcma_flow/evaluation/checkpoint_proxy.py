from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def joint_f1(precision: float, recall: float) -> float:
    """Return the harmonic mean of joint precision and joint recall."""
    precision = float(precision)
    recall = float(recall)
    if not math.isfinite(precision) or not math.isfinite(recall):
        raise ValueError("joint precision and recall must be finite")
    if precision < 0.0 or recall < 0.0:
        raise ValueError("joint precision and recall must be non-negative")
    denominator = precision + recall
    return 0.0 if denominator <= 0.0 else 2.0 * precision * recall / denominator


@dataclass(frozen=True)
class JointProxyDecision:
    selected: bool
    reason: str
    step: int
    mdd: float
    joint_precision: float
    joint_recall: float
    joint_f1: float

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "selected": self.selected,
            "reason": self.reason,
            "step": self.step,
            "MDD": self.mdd,
            "JointPrecision": self.joint_precision,
            "JointRecall": self.joint_recall,
            "JointF1": self.joint_f1,
        }


class JointProxyCheckpointSelector:
    """Select checkpoints by JointF1, with MDD as a guard and tie-breaker.

    JointF1 must improve by more than ``joint_f1_tolerance`` to replace the
    current best. Scores inside that tolerance are treated as ties and the
    lower-MDD checkpoint wins. No unrelated metric fallback is permitted.
    """

    def __init__(self, *, mdd_threshold: float, joint_f1_tolerance: float = 0.0) -> None:
        self.mdd_threshold = float(mdd_threshold)
        self.joint_f1_tolerance = float(joint_f1_tolerance)
        if not math.isfinite(self.mdd_threshold) or self.mdd_threshold <= 0.0:
            raise ValueError("mdd_threshold must be finite and positive")
        if not math.isfinite(self.joint_f1_tolerance) or self.joint_f1_tolerance < 0.0:
            raise ValueError("joint_f1_tolerance must be finite and non-negative")
        self.best_step: int | None = None
        self.best_joint_f1: float | None = None
        self.best_mdd: float | None = None

    def consider(self, metrics: Mapping[str, float], *, step: int) -> JointProxyDecision:
        mdd = _finite_metric(metrics, "MDD")
        precision = _finite_metric(metrics, "JointPrecision")
        recall = _finite_metric(metrics, "JointRecall")
        score = joint_f1(precision, recall)

        if mdd > self.mdd_threshold:
            return JointProxyDecision(False, "mdd_threshold", int(step), mdd, precision, recall, score)

        selected = False
        reason = "not_improved"
        if self.best_joint_f1 is None or self.best_mdd is None:
            selected = True
            reason = "first_eligible"
        else:
            delta = score - self.best_joint_f1
            if delta > self.joint_f1_tolerance:
                selected = True
                reason = "joint_f1_improved"
            elif abs(delta) <= self.joint_f1_tolerance and mdd < self.best_mdd:
                selected = True
                reason = "mdd_tiebreak"

        if selected:
            self.best_step = int(step)
            self.best_joint_f1 = score
            self.best_mdd = mdd
        return JointProxyDecision(selected, reason, int(step), mdd, precision, recall, score)


def _finite_metric(metrics: Mapping[str, float], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"proxy checkpoint metric {name!r} is missing")
    try:
        value = float(metrics[name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"proxy checkpoint metric {name!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"proxy checkpoint metric {name!r} must be finite")
    return value
