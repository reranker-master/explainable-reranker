from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HardConcreteGate:
    temperature: float = 0.67
    gamma: float = -0.1
    zeta: float = 1.1

    def probability(self, logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit))

    def deterministic(self, logit: float) -> float:
        stretched = self.probability(logit) * (self.zeta - self.gamma) + self.gamma
        return min(max(stretched, 0.0), 1.0)

    def sample(self, logit: float, *, uniform: float) -> float:
        if not 0.0 < uniform < 1.0:
            raise ValueError("uniform must be in (0, 1)")
        logistic_noise = math.log(uniform) - math.log(1.0 - uniform)
        relaxed = 1.0 / (1.0 + math.exp(-((logit + logistic_noise) / self.temperature)))
        stretched = relaxed * (self.zeta - self.gamma) + self.gamma
        return min(max(stretched, 0.0), 1.0)

    def expected_l0(self, logit: float) -> float:
        # Louizos et al. hard-concrete expected non-zero probability.
        offset = -self.temperature * math.log(-self.gamma / self.zeta)
        return self.probability(logit + offset)


def hard_select_from_logits(
    logits: list[float],
    *,
    threshold: float = 0.0,
    min_selected: int = 1,
    max_selected: int | None = None,
) -> list[int]:
    selected = [1 if logit > threshold else 0 for logit in logits]
    if sum(selected) < min_selected and logits:
        ranked_indices = sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)
        for idx in ranked_indices[:min_selected]:
            selected[idx] = 1
    if max_selected is not None and sum(selected) > max_selected:
        keep = set(sorted(range(len(logits)), key=lambda idx: logits[idx], reverse=True)[:max_selected])
        selected = [1 if idx in keep else 0 for idx in range(len(logits))]
    return selected
