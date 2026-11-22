"""Step 6.4/6.5 — calibrate rule weights on decided outcomes.

Each labelled case lists the rules that fired (driver, fact confidence) and the
fault share for A that an adjuster / arbitrator / court decided. We fit the
fixed rule weights w_r to minimise squared error between predicted and decided
splits, with an L2 pull toward the hand-set weights so a handful of cases can't
swing them wildly (a cheap stand-in for Bayesian updating).

The bundled cases are SIMULATED outcome data that demonstrate the mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .engine import load_rules

CASES_PATH = Path(__file__).with_name("calibration_cases.json")


def _predict(case: dict, w: dict[str, float]) -> float:
    s = {"A": 0.0, "B": 0.0}
    for r in case["fired"]:
        weight = r.get("weight") if r["rule_id"] not in w else w[r["rule_id"]]
        s[r["charged"]] += weight * r["confidence"] * r.get("causation", 1.0)
    tot = s["A"] + s["B"]
    return s["A"] / tot if tot > 0 else 0.5


def fit(cases: list[dict] | None = None, prior_strength: float = 0.02, iters: int = 600, lr: float = 0.5) -> dict:
    cases = cases or json.loads(CASES_PATH.read_text(encoding="utf-8"))
    rules = {r["id"]: r for r in load_rules()}
    base = {rid: float(r["weight"]) for rid, r in rules.items() if not isinstance(r["weight"], dict)}
    used = sorted({r["rule_id"] for c in cases for r in c["fired"] if r["rule_id"] in base})
    w = {k: base[k] for k in used}

    def loss(wd):
        err = sum((_predict(c, wd) - c["decided_fault_A"]) ** 2 for c in cases) / len(cases)
        reg = prior_strength * sum((wd[k] - base[k]) ** 2 for k in wd)
        return err + reg

    before = loss(w)
    mae_before = float(np.mean([abs(_predict(c, w) - c["decided_fault_A"]) for c in cases]))
    eps = 1e-4
    for _ in range(iters):
        g = {}
        for k in w:
            up = dict(w); up[k] += eps
            dn = dict(w); dn[k] -= eps
            g[k] = (loss(up) - loss(dn)) / (2 * eps)
        for k in w:
            w[k] = float(np.clip(w[k] - lr * g[k], 0.05, 1.0))
    mae_after = float(np.mean([abs(_predict(c, w) - c["decided_fault_A"]) for c in cases]))
    return {
        "n_cases": len(cases),
        "simulated": True,
        "weights": [{"rule_id": k, "title": rules[k]["title"], "before": round(base[k], 3), "after": round(w[k], 3)}
                    for k in used],
        "loss_before": round(before, 5), "loss_after": round(loss(w), 5),
        "mae_before_pct": round(mae_before * 100, 1), "mae_after_pct": round(mae_after * 100, 1),
        "cases": [{"id": c["id"], "decided_by": c["decided_by"], "decided_A": c["decided_fault_A"],
                   "predicted_before": round(_predict(c, base), 3), "predicted_after": round(_predict(c, w), 3)}
                  for c in cases],
    }
