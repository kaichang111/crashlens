"""Step 4 — deterministic fault engine.

Facts in, weighted Texas-statute rules fire, split out. No model, no LLM:
every point of the split traces to a rule and the facts behind it.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

RULES_PATH = Path(__file__).with_name("rules.yaml")
DRIVERS = ("A", "B")

# Step 6.2 — how much each kind of dispute evidence is worth.
EVIDENCE_WEIGHT = {"second_video": 0.9, "photo": 0.7, "witness": 0.6, "user_word": 0.3}

CONF_BAND = 0.10  # ± band used for the uncertainty range


def load_rules(weight_overrides: dict[str, float] | None = None) -> list[dict]:
    rules = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    if weight_overrides:
        for r in rules:
            if r["id"] in weight_overrides and not isinstance(r["weight"], dict):
                r["weight"] = float(weight_overrides[r["id"]])
    return rules


# ---------------------------------------------------------------- disputes
def apply_disputes(facts: list[dict], disputes: list[dict]) -> list[dict]:
    """Return effective facts. Nothing is overwritten: originals stay in `original`.

    Evidence for a correction combines as p = 1 - prod(1 - w_i).
    If p beats the original confidence the value flips; otherwise the fact is
    contested and its confidence is discounted to c * (1 - p).
    """
    out = []
    by_fact: dict[str, list[dict]] = {}
    for d in disputes:
        by_fact.setdefault(d["fact_id"], []).append(d)
    for f in facts:
        f = copy.deepcopy(f)
        ds = by_fact.get(f["id"], [])
        if ds:
            keep = 1.0
            for d in ds:
                keep *= 1.0 - EVIDENCE_WEIGHT.get(d["evidence_type"], 0.3)
            p = 1.0 - keep
            f["original"] = {"value": f["value"], "confidence": f["confidence"]}
            f["dispute_strength"] = round(p, 3)
            if p > f["confidence"]:
                f["value"] = _coerce(ds[-1]["new_value"], f["value"])
                f["confidence"] = round(p, 3)
                f["status"] = "overturned"
            else:
                f["confidence"] = round(f["confidence"] * (1.0 - p), 3)
                f["status"] = "contested"
                f["contested_value"] = _coerce(ds[-1]["new_value"], f["value"])
        else:
            f.setdefault("status", "measured")
        out.append(f)
    return out


def _coerce(new: Any, like: Any) -> Any:
    if isinstance(like, bool):
        return str(new).lower() in ("true", "1", "yes")
    if isinstance(like, (int, float)) and not isinstance(like, bool):
        try:
            return float(new)
        except (TypeError, ValueError):
            return new
    return new


# ---------------------------------------------------------------- rules
def _index(facts: list[dict]) -> dict[tuple[str, str], dict]:
    return {(f["type"], f["actor"]): f for f in facts}


def _cmp(value: Any, cond: dict) -> bool:
    if value is None:
        return False
    if "eq" in cond:
        return value == cond["eq"]
    if "ne" in cond:
        return value != cond["ne"]
    try:
        if "lt" in cond:
            return float(value) < cond["lt"]
        if "gt" in cond:
            return float(value) > cond["gt"]
    except (TypeError, ValueError):
        return False
    return bool(value)


def _eval(node: dict, idx, self_d: str, other_d: str, used: list, bound: dict) -> bool:
    if "all" in node:
        for c in node["all"]:
            if not _eval(c, idx, self_d, other_d, used, bound):
                return False
        return True
    if "any" in node:
        hit = False
        for c in node["any"]:
            trial: list = []
            if _eval(c, idx, self_d, other_d, trial, bound):
                used.extend(trial)
                hit = True
        return hit
    actor = self_d if node.get("of", "self") == "self" else other_d
    f = idx.get((node["fact"], actor))
    if f is None and node.get("missing_ok") and ("ne" in node):
        return True  # no evidence of the excusing condition; contributes no confidence
    if f is None or not _cmp(f["value"], node):
        return False
    used.append(f)
    if "as" in node:
        bound[node["as"]] = f["value"]
    return True


def _weight(rule: dict, bound: dict) -> float:
    w = rule["weight"]
    if isinstance(w, dict):
        x = float(bound.get(w["fact"], w["from"]))
        return round(min(w["cap"], w["base"] + w["per_unit"] * max(0.0, x - w["from"])), 4)
    return float(w)


def _fmt(template: str, ctx: dict) -> str:
    class Safe(dict):
        def __missing__(self, k):
            return "?"
    try:
        return template.format_map(Safe(ctx))
    except (ValueError, TypeError):
        return template


def fire_rules(facts: list[dict], rules: list[dict],
               conf_shift: dict[str, float] | None = None) -> list[dict]:
    idx = _index(facts)
    fired: list[dict] = []
    for pass_primary in (True, False):  # primaries first, so "only_if_not_primary" can see them
        for rule in rules:
            if bool(rule.get("primary")) != pass_primary:
                continue
            for d in DRIVERS:
                other = "B" if d == "A" else "A"
                if rule.get("only_if_not_primary") and any(
                        r["charged"] == d and r["primary"] for r in fired):
                    continue
                used: list[dict] = []
                bound: dict = {}
                if not _eval(rule["when"], idx, d, other, used, bound):
                    continue
                conf = min(f["confidence"] for f in used) if used else 0.0
                if conf_shift:
                    conf = min(1.0, max(0.0, conf + conf_shift.get(d, 0.0)))
                w = _weight(rule, bound)
                k = float(rule.get("causation", 1.0))
                fired.append({
                    "rule_id": rule["id"],
                    "title": rule["title"],
                    "statute": rule["statute"],
                    "url": rule.get("url", ""),
                    "charged": d,
                    "primary": bool(rule.get("primary")),
                    "weight": w,
                    "confidence": round(conf, 3),
                    "causation": k,
                    "contribution": round(w * conf * k, 4),
                    "explanation": _fmt(rule["explain"], {"self": d, "other": other, **bound}),
                    "fact_ids": sorted({f["id"] for f in used}, key=lambda s: int(s[1:])),
                    "contested": any(f.get("status") == "contested" for f in used),
                })
    return fired


def _scores(fired: list[dict]) -> tuple[float, float] | None:
    s = {d: sum(r["contribution"] for r in fired if r["charged"] == d) for d in DRIVERS}
    if s["A"] + s["B"] <= 0:
        return None
    return s["A"], s["B"]


def score(facts: list[dict], disputes: list[dict] | None = None,
          weight_overrides: dict[str, float] | None = None,
          claimant: str = "A") -> dict:
    rules = load_rules(weight_overrides)
    eff = apply_disputes(facts, disputes or [])
    fired = fire_rules(eff, rules)
    sc = _scores(fired)
    if sc is None:  # never invent a 50/50
        return {"status": "insufficient_evidence", "facts": eff, "fired_rules": fired,
                "split": None, "range": None, "scores": {"A": 0, "B": 0}, "recovery": None}
    s_a, s_b = sc
    fault_a = s_a / (s_a + s_b)
    lo = _scores(fire_rules(eff, rules, {"A": -CONF_BAND, "B": +CONF_BAND}))
    hi = _scores(fire_rules(eff, rules, {"A": +CONF_BAND, "B": -CONF_BAND}))
    a_lo = lo[0] / sum(lo) if lo else fault_a
    a_hi = hi[0] / sum(hi) if hi else fault_a
    a_lo, a_hi = sorted((a_lo, a_hi))
    share = fault_a if claimant == "A" else 1 - fault_a
    return {
        "status": "ok",
        "facts": eff,
        "fired_rules": fired,
        "scores": {"A": round(s_a, 4), "B": round(s_b, 4)},
        "split": {"A": round(fault_a, 4), "B": round(1 - fault_a, 4)},
        "range": {"A": [round(a_lo, 4), round(a_hi, 4)], "B": [round(1 - a_hi, 4), round(1 - a_lo, 4)]},
        "recovery": recovery(share),
    }


def recovery(share: float) -> dict:
    """Tex. Civ. Prac. & Rem. Code § 33.001 — proportionate responsibility."""
    pct = round(share * 100)
    if share > 0.5:
        return {"claimant_share": round(share, 4), "recoverable_pct": 0, "barred": True,
                "text": f"At {pct}% responsibility you would be over the 50% bar and recover nothing "
                        "under Texas proportionate responsibility (Civ. Prac. & Rem. Code § 33.001)."}
    return {"claimant_share": round(share, 4), "recoverable_pct": 100 - pct, "barred": False,
            "text": f"At {pct}% responsibility, you could recover about {100 - pct}% of your damages. "
                    "At over 50%, you would recover nothing, which is why this number matters."}


if __name__ == "__main__":
    import sys
    print(json.dumps(score(json.loads(Path(sys.argv[1]).read_text()))["split"], indent=2))
