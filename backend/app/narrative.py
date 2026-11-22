"""Step 5.2 — plain-English narrative, restricted to the structured facts.

The LLM (NVIDIA Nemotron via the OpenAI-compatible API, same as nemotron.py)
only restates facts. A code check then rejects any output that cites a fact
ID that doesn't exist or uses a number that isn't in the facts; on failure we
fall back to concatenating the rule cards' sentences. Results are cached by
input hash so the demo works offline after one run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from .store import DATA

MODEL = os.environ.get("CRASHLENS_LLM_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
BASE_URL = os.environ.get("CRASHLENS_LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
CACHE = DATA / "narrative_cache"

PROMPT = (
    "You are writing a neutral crash summary for a driver. Use ONLY the facts provided. "
    "Do not add facts, speeds, or causes that are not in the input. Cite each claim with its "
    "fact ID in brackets, like [F3]. If confidence is below 0.6, say \"likely\" rather than "
    "stating it as certain. Three short paragraphs: what happened, which traffic laws apply, "
    "what the fault split means under Texas law. Plain text only, no markdown."
)


def _payload(verdict: dict) -> dict:
    return {
        "facts": [{k: f.get(k) for k in ("id", "actor", "type", "value", "unit", "confidence", "label", "status")}
                  for f in verdict["facts"]],
        "fired_rules": [{k: r[k] for k in ("statute", "title", "charged", "explanation", "contribution", "fact_ids")}
                        for r in verdict["fired_rules"]],
        "split_percent": {k: round(v * 100) for k, v in (verdict.get("split") or {}).items()},
        "recovery": (verdict.get("recovery") or {}).get("text"),
    }


def _allowed_numbers(payload: dict) -> set[str]:
    blob = json.dumps(payload)
    nums = set()
    for m in re.findall(r"-?\d+(?:\.\d+)?", blob):
        x = float(m)
        for v in (x, round(x), round(x, 1), round(x * 100), round(x * 100, 1)):
            nums.add(_norm(v))
    return nums


def _norm(x) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) else f"{x:.1f}".rstrip("0").rstrip(".")


def check(text: str, payload: dict) -> list[str]:
    """Return problems; empty list means the narrative passes the hallucination guard."""
    problems = []
    ids = {f["id"] for f in payload["facts"]}
    for cited in re.findall(r"\[(F\d+)\]", text):
        if cited not in ids:
            problems.append(f"cites unknown fact {cited}")
    allowed = _allowed_numbers(payload)
    stripped = re.sub(r"\[F\d+\]", "", text)
    stripped = re.sub(r"§\s*[\d.]+", "", stripped)  # statute numbers come from the rules
    for m in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", stripped):
        if _norm(m) not in allowed:
            problems.append(f"number {m} not in facts")
    if not re.search(r"\[F\d+\]", text):
        problems.append("no citations")
    return problems


def fallback(verdict: dict) -> str:
    rules = verdict["fired_rules"]
    if not rules:
        return "The video does not support any of the traffic-law rules CrashLens checks, so no fault split is given."
    p1 = " ".join(f"{r['explanation']} [{', '.join(r['fact_ids'])}]" for r in rules)
    p2 = " ".join(f"{r['title']} ({r['statute']}) applies to driver {r['charged']}." for r in rules)
    sp = verdict.get("split") or {}
    p3 = (f"The rules assign {round(sp.get('A', 0) * 100)}% of responsibility to A and "
          f"{round(sp.get('B', 0) * 100)}% to B. " + ((verdict.get("recovery") or {}).get("text") or ""))
    return "\n\n".join((p1, p2, p3))


def _api_key() -> str | None:
    """Process env first; on Windows also the user-level variable (what a fresh terminal would see)."""
    key = os.environ.get("NVIDIA_API_KEY")
    if key or os.name != "nt":
        return key
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "NVIDIA_API_KEY")[0] or None
    except OSError:
        return None


def _llm(payload: dict) -> str | None:
    key = _api_key()
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(base_url=BASE_URL, api_key=key, timeout=40)
        # Nemotron is a reasoning model: give it room so the answer isn't cut off after its
        # hidden reasoning (reasoning_content), and ask it to keep that reasoning short.
        r = client.chat.completions.create(
            model=MODEL, temperature=0.2, max_tokens=4096,
            messages=[{"role": "system", "content": "detailed thinking off\n\n" + PROMPT},
                      {"role": "user", "content": json.dumps(payload)}])
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        if not text:
            print(f"[narrative] empty LLM content (finish_reason={r.choices[0].finish_reason})")
            return None
        return text
    except Exception as e:  # network / auth / model errors → fallback, but say why
        print(f"[narrative] LLM call failed: {e!r}")
        return None


def narrate(verdict: dict) -> dict:
    payload = _payload(verdict)
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))
    text = _llm(payload)
    if text is None:
        out = {"text": fallback(verdict), "source": "rule_cards", "problems": ["LLM unavailable"]}
    else:
        problems = check(text, payload)
        out = ({"text": text, "source": "llm", "model": MODEL, "problems": []} if not problems else
               {"text": fallback(verdict), "source": "rule_cards", "problems": problems, "rejected": text})
    if out["source"] == "llm":  # only cache real LLM output, so adding a key later takes effect
        cp.write_text(json.dumps(out), encoding="utf-8")
    return out
