"""Fault engine, disputes, narrative guard and API smoke tests."""
import pytest
from fastapi.testclient import TestClient

from app import calibrate, engine, facts, narrative, scenarios


@pytest.fixture(scope="module")
def left_turn_facts():
    return facts.extract(scenarios.left_turn())


def fid(fs, ftype, actor):
    return next(f["id"] for f in fs if f["type"] == ftype and f["actor"] == actor)


def test_left_turn_matches_spec_example(left_turn_facts):
    v = engine.score(left_turn_facts)
    ids = {(r["rule_id"], r["charged"]) for r in v["fired_rules"]}
    assert ("left_turn_yield", "B") in ids
    assert ("unsafe_speed", "A") in ids
    assert 0.85 <= v["split"]["B"] <= 0.93          # spec worked example: B ≈ 91 %
    assert v["range"]["A"][0] <= v["split"]["A"] <= v["range"]["A"][1]
    assert v["recovery"]["barred"] is False


def test_rear_end_charges_follower():
    v = engine.score(facts.extract(scenarios.rear_end()))
    assert v["fired_rules"][0]["rule_id"] == "following_too_closely"
    assert v["split"]["A"] == 1.0


def test_no_rules_means_insufficient_evidence():
    v = engine.score([{"id": "F1", "actor": "A", "type": "speed_mph", "value": 30, "confidence": 0.9}])
    assert v["status"] == "insufficient_evidence" and v["split"] is None


def test_dispute_user_word_contests_photo_moves_more(left_turn_facts):
    sig = fid(left_turn_facts, "signal_state", "B")
    base = engine.score(left_turn_facts)["split"]["A"]
    d1 = [{"fact_id": sig, "new_value": "green_arrow", "evidence_type": "user_word"}]
    v1 = engine.score(left_turn_facts, d1)
    f1 = next(f for f in v1["facts"] if f["id"] == sig)
    assert f1["status"] == "contested" and f1["value"] == "green"
    assert v1["split"]["A"] > base
    v2 = engine.score(left_turn_facts, d1 + [{"fact_id": sig, "new_value": "green_arrow", "evidence_type": "photo"}])
    assert v2["split"]["A"] > v1["split"]["A"]


def test_strong_evidence_overturns(left_turn_facts):
    sig = fid(left_turn_facts, "signal_state", "B")
    ds = [{"fact_id": sig, "new_value": "green_arrow", "evidence_type": "second_video"}]
    v = engine.score(left_turn_facts, ds)
    f = next(f for f in v["facts"] if f["id"] == sig)
    assert f["status"] == "overturned" and f["value"] == "green_arrow"
    assert all(r["rule_id"] != "left_turn_yield" for r in v["fired_rules"])
    assert f["original"]["value"] == "green"  # nothing overwritten


def test_narrative_guard_rejects_invented_numbers(left_turn_facts):
    v = engine.score(left_turn_facts)
    payload = narrative._payload(v)
    ttc = next(f for f in left_turn_facts if f["type"] == "oncoming_ttc_s")
    assert narrative.check(f"B turned left while A was {ttc['value']:.1f} s away [{ttc['id']}].", payload) == []
    assert narrative.check("B was doing 55 mph [F6].", payload)
    assert narrative.check("Something happened [F99].", payload)


def _track(points, dt=1 / 30):
    import numpy as np
    from app import reconstruct as rc
    p = np.array(points, float)
    t = np.arange(len(p)) * dt
    v = np.gradient(p, dt, axis=0)
    return rc.Track("B", "pickup", "#f00", t, np.c_[p, v], np.full(len(p), 0.3), np.arctan2(v[:, 1], v[:, 0]))


def test_slow_left_turn_detected_by_lane_crossing():
    import numpy as np
    # southbound (x < 0), creeping to a near stop, then drifting east across the centre line into the
    # northbound lanes: hit only ~20° into the turn, so heading alone never shows it
    n = 150
    x = np.r_[np.full(90, -1.8), np.linspace(-1.8, 0.9, 60)]
    y = np.r_[np.linspace(8.0, 6.0, 90), np.linspace(6.0, 3.5, 60)]
    res = facts._left_across_centreline(_track(np.c_[x, y]), t_imp=n / 30, half=7.32)
    assert res is not None
    t_turn, lat, h_app = res
    assert lat >= 1.5 and 2.5 <= t_turn <= 3.6 and abs(h_app + np.pi / 2) < 1e-9


def test_straight_vehicle_not_a_left_turn():
    import numpy as np
    # southbound, staying in its own lanes
    y = np.linspace(30.0, 2.0, 150)
    assert facts._left_across_centreline(_track(np.c_[np.full(150, -1.8), y]), t_imp=5.0, half=7.32) is None


def test_calibration_reduces_error():
    r = calibrate.fit()
    assert r["simulated"] and r["mae_after_pct"] <= r["mae_before_pct"]


def test_api_cached_flow(monkeypatch):
    monkeypatch.setattr(narrative, "_api_key", lambda: None)  # no network in tests
    from app.main import app
    with TestClient(app) as c:
        r = c.post("/cases?cached=left_turn", data={"location_type": "intersection_signal"})
        assert r.status_code == 200
        cid = r.json()["id"]
        assert len(r.json()["evidence_sha256"]) == 64
        assert c.get(f"/cases/{cid}/status").json()["status"] == "done"
        sc = c.get(f"/cases/{cid}/scene").json()
        assert {v["id"] for v in sc["vehicles"]} == {"A", "B"} and sc["impact"]["type"] == "angle"
        v = c.get(f"/cases/{cid}/verdict").json()
        sig = fid(v["facts"], "signal_state", "B")
        d = c.post(f"/cases/{cid}/disputes", json={"fact_id": sig, "new_value": "green_arrow",
                                                    "evidence_type": "user_word"}).json()
        assert d["verdict"]["split"]["A"] > d["old_split"]["A"]
        assert "left_turn_yield:B" in d["changed_rules"]
        assert len(c.get(f"/cases/{cid}/disputes").json()) == 1
