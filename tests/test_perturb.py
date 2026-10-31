import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cbench import perturb


def test_orders_are_seeded_non_identity_and_reversed_first():
    o = perturb.orders(3, 3, "ds")
    assert o[0] == [2, 1, 0] and all(p != [0, 1, 2] for p in o) and len({tuple(p) for p in o}) == 3
    assert o == perturb.orders(3, 3, "ds")
    assert perturb.orders(2, 5, "ds") == [[1, 0]]  # only one non-identity order exists for 2 options


def test_typo_is_one_small_edit_keeping_first_letter():
    text = "The service was absolutely wonderful today"
    for k in range(50):
        out = perturb.edits(text, 50, "ds", "id1")[k]
        assert out != text and abs(len(out) - len(text)) <= 1
        changed = [(a, b) for a, b in zip(text.split(), out.split()) if a != b]
        assert len(changed) == 1 and changed[0][0][0] == changed[0][1][0] and len(changed[0][0]) >= 4
    assert perturb.edits(text, 3, "ds", "id1") == perturb.edits(text, 3, "ds", "id1")
    assert perturb.edits("ok no", 2, "ds", "x") == ["ok no", "ok no"]  # nothing long enough to edit


def test_report_hand_computed():
    orig = [[0.8, 0.2], [0.4, 0.6]]
    preds = pd.DataFrame(
        {
            "dataset": "d",
            "id": ["a", "b", "a", "b", "a", "b"],
            "variant": ["orig", "orig", "edit:0", "edit:0", "edit:1", "edit:1"],
            "probs": orig + [[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.4, 0.6]],
        }
    )
    (r,) = perturb.report(preds).to_dict("records")
    # flips: a/e0 no, b/e0 yes, a/e1 yes, b/e1 no -> 2/4 ; any_flip: a yes, b yes -> 1.0
    assert (r["family"], r["items"], r["variants"], r["flip_rate"], r["any_flip"]) == ("edit", 2, 4, 0.5, 1.0)
    want = np.mean([0, 0.4 * np.log(0.4 / 0.6) + 0.6 * np.log(0.6 / 0.4),
                    0.8 * np.log(0.8 / 0.3) + 0.2 * np.log(0.2 / 0.7), 0])
    assert r["kl"] == pytest.approx(want)


def test_agreement_counts_only_marked_rows():
    sheet = pd.DataFrame({"same_meaning": ["y", "no", "YES", "", " 1 "]})
    assert perturb.agreement(sheet) == (0.75, 4)


class FakeClient:
    def __init__(self, stop="end_turn"):
        self.messages, self.stop, self.calls = self, stop, []

    def create(self, **kw):
        self.calls.append(kw)
        keys = kw["output_config"]["format"]["schema"]["required"]
        text = json.dumps({k: f"rewrite {k}" for k in keys})
        return SimpleNamespace(stop_reason=self.stop, model="claude-x", content=[SimpleNamespace(type="text", text=text)])


def test_generate_paraphrases_and_spotcheck(tmp_path, monkeypatch):
    import cbench.datasets as datasets

    df = pd.DataFrame({"id": ["a", "b", "c"], "text": ["t1", "t2", "t3"], "label": 0, "split": ["calib", "test", "test"]})
    monkeypatch.setattr(datasets, "load", lambda name: SimpleNamespace(df=df))
    monkeypatch.setattr(perturb, "PARAPHRASES", tmp_path)
    client = FakeClient()
    perturb.generate_paraphrases("sentiment_hard", n_items=5, k=2, client=client)
    para = perturb.paraphrases("sentiment_hard")
    assert para[["id", "k", "text"]].values.tolist() == [["b", 0, "rewrite p0"], ["b", 1, "rewrite p1"],
                                                          ["c", 0, "rewrite p0"], ["c", 1, "rewrite p1"]]
    assert "t2" in client.calls[0]["messages"][0]["content"] and "fallbacks" not in client.calls[0]
    sheet = perturb.spotcheck_sheet("sentiment_hard", n=3)
    assert len(sheet) == 3 and set(sheet.original) <= {"t2", "t3"} and (sheet.same_meaning == "").all()
    with pytest.raises(ValueError):
        perturb.generate_paraphrases("code_review_risk", client=client)


def test_word_edits_toggle_contractions_both_ways_keeping_case():
    got = perturb.word_edits("I do not like it. It is fine, it’s ok, Can't say.")
    assert "I don't like it. It is fine, it’s ok, Can't say." in got
    assert "I do not like it. It's fine, it’s ok, Can't say." in got  # "it is" is contracted
    assert "I do not like it. It is fine, it’s ok, Cannot say." in got  # curly or straight apostrophe, case kept
    assert not any("it is ok" in g for g in got)  # "it's" could mean "it has": never expanded
    assert perturb.word_edits("nothing to toggle here") == []


def test_attack_stops_at_the_first_flip_and_repeats_it():
    calls = []

    def predict(texts):  # answer 0 unless the text contains "don't"
        calls.append(len(texts))
        return [[0.2, 0.8] if "don't" in t else [0.9, 0.1] for t in texts]

    p0, steps = perturb.attack(predict, "I do not want this item", 3, "ds", "id1", n_typos=4)
    assert p0 == [0.9, 0.1] and len(steps) == 3
    assert steps[0] == ("I don't want this item", [0.2, 0.8])  # the contraction is the edit that flips it
    assert steps[1] == steps[2] == steps[0] and len(calls) == 2  # no queries after the flip
    _, steps = perturb.attack(lambda ts: [[0.9, 0.1]] * len(ts), "Nothing flips this sentence", 2, "ds", "id1")
    assert [np.argmax(p) for _, p in steps] == [0, 0] and steps[0][0] != "Nothing flips this sentence"


def test_attack_skips_typos_that_make_a_real_word():
    # the greedy attack found "that" -> "twat" and "great" -> "greta"; those change meaning, so they are not candidates
    assert perturb.makes_real_word("on top of that", "on top of twat")
    assert perturb.makes_real_word("great deal", "greta deal") and perturb.makes_real_word("good", "god")
    assert not perturb.makes_real_word("shut down", "sut down") and not perturb.makes_real_word("wonder", "woncer")
