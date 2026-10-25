import json
import os

import numpy as np
import pandas as pd
import pytest

import cbench.runners as runners
from cbench.datasets import Dataset, SOURCES
from cbench.runners import run, write

OPTIONS = ("negative", "neutral", "positive")


class Fake:
    manifest = {"runner": "fake", "model": "org/fake", "revision": "a" * 40}

    def __init__(self, probs=(0.2, 0.3, 0.5)):
        self.probs, self.calls = list(probs), []

    def predict(self, texts, question, options):
        self.calls.append((len(texts), question, options))
        return [self.probs for _ in texts]


@pytest.fixture(autouse=True)
def toy_dataset(monkeypatch):
    df = pd.DataFrame(
        {
            "id": [f"id{i}" for i in range(7)],
            "text": [f"t{i}" for i in range(7)],
            "label": [0, 1, 2, 0, 1, 2, 0],
            "split": ["calib", "calib"] + ["test"] * 5,
        }
    )
    monkeypatch.setattr(runners, "load", lambda name: Dataset(SOURCES[name], OPTIONS, df))


def test_run_batches_test_split_with_dataset_question():
    fake = Fake()
    preds = run(fake, "sentiment_hard", batch_size=2)
    assert preds.id.tolist() == ["id2", "id3", "id4", "id5", "id6"]
    assert [c[0] for c in fake.calls] == [2, 2, 1]
    assert fake.calls[0][1:] == (SOURCES["sentiment_hard"].question, OPTIONS)
    assert (preds.latency_ms >= 0).all() and preds.dataset.eq("sentiment_hard").all()


def test_run_rejects_invalid_distributions():
    with pytest.raises(ValueError):
        run(Fake(probs=(0.5, 0.5, 0.5)), "sentiment_hard")
    with pytest.raises(ValueError):
        run(Fake(probs=(0.5, 0.5)), "sentiment_hard")  # wrong number of options


def test_write_upserts_per_dataset_and_round_trips(tmp_path):
    info = {"split": "test", "limit": None, "batch_size": 8}
    write(tmp_path, run(Fake(), "sentiment_hard"), Fake.manifest, info)
    write(tmp_path, run(Fake(), "toxicity", limit=2), Fake.manifest, {**info, "limit": 2})
    write(tmp_path, run(Fake((0.6, 0.2, 0.2)), "sentiment_hard"), Fake.manifest, info)  # replaces, not appends
    df = pd.read_parquet(tmp_path / "predictions.parquet")
    assert df.groupby("dataset").size().to_dict() == {"sentiment_hard": 5, "toxicity": 2}
    np.testing.assert_allclose(df[df.dataset == "sentiment_hard"].probs.iloc[0], [0.6, 0.2, 0.2])
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["revision"] == "a" * 40 and m["runs"]["toxicity"]["limit"] == 2 and m["runs"]["sentiment_hard"]["n"] == 5


def test_write_refuses_other_model_without_touching_files(tmp_path):
    info = {"split": "test", "limit": None, "batch_size": 8}
    write(tmp_path, run(Fake(), "sentiment_hard"), Fake.manifest, info)
    before = (tmp_path / "predictions.parquet").read_bytes()
    with pytest.raises(ValueError, match="another directory"):
        write(tmp_path, run(Fake(), "toxicity"), {**Fake.manifest, "revision": "b" * 40}, info)
    assert (tmp_path / "predictions.parquet").read_bytes() == before


def test_d1_predict_sends_one_choice_question_and_unwraps_it():
    class Engine:  # mimics SystemOne.probabilities_batch: per request, one distribution per question
        def probabilities_batch(self, requests):
            self.requests = requests
            return [[[0.1, 0.9]] for _ in requests]

    d1 = runners.D1.__new__(runners.D1)
    d1.engine = Engine()
    assert d1.predict(["a", "b"], "Toxic?", ("non_toxic", "toxic")) == [[0.1, 0.9], [0.1, 0.9]]
    state, (q,) = d1.engine.requests[0]
    assert state == "a" and q == {"type": "choice", "instructions": "Toxic?",
                                  "criteria": {"non_toxic": "non toxic", "toxic": "toxic"}}


@pytest.mark.parametrize("rev", ["main", "v12", "051bcc46", "A" * 40])
def test_snapshot_requires_full_commit_sha(rev):
    with pytest.raises(ValueError, match="commit sha"):
        runners._snapshot("LiquidAI/d1-3B", rev)


@pytest.mark.skipif(not os.environ.get("CBENCH_NETWORK"), reason="set CBENCH_NETWORK=1 to download a tokenizer")
def test_end_of_turn_comes_from_chat_template_not_generation_config():
    pytest.importorskip("transformers")
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    # Qwen3.5-0.8B ships no generation_config.json; the config eos is <|endoftext|>, but replies end in <|im_end|>
    path = snapshot_download("Qwen/Qwen3.5-0.8B", revision="2fc06364715b967f1860aea9cf38778875588b17",
                             allow_patterns=["tokenizer*", "*.jinja", "*.json"])
    tok = AutoTokenizer.from_pretrained(path)
    eot = runners.end_of_turn(tok, fallback=tok.convert_tokens_to_ids("<|endoftext|>"))
    assert tok.convert_ids_to_tokens(eot) == "<|im_end|>"


@pytest.mark.skipif(not os.environ.get("CBENCH_NETWORK"), reason="set CBENCH_NETWORK=1 to download a tiny model")
def test_logprob_scores_equal_brute_force_sequence_logprob():
    torch = pytest.importorskip("torch")
    llm = runners.LogprobLLM("hf-internal-testing/tiny-random-LlamaForCausalLM", "9fb191250dd56d0ba7ec9785a025ed29c03d5998")
    llm.option_batch = 2  # force several chunks per length group
    options = ["no", "no_comment", "yes", "maybe later", "x"]
    got = llm.option_logprobs("some text", "Pick one.", options)
    prompt = llm._prompt_ids("some text", "Pick one.", options)
    for o, g in zip(options, got):
        cont = llm.tok(" " + o, add_special_tokens=False).input_ids + [llm.eos]
        ids = torch.tensor([prompt + cont])
        with torch.no_grad():
            lp = llm.model(input_ids=ids).logits[0].float().log_softmax(-1)
        want = sum(lp[len(prompt) - 1 + j, t].item() for j, t in enumerate(cont))
        assert g == pytest.approx(want, abs=1e-4), o
    probs = llm.predict(["some text"], "Pick one.", options)[0]
    assert sum(probs) == pytest.approx(1) and np.argmax(probs) == np.argmax(got)
