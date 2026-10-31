"""Model runners (M1): full probability vectors over each dataset's options.

A runner has `manifest` (dict) and `predict(texts, question, options) -> list[list[float]]`, one
distribution per text in the options' order. `run` drives it over a split and `write` stores the
result as submissions/<model>/predictions.parquet + manifest.json (SPEC §6).

Weights, tokenizer and any remote code all come from one snapshot at a pinned commit.
"""

from __future__ import annotations

import datetime
import json
import platform
import re
import time
from pathlib import Path

import pandas as pd

import cbench.perturb as perturb_mod
from cbench.datasets import load
from cbench.metrics import _check

SHA = re.compile(r"[0-9a-f]{40}")


def _snapshot(model_id: str, revision: str) -> str:
    # trust_remote_code runs whatever the repo holds, so a branch name is never accepted
    if not SHA.fullmatch(revision):
        raise ValueError(f"revision must be a full 40-char commit sha, got {revision!r}")
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, revision=revision)


def _env(device, dtype) -> dict:
    import torch
    import transformers

    return {
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "machine": f"{platform.system()} {platform.machine()}",
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


class D1:
    """Liquid AI d1 models via their own System One readout (one `choice` question, plain softmax)."""

    def __init__(self, model_id: str, revision: str, device: str | None = None, weights: str | None = None):
        """weights="REPO@SHA" runs another checkpoint of the same architecture through d1's readout, e.g. d1's
        base model, so the only difference from d1 is the decision training."""
        import torch
        from transformers import AutoModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if device == "cpu" else torch.bfloat16  # vendor's recommendation
        path = Path(_snapshot(model_id, revision))
        if weights:
            import tempfile

            # d1's config, code and tokenizer with the other checkpoint's weights, loaded by transformers itself so
            # its key renames and tied weights apply exactly as for d1
            repo, _, rev = weights.partition("@")
            mixed = Path(tempfile.mkdtemp(prefix="cbench-weights-"))
            for f in path.iterdir():
                if f.name != "model.safetensors":
                    (mixed / f.name).symlink_to(f)
            (mixed / "model.safetensors").symlink_to(Path(_snapshot(repo, rev)) / "model.safetensors")
            path = mixed
        model, info = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype, output_loading_info=True)
        if info["missing_keys"] or info["unexpected_keys"]:  # transformers would fill missing ones with random values
            raise ValueError(f"checkpoint does not match {model_id}: {info}")
        model = model.to(device)
        # d1-3B answers through a SystemOne engine; d1-omni has probabilities_batch on the model itself and applies
        # the per-type temperatures in its config.json to text answers
        self.engine = getattr(model, "engine", model)
        how = ("option codes scored at one slot, calibration=None" if hasattr(model, "engine")
               else "one marker per option, per-type temperatures from config.json")
        self.manifest = {
            "runner": "d1",
            "model": repo if weights else model_id,
            "revision": rev if weights else revision,
            "readout": f"System One choice question, {how} (vendor default)"
            + (f"; code from {model_id}@{revision}" if weights else ""),
            **_env(device, dtype),
        }

    def predict(self, texts, question, options):
        q = {"type": "choice", "instructions": question, "criteria": {o: label(o) for o in options}}
        return [per_q[0] for per_q in self.engine.probabilities_batch([(t, [q]) for t in texts])]  # one question each


def label(option: str) -> str:
    """How every runner shows an option to a model: dataset ids like non_toxic read as words (non toxic)."""
    return option.replace("_", " ")


def render(text: str, question: str, options) -> str:
    shown = ", ".join(label(o) for o in options)
    return f"{question}\n\nText:\n{text}\n\nOptions: {shown}\n\nAnswer with exactly one option."


def end_of_turn(tok, fallback: int | list[int]) -> int:
    """The token a chat template puts right after an assistant reply (Qwen: <|im_end|>, not <|endoftext|>).

    Read from the template, because generation_config can be missing or name a different eos.
    """
    if tok.chat_template is None:
        return fallback[0] if isinstance(fallback, list) else fallback
    full = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "XYZ"}], tokenize=False
    )
    suffix = full[full.rfind("XYZ") + 3 :]
    return tok(suffix, add_special_tokens=False).input_ids[0]


class LogprobLLM:
    """Any causal LM: P(option text + end of turn | prompt), renormalised over the options."""

    option_batch = 16  # options scored per forward pass

    def __init__(self, model_id: str, revision: str, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = _snapshot(model_id, revision)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        self.eos = end_of_turn(self.tok, self.model.generation_config.eos_token_id)
        self.device = device
        self.manifest = {
            "runner": "logprob",
            "model": model_id,
            "revision": revision,
            "readout": "sum of token logprobs of each option + end-of-turn token after the chat prompt, "
            "softmax over options",
            "prompt": render("{text}", "{question}", ["{options}"]),
            "chat_template": self.tok.chat_template is not None,
            "end_token": self.tok.convert_ids_to_tokens(self.eos),
            **_env(device, dtype),
        }

    def _prompt_ids(self, text, question, options) -> list[int]:
        msg = render(text, question, options)
        if self.tok.chat_template is None:
            return self.tok(msg + "\nAnswer:", add_special_tokens=True).input_ids
        # Qwen-style templates read enable_thinking; others ignore it
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.tok(prompt, add_special_tokens=False).input_ids

    def option_logprobs(self, text, question, options) -> list[float]:
        import copy

        import torch

        prompt = self._prompt_ids(text, question, options)
        lead = "" if self.tok.chat_template else " "
        conts = [self.tok(lead + label(o), add_special_tokens=False).input_ids + [self.eos] for o in options]
        b = min(self.option_batch, len(options))
        # The prompt (minus its last token) is encoded once, already batched b-wide: hybrid caches (e.g. Qwen3.5's
        # linear-attention layers) can't be batch-repeated later, but every cache type can be deep-copied.
        with torch.no_grad():
            head = torch.tensor([prompt[:-1]] * b, device=self.device)
            cache = self.model(input_ids=head, use_cache=True).past_key_values
        by_len: dict[int, list[int]] = {}
        for i, c in enumerate(conts):
            by_len.setdefault(len(c), []).append(i)
        out = [0.0] * len(options)
        for n, idx in by_len.items():  # equal-length rows need no padding
            for k in range(0, len(idx), b):
                chunk = idx[k : k + b]
                rows = [[prompt[-1]] + conts[i] for i in chunk]
                rows += [rows[-1]] * (b - len(rows))  # fill to the cache's batch size, extra rows are dropped
                ids = torch.tensor(rows, device=self.device)
                with torch.no_grad():
                    logits = self.model(input_ids=ids, past_key_values=copy.deepcopy(cache)).logits[:, :-1].float()
                lp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1).sum(-1)
                for i, v in zip(chunk, lp.tolist()):
                    out[i] = v
        return out

    def predict(self, texts, question, options):
        import numpy as np

        rows = []
        for t in texts:
            lp = np.array(self.option_logprobs(t, question, options))
            p = np.exp(lp - lp.max())
            rows.append((p / p.sum()).tolist())
        return rows


class Verbal:
    """API models that expose no token logprobs (the Claude Messages API has none): the model states a
    probability for every option through a JSON schema, renormalised to sum to 1.

    This is verbalized confidence, a different measurement from the logprob runners; report it separately.
    No `fallbacks`: a refusal must not be answered silently by another model, so refused items are dropped
    and counted instead.
    """

    def __init__(self, model_id: str = "claude-opus-5-5", effort: str = "low", client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model_id, effort
        self.usage = {"input_tokens": 0, "output_tokens": 0}  # billed tokens, for $ per 1k decisions
        self.manifest = {
            "runner": "verbal",
            "model": model_id,
            "revision": "api",  # hosted models can't be pinned; runs record the date and served model
            "readout": "probability per option stated through output_config json_schema, renormalised",
            "effort": effort,
            "prompt": render("{text}", "{question}", ["{options}"]) + VERBAL_SUFFIX,
        }

    def _ask(self, text, question, options):
        schema = {
            "type": "object",
            "properties": {label(o): {"type": "number"} for o in options},
            "required": [label(o) for o in options],
            "additionalProperties": False,
        }
        r = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": render(text, question, options) + VERBAL_SUFFIX}],
        )
        self.usage = {k: v + getattr(r.usage, k) for k, v in self.usage.items()}
        if r.stop_reason == "refusal":
            return None
        if r.stop_reason != "end_turn":
            raise RuntimeError(f"{self.model} stopped with {r.stop_reason}")
        self.manifest["served_model"] = r.model
        stated = json.loads(next(b.text for b in r.content if b.type == "text"))
        p = [max(float(stated[label(o)]), 0.0) for o in options]
        return [x / sum(p) for x in p] if sum(p) > 0 else None

    def predict(self, texts, question, options):
        return [self._ask(t, question, options) for t in texts]


VERBAL_SUFFIX = "\n\nFor each option, give the probability that it is the correct answer. The probabilities sum to 1."

PERTURBATIONS = ("order", "edit", "attack", "paraphrase")


def _variants(ds, df, perturb, n):
    """(variant name, ids, texts, option order) for one perturbation family; 'orig' when perturb is None."""
    k = len(ds.options)
    if perturb is None:
        yield "orig", df.id.tolist(), df.text.tolist(), list(range(k))
    elif perturb == "order":
        for j, order in enumerate(perturb_mod.orders(k, n, ds.source.name)):
            yield f"order:{j}", df.id.tolist(), df.text.tolist(), order
    elif ds.source.name in perturb_mod.NO_TEXT_PERTURBATION:
        raise ValueError(f"{ds.source.name} has no text perturbations: edits would change what is being labeled")
    elif perturb == "attack":
        return  # searched against the model itself, in _attack
    elif perturb == "edit":
        cands = [perturb_mod.edits(t, n, ds.source.name, i) for i, t in zip(df.id, df.text)]
        for j in range(n):
            yield f"edit:{j}", df.id.tolist(), [c[j] for c in cands], list(range(k))
    elif perturb == "paraphrase":
        para = perturb_mod.paraphrases(ds.source.name)
        para = para[para.id.isin(df.id) & (para.k < n)]
        if para.empty:
            raise ValueError(f"no stored paraphrases for {ds.source.name}; generate them first")
        for j, g in para.groupby("k"):
            yield f"paraphrase:{j}", g.id.tolist(), g.text.tolist(), list(range(k))
    else:
        raise ValueError(f"unknown perturbation {perturb!r}")


def _attack(runner, ds, df, budget):
    """Rows of the greedy minimal-edit attack: attack:k holds the answer after k+1 edits (or the flipped one)."""
    if ds.source.name in perturb_mod.NO_TEXT_PERTURBATION:
        raise ValueError(f"{ds.source.name} has no text perturbations: edits would change what is being labeled")
    predict = lambda texts: runner.predict(texts, ds.source.question, ds.options)  # noqa: E731
    rows, asked = [], 0
    for i, text in zip(df.id, df.text):
        t0 = time.perf_counter()
        p0, steps = perturb_mod.attack(predict, text, budget, ds.source.name, i)
        ms = (time.perf_counter() - t0) * 1000  # the whole search for this item
        asked += budget
        if p0 is not None:
            rows += [(ds.source.name, i, f"attack:{k}", list(p), ms) for k, (_, p) in enumerate(steps)]
    return rows, asked


def run(runner, dataset: str, split: str = "test", limit: int | None = None, batch_size: int = 8,
        perturb: str | None = None, n_variants: int = 3):
    """Predictions for one dataset split: columns dataset, id, variant, probs, latency_ms.

    Probabilities are stored in the dataset's option order, whatever order the model was shown.
    A runner may return None for an item it refused; those rows are left out and counted in df.attrs.
    """
    ds = load(dataset)
    df = ds.df[ds.df.split == split].head(limit)
    rows, asked = _attack(runner, ds, df, n_variants) if perturb == "attack" else ([], 0)
    usage = dict(getattr(runner, "usage", {}))
    for variant, ids, texts, order in _variants(ds, df, perturb, n_variants):
        asked += len(texts)
        shown = tuple(ds.options[i] for i in order)
        for k in range(0, len(texts), batch_size):
            batch = texts[k : k + batch_size]
            t0 = time.perf_counter()
            out = runner.predict(batch, ds.source.question, shown)
            # per-decision latency is batch wall time / batch size; p50/p99 of single calls needs batch_size=1
            ms = (time.perf_counter() - t0) * 1000 / len(batch)
            for i, p in zip(ids[k : k + batch_size], out):
                if p is not None:
                    if len(p) != len(order):
                        raise ValueError(f"runner returned {len(p)} probabilities for {len(order)} options")
                    canon = [0.0] * len(order)
                    for pos, opt in enumerate(order):
                        canon[opt] = p[pos]
                    rows.append((dataset, i, variant, canon, ms))
    preds = pd.DataFrame(rows, columns=["dataset", "id", "variant", "probs", "latency_ms"])
    labels = df.set_index("id").label.loc[preds.id].to_numpy()
    _check(preds.probs.tolist(), labels)  # every row a valid distribution over this dataset's options
    preds.attrs["refused"] = asked - len(preds)
    if usage:
        preds.attrs["usage"] = {k: runner.usage[k] - v for k, v in usage.items()}
    return preds


def write(out_dir: str | Path, preds: pd.DataFrame, manifest: dict, run_info: dict) -> Path:
    """Upsert one dataset's rows into predictions.parquet and record the run in manifest.json."""
    out = Path(out_dir)
    path, mpath = out / "predictions.parquet", out / "manifest.json"
    m = json.loads(mpath.read_text()) if mpath.exists() else {}
    if m and (m["runner"], m["model"], m["revision"]) != (manifest["runner"], manifest["model"], manifest["revision"]):
        raise ValueError(f"{out} holds predictions from {m['model']}@{m['revision']}; use another directory")
    (dataset,) = preds.dataset.unique()
    (family,) = preds.variant.str.split(":").str[0].unique()
    out.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(path) if path.exists() else preds.iloc[:0]
    replaced = (old.dataset == dataset) & (old.variant.str.split(":").str[0] == family)
    pd.concat([old[~replaced], preds], ignore_index=True).to_parquet(path, index=False)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    key = dataset if family == "orig" else f"{dataset}/{family}"
    info = {**run_info, "n": len(preds), "refused": preds.attrs.get("refused", 0), "date": now}
    if "usage" in preds.attrs:
        info["usage"] = preds.attrs["usage"]
    runs = {**m.get("runs", {}), key: info}
    mpath.write_text(json.dumps({**m, **manifest, "runs": runs}, indent=2) + "\n")
    return path
