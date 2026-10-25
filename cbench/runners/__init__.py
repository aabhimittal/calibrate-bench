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

    def __init__(self, model_id: str, revision: str, device: str | None = None):
        import torch
        from transformers import AutoModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32 if device == "cpu" else torch.bfloat16  # vendor's recommendation
        model = AutoModel.from_pretrained(_snapshot(model_id, revision), trust_remote_code=True, dtype=dtype)
        self.engine = model.to(device).engine
        self.manifest = {
            "runner": "d1",
            "model": model_id,
            "revision": revision,
            "readout": "System One choice question, option codes scored at one slot, "
            "calibration=None (vendor default)",
            **_env(device, dtype),
        }

    def predict(self, texts, question, options):
        q = {"type": "choice", "instructions": question, "criteria": {o: o.replace("_", " ") for o in options}}
        return [per_q[0] for per_q in self.engine.probabilities_batch([(t, [q]) for t in texts])]  # one question each


def render(text: str, question: str, options) -> str:
    return f"{question}\n\nText:\n{text}\n\nOptions: {', '.join(options)}\n\nAnswer with exactly one option."


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
        import torch

        prompt = self._prompt_ids(text, question, options)
        lead = "" if self.tok.chat_template else " "
        conts = [self.tok(lead + o, add_special_tokens=False).input_ids + [self.eos] for o in options]
        out = [0.0] * len(options)
        by_len: dict[int, list[int]] = {}
        for i, c in enumerate(conts):
            by_len.setdefault(len(c), []).append(i)
        # Equal-length rows need no padding. shortcut: the prompt is recomputed per option; share its KV cache
        # when running the 77/151-option datasets at scale.
        for n, idx in by_len.items():
            for k in range(0, len(idx), self.option_batch):
                chunk = idx[k : k + self.option_batch]
                ids = torch.tensor([prompt + conts[i] for i in chunk], device=self.device)
                with torch.no_grad():
                    logits = self.model(input_ids=ids, logits_to_keep=n + 1).logits[:, :-1].float()
                lp = logits.log_softmax(-1).gather(-1, ids[:, -n:, None]).squeeze(-1).sum(-1)
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


def run(runner, dataset: str, split: str = "test", limit: int | None = None, batch_size: int = 8):
    """Predictions for one dataset split: columns dataset, id, probs, latency_ms."""
    ds = load(dataset)
    df = ds.df[ds.df.split == split].head(limit)
    probs, latency = [], []
    for k in range(0, len(df), batch_size):
        texts = df.text.iloc[k : k + batch_size].tolist()
        t0 = time.perf_counter()
        out = runner.predict(texts, ds.source.question, ds.options)
        # per-decision latency is batch wall time / batch size; p50/p99 of single calls needs batch_size=1
        latency += [(time.perf_counter() - t0) * 1000 / len(texts)] * len(texts)
        probs += out
    _check(probs, df.label.to_numpy())  # every row a valid distribution over this dataset's options
    return pd.DataFrame({"dataset": dataset, "id": df.id.to_numpy(), "probs": probs, "latency_ms": latency})


def write(out_dir: str | Path, preds: pd.DataFrame, manifest: dict, run_info: dict) -> Path:
    """Upsert one dataset's rows into predictions.parquet and record the run in manifest.json."""
    out = Path(out_dir)
    path, mpath = out / "predictions.parquet", out / "manifest.json"
    m = json.loads(mpath.read_text()) if mpath.exists() else {}
    if m and (m["runner"], m["model"], m["revision"]) != (manifest["runner"], manifest["model"], manifest["revision"]):
        raise ValueError(f"{out} holds predictions from {m['model']}@{m['revision']}; use another directory")
    (dataset,) = preds.dataset.unique()
    out.mkdir(parents=True, exist_ok=True)
    old = pd.read_parquet(path) if path.exists() else preds.iloc[:0]
    pd.concat([old[old.dataset != dataset], preds], ignore_index=True).to_parquet(path, index=False)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    runs = {**m.get("runs", {}), dataset: {**run_info, "n": len(preds), "date": now}}
    mpath.write_text(json.dumps({**m, **manifest, "runs": runs}, indent=2) + "\n")
    return path
