from types import SimpleNamespace

import pytest

from cbench.contamination import verbatim_run

torch = pytest.importorskip("torch")


class Memorised:
    """Predicts the memorised sequence as the next token, so the run ends where the text departs from it."""

    device = "cpu"

    def __init__(self, memory):
        self.memory = memory

    def __call__(self, input_ids):
        n = input_ids.shape[1]
        logits = torch.zeros(1, n, 100)
        for i in range(n):
            logits[0, i, self.memory[i + 1] if i + 1 < len(self.memory) else 0] = 1
        return SimpleNamespace(logits=logits)


def test_verbatim_run_counts_leading_greedy_matches():
    tok = lambda text: SimpleNamespace(input_ids=[int(w) for w in text.split()])  # noqa: E731
    model = Memorised(list(range(1, 11)))
    # 10 tokens, second half is 6..10; text matches memory for 6, 7, 8 then departs: run 3 of 5
    assert verbatim_run(model, tok, "1 2 3 4 5 6 7 8 99 10") == (3, 5)
    assert verbatim_run(model, tok, "1 2 3 4 5 6 7 8 9 10") == (5, 5)
    assert verbatim_run(model, tok, "1 2 3 4 5 50 7 8 9 10") == (0, 5)  # first-half match alone counts for nothing
