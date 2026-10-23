"""Bounded JSON report choices for native prefix-constrained generation.

Constraints fix wire syntax and request identity, never the visual verdict. The
returned report remains an untrusted model assessment after strict parsing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product

from video_intelligence.reasoning.semantic import (
    SCHEMA_VERSION,
    SemanticReport,
    SemanticRequest,
    SemanticVerdict,
)

GENERATION_POLICY = "semantic_finite_json.v1"
MAX_CONSTRAINED_UNITS = 2
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 256


def _ids(values, *, nonempty=True):
    if (not isinstance(values, (list, tuple)) or nonempty and not values
            or any(type(value) is not int or value < 0 for value in values)):
        raise ValueError("token IDs must be nonnegative integers")
    return tuple(values)


@dataclass(frozen=True, slots=True, init=False)
class ReportChoices:
    """All legal verdict combinations for one complete, small bound request."""

    request_sha256: str
    wires: tuple[str, ...]

    def __init__(self, *args, **kwargs):
        raise TypeError("use ReportChoices.from_request")

    @classmethod
    def from_request(cls, request: SemanticRequest) -> ReportChoices:
        source = request.to_dict()
        units = source["units"]
        if not 1 <= len(units) <= MAX_CONSTRAINED_UNITS:
            raise ValueError("constrained generation requires one or two complete units")
        wires = []
        for verdicts in product(SemanticVerdict, repeat=len(units)):
            body = {
                "schema_version": SCHEMA_VERSION,
                "request_sha256": request.sha256,
                "backend_sha256": source["backend_sha256"],
                "assessments": [{"unit_id": unit["unit_id"], "verdict": verdict.value}
                                for unit, verdict in zip(units, verdicts, strict=True)],
            }
            wire = json.dumps(body, separators=(",", ":"), allow_nan=False)
            SemanticReport.from_json(request, wire)
            wires.append(wire)
        choices = object.__new__(cls)
        object.__setattr__(choices, "request_sha256", request.sha256)
        object.__setattr__(choices, "wires", tuple(wires))
        return choices

    def constrain(self, tokenizer, prompt_ids, *, eos_token_id: int,
                  max_new_tokens: int = MAX_NEW_TOKENS) -> ReportPrefix:
        """Build the Transformers prefix callback after CPU tokenization.

        Fail before generation if any legal verdict cannot fit, or the tokenizer
        cannot round-trip it exactly. No verdict is removed to fit the budget.
        """
        prompt = _ids(prompt_ids)
        if (len(prompt) > MAX_INPUT_TOKENS or type(eos_token_id) is not int
                or eos_token_id < 0 or type(max_new_tokens) is not int
                or not 1 <= max_new_tokens <= MAX_NEW_TOKENS):
            raise ValueError("constrained generation token budget exceeded")
        trie = {}
        longest = 0
        for wire in self.wires:
            ids = _ids(tokenizer.encode(wire, add_special_tokens=False))
            if eos_token_id in ids or len(ids) + 1 > max_new_tokens:
                raise ValueError("a complete verdict exceeds generation budget")
            decoded = tokenizer.decode(list(ids), skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
            if decoded != wire:
                raise ValueError("tokenizer changed the constrained JSON report")
            longest = max(longest, len(ids) + 1)
            node = trie
            for token in (*ids, eos_token_id):
                node = node.setdefault(token, {})
        return ReportPrefix(prompt, trie, longest, eos_token_id)


class ReportPrefix:
    """Single-batch prefix callback; unknown prefixes never fall back to full vocab."""

    def __init__(self, prompt, trie, longest, eos_token_id):
        self._prompt = prompt
        self._trie = trie
        self.max_sequence_tokens = longest
        self.eos_token_id = eos_token_id

    def __call__(self, batch_id, input_ids):
        if type(batch_id) is not int or batch_id != 0:
            raise ValueError("constrained generation supports batch size one only")
        sequence = _ids(input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids)
        if sequence[:len(self._prompt)] != self._prompt:
            raise ValueError("generation prompt tokens differ")
        node = self._trie
        for token in sequence[len(self._prompt):]:
            if token not in node:
                raise ValueError("generation left the bound report grammar")
            node = node[token]
        if not node:
            raise ValueError("generation continued after a complete report and EOS")
        return sorted(node)

    def generation_kwargs(self):
        """Native transport must apply these together, without overriding them."""
        return {"prefix_allowed_tokens_fn": self, "num_beams": 1,
                "num_return_sequences": 1, "do_sample": False,
                "max_new_tokens": self.max_sequence_tokens,
                "eos_token_id": self.eos_token_id, "forced_eos_token_id": None,
                "forced_bos_token_id": None, "min_length": 0, "min_new_tokens": 0,
                "stop_strings": None, "remove_invalid_values": False,
                "logits_processor": [RejectInvalidScores()]}


class RejectInvalidScores:
    """Fail after grammar masking on NaN/+inf or no finite legal continuation.

    Negative infinity is a normal mask value. This hook follows Transformers'
    built-in processors and never repairs invalid scores or chooses a verdict.
    """

    def __call__(self, input_ids, scores):
        if (scores.ndim != 2 or scores.shape[0] != 1 or scores.shape[1] == 0
                or scores.isnan().any().item() or scores.isposinf().any().item()
                or not scores.isfinite().any(dim=-1).all().item()):
            raise ValueError("semantic generation has invalid or exhausted logits")
        return scores
