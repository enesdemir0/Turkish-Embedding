"""Loads an already-published tokenizer instead of building one from scratch.

Used for exp001 (the exact paper reproduction): re-running SentencePiece training
over the Cosmos corpus ourselves would introduce its own source of divergence from
the paper (a different corpus snapshot, a different run producing different merge
rules) — that's a separate thing worth testing on its own later, not something to
bundle into the first reproduction. So exp001 points straight at the paper's own
published tokenizer, isolating the experiment to the parts actually being tested:
weight-preserving cloning + mean-composition init + cosine distillation.

Building a genuinely new, from-scratch tokenizer is `frequency_pruning.py` (future
work) — swapping between the two is a one-line config change, nothing else in the
pipeline needs to know which one is in use.

No actual loading happens here: `transformer_cloner`'s cloners take a tokenizer *id*
(HF repo id or local path) and resolve it themselves via `AutoTokenizer.
from_pretrained` internally, so this strategy just passes the id straight through.
"""

from __future__ import annotations

from src.tokenizer_surgery.base import TokenizerSurgeryStrategy
from src.tokenizer_surgery.registry import TOKENIZER_SURGERY_REGISTRY


@TOKENIZER_SURGERY_REGISTRY.register("reuse_pretrained")
class ReusePretrainedTokenizer(TokenizerSurgeryStrategy):
    """params:
        tokenizer_source: HF repo id (or local path) of the tokenizer to reuse.
    """

    def build(self) -> str:
        return self.params["tokenizer_source"]
