"""Hotword boosting for the greedy RNN-T: NeMo's word boosting (GPU-PB).

This follows NeMo's GPU phrase-boosting tree in shallow fusion
(``nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py`` and
TurboBias, arXiv:2508.07014) as it runs in greedy label-looping decoding:

* The unbiased argmax decides blank or not. On a blank frame nothing changes;
  otherwise every non-blank token gets ``alpha`` times its tree score added to
  its logit and the argmax is taken again over non-blank tokens.
* A token that starts a phrase scores ``context_score``; the token at depth
  ``d >= 1`` of a match scores ``context_score * depth_scaling + ln(d + 1)``.
* A token that breaks a partial match takes the failure arc, whose score gives
  back what the match had collected, so a wrong start costs its own bonus. A
  match that completed a phrase backs off at no cost. A token outside every
  phrase scores ``unk_score`` (0).

One difference: NeMo builds its tree from the tokenizer's ids for each phrase.
Here matching runs on the lowercased text a token appends (``▁`` as a space),
so every tokenization and casing of a hotword is boosted, as NeMo's
``case_insensitive`` mode intends, without needing a SentencePiece encoder. A
phrase starts only at a word boundary, and a bare ``▁`` piece moves a match
along without adding depth or score.
"""

from __future__ import annotations

import math
from typing import Iterable, NamedTuple, Optional

import numpy as np

from . import tokenizer

# NeMo's recommended values for RNN-T (word boosting docs).
CONTEXT_SCORE = 1.0
DEPTH_SCALING = 2.0
# NeMo's boosting_tree_alpha defaults to 0 (off) and is tuned per dataset.
DEFAULT_HOTWORDS_ALPHA = 1.0


class _Match(NamedTuple):
    text: str  # the phrase prefix the emitted text ends with
    depth: int  # tokens it took, bare "▁" pieces excluded
    score: float  # tree score collected along the way, alpha included


_ROOT = _Match("", 0, 0.0)


def normalize_hotwords(hotwords: Optional[Iterable[str]]) -> list[str]:
    """Strip each hotword and drop empty ones."""
    return [str(t).strip() for t in hotwords or [] if t is not None and str(t).strip()]


class HotwordBooster:
    """Per-stream boosting state for one hotword list.

    ``choose`` picks the token to emit at a frame; ``accept`` advances the
    matches with the token that was emitted.
    """

    def __init__(
        self,
        hotwords: Iterable[str],
        vocabulary: list[str],
        blank_id: int,
        alpha: float = DEFAULT_HOTWORDS_ALPHA,
    ):
        self.alpha = float(alpha)
        self.blank_id = blank_id
        self._keys = set()
        for term in normalize_hotwords(hotwords):
            self._keys.add(" " + " ".join(term.lower().split()))
        self._prefixes = {key[:i] for key in self._keys for i in range(1, len(key) + 1)}
        self._piece_ids: dict[str, list[int]] = {}
        self._text_pieces: set[int] = set()
        for token, piece in enumerate(vocabulary):
            if token == blank_id or tokenizer.is_special_piece(piece):
                continue
            text = tokenizer.piece_to_text(piece).lower()
            if not text:
                continue
            self._piece_ids.setdefault(text, []).append(token)
            if text.strip():
                self._text_pieces.add(token)
        self._emittable = {t for ids in self._piece_ids.values() for t in ids}
        self._max_piece = max((len(text) for text in self._piece_ids), default=0)
        self._extensions_cache: dict[str, list[tuple[int, str]]] = {}
        # Matches under way, longest first: NeMo's failure chain. The root is
        # implicit. The start of a transcript is a word boundary: a piece with no
        # leading "▁" still starts a word there.
        self._chain = [_Match(" ", 0, 0.0)]
        self._scores: Optional[tuple[float, dict[int, float]]] = None

    @property
    def is_empty(self) -> bool:
        return not self._keys

    def _arc(self, depth: int) -> float:
        if depth == 0:
            return self.alpha * CONTEXT_SCORE
        return self.alpha * (CONTEXT_SCORE * DEPTH_SCALING + math.log(depth + 1))

    def _extensions(self, text: str) -> list[tuple[int, str]]:
        """Tokens whose text extends ``text`` to a phrase prefix, with that prefix."""
        cached = self._extensions_cache.get(text)
        if cached is not None:
            return cached
        found: dict[int, str] = {}
        for key in self._keys:
            if len(key) <= len(text) or not key.startswith(text):
                continue
            rest = key[len(text) :]
            for end in range(1, min(len(rest), self._max_piece) + 1):
                for token in self._piece_ids.get(rest[:end], ()):
                    found.setdefault(token, text + rest[:end])
        result = list(found.items())
        self._extensions_cache[text] = result
        return result

    def _step(self, match: _Match, token: int, text: str) -> _Match:
        if token not in self._text_pieces:
            return _Match(text, match.depth, match.score)
        return _Match(text, match.depth + 1, match.score + self._arc(match.depth))

    def scores(self) -> tuple[float, dict[int, float]]:
        """The score of every token outside the matches, and of each token in one."""
        if self._scores is not None:
            return self._scores
        chain = self._chain + [_ROOT]
        token_scores: dict[int, float] = {}
        backoff = 0.0
        for index, match in enumerate(chain):
            for token, text in self._extensions(match.text):
                if token not in token_scores:
                    token_scores[token] = (
                        backoff + self._step(match, token, text).score - match.score
                    )
            if index + 1 < len(chain) and match.text not in self._keys:
                backoff += chain[index + 1].score - match.score
        self._scores = (backoff, token_scores)
        return self._scores

    def choose(self, logits: np.ndarray, greedy: int) -> int:
        """The token to emit instead of ``greedy``, the unbiased argmax of ``logits``."""
        if greedy == self.blank_id:
            return greedy
        unmatched, token_scores = self.scores()
        biased = np.asarray(logits, dtype=np.float32).reshape(-1) + unmatched
        for token, score in token_scores.items():
            biased[token] += score - unmatched
        biased[self.blank_id] = -np.inf
        return int(np.argmax(biased))

    def accept(self, token: int) -> None:
        """Advance the matches with an emitted token. Special tokens, such as a
        language tag, leave them as they are."""
        if token not in self._emittable:
            return
        chain = []
        for match in self._chain + [_ROOT]:
            for candidate, text in self._extensions(match.text):
                if candidate == token:
                    chain.append(self._step(match, token, text))
        chain.sort(key=lambda m: len(m.text), reverse=True)
        self._chain = chain
        self._scores = None


def make_booster(
    model, hotwords: Optional[Iterable[str]], alpha: float
) -> Optional[HotwordBooster]:
    """A booster for ``model``, or None when no hotword survives normalization."""
    booster = HotwordBooster(hotwords or [], model.vocabulary, model.blank_id, alpha)
    return None if booster.is_empty else booster
