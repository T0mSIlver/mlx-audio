"""Hotword boosting for the greedy RNN-T: NeMo's word boosting (GPU-PB).

This follows NeMo's GPU phrase-boosting tree in shallow fusion
(``nemo/collections/asr/parts/context_biasing/boosting_graph_batched.py`` and
TurboBias, arXiv:2508.07014) as it runs in greedy label-looping decoding:

* The unbiased argmax decides blank or not. On a blank frame nothing changes;
  otherwise every non-blank token gets ``alpha`` times its tree score added to
  its logit and the argmax is taken again over non-blank tokens. A frame whose
  argmax is a special token, such as a language tag, is left alone too, and
  special tokens never take a frame from a text token.
* A token that starts a phrase scores ``context_score``; the token at depth
  ``d >= 1`` of a match scores ``context_score * depth_scaling + ln(d + 1)``.
* A token that breaks a partial match takes the failure arc, whose score gives
  back what the match had collected, so a wrong start costs its own bonus. A
  match that completed a phrase backs off at no cost. A token outside every
  phrase scores ``unk_score`` (0).

Where this differs: NeMo builds its tree from one tokenization of each phrase.
There is no SentencePiece encoder here, so matching runs on the NFKC-normalized,
case-folded text each token appends (``▁`` as a space). Every tokenization and
casing of a hotword is boosted, and depth counts the tokens actually emitted,
so a spelling in more pieces collects more score than NeMo's single path would.
A phrase starts only at a word boundary (the start of the transcript counts), a
bare ``▁`` moves a match along without depth or score, and special tokens leave
the matches alone.
"""

from __future__ import annotations

import math
import unicodedata
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


def fold(text: str) -> str:
    """Lowercase one character at a time, with final sigma folded to sigma.

    A hotword and the pieces that spell it must fold the same way, so casing
    cannot depend on context: ``"ΟΣ".lower()`` ends in "ς" but the piece "Σ"
    lowers to "σ".
    """
    return "".join(c.lower() for c in text).replace("ς", "σ")


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
            words = fold(unicodedata.normalize("NFKC", term)).split()
            self._keys.add(" " + " ".join(words))
        self._piece_ids: dict[str, list[int]] = {}
        self._text_pieces: set[int] = set()
        for token, piece in enumerate(vocabulary):
            if token == blank_id or tokenizer.is_special_piece(piece):
                continue
            text = fold(tokenizer.piece_to_text(piece))
            if not text:
                continue
            self._piece_ids.setdefault(text, []).append(token)
            if text.strip(" "):
                self._text_pieces.add(token)
        self._emittable = {t for ids in self._piece_ids.values() for t in ids}
        # Blank and special tokens never take a frame from a text token.
        self._excluded = np.ones(len(vocabulary) + 1, dtype=bool)
        self._excluded[list(self._emittable)] = False
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
        if greedy not in self._emittable:
            return greedy
        unmatched, token_scores = self.scores()
        logits = np.asarray(logits, dtype=np.float32).reshape(-1)
        biased = logits + np.float32(unmatched)
        for token, score in token_scores.items():
            biased[token] = logits[token] + np.float32(score)
        biased[self._excluded[: biased.size]] = -np.inf
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
