"""Hotword boosting for Nemotron ASR's greedy RNN-T (NeMo's word boosting).

The booster tests run on plain logits over a toy vocabulary; the decode tests
script the joint of the tiny model, so nothing here needs weights.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.nemotron_asr import Model, ModelConfig
from mlx_audio.stt.models.nemotron_asr.hotwords import HotwordBooster, make_booster
from mlx_audio.stt.models.nemotron_asr.rnnt import GreedyDecoderState

VOCAB = ["<unk>", "<en-US>", "▁hello", "▁cl", "aude", "▁Cla", "ude", "▁", "cl", "▁code"]
BLANK = len(VOCAB)
HELLO, CL, AUDE, CLA_UPPER, UDE, SPACE, CL_MID, CODE = 2, 3, 4, 5, 6, 7, 8, 9
LANG = 1


def booster(terms=("Claude",), alpha=1.0, **kwargs):
    return HotwordBooster(list(terms), VOCAB, BLANK, alpha=alpha, **kwargs)


def logits(**values):
    row = np.full(BLANK + 1, -10.0, dtype=np.float32)
    for token, value in values.items():
        row[globals()[token.upper()] if token != "blank" else BLANK] = value
    return row


def score(b, token):
    unmatched, scores = b.scores()
    return scores.get(token, unmatched)


def test_a_phrase_start_takes_context_score_times_alpha():
    b = booster(alpha=1.5)
    assert score(b, CL) == pytest.approx(1.5)
    assert score(b, HELLO) == 0.0
    # The start within reach of the argmax wins; one further away does not.
    assert b.choose(logits(hello=2.0, cl=0.6), HELLO) == CL
    assert b.choose(logits(hello=2.0, cl=0.4), HELLO) == HELLO


def test_deeper_tokens_follow_nemo_depth_scaling():
    b = booster(["claude code"], alpha=2.0)
    b.accept(CL)
    assert score(b, AUDE) == pytest.approx(2.0 * (2.0 + math.log(2)))
    b.accept(AUDE)
    assert score(b, CODE) == pytest.approx(2.0 * (2.0 + math.log(3)))


def test_breaking_a_partial_match_gives_its_score_back():
    b = booster(["claude code"])
    b.accept(CL)
    assert score(b, HELLO) == pytest.approx(-1.0)
    b.accept(AUDE)
    collected = 1.0 + 2.0 + math.log(2)
    assert score(b, HELLO) == pytest.approx(-collected)
    b.accept(HELLO)
    assert b.scores() == (0.0, {CL: 1.0, CLA_UPPER: 1.0, SPACE: 0.0})


def test_a_completed_phrase_backs_off_at_no_cost():
    b = booster(["claude", "claude code"])
    b.accept(CL)
    b.accept(AUDE)
    assert score(b, HELLO) == 0.0
    assert score(b, CODE) == pytest.approx(2.0 + math.log(3))


def test_any_tokenization_and_casing_matches():
    b = booster(["CLAUDE"])
    assert score(b, CLA_UPPER) == pytest.approx(1.0)
    b.accept(CLA_UPPER)
    assert score(b, UDE) == pytest.approx(2.0 + math.log(2))


def test_a_phrase_starts_only_at_a_word_boundary():
    b = booster()
    # At the start of a transcript a piece without "▁" still starts a word.
    assert score(b, CL_MID) == pytest.approx(1.0)
    b.accept(HELLO)
    assert score(b, CL_MID) == 0.0
    # A bare "▁" moves the match along without depth or score of its own.
    assert score(b, SPACE) == 0.0
    b.accept(SPACE)
    assert score(b, CL_MID) == pytest.approx(1.0)


def test_blank_frames_and_blank_are_never_boosted():
    b = booster(alpha=100.0)
    assert b.choose(logits(blank=5.0, cl=4.9), BLANK) == BLANK
    assert b.choose(logits(blank=5.0, hello=5.1), HELLO) == CL


def test_a_language_tag_leaves_the_matches_alone():
    b = booster()
    b.accept(CL)
    before = b.scores()
    b.accept(LANG)
    assert b.scores() == before


def test_an_empty_list_turns_boosting_off():
    model = Model(ModelConfig.from_dict(_tiny_config()))
    assert make_booster(model, None, 1.0) is None
    assert make_booster(model, ["", "  ", None], 1.0) is None
    assert make_booster(model, [" Claude "], 1.0) is not None


def _tiny_config():
    from mlx_audio.stt.tests.test_nemotron_asr import _tiny_config as base

    config = base()
    config["vocabulary"] = VOCAB
    config["decoder"]["vocab_size"] = len(VOCAB)
    config["joint"]["num_classes"] = len(VOCAB)
    return config


def _scripted(rows):
    """A tiny model whose joint returns ``rows`` in order."""
    model = Model(ModelConfig.from_dict(_tiny_config()))
    rows = iter(rows)

    def predictor(token, hidden):
        return mx.zeros((1, 1, 16)), (mx.zeros((2, 1, 16)), mx.zeros((2, 1, 16)))

    model.decoder = predictor
    model.joint = lambda feature, prediction: mx.array(next(rows))
    return model


# "hello" narrowly beats the term's first piece, then the term's second piece
# narrowly loses to "hello" again; each frame emits one token, then a blank.
FRAMES = [logits(hello=2.0, cl=1.5), logits(blank=1.0)] + [
    logits(hello=2.0, aude=0.5),
    logits(blank=1.0),
]


@pytest.mark.parametrize(
    "hotwords, expected", [(None, [HELLO, HELLO]), (["Claude"], [CL, AUDE])]
)
def test_greedy_decode_emits_the_boosted_tokens(hotwords, expected):
    model = _scripted(FRAMES)
    decoder = GreedyDecoderState(model, make_booster(model, hotwords, 1.0))
    tokens = decoder.decode(mx.zeros((1, 2, 32)), 0)
    assert [t.id for t in tokens] == expected


def test_streaming_session_boosts_and_reset_starts_over():
    model = _scripted(FRAMES * 2)
    session = model.create_streaming_session(hotwords=["Claude"])
    for _ in range(2):
        session._encoded.append(mx.zeros((1, 2, 32)))
        session._flushed = True
        parts = []
        while not session.done:
            parts.extend(session.step())
        assert "".join(parts) == "claude"
        session.reset()
