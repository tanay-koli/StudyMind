"""StudyMind — the LangGraph agent: routing, corrective retrieval, self-check, persistence (offline)."""

import pytest

from core import agent


@pytest.mark.parametrize("msg,topics,hint,expected", [
    ("quiz me on attention", None, None, ("quiz", [])),
    ("Can you test me?", None, None, ("quiz", [])),
    ("I want to review today", None, None, ("review", [])),
    ("what links these?", ["Biology", "Chemistry"], None, ("connect", ["Biology", "Chemistry"])),
    ("What is the connection between attention and memory?", None, None, ("connect", ["attention", "memory"])),
    ("connect enzymes with catalysts", None, None, ("connect", ["enzymes", "catalysts"])),
    ("Explain backprop", None, None, ("explain", [])),
    ("how does dropout work", None, None, ("explain", [])),
    ("list the layers in the encoder", None, None, ("qa", [])),
    ("Explain backprop", None, "qa", ("qa", [])),            # UI hint wins
    ("what links these?", ["Biology"], None, ("qa", [])),     # one topic is not a connect
])
def test_classify(msg, topics, hint, expected):
    assert agent.classify(msg, topics, hint) == expected
