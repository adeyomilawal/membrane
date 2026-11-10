import builtins
import logging
import sys

import pytest

from membrane import lod


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(lod, "_embedding_model", None)
    monkeypatch.setattr(lod, "_numpy_warned", False)


@pytest.fixture
def no_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)


@pytest.fixture
def needs_numpy():
    return pytest.importorskip("numpy")


class Vec(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.seen = None

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("boom")
        self.seen = texts
        for t in texts:
            yield Vec([1.0, 0.0] if "cat" in t else [0.0, 1.0])


def test_get_embedding_model_failure_cached(monkeypatch, caplog):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("fastembed"):
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.WARNING):
        assert lod._get_embedding_model() is False
    assert "fastembed not available" in caplog.text
    assert lod._get_embedding_model() is False


def test_get_embedding_model_returns_cached(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(lod, "_embedding_model", sentinel)
    assert lod._get_embedding_model() is sentinel


def test_without_numpy_everything_degrades(no_numpy, caplog):
    with caplog.at_level(logging.WARNING):
        assert lod.embed_texts(["a"]) is None
        assert lod.embed_texts(["a"]) is None
    assert caplog.text.count("numpy not installed") == 1
    assert lod.cosine_distance([1, 0], [1, 0]) == 1.0
    assert lod.compute_attention_shift_with_embeddings([[1, 0]], [0, 1]) is False
    assert lod.compute_attention_shift(["a"], "b") is False


@pytest.mark.requires_extras
def test_embed_texts_uses_model(monkeypatch, needs_numpy):
    m = FakeModel()
    monkeypatch.setattr(lod, "_embedding_model", m)
    out = lod.embed_texts(["a cat", "  "])
    assert out.shape == (2, 2)
    assert m.seen == ["a cat", " "]


@pytest.mark.requires_extras
def test_embed_texts_none_cases(monkeypatch, needs_numpy):
    monkeypatch.setattr(lod, "_embedding_model", False)
    assert lod.embed_texts(["a"]) is None
    monkeypatch.setattr(lod, "_embedding_model", FakeModel())
    assert lod.embed_texts([]) is None
    monkeypatch.setattr(lod, "_embedding_model", FakeModel(fail=True))
    assert lod.embed_texts(["a"]) is None


@pytest.mark.requires_extras
def test_cosine_distance(needs_numpy):
    assert lod.cosine_distance([1, 0], [1, 0]) == pytest.approx(0.0)
    assert lod.cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert lod.cosine_distance([1, 0], [-1, 0]) == pytest.approx(2.0)
    assert lod.cosine_distance([1, 2], [2, 4]) == pytest.approx(0.0)
    assert lod.cosine_distance([1, 0], [1, 0, 0]) == 1.0
    assert lod.cosine_distance([], []) == 1.0
    assert lod.cosine_distance([0, 0], [1, 1]) == 1.0
    assert lod.cosine_distance([[1, 0]], [1, 0]) == pytest.approx(0.0)


@pytest.mark.requires_extras
def test_shift_with_embeddings(needs_numpy):
    f = lod.compute_attention_shift_with_embeddings
    assert f([[1, 0], [1, 0]], [0, 1]) is True
    assert f([[1, 0], [1, 0]], [1, 0.1]) is False
    assert f([], [1, 0]) is False
    assert f([[1, 0]], [0, 1], threshold=1.5) is False
    assert f([[1, 0]], [0, 1], threshold=0.5) is True


@pytest.mark.requires_extras
def test_shift_with_embeddings_uses_last_three(monkeypatch, needs_numpy):
    old = [[0, 1]] * 10
    recent = [[1, 0]] * 3
    assert lod.compute_attention_shift_with_embeddings(old + recent, [1, 0]) is False


@pytest.mark.requires_extras
def test_shift_default_threshold_env(monkeypatch, needs_numpy):
    monkeypatch.setenv("LOD_ATTENTION_SHIFT_COSINE_THRESHOLD", "0.9")
    # distance of 1 - cos(45deg) ~= 0.29, below 0.9
    assert lod.compute_attention_shift_with_embeddings([[1, 0]], [1, 1]) is False
    monkeypatch.setenv("LOD_ATTENTION_SHIFT_COSINE_THRESHOLD", "0.1")
    assert lod.compute_attention_shift_with_embeddings([[1, 0]], [1, 1]) is True


@pytest.mark.requires_extras
def test_compute_attention_shift(monkeypatch, needs_numpy):
    monkeypatch.setattr(lod, "_embedding_model", FakeModel())
    assert lod.compute_attention_shift(["cat a", "cat b"], "stocks") is True
    assert lod.compute_attention_shift(["cat a", "cat b"], "cat c") is False
    assert lod.compute_attention_shift([], "cat") is False
    monkeypatch.setattr(lod, "_embedding_model", False)
    assert lod.compute_attention_shift(["cat"], "dog") is False
