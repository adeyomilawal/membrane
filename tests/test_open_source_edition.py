"""Features left out of the open-source edition fail with a clear error."""

from __future__ import annotations

import pytest

from membrane import ContextCompiler, FeatureNotAvailable, MemoryItem
from membrane.profile import MembraneProfile


def _compile(profile=None):
    compiler = ContextCompiler(max_tokens=4096, model_id="gpt-4o-mini")
    return compiler.compile(
        memories=[MemoryItem(id="m1", content="Fact.", memory_type="semantic")],
        user_message="Hi",
        profile=profile,
    )


def test_enable_lineage_raises_feature_not_available():
    with pytest.raises(FeatureNotAvailable, match="open-source edition"):
        _compile(MembraneProfile(enable_lineage=True))


def test_compile_result_diff_raises_feature_not_available():
    result = _compile()
    with pytest.raises(FeatureNotAvailable):
        result.diff(result)


def test_feature_not_available_is_an_import_error():
    assert issubclass(FeatureNotAvailable, ImportError)


@pytest.mark.parametrize("name", ["minimal", "coding", "conversational"])
def test_free_profiles_compile(name):
    result = _compile(getattr(MembraneProfile, name)())
    assert result.messages
