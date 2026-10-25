"""Features left out of the open-source edition fail with a clear error."""

from __future__ import annotations

import json

import pytest

from membrane import ContextCompiler, FeatureNotAvailable, MemoryItem
from membrane.profile import PRESET_NAMES, MembraneProfile


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


@pytest.mark.parametrize("name", ["research", "full", "enterprise", "bogus"])
def test_preset_rejects_unknown_names(name):
    with pytest.raises(ValueError, match="unknown profile"):
        MembraneProfile.preset(name)


def test_cli_and_mcp_offer_only_free_presets():
    from membrane.adapters.mcp_resources import list_profile_resource_text
    from membrane.cli.diagnostics import PROFILE_DESCRIPTIONS, resolve_profile_name

    assert set(PROFILE_DESCRIPTIONS) == set(PRESET_NAMES)
    assert json.loads(list_profile_resource_text())["presets"] == list(PRESET_NAMES)
    for name in PRESET_NAMES:
        assert isinstance(resolve_profile_name(name), MembraneProfile)
