"""MCP server wiring (requires mcp)."""

import pytest

pytestmark = pytest.mark.requires_extras

pytest.importorskip("mcp", reason="mcp not installed")

from membrane.adapters.mcp_server import MembraneMCPServer, _resolve_profile
from membrane.compiler import ContextCompiler
from membrane.profile import MembraneProfile


def test_resolve_profile_minimal():
    p = _resolve_profile("minimal")
    assert isinstance(p, MembraneProfile)


def test_mcp_build_server():
    c = ContextCompiler(max_tokens=500, model_id="gpt-4o-mini")
    srv = MembraneMCPServer(c)
    assert srv.build_server() is not None


def test_tool_definitions():
    c = ContextCompiler(max_tokens=500, model_id="gpt-4o-mini")
    m = MembraneMCPServer(c)
    defs = m.as_tool_definitions()
    names = {d["name"] for d in defs}
    assert "membrane_compile" in names
    assert "membrane_compile_conversation" in names


def test_tool_definitions_no_conversation():
    c = ContextCompiler(max_tokens=500, model_id="gpt-4o-mini")
    m = MembraneMCPServer(c, enable_conversation_tool=False)
    names = {d["name"] for d in m.as_tool_definitions()}
    assert "membrane_compile_conversation" not in names
