"""Coverage tests for membrane.adapters.mcp_server.

Handlers are exercised directly; only tests that build the low-level ``mcp`` Server or
use ``mcp.types`` need the optional extra and are marked ``requires_extras``.
"""

import json
import logging
import sys
from types import SimpleNamespace

import pytest

from membrane.adapters import mcp_server as ms
from membrane.adapters.mcp_server import MembraneMCPServer
from membrane.memory_protocol import MemoryItem
from membrane.profile import MembraneProfile


class FakeResult:
    def __init__(self, tag="r", token_count=7):
        self.tag = tag
        self.metadata = {"token_count": token_count}

    def to_debug_dict(self):
        return {"tag": self.tag}


class FakeCompiler:
    def __init__(self, result=None):
        self.model_id = "orig-model"
        self.max_tokens = 111
        self.calls = []
        self.result = result if result is not None else FakeResult()

    def compile(self, *, memories, user_message, profile):
        # state is overridden only for the duration of the call
        self.calls.append(
            {
                "memories": memories,
                "user_message": user_message,
                "profile": profile,
                "model_id": self.model_id,
                "max_tokens": self.max_tokens,
            }
        )
        return self.result


class FakeConversationManager:
    def __init__(self, compiler):
        self.compiler = compiler
        self.turns = []
        self.compiled = []

    def add_turn(self, role, text):
        self.turns.append((role, text))

    def compile(self, *, memories, user_message, profile):
        self.compiled.append((memories, user_message, profile))
        return FakeResult("conv", token_count=3)


# ---------------------------------------------------------------- pure helpers


async def test_maybe_await():
    async def coro():
        return 5

    assert await ms._maybe_await(coro()) == 5
    assert await ms._maybe_await(6) == 6


def test_resolve_profile_unknown():
    with pytest.raises(ValueError, match="unknown profile"):
        ms._resolve_profile("nope")


def test_pick_profile_precedence():
    default = MembraneProfile.minimal()
    s = MembraneMCPServer(FakeCompiler(), profile=default)
    prof, name = s._pick_profile({})
    assert prof is default and name == "default"
    prof, name = s._pick_profile({"profile": "conversational"})
    assert name == "conversational" and prof.freshness is not None

    s2 = MembraneMCPServer(FakeCompiler())
    prof, name = s2._pick_profile({})
    assert name == "coding" and prof.policies is not None
    with pytest.raises(ValueError):
        s2._pick_profile({"profile": "bogus"})


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "[::1]", "LOCALHOST"])
def test_warn_if_exposed_loopback_silent(host, caplog):
    s = MembraneMCPServer(FakeCompiler(), host=host)
    with caplog.at_level(logging.WARNING, logger="membrane.adapters.mcp_server"):
        s._warn_if_exposed()
    assert caplog.text == ""


def test_warn_if_exposed_public(caplog):
    s = MembraneMCPServer(FakeCompiler(), host="0.0.0.0")
    with caplog.at_level(logging.WARNING, logger="membrane.adapters.mcp_server"):
        s._warn_if_exposed()
    assert "non-loopback" in caplog.text


def test_tool_definitions_shape():
    defs = {d["name"]: d for d in MembraneMCPServer(FakeCompiler()).as_tool_definitions()}
    assert set(defs) == {"membrane_compile", "membrane_inspect", "membrane_compile_conversation"}
    assert defs["membrane_compile"]["inputSchema"]["required"] == ["query"]
    assert defs["membrane_compile_conversation"]["inputSchema"]["required"] == ["user_message"]
    enum = defs["membrane_compile"]["inputSchema"]["properties"]["profile"]["enum"]
    assert enum == ["minimal", "coding", "conversational"]


def test_conversation_manager_explicit_and_lazy():
    cm = object()
    explicit = MembraneMCPServer(FakeCompiler(), conversation_manager=cm)
    assert explicit._get_conversation_manager() is cm

    from membrane.compiler import ContextCompiler
    from membrane.conversation.manager import ConversationManager

    s = MembraneMCPServer(ContextCompiler(max_tokens=500, model_id="gpt-4o"))
    lazy = s._get_conversation_manager()
    assert isinstance(lazy, ConversationManager)
    assert s._get_conversation_manager() is lazy


# ---------------------------------------------------------------- memory fetch


async def test_fetch_memories_empty_and_placeholder():
    s = MembraneMCPServer(FakeCompiler())
    assert await s._fetch_memories([]) == []
    got = await s._fetch_memories(["u1", "u2"])
    assert [m.id for m in got] == ["u1", "u2"]
    assert "no memory_provider" in got[0].content and "u1" in got[0].content


async def test_fetch_memories_sync_and_async_provider():
    seen = []

    def sync_provider(uris):
        seen.append(list(uris))
        return (MemoryItem(id=u, content="s") for u in uris)

    async def async_provider(uris):
        return [MemoryItem(id=u, content="a") for u in uris]

    got = await MembraneMCPServer(FakeCompiler(), memory_provider=sync_provider)._fetch_memories(
        ("x",)
    )
    assert [m.content for m in got] == ["s"] and seen == [["x"]]
    got = await MembraneMCPServer(FakeCompiler(), memory_provider=async_provider)._fetch_memories(
        ["y"]
    )
    assert [(m.id, m.content) for m in got] == [("y", "a")]


# ---------------------------------------------------------------- compile tool


async def test_compile_tool_happy_path_restores_state():
    comp = FakeCompiler()
    s = MembraneMCPServer(comp)
    out = await s._compile_tool(
        {
            "query": "hi",
            "memory_uris": ["m1"],
            "model": "gpt-4o-mini",
            "max_tokens": "256",
            "profile": "minimal",
        }
    )
    assert json.loads(out) == {"tag": "r"}
    call = comp.calls[0]
    assert call["user_message"] == "hi"
    assert call["model_id"] == "gpt-4o-mini" and call["max_tokens"] == 256
    assert [m.id for m in call["memories"]] == ["m1"]
    assert isinstance(call["profile"], MembraneProfile)
    # compiler state restored
    assert comp.model_id == "orig-model" and comp.max_tokens == 111
    assert s._metrics_snapshot == {"last_profile": "minimal", "memory_count": 1, "token_hint": 7}
    assert s._last_result is comp.result


async def test_compile_tool_defaults_and_missing_query():
    comp = FakeCompiler()
    s = MembraneMCPServer(comp)
    await s._compile_tool({})
    call = comp.calls[0]
    assert call["user_message"] == ""
    assert call["model_id"] == "gpt-4o" and call["max_tokens"] == 8192
    assert call["memories"] == []
    assert s._metrics_snapshot["last_profile"] == "coding"


async def test_compile_tool_result_without_debug_dict():
    comp = FakeCompiler(result=SimpleNamespace(metadata=None))
    s = MembraneMCPServer(comp)
    assert json.loads(await s._compile_tool({"query": "q"})) == {"ok": True}
    assert s._metrics_snapshot["token_hint"] is None


async def test_compile_tool_restores_state_on_error():
    class Boom(FakeCompiler):
        def compile(self, **kw):
            raise RuntimeError("fail")

    comp = Boom()
    s = MembraneMCPServer(comp)
    with pytest.raises(RuntimeError, match="fail"):
        await s._compile_tool({"query": "q", "model": "m", "max_tokens": 5})
    assert comp.model_id == "orig-model" and comp.max_tokens == 111
    assert s._last_result is None


async def test_compile_tool_requires_compile_method():
    s = MembraneMCPServer(SimpleNamespace())
    with pytest.raises(RuntimeError, match="no compile"):
        await s._compile_tool({"query": "q"})


async def test_compile_tool_invalid_max_tokens():
    s = MembraneMCPServer(FakeCompiler())
    with pytest.raises(ValueError):
        await s._compile_tool({"query": "q", "max_tokens": "lots"})


async def test_compile_tool_with_real_compiler():
    from membrane.compiler import ContextCompiler

    comp = ContextCompiler(max_tokens=2000, model_id="gpt-4o")
    s = MembraneMCPServer(comp)
    out = json.loads(await s._compile_tool({"query": "hello", "profile": "minimal"}))
    assert "messages" in out
    assert comp.max_tokens == 2000 and comp.model_id == "gpt-4o"


# ---------------------------------------------------------------- conversation tool


async def test_compile_conversation_tool():
    comp = FakeCompiler()
    cm = FakeConversationManager(comp)
    s = MembraneMCPServer(comp, conversation_manager=cm, profile=MembraneProfile.minimal())
    out = await s._compile_conversation_tool(
        {"user_message": "yo", "memory_uris": ["a"], "model": "m", "max_tokens": 9}
    )
    assert json.loads(out) == {"tag": "conv"}
    assert cm.turns == [("user", "yo")]
    mems, msg, _prof = cm.compiled[0]
    assert msg == "yo" and [m.id for m in mems] == ["a"]
    assert s._metrics_snapshot == {
        "last_profile": "default",
        "memory_count": 1,
        "conversation_tool": True,
        "token_hint": 3,
    }
    assert comp.model_id == "orig-model" and comp.max_tokens == 111


async def test_compile_conversation_tool_non_debug_result():
    comp = FakeCompiler()
    cm = FakeConversationManager(comp)
    cm.compile = lambda **kw: SimpleNamespace(metadata={})  # type: ignore[method-assign]
    s = MembraneMCPServer(comp, conversation_manager=cm)
    assert json.loads(await s._compile_conversation_tool({})) == {"ok": True}
    assert cm.turns == [("user", "")]


async def test_compile_conversation_tool_restores_on_error():
    comp = FakeCompiler()
    cm = FakeConversationManager(comp)

    def boom(**kw):
        raise RuntimeError("x")

    cm.compile = boom  # type: ignore[method-assign]
    s = MembraneMCPServer(comp, conversation_manager=cm)
    with pytest.raises(RuntimeError):
        await s._compile_conversation_tool({"user_message": "u", "model": "zz"})
    assert comp.model_id == "orig-model"


# ---------------------------------------------------------------- mcp-dependent


def test_require_mcp_reports_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp.server.stdio", None)
    with pytest.raises(ImportError, match="membrane-aol\\[mcp\\]"):
        ms._require_mcp()


async def test_serve_unsupported_transport():
    with pytest.raises(ValueError, match="Unsupported transport: ws"):
        await MembraneMCPServer(FakeCompiler()).serve(transport="ws")


async def test_serve_dispatch(monkeypatch):
    s = MembraneMCPServer(FakeCompiler())
    called = []

    async def fake_stdio():
        called.append("stdio")

    async def fake_sse():
        called.append("sse")

    monkeypatch.setattr(s, "_serve_stdio", fake_stdio)
    monkeypatch.setattr(s, "_serve_sse", fake_sse)
    await s.serve()
    await s.serve(transport="sse")
    assert called == ["stdio", "sse"]


@pytest.mark.requires_extras
async def test_serve_sse_without_uvicorn(monkeypatch, caplog):
    pytest.importorskip("mcp")
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    s = MembraneMCPServer(FakeCompiler(), host="0.0.0.0")
    with (
        caplog.at_level(logging.WARNING, logger="membrane.adapters.mcp_server"),
        pytest.raises(ImportError, match="uvicorn"),
    ):
        await s.serve(transport="sse")
    assert "non-loopback" in caplog.text


@pytest.mark.requires_extras
async def test_registered_handlers_tools_and_resources():
    pytest.importorskip("mcp")
    import mcp.types as types

    comp = FakeCompiler()
    s = MembraneMCPServer(comp, profile=MembraneProfile.minimal())
    server = s.build_server()
    handlers = server.request_handlers

    list_tools = await handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")
    )
    assert {t.name for t in list_tools.root.tools} == {
        "membrane_compile",
        "membrane_inspect",
        "membrane_compile_conversation",
    }

    def call(name, args):
        req = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments=args),
        )
        return handlers[types.CallToolRequest](req)

    # inspect before any compile
    res = await call("membrane_inspect", {})
    assert not res.root.isError
    assert json.loads(res.root.content[0].text) == {"error": "no_compilation_yet"}

    res = await call("membrane_compile", {"query": "hello"})
    assert json.loads(res.root.content[0].text) == {"tag": "r"}
    res = await call("membrane_inspect", {})
    assert json.loads(res.root.content[0].text) == {"tag": "r"}

    # unknown tool surfaces as an error result rather than raising
    res = await call("nope", {})
    assert res.root.isError is True

    res = await handlers[types.ListResourcesRequest](
        types.ListResourcesRequest(method="resources/list")
    )
    assert {str(r.uri) for r in res.root.resources} == {
        "membrane://profiles",
        "membrane://last_compile",
        "membrane://metrics",
    }


@pytest.mark.requires_extras
async def test_registered_conversation_tool_disabled_and_read_resource():
    pytest.importorskip("mcp")
    import mcp.types as types

    comp = FakeCompiler()
    s = MembraneMCPServer(comp, enable_conversation_tool=False)
    server = s.build_server()
    handlers = server.request_handlers

    req = types.CallToolRequest(
        method="tools/call",
        params=types.CallToolRequestParams(
            name="membrane_compile_conversation", arguments={"user_message": "x"}
        ),
    )
    res = await handlers[types.CallToolRequest](req)
    assert res.root.isError is True
    assert "disabled" in res.root.content[0].text

    def read(uri):
        return handlers[types.ReadResourceRequest](
            types.ReadResourceRequest(
                method="resources/read", params=types.ReadResourceRequestParams(uri=uri)
            )
        )

    res = await read("membrane://profiles")
    assert json.loads(res.root.contents[0].text) == {
        "presets": ["minimal", "coding", "conversational"]
    }
    res = await read("membrane://metrics")
    assert "membrane-mcp" in res.root.contents[0].text
    s._metrics_snapshot = {"k": 1}
    res = await read("membrane://metrics")
    assert json.loads(res.root.contents[0].text) == {"k": 1}
    res = await read("membrane://last_compile")
    assert json.loads(res.root.contents[0].text) == {"error": "no_compilation_yet"}
    with pytest.raises(ValueError, match="Unknown resource"):
        await read("membrane://bogus")
