"""
MCP (Model Context Protocol) server exposing Membrane as tools + resources.

Install: ``pip install membrane[mcp]`` (requires ``mcp>=1.0``).

Run (stdio, for Cursor / Claude Desktop):

.. code-block:: bash

   python -m membrane.adapters.mcp_server

**Single-client only:** the server keeps one session's state (last compile result, metrics
snapshot, conversation history) on the instance, shared by every connection. Use stdio (one
client per process). The SSE transport serves multiple connections from the same state and
has no authentication; do not expose it beyond loopback.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any
from collections.abc import Awaitable, Callable, Sequence

from membrane.profile import PRESET_NAMES, MembraneProfile

from membrane.adapters import mcp_resources

if TYPE_CHECKING:
    from membrane.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

MemoryProvider = (
    Callable[[Sequence[str]], Awaitable[Sequence[Any]]]
    | Callable[[Sequence[str]], Sequence[Any]]
)


def _require_mcp():
    try:
        import mcp.server.stdio  # noqa: F401
        import mcp.types as types  # noqa: F401
        from mcp.server.lowlevel import NotificationOptions, Server  # noqa: F401
        from mcp.server.models import InitializationOptions  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Membrane MCP server requires the 'mcp' package. Install: pip install membrane[mcp]"
        ) from e


def _resolve_profile(name: str) -> Any:
    return MembraneProfile.preset(name)


async def _maybe_await(res: Any) -> Any:
    if inspect.isawaitable(res):
        return await res
    return res


class MembraneMCPServer:
    """
    Registers tools: ``membrane_compile``, ``membrane_inspect``, and (unless disabled)
    ``membrane_compile_conversation`` for multi-turn state via
    :class:`~membrane.conversation.manager.ConversationManager`.

    Resources: ``membrane://profiles``, ``membrane://last_compile``, ``membrane://metrics``.

    **Single-client:** one instance holds one session's state (last result, metrics, conversation
    history) that all callers share. It is meant for a single client over stdio; run one process
    per client rather than sharing an SSE endpoint.
    """

    def __init__(
        self,
        compiler: Any,
        *,
        profile: Any | None = None,
        memory_provider: MemoryProvider | None = None,
        conversation_manager: ConversationManager | None = None,
        enable_conversation_tool: bool = True,
        host: str = "localhost",
        port: int = 3100,
    ) -> None:
        self._compiler = compiler
        self._default_profile = profile
        self._memory_provider = memory_provider
        self._conversation_manager = conversation_manager
        self._lazy_conversation_manager: Any | None = None
        self._enable_conversation_tool = enable_conversation_tool
        self.host = host
        self.port = port
        self._last_result: Any | None = None
        self._metrics_snapshot: dict[str, Any] = {}

    def _pick_profile(self, arguments: dict[str, Any]) -> tuple:
        """Per-call ``profile`` wins over the server default; returns ``(profile, name_used)``."""
        if arguments.get("profile"):
            name = str(arguments["profile"])
            return _resolve_profile(name), name
        if self._default_profile is not None:
            return self._default_profile, "default"
        return _resolve_profile("coding"), "coding"

    def _warn_if_exposed(self) -> None:
        if str(self.host).strip("[]").lower() not in ("localhost", "127.0.0.1", "::1"):
            logger.warning(
                "SSE transport on non-loopback host %r: this server shares one session's state "
                "across all clients and has no authentication",
                self.host,
            )

    def _get_conversation_manager(self) -> Any:
        if self._conversation_manager is not None:
            return self._conversation_manager
        if self._lazy_conversation_manager is None:
            from membrane.conversation.manager import ConversationManager

            self._lazy_conversation_manager = ConversationManager(self._compiler)
        return self._lazy_conversation_manager

    def as_tool_definitions(self) -> list[dict[str, Any]]:
        """JSON-schema style tool definitions for hosts that register tools manually."""
        tools: list[dict[str, Any]] = [
            {
                "name": "membrane_compile",
                "description": (
                    "Compile optimized context for an LLM query using Membrane's pipeline."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "memory_uris": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional URIs passed to memory_provider",
                        },
                        "profile": {
                            "type": "string",
                            "enum": list(PRESET_NAMES),
                            "default": "coding",
                        },
                        "model": {"type": "string", "default": "gpt-4o"},
                        "max_tokens": {"type": "integer", "default": 8192},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "membrane_inspect",
                "description": "Return debug JSON for the last membrane_compile result.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        if self._enable_conversation_tool:
            tools.append(
                {
                    "name": "membrane_compile_conversation",
                    "description": (
                        "Compile with multi-turn state: appends the user message to conversation "
                        "history then runs ContextCompiler (same parameters as membrane_compile)."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "user_message": {"type": "string"},
                            "memory_uris": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "profile": {
                                "type": "string",
                                "enum": list(PRESET_NAMES),
                                "default": "coding",
                            },
                            "model": {"type": "string", "default": "gpt-4o"},
                            "max_tokens": {"type": "integer", "default": 8192},
                        },
                        "required": ["user_message"],
                    },
                }
            )
        return tools

    async def _fetch_memories(self, uris: Sequence[str]) -> list[Any]:
        if not uris:
            return []
        if self._memory_provider is None:
            from membrane.memory_protocol import MemoryItem

            return [
                MemoryItem(id=u, content=f"(no memory_provider) placeholder for {u}")
                for u in uris
            ]
        out = self._memory_provider(list(uris))
        return list(await _maybe_await(out))

    async def _compile_tool(self, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query", ""))
        uris = arguments.get("memory_uris") or []
        model = str(arguments.get("model", "gpt-4o"))
        max_tokens = int(arguments.get("max_tokens", 8192))

        mems = await self._fetch_memories(uris)
        prof, prof_name = self._pick_profile(arguments)

        comp = self._compiler
        prev_model = getattr(comp, "model_id", None)
        prev_max = getattr(comp, "max_tokens", None)
        try:
            comp.model_id = model  # type: ignore[attr-defined]
            comp.max_tokens = max_tokens  # type: ignore[attr-defined]
            compile_fn = getattr(comp, "compile", None)
            if compile_fn is None:
                raise RuntimeError("compiler has no compile()")
            self._last_result = compile_fn(
                memories=mems,
                user_message=query,
                profile=prof,
            )
        finally:
            try:
                comp.model_id = prev_model  # type: ignore[attr-defined]
                comp.max_tokens = prev_max  # type: ignore[attr-defined]
            except Exception:
                pass

        self._metrics_snapshot = {
            "last_profile": prof_name,
            "memory_count": len(mems),
            "token_hint": (getattr(self._last_result, "metadata", {}) or {}).get("token_count"),
        }
        payload = (
            self._last_result.to_debug_dict()
            if self._last_result is not None and hasattr(self._last_result, "to_debug_dict")
            else {"ok": True}
        )
        return json.dumps(payload, default=str, indent=2)

    async def _compile_conversation_tool(self, arguments: dict[str, Any]) -> str:
        user_message = str(arguments.get("user_message", ""))
        uris = arguments.get("memory_uris") or []
        model = str(arguments.get("model", "gpt-4o"))
        max_tokens = int(arguments.get("max_tokens", 8192))

        mems = await self._fetch_memories(uris)
        prof, prof_name = self._pick_profile(arguments)
        cm = self._get_conversation_manager()
        cm.add_turn("user", user_message)

        comp = self._compiler
        prev_model = getattr(comp, "model_id", None)
        prev_max = getattr(comp, "max_tokens", None)
        try:
            comp.model_id = model  # type: ignore[attr-defined]
            comp.max_tokens = max_tokens  # type: ignore[attr-defined]
            self._last_result = cm.compile(
                memories=mems,
                user_message=user_message,
                profile=prof,
            )
        finally:
            try:
                comp.model_id = prev_model  # type: ignore[attr-defined]
                comp.max_tokens = prev_max  # type: ignore[attr-defined]
            except Exception:
                pass

        self._metrics_snapshot = {
            "last_profile": prof_name,
            "memory_count": len(mems),
            "conversation_tool": True,
            "token_hint": (getattr(self._last_result, "metadata", {}) or {}).get("token_count"),
        }
        payload = (
            self._last_result.to_debug_dict()
            if self._last_result is not None and hasattr(self._last_result, "to_debug_dict")
            else {"ok": True}
        )
        return json.dumps(payload, default=str, indent=2)

    def build_server(self) -> Any:
        """Build low-level MCP :class:`Server` with tools and resources."""
        _require_mcp()
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.models import InitializationOptions

        server = Server("membrane")
        parent = self

        @server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            defs = parent.as_tool_definitions()
            return [
                types.Tool(
                    name=d["name"],
                    description=d.get("description", ""),
                    inputSchema=d["inputSchema"],
                )
                for d in defs
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            if name == "membrane_compile":
                text = await parent._compile_tool(arguments or {})
                return [types.TextContent(type="text", text=text)]
            if name == "membrane_inspect":
                txt = mcp_resources.last_compile_resource_text(parent._last_result)
                return [types.TextContent(type="text", text=txt)]
            if name == "membrane_compile_conversation":
                if not parent._enable_conversation_tool:
                    raise ValueError("membrane_compile_conversation is disabled")
                text = await parent._compile_conversation_tool(arguments or {})
                return [types.TextContent(type="text", text=text)]
            raise ValueError(f"Unknown tool: {name}")

        @server.list_resources()
        async def _list_resources() -> list[types.Resource]:
            return [
                types.Resource(
                    uri="membrane://profiles",
                    name="profiles",
                    description="Available MembraneProfile presets",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="membrane://last_compile",
                    name="last_compile",
                    description="Last CompileResult snapshot",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="membrane://metrics",
                    name="metrics",
                    description="Session metrics snapshot",
                    mimeType="application/json",
                ),
            ]

        @server.read_resource()
        async def _read_resource(uri: Any) -> str:
            u = str(uri)
            if u == "membrane://profiles":
                return mcp_resources.list_profile_resource_text()
            if u == "membrane://last_compile":
                return mcp_resources.last_compile_resource_text(parent._last_result)
            if u == "membrane://metrics":
                return mcp_resources.metrics_resource_text(parent._metrics_snapshot)
            raise ValueError(f"Unknown resource: {u}")

        self._server = server
        self._init_options_type = InitializationOptions
        return server

    async def serve(self, *, transport: str = "stdio") -> None:
        """
        Start the MCP server. ``transport`` is ``stdio`` (default) or ``sse`` (requires uvicorn).

        For stdio, logging must go to stderr only.
        """
        if transport == "stdio":
            await self._serve_stdio()
            return
        if transport == "sse":
            await self._serve_sse()
            return
        raise ValueError(f"Unsupported transport: {transport}")

    async def _serve_stdio(self) -> None:
        import mcp.server.stdio
        from mcp.server.lowlevel import NotificationOptions

        server = self.build_server()
        init_cls = self._init_options_type
        notif_cls = NotificationOptions
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                init_cls(
                    server_name="membrane",
                    server_version="0.1.0",
                    capabilities=server.get_capabilities(
                        notification_options=notif_cls(),
                        experimental_capabilities={},
                    ),
                ),
            )

    async def _serve_sse(self) -> None:
        self._warn_if_exposed()
        try:
            import uvicorn  # type: ignore
        except ImportError as e:
            raise ImportError("SSE transport requires uvicorn: pip install uvicorn") from e
        from mcp.server.lowlevel import NotificationOptions
        from mcp.server.sse import SseServerTransport  # type: ignore
        from starlette.applications import Starlette  # type: ignore
        from starlette.requests import Request  # type: ignore
        from starlette.responses import Response  # type: ignore
        from starlette.routing import Mount, Route  # type: ignore

        mcp_server = self.build_server()
        sse = SseServerTransport("/messages/")
        init_opts = self._init_options_type(
            server_name="membrane",
            server_version="0.1.0",
            capabilities=mcp_server.get_capabilities(
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        )

        async def handle_sse(request: Request) -> Response:
            async with sse.connect_sse(
                request.scope,
                request.receive,
                request._send,
            ) as streams:
                await mcp_server.run(streams[0], streams[1], init_opts)
            return Response()

        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        await uvicorn.Server(config).serve()


def main() -> None:
    """CLI: stdio MCP with a default :class:`~membrane.compiler.ContextCompiler`."""
    logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
    from membrane.compiler import ContextCompiler

    compiler = ContextCompiler(max_tokens=8192, model_id="gpt-4o")
    mcp = MembraneMCPServer(compiler)

    asyncio.run(mcp.serve(transport="stdio"))


if __name__ == "__main__":
    main()
