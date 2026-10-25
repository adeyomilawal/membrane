# Drop 3 — MCP server & framework adapters

## MCP (stdio)

Install: `pip install "membrane[mcp]"`.

```bash
python -m membrane.adapters.mcp_server
```

Wire a custom `memory_provider` in Python to resolve `memory_uris` to `MemoryItem` rows before compiling.

## LangChain

Install: `pip install "membrane[langchain]"`.

Use `MembranePromptTemplate` **instead of** `ConversationBufferMemory` for the same chain. See
`membrane.adapters.langchain.disable_langchain_memory` and `MemoryCollisionError`.

## LlamaIndex / CrewAI / Haystack

See `membrane.adapters.llamaindex`, `membrane.adapters.crewai`, `membrane.adapters.haystack`.
