# Interrupt protocol

The kernel can inject instructions so that, when context is incomplete, the model emits a machine-readable tag instead of hallucinating:

`<SYSTEM_INTERRUPT_FETCH[precise query]>`

## Parsing

- `parse_interrupt(text)` → `(before, query, after)` or `None`
- `contains_interrupt(text)` → bool
- `strip_all_interrupts(text)` → text with all tags removed (e.g. after max “page faults”)

Hosts should detect the tag in the stream or final string, run retrieval with `query`, inject results, and continue generation per their orchestration policy.

## Local memory references

Dense serialization prefixes lines with `[Mem:N]`. The compiler builds `ref_map: N → global_id` and optional `ref_map_type`. Persist with `set_ref_map_for_conversation` (Redis) so tools like `request_memory_expansion(N)` can resolve IDs on subsequent turns.
