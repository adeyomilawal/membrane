# U-shaped context layout

## Academic foundation

Large language models often attend more strongly to the **beginning** and **end** of long contexts than to the middle. This positional attention bias was demonstrated empirically by:

> Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). *Lost in the Middle: How Language Models Use Long Contexts*. Transactions of the Association for Computational Linguistics, 12, 157–173. https://doi.org/10.1162/tacl_a_00638

The authors tested 16 models on multi-document QA and key-value retrieval tasks and found that performance is highest when relevant information appears at the very beginning or very end of the input context, and lowest when it appears in the middle — even for models explicitly trained for long contexts.

## How the kernel encodes this

The `ContextLayoutEngine` constructs prompts with three positional regions that exploit this U-shaped attention curve:

### Start (high attention)

System rules, persona definition, interrupt protocol instructions, and anything that must constrain every token the model generates. This is the **core** block — it is never trimmed by the firewall.

### Middle (sacrifice zone)

Retrieved memories and digests, serialised as `[Mem:N]` indexed lines. This is the content that `PreFlightContextFirewall` trims first under budget pressure. Segments are ordered by priority tier:

| Tier constant | Value | Retention priority |
|---------------|-------|--------------------|
| `TIER_PROCEDURAL` | 1 | Highest — trimmed last |
| `TIER_SEMANTIC` | 2 | Medium |
| `TIER_EPISODIC` | 3 | Lowest — trimmed first |

Within a tier, the firewall removes segments furthest from the positional edges first, maintaining the U-shaped property even within the sacrifice zone.

### End (high attention)

The immediate dialogue context: last assistant turn and the current user message. This ensures the model's answer aligns with the latest intent. `ContextLayoutEngine.build_end()` formats this block explicitly so it is easy to audit in logs and tests.

`ACTIVE_SYSTEM_DIRECTIVES` (dynamically injected system instructions) are appended after the core block via `build_directives_block()`, keeping them close to the high-attention start region.

## Practical implications

1. **Critical instructions belong in core or end** — never in the middle. If the model must follow a rule, place it in the system prompt (start) or the user message (end).
2. **Memory ordering matters less than position** — the sacrifice zone exists precisely because models attend to it less. The kernel's job is to keep the budget honest and ensure the most important memories survive trimming.
3. **The layout is configurable** — hosts can override which content goes into each region. The kernel provides the structure; you decide what fills it.
