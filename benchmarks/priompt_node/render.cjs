// Runs the real @anysphere/priompt renderer for benchmarks/hard_suite.py.
//
// Reads one JSON request per line on stdin and writes one JSON reply per line on stdout:
//   {"budget": N, "tokenizer": "cl100k_base", "core": "...", "header": "...",
//    "memories": [{"prio": P, "text": "..."}],
//    "history": [{"prio": P, "messages": [{"role", "content", "name"?}]}],
//    "question": "..."}
//   -> {"cutoff": P, "tokens": N}   or   {"error": "..."}
// Every memory line and every conversation turn is one <scope p={P}>; the core and the
// question have no priority, so they are always rendered. Python rebuilds the request from
// the cutoff (everything with prio >= cutoff is kept), so the messages every strategy
// sends are assembled by the same code.
"use strict";
const readline = require("node:readline");
const P = require("@anysphere/priompt");

const h = P.createElement;
const ROLES = {
  user: P.UserMessage,
  assistant: P.AssistantMessage,
  system: P.SystemMessage,
  tool: P.ToolResultMessage,
};

function build(req) {
  const memories = req.memories.map((m) => h("scope", { p: m.prio }, m.text));
  const system = h(P.SystemMessage, null, req.core, ...(memories.length ? [req.header] : []), ...memories);
  const history = req.history.map((turn) =>
    h(
      "scope",
      { p: turn.prio },
      ...turn.messages.map((m) => h(ROLES[m.role], m.name ? { name: m.name } : null, m.content)),
    ),
  );
  return h(P.Fragment, null, system, ...history, h(P.UserMessage, null, req.question));
}

async function answer(line) {
  try {
    const req = JSON.parse(line);
    const out = await P.render(build(req), {
      tokenLimit: req.budget,
      tokenizer: P_TOKENIZERS(req.tokenizer || "cl100k_base"),
    });
    return { cutoff: out.priorityCutoff, tokens: out.tokenCount };
  } catch (err) {
    return { error: String((err && err.message) || err) };
  }
}

const { getTokenizerByName_ONLY_FOR_OPENAI_TOKENIZERS: P_TOKENIZERS } = require(
  "@anysphere/priompt/dist/tokenizer",
);

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let chain = Promise.resolve();
rl.on("line", (line) => {
  if (!line.trim()) return;
  chain = chain.then(async () => {
    process.stdout.write(JSON.stringify(await answer(line)) + "\n");
  });
});
