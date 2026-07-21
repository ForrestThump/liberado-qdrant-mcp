# Roadmap

**Forward-looking only.** What has already shipped lives in:

| Doc | Role |
|-----|------|
| [`README.md`](../README.md) | Capabilities overview, quick start, storage model |
| [`docs/AGENTS.md`](AGENTS.md) | MCP ↔ HTTP tool matrix, payload schema, host setup |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | Crate graph, data flows, scaling boundaries |
| [`docs/DECISIONS.md`](DECISIONS.md) | Why we chose what we chose |
| [`docs/PLAN.md`](PLAN.md) | Design rationale (historical milestones) |
| [AnythingLLM gap map](../liberado-qdrant-mcp_vs_AnythingLLM_Analysis_and_Implementation_Roadmap.md) | Capability matrix vs AnythingLLM knowledge layer |

When a roadmap item lands: implement it, document it in the rows above, then
**remove it from this file** (or move it only into historical docs if needed).
Do not accumulate a “Shipped” checklist here.

---

## Goal

Headless agent knowledge layer: **collections + ingest + lifecycle + retrieval
+ memories** over Qdrant via MCP and HTTP — enough to replace AnythingLLM’s
backend knowledge/MCP path for agents. Not a chat UI, multi-user product, or
agent orchestrator.

**Architecture rule:** new capability → `lqm-core` first → thin MCP + API
(+ CLI if ops-useful) in the same change. Live smoke against Qdrant for every
new tool.

**Storage model (fixed):** Qdrant holds **chunk text + metadata + vectors**.
`source` is a **pointer** (path/URL/id) for agents and other MCPs — originals
are not copied into a blob store. See ARCHITECTURE and DECISIONS.

---

## Active sequence (by leverage)

Do these **in order** unless a deployment constraint forces a jump.

_No active sequenced items right now._ Prefer items from **Later / optional**
only when a concrete deployment need appears. Shipped work lives in README /
AGENTS / ARCHITECTURE / DECISIONS (and crate ARCHITECTURE files).

---

## Later / optional (not sequenced)

Pick only if a concrete need appears:

| Item | Notes |
|------|--------|
| Full offline Qdrant double for ingest/search | MVP uses `McpTestClient` + FakeEmbedder + lazy client; full mock needs a seam |
| HTTP router tests (`tower::ServiceExt`) | Companion to offline MCP harness |

| Richer `list_sources` previews | Sample title / first-chunk preview / total chars |
| Background re-index workers | Only if bulk refresh becomes painful |
| Dioxus SPA | Demo UX; agents should prefer MCP/HTTP |
| WASM core | Browser-side story, not agent replacement |
| Chat-with-context tool | Only if a host cannot generate from `get_relevant_context` |
| Magic-values audit: `SourceType` on domain types | Still `Option<String>` on chunks/filters; enum exists in `source_type.rs` |
| Magic-values audit: `Importance` / `Scope` newtypes | Clamp/reject empty at construction; not started |
| Magic-values audit: full `response_keys` migration | Partial wiring in MCP/API; remaining JSON builders still use string literals |
| Magic-values audit: ingest/ASR string constants | `lqm-ingest` payload keys / content-type / API path literals |
| Magic-values audit: FNV + binary port defaults | Named constants in hybrid.rs and CLI/MCP/API listen defaults |

---

## Explicit non-goals

Do not prioritize these for the headless agent-KB goal:

- Multi-user auth, API-key admin consoles, multi-tenancy
- AnythingLLM-style agent CRUD / `invoke_agent`
- Full product chat UI and workspace chat history
- Breadth-first SaaS connectors (Drive, Notion, …) — agents use other MCPs +
  `ingest_*` with `source` pointers
- Storing original media binaries inside lqm (paths/URLs as pointers only)

---

## How to use this doc

1. Implement the **next unchecked item** in the active sequence (or jump with a
   written reason in the PR).
2. Update AGENTS / ARCHITECTURE / DECISIONS / README as needed.
3. Drop the item from this roadmap when merged.
4. Keep the AnythingLLM gap map in sync if capability vs product changes.
