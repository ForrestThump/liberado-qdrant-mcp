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

Do these **in order** unless a deployment constraint forces a jump (e.g. large
corpus already needs sparse hybrid before other work).

### 1. Sparse / scalable hybrid retrieval

**Why:** Hybrid dense + keyword is shipped, but the keyword path **scrolls the
full collection** (`O(n)` payloads). Fine for small/medium homelab corpora;
the main quality-at-scale gap as vaults grow.

**Ship:**

- [ ] Replace full-collection keyword scroll with a scalable approach (prefer
      **native Qdrant sparse vectors**, or a dedicated keyword index) while
      keeping dense-only as default.
- [ ] Preserve MCP/HTTP `hybrid` / `hybrid_alpha` ergonomics; document scaling
      in ARCHITECTURE.
- [ ] Unit tests for fusion/index wiring; live smoke for rare-token recall.

**Done when:** hybrid stays useful on large collections without full payload
walks.

**Defer until:** corpora are large enough that current hybrid is measurably slow
— unless building this alongside other retrieval work.

---

### 2. Real audio transcription (only if needed)

**Why:** Audio paths currently emit `source_type=audio_placeholder` stubs.
Worth doing **only if** agents ingest voice notes / podcasts into the KB.

**Ship:**

- [ ] Transcribe via whisper-rs or external ASR; store **transcript text**
      chunks + `source` pointer (same storage model as PDF: text in Qdrant,
      file stays on disk).
- [ ] Prefer `source_type=audio` (or equivalent) for real transcripts; keep
      placeholder filterable or remove once unused.
- [ ] Feature-gate heavy deps; MCP/API path parity.

**Done when:** an agent can retrieve meaningful audio content by meaning, not
metadata stubs.

**Skip if:** no audio ingest in your deployment.

---

### 3. Offline MCP integration tests

**Why:** Live smokes skip without Qdrant. PLAN called for turbomcp `channel` +
`McpTestClient`; hermetic tool tests protect the surface as it grows.

**Ship:**

- [ ] Channel-transport harness with FakeEmbedder (and Qdrant mock or test
      double as needed).
- [ ] Cover tool registration + a representative subset of tool behaviors
      offline; keep live smokes for real Qdrant.
- [ ] Optional companion: HTTP router tests via `tower::ServiceExt` (same
      spirit, lower priority than MCP harness).

**Done when:** `cargo test -p lqm-mcp` exercises tools without requiring a live
Qdrant for the core contract.

---

## Later / optional (not sequenced)

Pick only if a concrete need appears:

| Item | Notes |
|------|--------|
| Collection ↔ embedder hard guarantees | Refuse search on dim/model mismatch; optional model label at create |
| Richer `list_sources` previews | Sample title / first-chunk preview / total chars |
| Background re-index workers | Only if bulk refresh becomes painful |
| Dioxus SPA | Demo UX; agents should prefer MCP/HTTP |
| WASM core | Browser-side story, not agent replacement |
| Chat-with-context tool | Only if a host cannot generate from `get_relevant_context` |

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
