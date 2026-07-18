# Plan: Real audio transcription via DeepInfra (Whisper + Nemotron)

## Goal kind
code-change

## Operator setup (before `/goal`)

Open a **new shell** with at least:

```powershell
# Required for live ASR smoke (plan allows skip-safe if missing, but
# LQM_LIVE_ASR=1 forces real network proof).
$env:DEEPINFRA_API_KEY = "<your key>"   # also accept DEEPINFRA_TOKEN

# Optional but recommended for full path proof (Qdrant + embed + search):
$env:QDRANT_URL = "http://127.0.0.1:6334"
$env:LQM_LIVE = "1"          # hard-require Qdrant for existing live smokes
$env:LQM_LIVE_ASR = "1"      # hard-require DeepInfra for ASR live smoke

# Then from repo root:
#   /goal  (point at this file: docs/plans/2026-07-18-audio-transcription-deepinfra.md)
```

**Do not** put secrets in the plan, commit, or docs. Implementer must read key from env only.

---

## Locked product decisions (do not re-litigate)

| Decision | Choice |
|----------|--------|
| Pipeline | **STT → text chunks → existing dense embed** (not multimodal audio embeddings) |
| Input mode | **File ingest only** (no streaming mic/WebSocket ASR) |
| Provider | **DeepInfra** HTTP inference API |
| Backends (both ship in one change) | (1) `whisper` → slug `openai/whisper-large-v3-turbo` (2) `nemotron` → slug `nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b` |
| Default backend | **`whisper`** |
| Storage | Transcript **text** + `source` path pointer in Qdrant; **never** store audio bytes |
| `source_type` on success | `audio` (`SOURCE_TYPE_AUDIO`) |
| Placeholder | Keep `audio_placeholder` path when ASR feature/key unavailable and policy is placeholder; **do not** silently write placeholders when ASR was explicitly required / live-forced |
| Streaming Nemotron | Model name contains “Streaming” but v1 uses **batch file** multipart only (same as Whisper on DeepInfra) |

### DeepInfra API (verified shape — both models)

Both models use the same HTTP contract (only path slug differs):

```http
POST https://api.deepinfra.com/v1/inference/{model_slug}
Authorization: Bearer <DEEPINFRA_API_KEY or DEEPINFRA_TOKEN>
Content-Type: multipart/form-data
Field: audio=@<file>
```

Example response (parse flexibly; require non-empty `text` when status OK):

```json
{
  "text": "…",
  "segments": [ { "start": 0.0, "end": 1.0, "text": "…" } ],
  "language": "en",
  "duration": 0.0
}
```

Slugs (exact):

- Whisper (default): `openai/whisper-large-v3-turbo`
- Nemotron: `nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`

Auth header: `Authorization: Bearer …` (DeepInfra docs also show `bearer`; accept either casing in client).

---

## Acceptance criteria

1. **Two configurable ASR backends** are implemented and selectable without code changes:
   - Enum / config: at least `whisper` | `nemotron` (aliases OK).
   - Env (recommended names — implement exactly or document if you must rename):
     - `LQM_ASR_BACKEND` — `whisper` (default) | `nemotron`
     - `LQM_ASR_MODEL` — optional override of full DeepInfra model slug
     - `DEEPINFRA_API_KEY` **or** `DEEPINFRA_TOKEN` — API key (prefer first if both set)
     - Optional: `DEEPINFRA_BASE_URL` default `https://api.deepinfra.com`
   - Default backend is **whisper** / slug `openai/whisper-large-v3-turbo`.

2. **File ingest path** for audio extensions (at least existing set: mp3, wav, flac, ogg, m4a, opus, aac, wma) can produce **real transcript text** and ingest with `source_type=audio`, `source` = path pointer, then normal chunk/embed/upsert via `RagCore`.

3. **No multimodal embeddings.** No audio vectors in Qdrant. Same text hybrid/search path as other docs.

4. **Feature gate:** heavy HTTP ASR behind a cargo feature on `lqm-ingest` (suggested name: `asr-deepinfra`). MCP and API enable it (same pattern as `pdf`). Without feature: either compile-time no ASR or clear validation error — document which. Prefer: feature on for mcp/api like pdf; placeholder remains for builds without feature.

5. **Failure modes:**
   - Missing key when ASR is attempted → clear `LqmError` / extract error (not a successful placeholder ingest that looks like content).
   - Network/API error → surfaced; no silent empty success.
   - Optional env `LQM_ASR_FALLBACK_PLACEHOLDER=1` may restore placeholder behavior for ops; default **off**.

6. **Tests (no theater):**
   - **Offline unit tests** (no network): config parse (backend + model slug resolution); response JSON → text extraction (fixture JSON); multipart request builder shape if pure; placeholder still tested when ASR off.
   - **Live smoke (skip-safe):** if `DEEPINFRA_API_KEY`/`DEEPINFRA_TOKEN` missing → skip with message. If `LQM_LIVE_ASR=1` and key missing → **panic/fail** (hard require). With key: transcribe a tiny committed fixture (or generated short WAV) for **both** backends (or default whisper mandatory + nemotron if key present), assert non-empty text that is **not** the old placeholder string.
   - Prefer also: with Qdrant available, ingest audio file → `search` finds a distinctive spoken token / phrase from the fixture transcript.

7. **CI gate:** `cargo fmt --all -- --check`, `cargo clippy --workspace -- -D warnings`, `cargo test --workspace` all exit 0. Offline suite must pass **without** DeepInfra key (live ASR skips).

8. **Docs:** remove “Real audio transcription” from active `docs/ROADMAP.md` sequence; update ARCHITECTURE / AGENTS / DECISIONS / crate ARCHITECTURE / README as needed. Record decision: STT-via-DeepInfra dual backend, not multimodal.

---

## Verification plan

1. **gating:** `cargo fmt --all -- --check` && `cargo clippy --workspace -- -D warnings` && `cargo test --workspace` → capture to `{SCRATCH}/ci-suite.log`. All exit 0; no failures. Live ASR may skip.

2. **gating:** Offline ASR unit tests: `cargo test -p lqm-ingest asr -- --nocapture` (or actual filter) → `{SCRATCH}/unit-asr.log`. Must cover backend parse + response parse for whisper-shaped JSON; **not** network.

3. **gating:** Config defaults: unit test proves default backend is whisper and default slug is `openai/whisper-large-v3-turbo`; nemotron maps to `nvidia/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`.

4. **evidence (live):** If key present, run live smoke(s) for whisper and nemotron; capture `{SCRATCH}/live-asr.log`. If `LQM_LIVE_ASR=1`, must pass (not skip). If key absent and `LQM_LIVE_ASR` unset, skip is OK.

5. **docs:** ROADMAP audio item gone; AGENTS documents env + `source_type=audio`; DECISIONS entry for dual DeepInfra ASR backends.

---

## Non-goals

- Streaming / real-time ASR sessions (mic, WebSocket, chunked upload protocols)
- Multimodal or audio-native embeddings
- Storing original audio binaries in Qdrant or a blob store
- Local whisper-rs / ONNX / CUDA offline STT (may be later optional backend)
- OpenAI-hosted `/v1/audio/transcriptions` as required path (DeepInfra only for this plan; OpenAI-compatible STT optional only if free and not required)
- Full diarization productization (timestamps in payload optional; joining `text` is enough)
- Auth product / multi-user

---

## Assumed scope

| Area | Work |
|------|------|
| `crates/lqm-ingest` | ASR module (config, HTTP client, parse); AudioExtractor or async extract path; feature `asr-deepinfra`; fixture + unit/live tests |
| `crates/lqm-mcp` / `lqm-api` / `lqm-cli` | Enable feature; call async extract for audio on file ingest paths; surface errors |
| `lqm-core` | Only if shared constants/error codes needed; prefer keep STT in ingest |
| Docs | ROADMAP, ARCHITECTURE, AGENTS, DECISIONS, crate ARCHITECTURE, README |

**Deps:** `reqwest` with `multipart` (may already be optional via `fetch-url`; ASR feature may re-export/require reqwest + multipart). No heavy local ML crates.

---

## Implementation approach

### 1. Config / backend enum (pure)

```text
AsrBackend::Whisper | AsrBackend::Nemotron
default model slugs as above
LQM_ASR_MODEL overrides slug for either backend
resolve_api_key: DEEPINFRA_API_KEY then DEEPINFRA_TOKEN
```

Unit-test parse + defaults offline.

### 2. DeepInfra client

```text
async fn transcribe_file(path, config) -> Result<Transcript, AsrError>
  - read file bytes / stream multipart field `audio`
  - POST {base}/v1/inference/{slug}
  - parse JSON; use `text` field (trim); error if empty after OK HTTP
```

Share one client for both backends (slug differs only). Timeouts configurable (sensible default e.g. 120s for long files).

### 3. Extractor integration (critical seam)

Today: `Extractor::extract_text` is **sync**; MCP/API call `extract_file` from async tools. ASR needs async HTTP.

**Preferred pattern (implement this unless blocked):**

- Add `extract_file_async(path, base_payload) -> Result<Vec<DocumentChunk>, ExtractError>` that:
  - for non-audio: same as sync extract
  - for audio + feature + configured ASR: `transcribe_file` → `source_type=audio`, text = transcript
  - for audio without feature/key: either error or placeholder per policy above
- Keep sync `extract_file` for text/pdf tests; for audio sync path either call block_in_place only in CLI single-thread contexts **or** return clear error “use async path”. Prefer MCP/API/CLI all use async for file ingest.

### 4. Binary wiring

- `lqm-mcp` / `lqm-api` Cargo.toml: `lqm-ingest` features include `pdf` **and** `asr-deepinfra` (and `fetch-url` as today).
- `ingest_path` / `ingest_many` (file items) / CLI single-file: use async extract for audio.
- Log model slug + backend at info on successful transcription (no key logging).

### 5. Fixture

- Small checked-in audio under e.g. `crates/lqm-ingest/tests/fixtures/asr_hello.wav` (or generate minimal valid WAV in test if licensing/size is a concern).
- Spoken content should be short English so Whisper/Nemotron return non-empty text.
- Live smoke asserts transcript does not contain `Audio placeholder`.

### 6. Docs / roadmap

- Remove active ROADMAP item “Real audio transcription”.
- DECISIONS: STT DeepInfra dual backend; pointer-only storage unchanged.
- AGENTS: how to set env; filter `source_type=audio` vs `audio_placeholder`.

---

## Task checklist

Implementer: flip each box to `[x]` in **this file** as completed (or track in goal harness plan copy — keep one source of truth).

- [x] Add `AsrBackend` + env resolution + default slugs; offline unit tests
- [x] Implement DeepInfra multipart `transcribe_file` + JSON parse; offline parse tests with fixture JSON
- [x] Wire audio extract path to ASR (`source_type=audio`); placeholder policy + feature gate
- [x] Enable feature on MCP/API (and CLI if file ingest); async file ingest call sites
- [x] Live skip-safe smoke(s) for whisper + nemotron; optional Qdrant search proof
- [x] Docs: ROADMAP remove item; ARCHITECTURE / AGENTS / DECISIONS / README / crate ARCHITECTURE
- [x] Full CI: fmt, clippy `-D warnings`, `cargo test --workspace`; capture logs under scratch

---

## Risks / mitigations

| Risk | Mitigation |
|------|------------|
| Sync extractor + async HTTP | Introduce `extract_file_async`; do not `block_on` inside multi-thread Tokio without care |
| Nemotron response differs slightly | Parse `text` first; tolerate missing segments; add second fixture response if needed |
| Large audio files / timeouts | Timeout + clear error; optional max bytes env later (not required v1) |
| CI without key | Skip-safe live tests; offline units always run |
| Key name mismatch | Accept `DEEPINFRA_API_KEY` and `DEEPINFRA_TOKEN` |
| Accidental placeholder pollution | Default: fail closed when ASR feature on and key missing on audio ingest; only placeholder if feature off or explicit fallback env |

---

## How to run this plan with `/goal`

1. New terminal; export `DEEPINFRA_API_KEY` (and optionally Qdrant + `LQM_LIVE_ASR=1`).
2. `cd` to repo root (`liberado-qdrant-mcp`).
3. Start goal mode pointing at:

   **`docs/plans/2026-07-18-audio-transcription-deepinfra.md`**

4. Instruct the agent to treat this file as the source of truth for acceptance + verification (same discipline as hybrid goal: checklist, tests that drive shipped code, scratch evidence, no test theater).
5. Expect: commit-ready tree on `develop` (or feature branch if agent policy prefers); ROADMAP updated; suite green offline.

### Definition of done (operator checklist)

- [ ] Audio file ingest produces searchable transcript text with `source_type=audio`
- [ ] `LQM_ASR_BACKEND=whisper` (default) and `=nemotron` both work against DeepInfra file API
- [ ] Workspace CI green without requiring DeepInfra in default CI
- [ ] With key + `LQM_LIVE_ASR=1`, live ASR smoke passes for at least Whisper (Nemotron too if API up)
- [ ] ROADMAP audio item removed; env documented in AGENTS

---

## Out-of-band notes for the implementer

- Project rules: read `docs/PLAN.md` / `Agents.md`; prefer small pure modules; log via `log::`, not `println!`; `cargo fmt && cargo clippy -- -D warnings`.
- Do not break dense-only search or hybrid backends.
- Do not add connectors or blob storage.
- Naming: `lqm-*`, never “myrag”.
)
