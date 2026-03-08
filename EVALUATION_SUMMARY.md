# Math Mentor — Evaluation Summary

**Project:** Multi-Agent RAG Math Tutoring System  
**Scope:** JEE-level Mathematics (Algebra, Calculus, Probability, Linear Algebra)  
**Stack:** Google Gemini 2.5 Flash · Pinecone · LangChain · Streamlit  
**Codebase:** ~4,400 lines across 13 Python modules  

---

## 1. System Architecture Assessment

### Agent Pipeline

| Agent | Role | Model | Status |
|---|---|---|---|
| ParserAgent | Structures raw input into problem dict | Gemini 2.5 Flash | ✅ Working |
| RouterAgent | Classifies topic, decides if calculator needed | Gemini 2.5 Flash | ✅ Working |
| SolverAgent | RAG + reasoning + safe Python calculator | Gemini 2.5 Flash | ✅ Working |
| VerifierAgent | Checks correctness, domain, edge cases | Gemini 2.5 Flash | ✅ Working |
| ExplainerAgent | Student-friendly walkthrough on demand | Gemini 2.5 Flash | ✅ Working |

The pipeline is **strictly sequential** — each agent receives the output of the previous one. All five agents use direct Gemini API calls with manually built prompts, which proved more reliable than LangChain chain abstractions for multi-variable prompt injection.

### RAG Setup

| Parameter | Value |
|---|---|
| Source corpus | JEE Mathematics PDFs |
| Pages ingested | 643 |
| Chunks created | 5,959 |
| Embedding model | all-MiniLM-L6-v2 (dim=384) |
| Vector store | Pinecone serverless |
| Retrieval | Top-3 semantic nearest neighbours |
| Similarity metric | Cosine |

---

## 2. Feature Completeness

| Feature | Implemented | Notes |
|---|---|---|
| Text input | ✅ | Direct problem entry |
| Image input (OCR) | ✅ | Gemini Vision, plain-text output |
| Audio input (mic) | ✅ | Live recording via browser mic |
| Step-by-step solution | ✅ | Notebook-style, Georgia serif |
| On-demand explanation | ✅ | Click-to-reveal, not auto-shown |
| Solution verification | ✅ | 3-dimension check by VerifierAgent |
| Calculator tool | ✅ | Safe Python eval, sandboxed |
| Memory / retrieval | ✅ | JSONL store, semantic similarity |
| Human-in-the-loop (HITL) | ✅ | Manual trigger only |
| OCR correction learning | ✅ | Stores raw→clean pairs |
| Streamlit Cloud deploy | ✅ | Live at streamlit.app |

---

## 3. Key Engineering Decisions

### Prompt Injection — `.format()` → `.replace()`
All three core agents (Solver, Verifier, Explainer) originally used Python's `.format()` for prompt variable injection. This caused `KeyError` crashes whenever retrieved math content contained curly braces like `{b}`, `{x}`, or `{2}` — common in JEE problems. Switched to `str.replace("<<VAR>>", value)` which is fully immune to content with curly braces.

### Direct Gemini Calls over LangChain Chains
The original solver used `create_retrieval_chain` + `create_stuff_documents_chain`. This only supported `{context}` and `{input}` injection — custom variables like `{memory_context}` and `{calculator_results}` were silently dropped. Replaced with direct `ChatGoogleGenerativeAI.invoke([SystemMessage, HumanMessage])` which gives full control over prompt content.

### Token Limits
Default `max_tokens=2048` caused Gemini to truncate mid-JSON, producing unparseable responses. Raised to 16,000 for the Solver and 4,096 for Verifier and Explainer. Added robust JSON extraction that finds the outermost `{}` block even if the response has surrounding text.

### `chunk.py` → `chunker.py` Rename
The project's chunking module was named `chunk.py`, which shadowed Python's built-in `chunk` module. This caused `ImportError: cannot import name 'Chunk'` deep inside `speech_recognition` → `aifc` → `chunk`. Renaming to `chunker.py` resolved the conflict across all dependents.

### HITL Scope Reduction
Initially HITL triggered automatically on parser ambiguity, verifier low confidence, and before explanation. This created blocking UX where students couldn't see results. Reduced HITL to manual-only trigger via a "Request Review" button, keeping it available without blocking normal flow.

---

## 4. Bugs Found and Fixed

| # | Bug | Root Cause | Fix |
|---|---|---|---|
| 1 | Every question returned same probability answer | `use_mock=True` hardcoded | Set `use_mock=False`, removed toggle |
| 2 | Custom prompt variables silently dropped | LangChain chain only injects `{context}` and `{input}` | Switched to direct Gemini API call |
| 3 | `from chunk import embeddings` failed | `chunk.py` shadowed Python built-in | Renamed to `chunker.py` |
| 4 | JSON truncated mid-response | `max_tokens=2048` too small | Raised to 16,000 (solver) |
| 5 | `Prompt formatting failed: 'b'` | `.format()` interpreted `{b}` in math content as variable | Replaced all `.format()` with `.replace()` |
| 6 | Explanation shown, solution empty | Agent trace confirmed steps array empty due to prompt crash | Fixed prompt injection, added fallback display |
| 7 | `streamlit_mic_recorder` import error | `chunk.py` name collision broke `speech_recognition` dependency | Same fix as bug #3 |
| 8 | Streamlit Cloud deploy failed | Python 3.14 used; `langchain-pinecone` requires <3.14 | Added `.python-version = 3.11` |
| 9 | Pinecone version conflict | `pinecone-client` vs `pinecone` package name + version range clash | Pinned `pinecone>=3.0.0,<6.0.0` with `langchain-pinecone<0.2.0` |

---

## 5. Known Limitations

### API Quota
Gemini 2.5 Flash free tier allows approximately 20 requests per day. Each problem solve consumes 4–5 API calls (one per agent). This limits the free deployment to roughly 4–5 problems per day. Upgrade to the paid tier or switch to `gemini-1.5-flash` (1,500 requests/day free) for production use.

### Memory Persistence on Streamlit Cloud
`memory_store.jsonl` is written to the container's filesystem at runtime. Streamlit Cloud containers are ephemeral — the file is lost on every redeploy or restart. For production, this should be replaced with a persistent store such as Pinecone, Supabase, or a cloud database.

### Audio on HTTPS
The browser microphone API (`getUserMedia`) requires HTTPS. Streamlit Cloud serves over HTTPS so this works correctly in production. It will not work on plain `http://localhost` without special browser flags.

### No Authentication
The current deployment has no user authentication. Anyone with the URL can use the app and consume API quota. For a production rollout, add Streamlit's built-in authentication or a login gate.

---

## 6. Performance Observations

| Metric | Observed |
|---|---|
| End-to-end solve time | 15–30 seconds (5 sequential LLM calls) |
| OCR extraction time | 3–6 seconds |
| Audio transcription time | 5–10 seconds |
| Memory retrieval time | <1 second (local JSONL) |
| Pinecone retrieval time | 1–2 seconds |

The main latency bottleneck is the sequential nature of the 5-agent pipeline. Verifier and Explainer could potentially run in parallel after the Solver completes, which could reduce total time by ~30%.

---

## 7. Recommendations for Next Version

1. **Switch memory store to Pinecone or Supabase** — persistent across restarts, scales beyond single-user
2. **Parallelise Verifier + Explainer** — both depend only on Solver output, can run concurrently
3. **Add user authentication** — tie memory records to individual users
4. **Stream Gemini responses** — use `stream=True` to show solution steps as they generate instead of waiting for full response
5. **Upgrade to `gemini-1.5-flash`** — 1,500 free requests/day vs 20, sufficient for classroom use without payment
6. **Add topic performance tracking** — show students which topics they struggle with based on verifier verdicts across sessions

---

*Evaluation completed: March 2026*
