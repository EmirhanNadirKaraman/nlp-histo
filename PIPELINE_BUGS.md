# Summarization Pipeline — Bugs, Logical Mistakes & Fixes

Audit of the MAP batch pipeline and related infrastructure.

---

## BUG 1 — `cache_path` variable shadowed in `map_theta_sweep.py` [HIGH]

**File:** `eval/silver/map_theta_sweep.py:688,767`

```python
cache_path = primer_dir / "voter_cache.json"      # line 688 — voter cache
...
cache_path = Path(args.embed_cache) if args.embed_cache else DEFAULT_GEMINI_CACHE_PATH  # line 767 — overwritten!
embed_cache = EmbeddingCache(cache_path, ...)
```

After line 767, `cache_path` no longer points to `voter_cache.json` — it points to the embedding cache. The voter cache is read correctly before this line (line 755), so there is no crash today. But any code added between lines 755 and 767 that references `cache_path` expecting the voter cache will silently read/check the embedding cache path instead.

**Fix:** Use distinct names:

```python
voter_cache_path = primer_dir / "voter_cache.json"
...
embed_cache_path = Path(args.embed_cache) if args.embed_cache else DEFAULT_GEMINI_CACHE_PATH
embed_cache = EmbeddingCache(embed_cache_path, ...)
```

---

## BUG 2 — Same model at two temperatures is not meaningful voter diversity [HIGH]

**Files:** `scripts/run_paper.py:90-101`, `eval/silver/map_theta_sweep.py:87-95`

```python
# L1 in both production and sweep
VoterBatchConfig("gemini-2.5-flash-lite", provider="gemini", temperature=0.1),
VoterBatchConfig("gemini-2.5-flash-lite", provider="gemini", temperature=0.3),  # same model!
VoterBatchConfig("claude-haiku-4-5-20251001", provider="claude", temperature=0.1),

# L2 in both production and sweep
VoterBatchConfig("gemini-2.5-flash",          provider="gemini", temperature=0.1),
VoterBatchConfig("gemini-2.5-flash",          provider="gemini", temperature=0.3),  # same model!
VoterBatchConfig("claude-haiku-4-5-20251001", provider="claude", temperature=0.3),
```

2 out of 3 voters at each level are the same model at different temperatures. Same-model voters are highly correlated — they produce nearly identical outputs and will always "agree" with each other regardless of whether the answer is correct. This means:

1. The agreement score is inflated — 2/3 voters always agree by construction.
2. The theta calibration plateau observed at 0.40–0.70 is caused by this: same-model pairs score above 0.40 for almost all chunks.
3. An escalation that _should_ happen (because the answer is wrong) is suppressed by the artificial same-model agreement.

The whole point of the cascade is to use cheap models first and only escalate when they genuinely disagree — which requires architectural diversity.

**Fix:** Use 2 distinct models per level (1 Gemini + 1 Claude Haiku), or add OpenAI/Mistral as a third distinct architecture:

```python
# L1 — 2 distinct architectures
VoterBatchConfig("gemini-2.5-flash-lite",     provider="gemini", temperature=0.1),
VoterBatchConfig("claude-haiku-4-5-20251001", provider="claude", temperature=0.1),

# L2 — 2 distinct architectures
VoterBatchConfig("gemini-2.5-flash",          provider="gemini", temperature=0.1),
VoterBatchConfig("claude-haiku-4-5-20251001", provider="claude", temperature=0.3),
```

After changing voter configs, re-run the theta sweep to recalibrate (zero API cost if `voter_cache.json` already exists — replay with the new voter index subset).

---

## BUG 3 — Baseline cost in escalation report overcounts by voter count [MEDIUM]

**File:** `scripts/run_paper.py:283-284`

```python
l1_inp = usage.get("l1", {}).get("input", 0)   # TOTAL across ALL L1 voters
l1_out = usage.get("l1", {}).get("output", 0)
baseline_cost = (l1_inp * _INPUT_PRICE_PER_M["l3"] + l1_out * _OUTPUT_PRICE_PER_M["l3"]) / 1_000_000
```

`l1_inp` is the sum of input tokens across ALL L1 voters (3 voters × prompt length). The "all-L3 baseline" scenario uses exactly 1 L3 voter per chunk. So the baseline inflates token counts by ×3, making cost savings appear ~3× larger than reality.

**Fix:** Divide by the voter count for a fair per-chunk baseline:

```python
n_l1_voters = 3  # or pass as parameter
baseline_inp = l1_inp / n_l1_voters
baseline_out = l1_out / n_l1_voters
baseline_cost = (baseline_inp * _INPUT_PRICE_PER_M["l3"] + baseline_out * _OUTPUT_PRICE_PER_M["l3"]) / 1_000_000
```

Or: store the voter count in `BatchHandle.token_usage` so `_escalation_stats` doesn't need to hardcode it.

---

## BUG 4 — L3 model name inconsistency: sweep uses `claude-sonnet-4-6-20251001`, production uses `claude-sonnet-4-6` [MEDIUM]

**Files:** `eval/silver/map_theta_sweep.py:96`, `scripts/run_paper.py:102`

```python
# Sweep
L3 = VoterBatchConfig(model="claude-sonnet-4-6-20251001", provider="claude")

# Production
l3_model = VoterBatchConfig("claude-sonnet-4-6", provider="claude")
```

The `_resolve_model` function in `claude_batch.py` only maps Vertex AI version strings (`claude-sonnet-4-6@default`). It does not map `claude-sonnet-4-6`, so that alias passes through to the Anthropic API as-is. This works today because Anthropic accepts the alias, but will break silently when a newer Sonnet is released and the alias is reassigned. It also means the sweep is calibrated against a slightly different effective model than production uses.

**Fix:** Use the explicit date-stamped ID in production:

```python
l3_model = VoterBatchConfig("claude-sonnet-4-6-20251001", provider="claude")
```

---

## BUG 5 — `_run_all_batch` poll loop advances papers sequentially [MEDIUM]

**File:** `scripts/run_paper.py:393-395`

```python
for pmcid in pending:
    handles[pmcid] = runner.advance(handles[pmcid])   # one at a time
```

Each `advance()` makes HTTP requests to Gemini and Anthropic batch APIs to check job status and retrieve results. These are independent per-paper. Sequential advancing serializes these network calls — for 15 papers, the poll cycle is 15× longer than it needs to be.

**Fix:** Use `ThreadPoolExecutor`:

```python
def _advance_one(pmcid: str) -> tuple[str, object]:
    return pmcid, runner.advance(handles[pmcid])

with ThreadPoolExecutor(max_workers=min(len(pending), 8)) as ex:
    for pmcid, handle in ex.map(lambda p: _advance_one(p), pending):
        handles[pmcid] = handle
```

---

## BUG 6 — `build_batch_runner()` is called during `--dry-run` [MEDIUM]

**File:** `scripts/run_paper.py:184`

```python
if args.dry_run:
    runner = build_batch_runner()   # instantiates BatchSummarizationRunner, creates LLM clients
    print(...)
    return
```

`build_batch_runner()` calls `anthropic_direct_chat(...)` which imports `anthropic` and initializes a client. If `ANTHROPIC_API_KEY` is not set, this crashes even in dry-run mode. It also triggers the `GOOGLE_API_KEY` requirement for `GeminiEmbedder`.

**Fix:** Defer the runner construction, or extract the model names from the voter config structs without instantiating them:

```python
if args.dry_run:
    # Build config without API clients
    l1 = [...VoterBatchConfig definitions...]
    l2 = [...VoterBatchConfig definitions...]
    l3 = VoterBatchConfig(...)
    print(f"Mode: {mode}")
    for v in l1: print(f"  L1  {v.model} ...")
    # No API client needed
    return
```

---

## BUG 7 — `_VERTEX_TO_DIRECT` is dead code [MEDIUM]

**File:** `pipeline/stages/summarization/batch/claude_batch.py:23`

```python
_VERTEX_TO_DIRECT: dict[str, str] = {
    "claude-haiku-4-5@20251001": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6@default": "claude-sonnet-4-6-20251001",
}
```

Both haiku and sonnet are configured with direct-API format IDs in `run_paper.py`, so the mapping table is never triggered. Its existence creates a false impression that Vertex AI is already wired up.

**Fix:** Add a comment clarifying it's a forward-compatibility stub:

```python
# Maps Vertex AI version strings to Anthropic direct-API IDs.
# Only needed if using provider="vertex_claude"; currently unused because
# all Claude voters use the direct Anthropic API.
_VERTEX_TO_DIRECT: dict[str, str] = { ... }
```

---

## DESIGN ISSUE 1 — Theta calibrated on a voter setup that no longer exists [HIGH]

The current `MapConfig.theta = 0.80` was calibrated against a 3-voter L1 setup where 2 voters are the same model at different temperatures. After fixing BUG 2 (voter diversity), the agreement score distribution will change significantly — same-model pairs no longer inflate the score, so most chunks will score lower. The calibrated theta of 0.80 will then escalate nearly every chunk to L2/L3, eliminating the cost benefit.

**Action required:** After changing the voter configs (BUG 2 fix), re-run the theta sweep:

```bash
python -m eval.silver.map_theta_sweep sweep --embedder gemini
```

This uses the existing `voter_cache.json` for the L1/L2/L3 outputs already collected, but replays with only the voter indices that correspond to the new 2-voter setup. Zero additional API calls needed.

---

## DESIGN ISSUE 2 — `L3` has no agreement check — it always keeps whatever it returns [MEDIUM]

**File:** `pipeline/stages/summarization/batch/runner.py:415-424`

```python
def _collect_l3(self, handle, raw_results):
    for chunk_id in targets:
        results = by_chunk.get(chunk_id, [])
        parsed = parse_result(results[0], strip_thinking=handle.l3_strip)
        if parsed is not None:
            handle.finalized[chunk_id] = parsed.model_dump()  # accepted unconditionally
```

L3 is a single model (no agreement check possible), so there is no quality signal before finalizing. If the L3 model returns a malformed or hallucinated summary, it is accepted without any filter. The grounding filter (`GroundingFilter`) is applied later in `finalize()`, but that only runs after all chunks are done — there is no per-chunk quality gate at L3.

**Fix:** Apply grounding check immediately inside `_collect_l3` before persisting:

```python
if parsed is not None:
    if self._grounding is not None:
        filtered = self._grounding.filter_findings(parsed)
        handle.finalized[chunk_id] = filtered.model_dump()
    else:
        handle.finalized[chunk_id] = parsed.model_dump()
```

---

## DESIGN ISSUE 3 — Escalation report cost model constants hardcoded for 3-voter setup [LOW]

**File:** `scripts/run_paper.py:249-258`

```python
_INPUT_PRICE_PER_M = {
    "l1": (2 * 0.10  + 1 * 0.80)  / 3,   # ≈ 0.333
    "l2": (2 * 0.15  + 1 * 0.80)  / 3,   # ≈ 0.367
    "l3":  3.00,
}
```

When voter configs change (BUG 2 — 2 voters instead of 3), the blended rate constants are wrong: they still assume 3 voters. The denominators and weights need to be updated whenever voter configs change.

**Fix:** Derive per-level prices from the actual voter configs at runtime rather than hardcoding them.

---

## SUMMARY TABLE

| #   | Severity | File | Issue |
|-----|----------|------|-------|
| 1   | **HIGH** | `map_theta_sweep.py:767` | `cache_path` variable shadow (voter→embedding) |
| 2   | **HIGH** | `run_paper.py:90-101` | Same model at two temperatures ≠ voter diversity |
| 3   | MEDIUM | `run_paper.py:283` | Baseline cost inflated by voter count ×3 |
| 4   | MEDIUM | `run_paper.py:102` | L3 model name alias vs explicit ID inconsistency |
| 5   | MEDIUM | `run_paper.py:393` | Sequential advance in poll loop (should be parallel) |
| 6   | MEDIUM | `run_paper.py:184` | `build_batch_runner()` called in dry-run, fails without keys |
| 7   | MEDIUM | `claude_batch.py:23` | `_VERTEX_TO_DIRECT` is dead code, no Vertex voter exists |
| D1  | **HIGH** | `config.py` | `theta=0.80` calibrated for wrong voter setup — must recalibrate after BUG 2 fix |
| D2  | MEDIUM | `batch/runner.py:415` | L3 results finalized without quality gate |
| D3  | LOW | `run_paper.py:249` | Cost model constants hardcoded for 3-voter setup |
