# Summarization Pipeline — Bugs, Logical Mistakes & Fixes

Audit of the MAP batch pipeline and related infrastructure.

---

## DESIGN ISSUE 1 — Theta needs recalibration after voter config change [HIGH]

The previous voter setup had 2× same model at different temperatures per level, which inflated agreement scores artificially. The voter configs have been fixed (gemini + openai + claude per level), but `MapConfig.theta = 0.65` was not calibrated against this new setup. The agreement score distribution will be different — same-model pairs no longer inflate the score, so chunks will score lower on average and the optimal theta will shift.

**Action required:** Re-run the sweep against the new voter outputs:

```bash
python -m eval.silver.map_theta_sweep all
```

---

## DESIGN ISSUE 2 — Escalation report cost model constants hardcoded [LOW]

**File:** `scripts/run_paper.py:249-258`

When voter configs change, the blended rate constants need manual updating. The denominators and model prices are hardcoded.

**Fix:** Derive per-level prices from the actual voter configs at runtime rather than hardcoding them.

---

## SUMMARY TABLE

| #   | Severity | File | Issue |
|-----|----------|------|-------|
| D1  | **HIGH** | `config.py` | Theta needs recalibration for new voter setup |
| D2  | LOW | `run_paper.py:249` | Cost model constants hardcoded |
