"""
MAP theta sweep harness.

Submits all L1/L2/L3 batch requests for every dev source case simultaneously,
caches the raw voter outputs, then replays different theta values offline using
only the local EmbeddingScorer — no additional API calls needed.

Modes
-----
  prime     Build chunk maps, submit 6 batch jobs (one per provider×model),
            save primer state to eval/data/map_primer/primer.json.
  collect   Check job statuses; when all complete, build voter_cache.json.
            Re-run until it prints "Voter cache written."
  sweep     Load voter_cache.json, replay theta grid, emit CSV to eval/reports/.
  all       prime → poll loop (120 s sleep) → collect → sweep in one shot.

Usage
-----
  python -m eval.silver.map_theta_sweep all
  python -m eval.silver.map_theta_sweep prime
  python -m eval.silver.map_theta_sweep collect
  python -m eval.silver.map_theta_sweep sweep --embedder gemini

Notes
-----
- All three levels (L1, L2, L3) are submitted simultaneously for every case and
  every chunk so that we can replay any theta value without additional API calls.
- reject_theta is fixed at -1.0 during replay (never reject) so the sweep
  isolates the effect of the escalation threshold theta alone.
- EmbeddingScorer uses OpenAI text-embedding-3-small by default; pass
  --embed-fn gemini to use the Gemini embedding model instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

from eval.silver.embedders import GeminiEmbedder
from eval.silver.jsonl_utils import read_jsonl
from eval.silver.matcher import (
    DEFAULT_GEMINI_CACHE_PATH,
    GEMINI_EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    EmbeddingCache,
)
from eval.silver.pipeline_sweep import case_to_file_data, _evaluate_outputs
from eval.silver.schemas import (
    PipelineCaseOutput,
    PipelineFinding,
    SilverCaseResult,
    SourceCase,
)
from eval.silver.split import filter_by_split

# ── Paths ─────────────────────────────────────────────────────────────────────

PRIMER_DIR   = Path("eval/data/map_primer")
PRIMER_PATH  = PRIMER_DIR / "primer.json"
CACHE_PATH   = PRIMER_DIR / "voter_cache.json"
REPORTS_DIR  = Path("eval/reports")
SOURCE_PATH  = Path("eval/data/source_cases.jsonl")
SILVER_PATH  = Path("eval/data/silver_findings.jsonl")

# ── Voter configs (production setup) ─────────────────────────────────────────

def _make_voters():
    from pipeline.stages.summarization.batch.voter_configs import (
        make_l1_voters, make_l2_voters, make_l3_voter,
    )
    return make_l1_voters(), make_l2_voters(), make_l3_voter()

THETA_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

# ── PrimerHandle ──────────────────────────────────────────────────────────────

@dataclass
class PrimerHandle:
    """Serialisable state for the multi-case batch primer."""

    phase: str                           # "submitted" | "complete"
    case_ids: list[str]                  # original case_ids (contain "|")

    # safe_id → {chunk_id → [sentence dicts]}
    chunk_maps: dict[str, dict[str, list[dict]]] = field(default_factory=dict)

    # safe_id → {chunk_id → formatted source text string}
    source_texts: dict[str, dict[str, str]] = field(default_factory=dict)

    # safe_id → {case_id, pmcid, te_id}
    case_meta: dict[str, dict] = field(default_factory=dict)

    # submitted batch jobs
    jobs: list[dict] = field(default_factory=list)  # ProviderJob.to_dict()

    # strip_thinking flags per (level, voter_idx) — stored so retrieve knows how to parse
    l1_strip: list[bool] = field(default_factory=list)
    l2_strip: list[bool] = field(default_factory=list)
    l3_strip: bool = False

    # custom_id → raw content string (accumulated during collect)
    raw: dict[str, str] = field(default_factory=dict)

    # job_ids already retrieved (so we don't re-download)
    retrieved_job_ids: list[str] = field(default_factory=list)

    def save(self, path: Path = PRIMER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _to_dict(self) -> dict:
        return {
            "phase":              self.phase,
            "case_ids":           self.case_ids,
            "chunk_maps":         self.chunk_maps,
            "source_texts":       self.source_texts,
            "case_meta":          self.case_meta,
            "jobs":               self.jobs,
            "l1_strip":           self.l1_strip,
            "l2_strip":           self.l2_strip,
            "l3_strip":           self.l3_strip,
            "raw":                self.raw,
            "retrieved_job_ids":  self.retrieved_job_ids,
        }

    @classmethod
    def load(cls, path: Path = PRIMER_PATH) -> PrimerHandle:
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            phase=d["phase"],
            case_ids=d["case_ids"],
            chunk_maps=d.get("chunk_maps", {}),
            source_texts=d.get("source_texts", {}),
            case_meta=d.get("case_meta", {}),
            jobs=d.get("jobs", []),
            l1_strip=d.get("l1_strip", []),
            l2_strip=d.get("l2_strip", []),
            l3_strip=d.get("l3_strip", False),
            raw=d.get("raw", {}),
            retrieved_job_ids=d.get("retrieved_job_ids", []),
        )


# ── PRIME ─────────────────────────────────────────────────────────────────────

def run_prime(cases: list[SourceCase], primer_path: Path = PRIMER_PATH) -> PrimerHandle:
    """Build chunk maps for all cases, submit all L1+L2+L3 batch jobs, save primer."""
    from pipeline.stages.summarization.batch.dispatch import (
        build_requests,
        build_providers,
        OPENAI_MAP_TOOL,
    )
    from pipeline.stages.summarization.current_stages.map_stage import _format_sentences
    from pipeline.stages.summarization.config import MapConfig

    L1, L2, L3 = _make_voters()

    chunk_size = MapConfig().chunk_size  # default 10

    # Build chunk maps and source texts for every case
    chunk_maps: dict[str, dict[str, list[dict]]] = {}
    source_texts: dict[str, dict[str, str]] = {}
    case_meta: dict[str, dict] = {}

    for case in cases:
        safe_id = case.case_id.replace("|", "_")
        file_data = case_to_file_data(case)
        sentences = file_data["sentences_with_provenance"]

        cmap: dict[str, list[dict]] = {}
        for i in range(0, max(len(sentences), 1), chunk_size):
            cid = f"C{i // chunk_size + 1}"
            cmap[cid] = sentences[i : i + chunk_size]
        if not cmap:
            cmap["C1"] = []

        stexts = {cid: _format_sentences(sents) for cid, sents in cmap.items()}
        chunk_maps[safe_id] = cmap
        source_texts[safe_id] = stexts
        case_meta[safe_id] = {
            "case_id": case.case_id,
            "pmcid":   case.pmcid,
            "te_id":   case.te_id,
        }

    logger.info("Built chunk maps for %d cases", len(cases))

    # Build all requests grouped by (provider, model)
    from pipeline.stages.summarization.batch.models import BatchRequest
    by_prov_model: dict[tuple[str, str], list[BatchRequest]] = {}

    for safe_id, cmap in chunk_maps.items():
        for level, voters in [("l1", L1), ("l2", L2), ("l3", [L3])]:
            reqs = build_requests(cmap, safe_id, voters, level=level)
            for req in reqs:
                key = (req.provider, req.model)
                by_prov_model.setdefault(key, []).append(req)

    total_reqs = sum(len(v) for v in by_prov_model.values())
    logger.info(
        "Total requests: %d across %d (provider, model) groups",
        total_reqs, len(by_prov_model),
    )
    for (prov, model), reqs in by_prov_model.items():
        logger.info("  %s / %s  → %d requests", prov, model, len(reqs))

    # Build handle first so we can save incrementally after each submission.
    handle = PrimerHandle(
        phase="submitted",
        case_ids=[c.case_id for c in cases],
        chunk_maps=chunk_maps,
        source_texts=source_texts,
        case_meta=case_meta,
        l1_strip=[v.strip_thinking for v in L1],
        l2_strip=[v.strip_thinking for v in L2],
        l3_strip=L3.strip_thinking,
    )
    handle.save(primer_path)  # write skeleton so partial progress survives a crash

    providers = build_providers({k[0] for k in by_prov_model})
    already_submitted = {j["job_id"] for j in handle.jobs}  # empty on first run

    for (prov, model), reqs in by_prov_model.items():
        job = providers[prov].submit(reqs, OPENAI_MAP_TOOL)
        logger.info("Submitted %s/%s: job_id=%s", prov, model, job.job_id)
        handle.jobs.append(job.to_dict())
        handle.save(primer_path)  # persist after each successful submission
    handle.save(primer_path)
    logger.info("Primer state saved → %s", primer_path)
    return handle


# ── COLLECT ───────────────────────────────────────────────────────────────────

def _strip_flag(custom_id: str, handle: PrimerHandle) -> bool:
    """Determine strip_thinking flag from custom_id level and voter index."""
    parts = custom_id.split("__")
    if len(parts) < 4:
        return False
    level, vi_str = parts[-2], parts[-1]
    vi = int(vi_str) if vi_str.isdigit() else 0
    if level == "l1":
        return handle.l1_strip[vi] if vi < len(handle.l1_strip) else False
    if level == "l2":
        return handle.l2_strip[vi] if vi < len(handle.l2_strip) else False
    return handle.l3_strip


def run_collect(
    handle: PrimerHandle,
    primer_path: Path = PRIMER_PATH,
    cache_path: Path = CACHE_PATH,
) -> tuple[PrimerHandle, bool]:
    """
    Check all jobs; retrieve results for completed jobs.
    Returns (updated_handle, is_complete).
    """
    from pipeline.stages.summarization.batch.dispatch import build_providers
    from pipeline.stages.summarization.batch.models import ProviderJob

    jobs = [ProviderJob.from_dict(d) for d in handle.jobs]
    providers_needed = {j.provider for j in jobs}
    providers = build_providers(providers_needed)

    # Refresh statuses
    for job in jobs:
        if job.status in ("completed", "failed"):
            continue
        p = providers.get(job.provider)
        if p:
            p.check(job)

    # Retrieve results for newly completed jobs
    for job in jobs:
        if job.job_id in handle.retrieved_job_ids:
            continue
        if job.status == "completed":
            p = providers.get(job.provider)
            if p:
                results = p.retrieve(job)
                for res in results:
                    if res.content:
                        handle.raw[res.custom_id] = res.content
                handle.retrieved_job_ids.append(job.job_id)
                logger.info("Retrieved %d results from job %s", len(results), job.job_id)
        elif job.status == "failed":
            logger.warning("Job %s failed", job.job_id)
            handle.retrieved_job_ids.append(job.job_id)  # mark as done to avoid re-checking

    # Update serialised job list with refreshed statuses
    handle.jobs = [j.to_dict() for j in jobs]

    pending = [j for j in jobs if j.status not in ("completed", "failed")]
    done    = [j for j in jobs if j.status in ("completed", "failed")]
    logger.info("%d/%d jobs complete", len(done), len(jobs))

    all_done = len(pending) == 0
    if all_done:
        handle.phase = "complete"
        handle.save(primer_path)  # raw content persisted before cache build
        _build_voter_cache(handle, cache_path)
    else:
        handle.save(primer_path)

    return handle, all_done


def rebuild_cache_from_primer(primer_path: Path = PRIMER_PATH,
                               cache_path: Path = CACHE_PATH) -> None:
    """Rebuild voter_cache.json from a saved primer without any API calls.

    Use this after a parse failure in collect — the raw content is already
    in primer.json, so no jobs need to be re-submitted or re-retrieved.
    """
    handle = PrimerHandle.load(primer_path)
    _build_voter_cache(handle, cache_path)


def _build_voter_cache(handle: PrimerHandle, cache_path: Path = CACHE_PATH) -> None:
    """Parse handle.raw into a per-case voter cache and write to CACHE_PATH."""
    from pipeline.stages.summarization.batch.dispatch import parse_result
    from pipeline.stages.summarization.batch.models import BatchResult

    cache: dict[str, dict] = {}

    for safe_id, meta in handle.case_meta.items():
        cmap = handle.chunk_maps.get(safe_id, {})
        stexts = handle.source_texts.get(safe_id, {})
        entry: dict[str, Any] = {
            "case_id":      meta["case_id"],
            "pmcid":        meta["pmcid"],
            "te_id":        meta["te_id"],
            "source_texts": stexts,
            "chunk_map":    cmap,
            "l1":           {cid: [] for cid in cmap},
            "l2":           {cid: [] for cid in cmap},
            "l3":           {cid: None for cid in cmap},
        }
        cache[safe_id] = entry

    # Parse every raw result into the appropriate slot
    for custom_id, content in handle.raw.items():
        parts = custom_id.split("__")
        if len(parts) != 4:
            logger.debug("Skipping malformed custom_id: %s", custom_id)
            continue
        safe_id, chunk_id, level, vi_str = parts
        if safe_id not in cache:
            logger.debug("Unknown safe_id in custom_id: %s", custom_id)
            continue

        strip = _strip_flag(custom_id, handle)
        parsed = parse_result(
            BatchResult(custom_id=custom_id, content=content),
            strip_thinking=strip,
        )
        if parsed is None:
            logger.warning("Failed to parse %s", custom_id)
            continue

        vi = int(vi_str) if vi_str.isdigit() else 0
        entry = cache[safe_id]

        if level == "l3":
            entry["l3"][chunk_id] = parsed.model_dump()
        elif level in ("l1", "l2"):
            slot: list = entry[level].setdefault(chunk_id, [])
            # Expand list to hold slot for voter vi
            while len(slot) <= vi:
                slot.append(None)
            slot[vi] = parsed.model_dump()

    # Count parsed outputs
    n_l1 = sum(
        sum(1 for v in cmap.values() for x in (v if isinstance(v, list) else []) if x)
        for cmap in [entry["l1"] for entry in cache.values()]
    )
    logger.info("Voter cache: %d cases, %d L1 outputs parsed", len(cache), n_l1)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Voter cache written → %s", cache_path)


# ── SWEEP ─────────────────────────────────────────────────────────────────────

def _finding_to_pipeline(f: dict, pmcid: str, chunk_id: str, theta: float) -> PipelineFinding:
    scope = f.get("scope") or {}
    return PipelineFinding(
        pipeline_run_id=0,
        run_id=f"map_theta_{theta:.2f}",
        pmcid=pmcid,
        chunk_id=chunk_id,
        category=f.get("category", ""),
        claim=f.get("claim", ""),
        subject_entity=f.get("subject_entity"),
        outcome_entity=f.get("outcome_entity"),
        relation_type=str(f.get("relation_type", "unclear")),
        direction=str(f["direction"]) if f.get("direction") else None,
        confidence=str(f.get("confidence", "medium")),
        verbatim_support=f.get("verbatim_support", ""),
        grounding_score=f.get("grounding_score"),
        scope_disease_subtype=scope.get("disease_subtype"),
        scope_cohort_n=scope.get("cohort_n"),
        scope_assay_method=scope.get("assay_method"),
        scope_biomarker_cutoff=scope.get("biomarker_cutoff"),
        scope_tissue_site=scope.get("tissue_site"),
        scope_treatment_context=scope.get("treatment_context"),
        scope_endpoint=scope.get("endpoint"),
        scope_study_design=scope.get("study_design"),
    )


def _prewarm_agreement_cache(
    voter_cache: dict[str, dict],
    embed_fn,
    disk_cache: "EmbeddingCache",
    batch_size: int = 100,
) -> None:
    """Extract every unique claim string from the voter cache and embed any misses.

    Results are written into disk_cache (persistent JSON file) so subsequent
    sweep runs skip this step entirely.  disk_cache.save() is called after each
    batch so progress survives a crash.
    """
    from pipeline.stages.summarization.agreement.embedding import _claims
    from pipeline.stages.summarization.models import AuditableSummary

    texts: set[str] = set()
    for entry in voter_cache.values():
        for level in ("l1", "l2"):
            for data in entry.get(level, {}).values():
                for d in (data or []):
                    if d:
                        try:
                            s = AuditableSummary.model_validate(d)
                            texts.update(_claims(s))
                        except Exception:
                            pass
        for d in entry.get("l3", {}).values():
            if d:
                try:
                    s = AuditableSummary.model_validate(d)
                    texts.update(_claims(s))
                except Exception:
                    pass

    misses = [t for t in sorted(texts) if disk_cache.get(t) is None]
    logger.info(
        "Agreement embed pre-warm: %d unique claims, %d cache misses",
        len(texts), len(misses),
    )
    for i in range(0, len(misses), batch_size):
        batch = misses[i : i + batch_size]
        embs = embed_fn(batch)
        for t, e in zip(batch, embs):
            disk_cache.set(t, e)
        disk_cache.save()
        logger.info("  embedded %d / %d", min(i + batch_size, len(misses)), len(misses))
    logger.info("Agreement embed pre-warm complete")


def _make_cached_embed_fn(disk_cache: "EmbeddingCache", embed_fn):
    """Wrap embed_fn with disk_cache so any remaining misses are persisted."""
    def _fn(texts: list[str]) -> list[list[float]]:
        result: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, t in enumerate(texts):
            cached = disk_cache.get(t)
            if cached is not None:
                result[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            new_embs = embed_fn(miss_texts)
            for idx, t, e in zip(miss_idx, miss_texts, new_embs):
                disk_cache.set(t, e)
                result[idx] = e
            disk_cache.save()
        return result  # type: ignore[return-value]
    return _fn


def _replay_theta(
    voter_cache: dict[str, dict],
    theta: float,
    embed_fn=None,
) -> list[PipelineCaseOutput]:
    """Apply theta to cached voter outputs; return PipelineCaseOutput per case."""
    from pipeline.stages.summarization.agreement import AgreementChecker, EmbeddingScorer
    from pipeline.stages.summarization.interfaces.scoring import ChunkDecision
    from pipeline.stages.summarization.models import AuditableSummary

    checker = AgreementChecker(EmbeddingScorer(embed_fn), theta=theta, reject_theta=-1.0)
    outputs: list[PipelineCaseOutput] = []

    for safe_id, entry in voter_cache.items():
        case_id  = entry["case_id"]
        pmcid    = entry["pmcid"]
        te_id    = entry["te_id"]
        findings: list[PipelineFinding] = []

        for chunk_id in entry.get("chunk_map", {}):
            source_text = entry.get("source_texts", {}).get(chunk_id, "")

            l1_raw = entry.get("l1", {}).get(chunk_id) or []
            l1_out = [AuditableSummary.model_validate(d) for d in l1_raw if d]

            selected: dict | None = None
            escalate_to_l2 = False

            if not l1_out:
                escalate_to_l2 = True  # no L1 data → skip to L2/L3
            else:
                bundle = checker.compute(l1_out, source_text=source_text)
                if bundle.decision == ChunkDecision.KEEP:
                    selected = checker.best(l1_out, bundle).model_dump()
                elif bundle.decision == ChunkDecision.REJECT:
                    pass  # discard chunk; selected stays None
                else:
                    escalate_to_l2 = True

            if escalate_to_l2:
                l2_raw = entry.get("l2", {}).get(chunk_id) or []
                l2_out = [AuditableSummary.model_validate(d) for d in l2_raw if d]
                if l2_out:
                    bundle2 = checker.compute(l2_out, source_text=source_text)
                    if bundle2.decision == ChunkDecision.KEEP:
                        selected = checker.best(l2_out, bundle2).model_dump()
                    elif bundle2.decision == ChunkDecision.REJECT:
                        pass  # discard chunk
                    else:
                        # ESCALATE to L3
                        l3_raw = entry.get("l3", {}).get(chunk_id)
                        if l3_raw:
                            selected = l3_raw
                else:
                    # No L2 data — fall to L3
                    l3_raw = entry.get("l3", {}).get(chunk_id)
                    if l3_raw:
                        selected = l3_raw

            if selected is not None:
                for f in selected.get("findings", []):
                    findings.append(_finding_to_pipeline(f, pmcid, chunk_id, theta))

        outputs.append(PipelineCaseOutput(
            case_id=case_id,
            pmcid=pmcid,
            te_id=te_id,
            run_id=f"map_theta_{theta:.2f}",
            pipeline_run_id=0,
            findings=findings,
        ))

    return outputs


def run_sweep(
    voter_cache: dict[str, dict],
    silver_by_case: dict[str, SilverCaseResult],
    embedder: object,
    embed_cache: EmbeddingCache,
    sim_threshold: float,
    thetas: list[float],
    split: str,
    seed: int,
    dev_fraction: float,
    agreement_embed_fn=None,
) -> list[dict]:
    rows: list[dict] = []
    for theta in thetas:
        case_outputs = _replay_theta(voter_cache, theta, embed_fn=agreement_embed_fn)
        # Filter to only cases present in silver (dev split already filtered at load time)
        case_outputs = [co for co in case_outputs if co.case_id in silver_by_case]
        metrics = _evaluate_outputs(case_outputs, silver_by_case, embedder, embed_cache, sim_threshold)
        row = {
            "theta":         round(theta, 2),
            "split":         split,
            "seed":          seed,
            "dev_fraction":  dev_fraction,
            "sim_threshold": sim_threshold,
            **metrics,
        }
        rows.append(row)
        logger.info(
            "  theta=%.2f  P=%.3f  R=%.3f  F1=%.3f  strict_F1=%.3f  "
            "matched=%d / silver=%d / pipeline=%d",
            theta,
            metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["strict_f1"],
            metrics["n_matched"], metrics["n_silver"], metrics["n_pipeline"],
        )
    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows → %s", len(rows), path)


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    best = max(rows, key=lambda r: float(r["f1"]))
    print(f"\n{'─'*72}")
    print("MAP Theta Sweep")
    print(f"{'─'*72}")
    print(f"{'Theta':>6}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  "
          f"{'Strict F1':>9}  {'Pipeline':>8}")
    print(f"{'─'*72}")
    for r in rows:
        marker = " ← best F1" if r["theta"] == best["theta"] else ""
        print(f"{float(r['theta']):>6.2f}  "
              f"{float(r['precision']):>9.3f}  {float(r['recall']):>7.3f}  "
              f"{float(r['f1']):>7.3f}  {float(r['strict_f1']):>9.3f}  "
              f"{int(r['n_pipeline']):>8}{marker}")
    print(f"{'─'*72}")
    print(f"\nBest F1: theta={best['theta']:.2f}  F1={float(best['f1']):.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MAP theta sweep harness")
    parser.add_argument("mode", choices=["prime", "collect", "rebuild-cache", "sweep", "all"])
    parser.add_argument("--source",        default=str(SOURCE_PATH))
    parser.add_argument("--silver",        default=str(SILVER_PATH))
    parser.add_argument("--reports",       default=str(REPORTS_DIR))
    parser.add_argument("--embedder",      default="openai", choices=["openai", "gemini"])
    parser.add_argument("--embed-cache",   default=None)
    parser.add_argument("--sim-threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--split",         default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--dev-fraction",  type=float, default=0.8)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls in collect/all mode")
    parser.add_argument("--n-cases", type=int, default=None,
                        help="Limit to first N cases (for smoke testing)")
    parser.add_argument("--primer-dir", default=None,
                        help="Override primer/cache directory (default: eval/data/map_primer)")
    args = parser.parse_args()

    source_path = Path(args.source)
    silver_path = Path(args.silver)
    reports_dir = Path(args.reports)
    timestamp   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    primer_dir       = Path(args.primer_dir) if args.primer_dir else PRIMER_DIR
    primer_path      = primer_dir / "primer.json"
    voter_cache_path = primer_dir / "voter_cache.json"

    # ── Load source cases ──
    if not source_path.exists():
        print(f"source_cases.jsonl not found: {source_path}", file=sys.stderr)
        sys.exit(1)
    all_cases = list(read_jsonl(source_path, SourceCase))
    filtered_cases = filter_by_split(
        all_cases, args.split, dev_fraction=args.dev_fraction, seed=args.seed,
    )
    if args.n_cases is not None:
        filtered_cases = filtered_cases[: args.n_cases]
    logger.info("Split=%s  cases=%d  (seed=%d, dev_fraction=%.2f)",
                args.split, len(filtered_cases), args.seed, args.dev_fraction)

    # ── PRIME ──
    if args.mode in ("prime", "all"):
        if primer_path.exists():
            logger.info("Primer already exists at %s — loading", primer_path)
            handle = PrimerHandle.load(primer_path)
        else:
            handle = run_prime(filtered_cases, primer_path)

    # ── REBUILD CACHE (no API calls) ──
    if args.mode == "rebuild-cache":
        if not primer_path.exists():
            print(f"No primer found at {primer_path}.", file=sys.stderr)
            sys.exit(1)
        rebuild_cache_from_primer(primer_path, voter_cache_path)
        print("Voter cache rebuilt from saved primer. Ready to sweep.")
        return

    # ── COLLECT / poll loop ──
    if args.mode == "collect":
        if not primer_path.exists():
            print("No primer found. Run `prime` first.", file=sys.stderr)
            sys.exit(1)
        handle = PrimerHandle.load(primer_path)
        while True:
            handle, done = run_collect(handle, primer_path, voter_cache_path)
            if done:
                print("Voter cache written. Ready to sweep.")
                break
            from pipeline.stages.summarization.batch.models import ProviderJob
            for jd in handle.jobs:
                j = ProviderJob.from_dict(jd)
                logger.info("  job %s  provider=%s  status=%s", j.job_id, j.provider, j.status)
            logger.info("Jobs still running — sleeping %ds…", args.poll_interval)
            time.sleep(args.poll_interval)

    if args.mode == "all":
        while handle.phase != "complete":
            logger.info("Polling jobs… (sleeping %ds)", args.poll_interval)
            time.sleep(args.poll_interval)
            handle, done = run_collect(handle, primer_path, voter_cache_path)
            if done:
                logger.info("All jobs complete. Voter cache ready.")
                break
            # Show job statuses
            from pipeline.stages.summarization.batch.models import ProviderJob
            for jd in handle.jobs:
                j = ProviderJob.from_dict(jd)
                logger.info("  job %s  provider=%s  status=%s", j.job_id, j.provider, j.status)

    # ── SWEEP ──
    if args.mode in ("sweep", "all"):
        if not voter_cache_path.exists():
            print(f"Voter cache not found: {voter_cache_path}. Run prime + collect first.", file=sys.stderr)
            sys.exit(1)
        if not silver_path.exists():
            print(f"silver_findings.jsonl not found: {silver_path}", file=sys.stderr)
            sys.exit(1)

        voter_cache = json.loads(voter_cache_path.read_text(encoding="utf-8"))

        silver_by_case: dict[str, SilverCaseResult] = {}
        for rec in read_jsonl(silver_path, SilverCaseResult):
            silver_by_case[rec.case_id] = rec

        # Build embedder (for eval matching) and agreement embed_fn (for EmbeddingScorer)
        if args.embedder == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                print("GOOGLE_API_KEY not set", file=sys.stderr); sys.exit(1)
            embedder = GeminiEmbedder(api_key)
            embed_cache_path = Path(args.embed_cache) if args.embed_cache else DEFAULT_GEMINI_CACHE_PATH
            embed_cache = EmbeddingCache(embed_cache_path, GEMINI_EMBEDDING_MODEL)
            from pipeline.stages.summarization.agreement.providers import GeminiEmbedder as AgreementGeminiEmbedder
            _raw_agreement_fn = AgreementGeminiEmbedder()
        else:
            from eval.silver.embedders import OpenAIEmbedder
            from eval.silver.matcher import DEFAULT_CACHE_PATH, EMBEDDING_MODEL
            from pipeline.stages.summarization.agreement.providers import OpenAIEmbedder as AgreementOpenAIEmbedder
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("OPENAI_API_KEY not set", file=sys.stderr); sys.exit(1)
            embedder = OpenAIEmbedder(api_key)
            embed_cache_path = Path(args.embed_cache) if args.embed_cache else DEFAULT_CACHE_PATH
            embed_cache = EmbeddingCache(embed_cache_path, EMBEDDING_MODEL)
            _raw_agreement_fn = AgreementOpenAIEmbedder()

        # Pre-compute all agreement embeddings upfront so the theta replay loop
        # makes zero API calls (same texts are used for all 7 theta values).
        # Reuses the same disk cache as eval matching — keys are sha256(model+text)
        # so agreement claims and eval finding strings coexist without collision.
        _prewarm_agreement_cache(voter_cache, _raw_agreement_fn, embed_cache)
        agreement_embed_fn = _make_cached_embed_fn(embed_cache, _raw_agreement_fn)

        sweep_rows = run_sweep(
            voter_cache=voter_cache,
            silver_by_case=silver_by_case,
            embedder=embedder,
            embed_cache=embed_cache,
            sim_threshold=args.sim_threshold,
            thetas=THETA_GRID,
            split=args.split,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            agreement_embed_fn=agreement_embed_fn,
        )
        if sweep_rows:
            csv_path = reports_dir / f"map_theta_sweep_{timestamp}.csv"
            _write_csv(csv_path, sweep_rows, fieldnames=[
                "theta", "precision", "recall", "f1", "strict_f1",
                "n_matched", "n_silver", "n_pipeline",
                "split", "seed", "dev_fraction", "sim_threshold",
            ])
            _print_table(sweep_rows)


if __name__ == "__main__":
    main()
