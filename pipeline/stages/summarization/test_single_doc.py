"""
Quick smoke-test for SummarizationRunner on a single concept file.

Usage:
    cd /Users/emir/Documents/GitHub/nlp-histo
    python -m pipeline.stages.summarization.test_single_doc
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parents[3]
INPUT_FILE = (
    REPO_ROOT
    / "langchain-summarization"
    / "test_results_50_docs"
    / "relevant_texts"
    / "C0000737_Abdominal_Pain.json"
)

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s  %(name)s  %(message)s",
    stream=sys.stdout,
)
# Silence noisy third-party loggers
for noisy in ("httpx", "openai", "httpcore", "langchain", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ── load input ─────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

file_data = {
    "cui": raw["umls_cui"],
    "concept_name": raw["canonical_name"],
    "sentences_with_provenance": [
        {
            "pmcid": s.get("pmcid"),
            "text_element_id": s.get("text_element_id"),
            "sentence": s.get("sentence"),
        }
        for s in raw.get("sentences", [])
    ],
}

print(f"\nConcept : {file_data['concept_name']}  ({file_data['cui']})")
print(f"Sentences: {len(file_data['sentences_with_provenance'])}\n")

# ── output dirs ────────────────────────────────────────────────────────────────
OUTPUT_DIR        = Path(__file__).parent / "output"
MAP_DIR           = OUTPUT_DIR / "map"
REDUCE_DIR        = OUTPUT_DIR / "reduce"
RULES_DIR         = OUTPUT_DIR / "rules"
CONTRADICTION_DIR = OUTPUT_DIR / "contradictions"
for d in (MAP_DIR, REDUCE_DIR, RULES_DIR, CONTRADICTION_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── build runner ───────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from pipeline.stages.summarization import SummarizationRunner

voter_llms = [
    ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
    ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
]
escalation_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

runner = SummarizationRunner(
    voter_llms=voter_llms,
    escalation_llm=escalation_llm,
    theta=0.6,
    chunk_size=10,
    output_dir=OUTPUT_DIR,
)

# ── run ────────────────────────────────────────────────────────────────────────
result = runner.process(file_data)

# ── save per-stage outputs ─────────────────────────────────────────────────────
cui = file_data["cui"]
if result["status"] == "success":
    (MAP_DIR / f"{cui}.json").write_text(
        json.dumps(result["audit_trail"]["map_chunks"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REDUCE_DIR / f"{cui}.json").write_text(
        json.dumps(result["audit_trail"]["master_summary"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (RULES_DIR / f"{cui}.json").write_text(
        json.dumps(result["audit_trail"]["rules_provenance"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if result["contradiction_report"]:
        (CONTRADICTION_DIR / f"{cui}.json").write_text(
            json.dumps(result["contradiction_report"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

# ── print result ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
if result["status"] == "success":
    print("STATUS : success")
    print(f"SUMMARY:\n{result['summary']}\n")
    print(f"RULES  : {len(result['rules'])}")
    for r in result["rules"]:
        print(f"  [{r['rule_id']}] {r['type']}  conf={r['confidence']}")
        print(f"    {r['condition']}")
        print(f"    {r['action']}")
    print(f"\nOutputs saved to {OUTPUT_DIR}")
else:
    print(f"STATUS : error\n{result['error']}")
print("=" * 70)
