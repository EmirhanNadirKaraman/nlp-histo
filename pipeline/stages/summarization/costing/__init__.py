"""
Cost / usage accounting for the summarization pipeline.

Public surface
--------------
``LLMUsageRecord``     — one normalized record per LLM call.
``UsageCollector``     — thread-safe accumulator + LangChain callback handler.
``PriceBook``          — configurable per-model unit prices.
``write_cost_reports`` — write markdown + JSON cost report from a collector.
``load_records``       — replay JSONL records (used by the offline script).
"""
from .models import LLMUsageRecord, append_records_jsonl, load_records
from .pricing import PriceBook
from .collector import UsageCollector, UsageCallbackHandler
from .report import write_cost_reports, build_report_from_records

__all__ = [
    "LLMUsageRecord",
    "UsageCollector",
    "UsageCallbackHandler",
    "PriceBook",
    "write_cost_reports",
    "build_report_from_records",
    "append_records_jsonl",
    "load_records",
]
