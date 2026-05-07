"""
Shared helpers: prompt formatting, tool schema, provider instantiation, result parsing.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate

from ..models import AuditableSummary
from ..prompts import _MAP_SYSTEM, _MAP_USER
from .models import BatchRequest, BatchResult, ProviderJob, VoterBatchConfig

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MAP_PROMPT = ChatPromptTemplate([("system", _MAP_SYSTEM), ("user", _MAP_USER)])


def _build_openai_tool() -> dict:
    from langchain_core.utils.function_calling import convert_to_openai_tool
    tool = convert_to_openai_tool(AuditableSummary)
    tool["function"]["strict"] = True
    return tool


# Computed once at import time — shared across all batch providers
OPENAI_MAP_TOOL: dict = _build_openai_tool()


def format_messages(pmcid: str, chunk_id: str, text: str) -> list[dict]:
    """Render the MAP prompt into a list of OpenAI-style role/content dicts."""
    rendered = _MAP_PROMPT.invoke({"pmcid": pmcid, "chunk_id": chunk_id, "text": text})
    out = []
    for msg in rendered.messages:
        role = {"system": "system", "human": "user", "ai": "assistant"}.get(
            msg.type, msg.type
        )
        out.append({"role": role, "content": msg.content})
    return out


def build_requests(
    chunk_map: dict[str, list[dict]],
    pmcid: str,
    voter_configs: list[VoterBatchConfig],
    level: str,
    chunk_ids: list[str] | None = None,
) -> list[BatchRequest]:
    """
    Create a BatchRequest for every (chunk, voter) combination.

    Parameters
    ----------
    chunk_map:   chunk_id → list of sentence dicts
    pmcid:       paper ID
    voter_configs: ordered list of voter configs for this level
    level:       "l1" | "l2" | "l3"
    chunk_ids:   subset of chunks to process; None = all chunks
    """
    from ..current_stages.map_stage import _format_sentences  # module-private helper

    targets = chunk_ids if chunk_ids is not None else list(chunk_map.keys())
    requests: list[BatchRequest] = []
    for chunk_id in targets:
        sentences = chunk_map[chunk_id]
        text = _format_sentences(sentences)
        messages = format_messages(pmcid, chunk_id, text)
        for vi, cfg in enumerate(voter_configs):
            requests.append(BatchRequest(
                custom_id=f"{pmcid}__{chunk_id}__{level}__{vi}",
                messages=messages,
                model=cfg.model,
                provider=cfg.provider,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            ))
    return requests


def parse_result(result: BatchResult, strip_thinking: bool = False) -> AuditableSummary | None:
    """Parse a BatchResult's content string into an AuditableSummary."""
    if result.error or not result.content:
        logger.warning("Batch result %s empty/error: %s", result.custom_id, result.error)
        return None

    content = result.content
    if strip_thinking:
        content = _THINK_RE.sub("", content).strip()

    # Strip markdown code fences if present
    if "```" in content:
        m = re.search(r"```(?:json)?\s*(\{.*?})\s*```", content, re.DOTALL)
        if m:
            content = m.group(1)

    try:
        return AuditableSummary.model_validate(json.loads(content))
    except Exception as exc:
        logger.warning("Failed to parse batch result %s: %s", result.custom_id, exc)
        return None


def build_providers(providers_needed: set[str]) -> dict:
    """Lazily instantiate only the required provider clients."""
    providers: dict = {}
    if "azure" in providers_needed:
        from .azure_batch import AzureBatchProvider
        providers["azure"] = AzureBatchProvider()
    if "claude" in providers_needed:
        from .claude_batch import AnthropicBatchProvider
        providers["claude"] = AnthropicBatchProvider()
    if "gemini" in providers_needed:
        from .gemini_batch import GeminiBatchProvider
        providers["gemini"] = GeminiBatchProvider()
    if "vertex_gemini" in providers_needed:
        from .vertex_batch import VertexGeminiBatchProvider
        providers["vertex_gemini"] = VertexGeminiBatchProvider()
    if "openai" in providers_needed:
        from .openai_batch import OpenAIBatchProvider
        providers["openai"] = OpenAIBatchProvider()
    return providers


def submit_level(
    chunk_map: dict[str, list[dict]],
    pmcid: str,
    voter_configs: list[VoterBatchConfig],
    level: str,
    chunk_ids: list[str] | None = None,
) -> list[ProviderJob]:
    """Build requests, group by provider, submit, return list of ProviderJob."""
    all_requests = build_requests(chunk_map, pmcid, voter_configs, level, chunk_ids)
    by_provider: dict[str, list[BatchRequest]] = {}
    for req in all_requests:
        by_provider.setdefault(req.provider, []).append(req)

    providers = build_providers(set(by_provider.keys()))
    jobs: list[ProviderJob] = []
    for pname, reqs in by_provider.items():
        job = providers[pname].submit(reqs, OPENAI_MAP_TOOL)
        jobs.append(job)
    return jobs
