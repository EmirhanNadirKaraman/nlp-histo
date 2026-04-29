"""Call Opus to generate silver findings from source cases."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .jsonl_utils import exists_in_jsonl, read_jsonl
from .prompts import EXTRACT_FINDINGS_TOOL, PROMPT_VERSION, SYSTEM_PROMPT, make_user_prompt
from .schemas import SilverCaseResult, SilverFinding, SilverFindingScope, SourceCase

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_MAX_TOKENS = 4096


def _call_opus(
    text: str,
    path_string: str,
    model: str,
    api_key: str,
) -> list[SilverFinding]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[EXTRACT_FINDINGS_TOOL],
        tool_choice={"type": "tool", "name": "extract_findings"},
        messages=[{"role": "user", "content": make_user_prompt(text, path_string)}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    findings = []
    for item in tool_use.input.get("findings", []):
        try:
            scope_raw = item.pop("scope", {}) or {}
            scope = SilverFindingScope(**{k: v for k, v in scope_raw.items()
                                         if k in SilverFindingScope.model_fields})
            finding = SilverFinding(scope=scope, **{k: v for k, v in item.items()
                                                    if k in SilverFinding.model_fields})
            findings.append(finding)
        except Exception as exc:
            logger.warning("Skipping malformed finding: %s — %s", item, exc)
    return findings


class SilverGenerator:
    def __init__(
        self,
        source_cases_path: Path,
        output_path: Path,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
    ):
        self.source_cases_path = source_cases_path
        self.output_path = output_path
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def run(self) -> None:
        cases = list(read_jsonl(self.source_cases_path, SourceCase))
        logger.info("Loaded %d source cases from %s", len(cases), self.source_cases_path)

        skipped = 0
        generated = 0

        for case in cases:
            cache_key = f"{case.case_id}|{PROMPT_VERSION}|{self.model}"
            if exists_in_jsonl(self.output_path, "_cache_key", cache_key):
                logger.debug("Cache hit: %s", cache_key)
                skipped += 1
                continue

            logger.info("Generating silver labels for %s", case.case_id)
            try:
                findings = _call_opus(case.text, case.path_string, self.model, self.api_key)
            except Exception as exc:
                logger.error("Failed for %s: %s", case.case_id, exc)
                continue

            result = SilverCaseResult(
                case_id=case.case_id,
                pmcid=case.pmcid,
                te_id=case.te_id,
                prompt_version=PROMPT_VERSION,
                model=self.model,
                findings=findings,
            )

            raw_dict = result.model_dump()
            raw_dict["_cache_key"] = cache_key

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a") as fh:
                fh.write(json.dumps(raw_dict) + "\n")

            logger.info("  → %d finding(s)", len(findings))
            generated += 1

        logger.info("Done. Generated: %d  Skipped (cached): %d", generated, skipped)
