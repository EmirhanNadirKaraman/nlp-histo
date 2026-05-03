"""
Google Gemini Batch API client (direct API, GOOGLE_API_KEY).

Submits all requests as a single inline batch job via google.genai.
Uses response_mime_type='application/json' so the model returns JSON
that parse_result() can handle like any other batch provider.
"""
from __future__ import annotations

import logging
import os

from .models import BatchRequest, BatchResult, ProviderJob

logger = logging.getLogger(__name__)

_ID_SEP = "|||"  # separates job name from custom_id list in output_location


class GeminiBatchProvider:
    """Submit and retrieve batches via the Gemini Batch API (google.genai)."""

    def __init__(self) -> None:
        import google.genai as genai
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def submit(self, requests: list[BatchRequest], openai_tool: dict) -> ProviderJob:  # noqa: ARG002
        if not requests:
            raise ValueError("Cannot submit an empty batch.")

        models = {r.model for r in requests}
        if len(models) > 1:
            raise ValueError(
                f"GeminiBatchProvider requires all requests to use the same model; "
                f"got: {models}. Use separate VoterBatchConfig entries per model."
            )
        model = requests[0].model

        inline_requests = []
        custom_ids = []
        for req in requests:
            system_content: str | None = None
            user_messages: list[dict] = []
            for msg in req.messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    user_messages.append(msg)

            gemini_req: dict = {
                "contents": [
                    {"role": "user", "parts": [{"text": m["content"]}]}
                    for m in user_messages
                ],
                "config": {
                    "response_mime_type": "application/json",
                    "temperature": req.temperature,
                    "max_output_tokens": req.max_tokens,
                },
            }
            if system_content:
                gemini_req["system_instruction"] = {
                    "parts": [{"text": system_content}]
                }

            inline_requests.append(gemini_req)
            custom_ids.append(req.custom_id)

        batch_job = self._client.batches.create(
            model=model,
            src=inline_requests,
            config={"display_name": f"nlp-histo-{model}"},
        )
        logger.info(
            "Gemini batch submitted: %s  (%d requests, model=%s)",
            batch_job.name, len(requests), model,
        )
        return ProviderJob(
            provider="gemini",
            job_id=batch_job.name,
            status="submitted",
            model=model,
            request_count=len(requests),
            output_location=f"{batch_job.name}{_ID_SEP}{','.join(custom_ids)}",
        )

    def check(self, job: ProviderJob) -> ProviderJob:
        batch_job = self._client.batches.get(name=job.job_id)
        state = batch_job.state.name
        if state == "JOB_STATE_SUCCEEDED":
            job.status = "completed"
        elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"):
            job.status = "failed"
        else:
            job.status = "in_progress"
        return job

    def retrieve(self, job: ProviderJob) -> list[BatchResult]:
        _, ids_str = job.output_location.split(_ID_SEP, 1)
        custom_ids = ids_str.split(",")

        batch_job = self._client.batches.get(name=job.job_id)
        inline_responses = batch_job.dest.inlined_responses or []

        results: list[BatchResult] = []
        for i, inline_response in enumerate(inline_responses):
            custom_id = custom_ids[i] if i < len(custom_ids) else f"unknown_{i}"
            if inline_response.error:
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error=str(inline_response.error)
                ))
            elif inline_response.response:
                try:
                    content = inline_response.response.text
                    results.append(BatchResult(custom_id=custom_id, content=content))
                except Exception as exc:
                    results.append(BatchResult(
                        custom_id=custom_id, content=None, error=str(exc)
                    ))
            else:
                results.append(BatchResult(
                    custom_id=custom_id, content=None, error="empty response"
                ))
        return results
