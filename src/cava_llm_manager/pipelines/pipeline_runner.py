import asyncio
import httpx
import logging
from pydantic import BaseModel
from ..models.registry import get_model
from ..utils.json_utils import extract_json
from ..utils.config import OLLAMA_URL
from .base import PipelineSpec

from typing import Any, TypeAlias

JSONDict: TypeAlias = dict[str, Any]

logger = logging.getLogger(__name__)


def _coerce_report_id(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        report_id = int(value)
    except (TypeError, ValueError):
        return None

    return report_id if report_id > 0 else None


def _merge_batch_metadata(
    parsed: JSONDict | list[Any],
    items: list[str],
    root_name: str,
) -> JSONDict:
    if isinstance(parsed, dict):
        data = dict(parsed)
    elif isinstance(parsed, list):
        data = {root_name: parsed}
    else:
        data = {}

    reports = data.get(root_name, [])

    if isinstance(reports, dict):
        reports = [reports]
    elif not isinstance(reports, list):
        reports = []

    aligned_reports: list[JSONDict | None] = [None] * len(items)
    next_open_slot = 0

    for raw_report in reports:
        report = raw_report if isinstance(raw_report, dict) else {}
        report_id = _coerce_report_id(report.get("report_id"))
        target_index: int | None = None

        if report_id is not None:
            candidate_index = report_id - 1
            if (
                0 <= candidate_index < len(aligned_reports)
                and aligned_reports[candidate_index] is None
            ):
                target_index = candidate_index

        while (
            target_index is None
            and next_open_slot < len(aligned_reports)
            and aligned_reports[next_open_slot] is not None
        ):
            next_open_slot += 1

        if target_index is None:
            if next_open_slot >= len(aligned_reports):
                break
            target_index = next_open_slot

        merged_report = dict(report)
        merged_report["report_id"] = target_index + 1
        merged_report["input_text"] = items[target_index]
        aligned_reports[target_index] = merged_report

    data[root_name] = [
        report
        if report is not None
        else {"report_id": i + 1, "input_text": item}
        for i, (report, item) in enumerate(zip(aligned_reports, items))
    ]

    return data

def extract_ollama_message(data: dict) -> str:

    if "error" in data:
        logger.error("Ollama returned error: %s", data["error"])
        raise RuntimeError(f"Ollama error: {data['error']}")

    if "message" not in data:
        logger.error("Unexpected Ollama response: %s", data)
        raise RuntimeError(
            f"Unexpected Ollama response structure:\n{data}"
        )

    msg = data["message"]

    if "content" not in msg:
        logger.error("Ollama message missing content: %s", data)
        raise RuntimeError(
            f"Ollama response missing message content:\n{data}"
        )

    return msg["content"]

async def retry_async(fn, retries=3):

    for attempt in range(retries):

        try:
            return await fn()

        except Exception as e:
            if attempt == retries - 1:
                logger.exception(
                    "Retry failed after %d attempts",
                    retries
                )
                raise
            logger.warning(
                "Retrying async call (attempt %d/%d): %s",
                attempt + 1,
                retries,
                str(e),
            )
            await asyncio.sleep(1)

async def run_pipeline_batch(
    pipeline: PipelineSpec,
    items: list[str],
    client: httpx.AsyncClient,
    *,
    validate_output: bool,
    attempts: int,
    raise_on_failure: bool
) -> BaseModel | JSONDict | None:
    
    model = get_model(pipeline.model_id)

    logger.debug(
        "Running pipeline batch | pipeline=%s | model=%s | batch_size=%d",
        pipeline.name,
        model.id,
        len(items),
    )

    system_prompt = pipeline.system_prompt_text
    prompt = pipeline.build_prompt(items)

    payload = {
        "model": model.server_label,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    async def call() -> BaseModel | JSONDict:

        logger.debug(
            "Sending request to Ollama | model=%s | batch_size=%d",
            model.server_label,
            len(items),
        )

        r = await client.post(OLLAMA_URL, json=payload, timeout=600)

        data = r.json()

        raw = extract_ollama_message(data)

        try:
            parsed = extract_json(raw)
        except Exception:
            logger.error(
                "Failed to extract JSON from model output | model=%s | preview=%s",
                model.server_label,
                raw[:300],
            )
            raise ValueError("Failed to extract JSON from model output")

        if pipeline.return_schema is not None:
            parsed = _merge_batch_metadata(
                parsed,
                items,
                pipeline.root_collection_name,
            )

        if not validate_output:
            return parsed

        if pipeline.return_schema is None:
            raise ValueError(
                f"Pipeline '{pipeline.name}' has no return_schema"
            )

        return pipeline.return_schema.model_validate(parsed)


    try:
        return await retry_async(call, retries=attempts)

    except Exception:

        logger.exception(
            "Batch failed | pipeline=%s | batch_size=%d",
            pipeline.name,
            len(items),
        )

        if raise_on_failure:
            raise

        return None
    
async def run_pipeline(
    pipeline: PipelineSpec,
    items: list[str],
    batch_size=5,
    workers=4,
    validate_output: bool = True,
    raise_on_failure: bool = True,
    attempts=3
) -> list[BaseModel | None | JSONDict]:

    logger.info(
        "Starting pipeline | pipeline=%s | items=%d | batch_size=%d | workers=%d",
        pipeline.name,
        len(items),
        batch_size,
        workers,
    )

    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]

    logger.debug("Created %d batches", len(batches))

    sem = asyncio.Semaphore(workers)

    async with httpx.AsyncClient() as client:

        async def task(batch_id, batch):

            async with sem:

                logger.debug(
                    "Processing batch %d | size=%d",
                    batch_id,
                    len(batch),
                )

                return await run_pipeline_batch(
                    pipeline,
                    batch,
                    client,
                    validate_output=validate_output,
                    attempts=attempts,
                    raise_on_failure=raise_on_failure
                )   

        tasks = [task(i, b) for i, b in enumerate(batches)]

        results = await asyncio.gather(*tasks)

    logger.info(
        "Pipeline completed | pipeline=%s | batches=%d",
        pipeline.name,
        len(batches),
    )

    return results
