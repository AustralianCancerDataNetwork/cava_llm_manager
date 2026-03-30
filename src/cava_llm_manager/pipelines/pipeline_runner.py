import asyncio
import json
import httpx
import logging
from pydantic import BaseModel

from ..models.registry import get_model
from ..utils.config import OLLAMA_URL
from ..utils.json_utils import extract_json
from .base import (
    ConfidenceBatchResult,
    ConfidenceBatchSummary,
    PipelineSpec,
    ReportConfidenceTrace,
)

from typing import Any, TypeAlias

JSONDict: TypeAlias = dict[str, Any]

logger = logging.getLogger(__name__)

LLAMA_CPP_PROMPT_TEMPLATE = (
    "<|system|>\n"
    "{system_prompt}\n\n"
    "<|user|>\n"
    "{user_prompt}\n\n"
    "<|assistant|>\n"
)


def _coerce_report_id(value: Any) -> int | None:
    # Treat missing, invalid, or non-positive ids as unusable so the caller
    # can fall back to positional alignment within the batch.
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
    # Normalise the parsed payload into a dict keyed by the batch collection
    # name so downstream logic can work with either `{"reports": [...]}` or
    # a bare top-level list.
    if isinstance(parsed, dict):
        data = dict(parsed)
        if root_name not in data:
            raise ValueError(
                f"Parsed model output did not contain expected root key '{root_name}'"
            )
    elif isinstance(parsed, list):
        data = {root_name: parsed}
    else:
        raise ValueError(
            "Parsed model output must be a JSON object or top-level list"
        )

    reports = data[root_name]

    if reports is None:
        reports = []
    elif isinstance(reports, dict):
        reports = [reports]
    elif not isinstance(reports, list):
        raise ValueError(
            f"Parsed model output field '{root_name}' must be a list, object, or null"
        )

    # Stage results in original input order. Missing slots stay as `None`,
    # and are dropped later rather than being synthesized into empty reports.
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

        # If the model omitted a usable report_id, or duplicated one, place
        # the result into the next available batch slot instead.
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

        # Reattach the resolved report position and original input text so the
        # validated output stays traceable back to the batch input.
        merged_report = dict(report)
        merged_report["report_id"] = target_index + 1
        merged_report["input_text"] = items[target_index]
        aligned_reports[target_index] = merged_report

    data[root_name] = [
        report for report in aligned_reports if report is not None
    ]

    return data


def extract_ollama_message(data: dict) -> str:
    # Ollama chat responses wrap the generated text under
    # `message.content`; raise early if that envelope is missing.
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


def extract_llama_cpp_message(data: dict) -> str:
    if "error" in data:
        logger.error("llama.cpp returned error: %s", data["error"])
        raise RuntimeError(f"llama.cpp error: {data['error']}")

    if isinstance(data.get("content"), str):
        return data["content"]

    if isinstance(data.get("response"), str):
        return data["response"]

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and isinstance(first.get("text"), str):
            return first["text"]

    logger.error("Unexpected llama.cpp response: %s", data)
    raise RuntimeError(f"Unexpected llama.cpp response structure:\n{data}")


def _build_ollama_payload(
    model_label: str,
    system_prompt: str,
    user_prompt: str,
    host_options: JSONDict | None = None,
) -> JSONDict:
    options = {"temperature": 0}
    if host_options:
        options.update(host_options)

    return {
        "model": model_label,
        "format": "json",
        "stream": False,
        "options": options,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }


def _build_llama_cpp_prompt(
    pipeline: PipelineSpec,
    system_prompt: str,
    user_prompt: str,
) -> str:
    template = pipeline.prompt_template or LLAMA_CPP_PROMPT_TEMPLATE
    return template.format(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _build_llama_cpp_payload(
    pipeline: PipelineSpec,
    system_prompt: str,
    user_prompt: str,
) -> JSONDict:
    payload: JSONDict = {
        "prompt": _build_llama_cpp_prompt(
            pipeline,
            system_prompt,
            user_prompt,
        ),
        "stream": False,
        "temperature": 0,
        "cache_prompt": True,
        "n_predict": 2048,
    }
    payload.update(pipeline.host_options)
    return payload


def _resolve_pipeline_url(pipeline: PipelineSpec) -> str:
    if pipeline.host_type == "ollama":
        return pipeline.host_url or OLLAMA_URL

    if pipeline.host_type == "llama_cpp":
        if not pipeline.host_url:
            raise ValueError(
                f"Pipeline '{pipeline.name}' requires host_url for llama_cpp"
            )
        return pipeline.host_url

    raise ValueError(
        f"Unsupported host_type '{pipeline.host_type}' for pipeline '{pipeline.name}'"
    )


async def retry_async(fn, retries=3):
    # Keep transient API or parsing failures from failing the entire batch.
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


async def _call_model_json(
    pipeline: PipelineSpec,
    system_prompt: str,
    user_prompt: str,
    client: httpx.AsyncClient,
    *,
    attempts: int,
) -> JSONDict | list[Any]:
    model = get_model(pipeline.model_id)
    url = _resolve_pipeline_url(pipeline)

    if pipeline.host_type == "ollama":
        payload = _build_ollama_payload(
            model.server_label,
            system_prompt,
            user_prompt,
            pipeline.host_options,
        )
        response_extractor = extract_ollama_message
    elif pipeline.host_type == "llama_cpp":
        payload = _build_llama_cpp_payload(
            pipeline,
            system_prompt,
            user_prompt,
        )
        response_extractor = extract_llama_cpp_message
    else:
        raise ValueError(
            f"Unsupported host_type '{pipeline.host_type}' for pipeline '{pipeline.name}'"
        )

    async def call() -> JSONDict | list[Any]:
        response = await client.post(url, json=payload, timeout=600)
        data = response.json()
        raw = response_extractor(data)

        try:
            return extract_json(raw)
        except Exception:
            logger.error(
                "Failed to extract JSON from model output | model=%s | host_type=%s | preview=%s",
                model.server_label,
                pipeline.host_type,
                raw[:300],
            )
            raise ValueError("Failed to extract JSON from model output")

    return await retry_async(call, retries=attempts)


def _validate_pipeline_output(
    pipeline: PipelineSpec,
    parsed: JSONDict | list[Any],
    items: list[str],
) -> BaseModel:
    if pipeline.return_schema is None:
        raise ValueError(
            f"Pipeline '{pipeline.name}' has no return_schema"
        )

    merged = _merge_batch_metadata(
        parsed,
        items,
        pipeline.root_collection_name,
    )

    return pipeline.return_schema.model_validate(merged)


def _build_empty_pipeline_result(
    pipeline: PipelineSpec,
) -> BaseModel | JSONDict | ConfidenceBatchResult:
    if pipeline.return_schema is None:
        return {}

    empty_batch = pipeline.return_schema.model_validate(
        {pipeline.root_collection_name: []}
    )

    if (
        pipeline.confidence is not None
        and pipeline.confidence.enabled
        and pipeline.confidence.return_sidecar
    ):
        return ConfidenceBatchResult(
            result=empty_batch,
            original_result=empty_batch,
        )

    return empty_batch


def _build_confidence_review_item(
    input_text: str,
    report_data: JSONDict,
) -> str:
    return (
        "SOURCE INPUT\n"
        f"{input_text}\n\n"
        "EXTRACTED STRUCTURED OUTPUT\n"
        f"{json.dumps(report_data, indent=2)}\n\n"
        "Assess whether the structured output is faithful to the source input. "
        "Return only JSON matching the configured review schema."
    )


def _build_confidence_revision_item(
    pipeline: PipelineSpec,
    input_text: str,
    report_data: JSONDict,
    issues: list[str],
    rationale: str | None,
) -> str:
    sections = [
        "SOURCE INPUT",
        input_text,
        "",
        "CURRENT STRUCTURED OUTPUT",
        json.dumps(report_data, indent=2),
        "",
        "TARGET REPORT JSON SCHEMA",
        pipeline.serialized_report_schema,
        "",
        "REVIEW ISSUES",
        json.dumps(issues, indent=2),
    ]

    if rationale:
        sections.extend(["", "REVIEW RATIONALE", rationale])

    sections.extend(
        [
            "",
            "Produce a corrected JSON object for this single report only. "
            "Return only JSON.",
        ]
    )

    return "\n".join(sections)


def _extract_collection_items(
    model: BaseModel,
    root_name: str,
) -> list[BaseModel]:
    return list(getattr(model, root_name, []))


async def _run_review_pass(
    review_pipeline: PipelineSpec,
    input_text: str,
    report_data: JSONDict,
    client: httpx.AsyncClient,
    *,
    attempts: int,
) -> BaseModel | None:
    if review_pipeline.return_schema is None:
        raise ValueError(
            "Confidence review requires review_pipeline.return_schema"
        )

    review_item = _build_confidence_review_item(input_text, report_data)
    review_prompt = review_pipeline.build_prompt([review_item])
    parsed = await _call_model_json(
        review_pipeline,
        review_pipeline.system_prompt_text,
        review_prompt,
        client,
        attempts=attempts,
    )

    validated = _validate_pipeline_output(
        review_pipeline,
        parsed,
        [input_text],
    )
    reviews = _extract_collection_items(
        validated,
        review_pipeline.root_collection_name,
    )
    return reviews[0] if reviews else None


def _validate_single_report_output(
    pipeline: PipelineSpec,
    input_text: str,
    report_data: JSONDict,
) -> BaseModel:
    wrapped = {
        pipeline.root_collection_name: [dict(report_data)],
    }
    validated = _validate_pipeline_output(
        pipeline,
        wrapped,
        [input_text],
    )
    reports = _extract_collection_items(validated, pipeline.root_collection_name)

    if not reports:
        raise ValueError("Validated revised output did not contain a report")

    return reports[0]


async def _run_revision_pass(
    pipeline: PipelineSpec,
    revision_pipeline: PipelineSpec,
    input_text: str,
    report_data: JSONDict,
    issues: list[str],
    rationale: str | None,
    client: httpx.AsyncClient,
    *,
    attempts: int,
) -> BaseModel:
    revision_item = _build_confidence_revision_item(
        pipeline,
        input_text,
        report_data,
        issues,
        rationale,
    )
    revision_prompt = (
        revision_pipeline.build_prompt([revision_item])
        if revision_pipeline.return_schema is not None
        else revision_item
    )
    parsed = await _call_model_json(
        revision_pipeline,
        revision_pipeline.system_prompt_text,
        revision_prompt,
        client,
        attempts=attempts,
    )
    if isinstance(parsed, dict) and pipeline.root_collection_name not in parsed:
        parsed = {pipeline.root_collection_name: [parsed]}

    validated = _validate_pipeline_output(
        pipeline,
        parsed,
        [input_text],
    )
    reports = _extract_collection_items(validated, pipeline.root_collection_name)

    if not reports:
        raise ValueError("Validated revised output did not contain a report")

    return reports[0]


async def _apply_confidence_loop(
    pipeline: PipelineSpec,
    batch_result: BaseModel,
    client: httpx.AsyncClient,
    *,
    attempts: int,
    raise_on_failure: bool,
) -> ConfidenceBatchResult:
    confidence = pipeline.confidence
    if confidence is None or not confidence.enabled:
        return ConfidenceBatchResult(
            result=batch_result,
            original_result=batch_result,
        )

    if confidence.review_pipeline is None:
        raise ValueError(
            f"Pipeline '{pipeline.name}' has confidence enabled but no review_pipeline"
        )

    root_name = pipeline.root_collection_name
    original_reports = _extract_collection_items(batch_result, root_name)
    final_reports: list[JSONDict] = []
    traces: list[ReportConfidenceTrace] = []
    summary = ConfidenceBatchSummary()

    for report in original_reports:
        summary.reviewed_reports += 1

        report_dict = report.model_dump(mode="json")
        final_report_dict = dict(report_dict)
        trace = ReportConfidenceTrace(
            report_id=report.report_id,
            input_text=report.input_text,
            original_report=report_dict,
            final_report=final_report_dict,
        )

        try:
            initial_review = await _run_review_pass(
                confidence.review_pipeline,
                report.input_text,
                report_dict,
                client,
                attempts=attempts,
            )

            if initial_review is None:
                raise ValueError("Confidence review returned no result")

            trace.initial_review = initial_review.model_dump(mode="json")
            decision = getattr(initial_review, "decision").value
            issues = list(getattr(initial_review, "issues", []))
            rationale = getattr(initial_review, "rationale", None)

            if decision == "accept":
                summary.accepted_reports += 1

            elif (
                decision == "revise"
                and confidence.acceptance_policy == "revise_once_then_annotate"
                and confidence.attempt_revision
                and confidence.max_revision_rounds > 0
            ):
                revised_report: BaseModel | None = None

                try:
                    if (
                        getattr(initial_review, "corrected_output", None)
                        and isinstance(initial_review.corrected_output, dict)
                    ):
                        revised_report = _validate_single_report_output(
                            pipeline,
                            report.input_text,
                            initial_review.corrected_output,
                        )
                    elif confidence.revision_pipeline is not None:
                        revised_report = await _run_revision_pass(
                            pipeline,
                            confidence.revision_pipeline,
                            report.input_text,
                            report_dict,
                            issues,
                            rationale,
                            client,
                            attempts=attempts,
                        )
                    else:
                        trace.revision_error = (
                            "Revision requested but no revision_pipeline or corrected_output was provided"
                        )
                except Exception as exc:
                    trace.revision_error = str(exc)

                if revised_report is not None:
                    final_report_dict = revised_report.model_dump(mode="json")
                    trace.final_report = final_report_dict
                    trace.revised = True
                    summary.revised_reports += 1

                    try:
                        revision_review = await _run_review_pass(
                            confidence.review_pipeline,
                            report.input_text,
                            final_report_dict,
                            client,
                            attempts=attempts,
                        )
                    except Exception as exc:
                        trace.revision_error = str(exc)
                        revision_review = None

                    if revision_review is not None:
                        trace.revision_review = revision_review.model_dump(mode="json")
                        revision_decision = getattr(revision_review, "decision").value
                        if revision_decision == "accept":
                            summary.accepted_reports += 1
                        else:
                            trace.unresolved = True
                    else:
                        trace.unresolved = True
                else:
                    trace.unresolved = True

            else:
                trace.unresolved = True

        except Exception as exc:
            trace.review_error = str(exc)
            trace.unresolved = True
            summary.review_failures += 1
            if raise_on_failure:
                raise

        if trace.unresolved:
            summary.unresolved_reports += 1

        final_reports.append(final_report_dict)
        traces.append(trace)

    final_batch_data = batch_result.model_dump(mode="json")
    final_batch_data[root_name] = final_reports
    final_result = pipeline.return_schema.model_validate(final_batch_data)

    return ConfidenceBatchResult(
        result=final_result,
        original_result=batch_result,
        report_confidence=traces,
        summary=summary,
    )


async def run_pipeline_batch(
    pipeline: PipelineSpec,
    items: list[str],
    client: httpx.AsyncClient,
    *,
    validate_output: bool,
    attempts: int,
    raise_on_failure: bool
) -> BaseModel | JSONDict | ConfidenceBatchResult | None:

    model = get_model(pipeline.model_id)

    logger.debug(
        "Running pipeline batch | pipeline=%s | model=%s | batch_size=%d",
        pipeline.name,
        model.id,
        len(items),
    )

    if (
        pipeline.confidence is not None
        and pipeline.confidence.enabled
        and not validate_output
    ):
        raise ValueError(
            "Confidence loop requires validate_output=True"
        )

    system_prompt = pipeline.system_prompt_text
    prompt = pipeline.build_prompt(items)

    async def call() -> BaseModel | JSONDict | ConfidenceBatchResult:
        # Send the batch prompt, extract the raw model text, then parse and
        # validate it into the configured return schema.
        logger.debug(
            "Sending request | host_type=%s | model=%s | batch_size=%d",
            pipeline.host_type,
            model.server_label,
            len(items),
        )

        parsed = await _call_model_json(
            pipeline,
            system_prompt,
            prompt,
            client,
            attempts=attempts,
        )

        if pipeline.return_schema is None:
            if validate_output:
                raise ValueError(
                    f"Pipeline '{pipeline.name}' has no return_schema"
                )
            return parsed

        if not validate_output:
            return _merge_batch_metadata(
                parsed,
                items,
                pipeline.root_collection_name,
            )

        validated = _validate_pipeline_output(
            pipeline,
            parsed,
            items,
        )

        if pipeline.confidence is None or not pipeline.confidence.enabled:
            return validated

        reviewed = await _apply_confidence_loop(
            pipeline,
            validated,
            client,
            attempts=attempts,
            raise_on_failure=raise_on_failure,
        )

        if pipeline.confidence.return_sidecar:
            return reviewed

        return reviewed.result

    try:
        return await call()

    except Exception:

        logger.exception(
            "Batch failed | pipeline=%s | batch_size=%d",
            pipeline.name,
            len(items),
        )

        if raise_on_failure:
            raise

        return _build_empty_pipeline_result(pipeline)


async def _run_batch_with_backoff(
    pipeline: PipelineSpec,
    batch: list[str],
    client: httpx.AsyncClient,
    *,
    validate_output: bool,
    attempts: int,
    raise_on_failure: bool,
) -> list[BaseModel | JSONDict | ConfidenceBatchResult | None]:
    try:
        result = await run_pipeline_batch(
            pipeline,
            batch,
            client,
            validate_output=validate_output,
            attempts=attempts,
            raise_on_failure=raise_on_failure,
        )
        return [result]
    except Exception:
        if not raise_on_failure or len(batch) <= 1:
            raise

        next_batch_size = max(1, len(batch) // 2)
        logger.warning(
            "Batch failed; retrying with smaller chunks | pipeline=%s | original_size=%d | next_batch_size=%d",
            pipeline.name,
            len(batch),
            next_batch_size,
        )

        results: list[BaseModel | JSONDict | ConfidenceBatchResult | None] = []
        for i in range(0, len(batch), next_batch_size):
            child_batch = batch[i:i + next_batch_size]
            child_results = await _run_batch_with_backoff(
                pipeline,
                child_batch,
                client,
                validate_output=validate_output,
                attempts=attempts,
                raise_on_failure=raise_on_failure,
            )
            results.extend(child_results)

        if pipeline.return_schema is None:
            return results

        return [
            _reassemble_backoff_results(
                pipeline,
                results,
                validate_output=validate_output,
            )
        ]


def _extract_collection_dicts(
    pipeline: PipelineSpec,
    batch_result: BaseModel | JSONDict | None,
) -> list[JSONDict]:
    if batch_result is None:
        return []

    root_name = pipeline.root_collection_name

    if isinstance(batch_result, BaseModel):
        collection = getattr(batch_result, root_name, [])
        return [
            item.model_dump(mode="json")
            if isinstance(item, BaseModel)
            else dict(item)
            for item in collection
        ]

    if isinstance(batch_result, dict):
        collection = batch_result.get(root_name, [])
        if isinstance(collection, dict):
            collection = [collection]
        if not isinstance(collection, list):
            return []
        return [dict(item) if isinstance(item, dict) else {} for item in collection]

    return []


def _renumber_collection_items(items: list[JSONDict]) -> list[JSONDict]:
    renumbered: list[JSONDict] = []

    for i, item in enumerate(items, start=1):
        merged = dict(item)
        merged["report_id"] = i
        renumbered.append(merged)

    return renumbered


def _combine_batch_results(
    pipeline: PipelineSpec,
    results: list[BaseModel | JSONDict | None],
    *,
    validate_output: bool,
) -> BaseModel | JSONDict:
    combined_items: list[JSONDict] = []

    for result in results:
        combined_items.extend(_extract_collection_dicts(pipeline, result))

    merged = {
        pipeline.root_collection_name: _renumber_collection_items(combined_items)
    }

    if validate_output:
        return pipeline.return_schema.model_validate(merged)

    return merged


def _renumber_review_payload(
    review: JSONDict | None,
    report_id: int,
) -> JSONDict | None:
    if review is None:
        return None

    updated = dict(review)
    updated["report_id"] = report_id
    return updated


def _combine_confidence_results(
    pipeline: PipelineSpec,
    results: list[ConfidenceBatchResult],
    *,
    validate_output: bool,
) -> ConfidenceBatchResult:
    combined_result = _combine_batch_results(
        pipeline,
        [result.result for result in results],
        validate_output=validate_output,
    )
    combined_original_result = _combine_batch_results(
        pipeline,
        [result.original_result for result in results],
        validate_output=validate_output,
    )

    traces: list[ReportConfidenceTrace] = []
    next_report_id = 1

    for result in results:
        for trace in result.report_confidence:
            original_report = dict(trace.original_report)
            original_report["report_id"] = next_report_id

            final_report = dict(trace.final_report)
            final_report["report_id"] = next_report_id

            traces.append(
                ReportConfidenceTrace(
                    report_id=next_report_id,
                    input_text=trace.input_text,
                    original_report=original_report,
                    final_report=final_report,
                    initial_review=_renumber_review_payload(
                        trace.initial_review,
                        next_report_id,
                    ),
                    revision_review=_renumber_review_payload(
                        trace.revision_review,
                        next_report_id,
                    ),
                    revised=trace.revised,
                    unresolved=trace.unresolved,
                    revision_error=trace.revision_error,
                    review_error=trace.review_error,
                )
            )
            next_report_id += 1

    summary = ConfidenceBatchSummary(
        reviewed_reports=sum(result.summary.reviewed_reports for result in results),
        accepted_reports=sum(result.summary.accepted_reports for result in results),
        revised_reports=sum(result.summary.revised_reports for result in results),
        unresolved_reports=sum(result.summary.unresolved_reports for result in results),
        review_failures=sum(result.summary.review_failures for result in results),
    )

    return ConfidenceBatchResult(
        result=combined_result,
        original_result=combined_original_result,
        report_confidence=traces,
        summary=summary,
    )


def _reassemble_backoff_results(
    pipeline: PipelineSpec,
    results: list[BaseModel | JSONDict | ConfidenceBatchResult | None],
    *,
    validate_output: bool,
) -> BaseModel | JSONDict | ConfidenceBatchResult:
    confidence_results = [
        result for result in results if isinstance(result, ConfidenceBatchResult)
    ]

    if confidence_results:
        if len(confidence_results) != len(results):
            raise ValueError(
                "Backoff reassembly received a mixed set of confidence and non-confidence results"
            )
        return _combine_confidence_results(
            pipeline,
            confidence_results,
            validate_output=validate_output,
        )

    return _combine_batch_results(
        pipeline,
        [
            result
            for result in results
            if not isinstance(result, ConfidenceBatchResult)
        ],
        validate_output=validate_output,
    )


async def run_pipeline(
    pipeline: PipelineSpec,
    items: list[str],
    batch_size=5,
    workers=4,
    validate_output: bool = True,
    raise_on_failure: bool = True,
    attempts=3
) -> list[BaseModel | None | JSONDict | ConfidenceBatchResult]:
    # Split the workload into batches and run them concurrently, while
    # preserving result order via `asyncio.gather`.
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

                return await _run_batch_with_backoff(
                    pipeline,
                    batch,
                    client,
                    validate_output=validate_output,
                    attempts=attempts,
                    raise_on_failure=raise_on_failure
                )

        tasks = [task(i, b) for i, b in enumerate(batches)]

        nested_results = await asyncio.gather(*tasks)
        results = [
            result
            for batch_results in nested_results
            for result in batch_results
        ]

    logger.info(
        "Pipeline completed | pipeline=%s | batches=%d",
        pipeline.name,
        len(batches),
    )

    return results
