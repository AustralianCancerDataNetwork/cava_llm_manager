import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

import cava_llm_manager.pipelines.base as pipeline_base_module
import cava_llm_manager.pipelines.pipeline_runner as pipeline_runner_module
from cava_llm_manager.pipelines.base import PipelineSpec
from cava_llm_manager.pipelines.pipeline_runner import run_pipeline
from cava_llm_manager.schemas.genomic import GenomicBatchResult


@dataclass
class StubTransportClient:
    known_items: list[str]
    failures_remaining: dict[tuple[str, ...], int] = field(default_factory=dict)
    permanently_failing_batches: set[tuple[str, ...]] = field(default_factory=set)
    call_log: list[tuple[str, ...]] = field(default_factory=list)

    def _batch_from_payload(self, payload: dict) -> tuple[str, ...]:
        serialized = json.dumps(payload, sort_keys=True)
        batch = tuple(item for item in self.known_items if item in serialized)
        if not batch:
            raise AssertionError(
                f"Could not infer batch contents from payload: {payload}"
            )
        return batch

    def _success_payload(self, batch: tuple[str, ...]) -> dict:
        reports = [
            {
                "report_id": i,
                "tests": [],
            }
            for i, _ in enumerate(batch, start=1)
        ]
        return {"message": {"content": json.dumps({"reports": reports})}}

    async def post(self, url: str, *, json: dict, timeout: int):
        batch = self._batch_from_payload(json)
        self.call_log.append(batch)

        if batch in self.permanently_failing_batches:
            raise httpx.ReadTimeout(f"Persistent simulated failure for batch {batch}")

        remaining = self.failures_remaining.get(batch, 0)
        if remaining > 0:
            self.failures_remaining[batch] = remaining - 1
            raise httpx.ReadTimeout(f"Transient simulated failure for batch {batch}")

        return SimpleNamespace(json=lambda: self._success_payload(batch))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


async def immediate_sleep(_: float) -> None:
    return None


def make_pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="genomic_variants",
        model_id="stub-model",
        system_prompt_id="stub-system-prompt",
        fewshot_id=None,
        return_schema=GenomicBatchResult,
        inject_schema=False,
    )


def run_with_stub(
    scenario: StubTransportClient,
    items: list[str],
    *,
    batch_size: int,
    attempts: int,
    raise_on_failure: bool,
):
    pipeline = make_pipeline()

    with patch.object(
        pipeline_runner_module.httpx,
        "AsyncClient",
        lambda *args, **kwargs: scenario,
    ):
        with patch.object(pipeline_runner_module.asyncio, "sleep", immediate_sleep):
            with patch.object(
                pipeline_runner_module,
                "get_model",
                lambda model_id: SimpleNamespace(id=model_id, server_label=model_id),
            ):
                with patch.object(
                    pipeline_base_module,
                    "get_system_prompt",
                    lambda prompt_id: "Return only JSON.",
                ):
                    return asyncio.run(
                        run_pipeline(
                            pipeline,
                            items,
                            batch_size=batch_size,
                            workers=1,
                            validate_output=True,
                            attempts=attempts,
                            raise_on_failure=raise_on_failure,
                        )
                    )


def flatten_report_inputs(results: list[GenomicBatchResult]) -> list[str]:
    inputs: list[str] = []
    for batch in results:
        inputs.extend(report.input_text for report in batch.reports)
    return inputs


def test_run_pipeline_retries_transient_failure_on_later_batch_and_preserves_order():
    items = ["report-a", "report-b", "report-c", "report-d", "report-e", "report-f"]
    scenario = StubTransportClient(
        known_items=items,
        failures_remaining={
            ("report-c", "report-d"): 1,
        },
    )

    results = run_with_stub(
        scenario,
        items,
        batch_size=2,
        attempts=2,
        raise_on_failure=True,
    )

    assert [len(batch.reports) for batch in results] == [2, 2, 2]
    assert flatten_report_inputs(results) == items
    assert scenario.call_log == [
        ("report-a", "report-b"),
        ("report-c", "report-d"),
        ("report-c", "report-d"),
        ("report-e", "report-f"),
    ]


def test_run_pipeline_backoff_splits_batch_after_retries_are_exhausted():
    items = ["report-a", "report-b", "report-c", "report-d"]
    scenario = StubTransportClient(
        known_items=items,
        permanently_failing_batches={
            tuple(items),
            ("report-a", "report-b"),
            ("report-c", "report-d"),
        },
    )

    results = run_with_stub(
        scenario,
        items,
        batch_size=4,
        attempts=1,
        raise_on_failure=True,
    )

    assert len(results) == 1
    assert [len(batch.reports) for batch in results] == [4]
    assert flatten_report_inputs(results) == items
    assert scenario.call_log == [
        tuple(items),
        ("report-a", "report-b"),
        ("report-a",),
        ("report-b",),
        ("report-c", "report-d"),
        ("report-c",),
        ("report-d",),
    ]


def test_run_pipeline_reassembles_only_the_failed_top_level_batch():
    items = [
        "report-a",
        "report-b",
        "report-c",
        "report-d",
        "report-e",
        "report-f",
    ]
    scenario = StubTransportClient(
        known_items=items,
        permanently_failing_batches={
            ("report-c", "report-d"),
        },
    )

    results = run_with_stub(
        scenario,
        items,
        batch_size=2,
        attempts=1,
        raise_on_failure=True,
    )

    assert len(results) == 3
    assert [len(batch.reports) for batch in results] == [2, 2, 2]
    assert [[report.input_text for report in batch.reports] for batch in results] == [
        ["report-a", "report-b"],
        ["report-c", "report-d"],
        ["report-e", "report-f"],
    ]
    assert scenario.call_log == [
        ("report-a", "report-b"),
        ("report-c", "report-d"),
        ("report-c",),
        ("report-d",),
        ("report-e", "report-f"),
    ]


def test_run_pipeline_raises_when_singleton_batch_still_fails_after_backoff():
    items = ["report-a", "report-b"]
    scenario = StubTransportClient(
        known_items=items,
        permanently_failing_batches={
            tuple(items),
            ("report-a",),
        },
    )

    with pytest.raises(httpx.ReadTimeout, match=r"\('report-a',\)"):
        run_with_stub(
            scenario,
            items,
            batch_size=2,
            attempts=1,
            raise_on_failure=True,
        )

    assert scenario.call_log == [
        tuple(items),
        ("report-a",),
    ]


def test_run_pipeline_returns_empty_batch_without_backoff_when_failures_are_suppressed():
    items = ["report-a", "report-b"]
    scenario = StubTransportClient(
        known_items=items,
        permanently_failing_batches={tuple(items)},
    )

    results = run_with_stub(
        scenario,
        items,
        batch_size=2,
        attempts=1,
        raise_on_failure=False,
    )

    assert len(results) == 1
    assert isinstance(results[0], GenomicBatchResult)
    assert results[0].reports == []
    assert scenario.call_log == [tuple(items)]
