import asyncio
from unittest.mock import patch

from cava_llm_manager.pipelines.base import ConfidenceBatchResult, ConfidenceConfig, PipelineSpec
from cava_llm_manager.pipelines.pipeline_runner import run_pipeline, run_pipeline_batch
import cava_llm_manager.pipelines.pipeline_runner as pipeline_runner_module
from cava_llm_manager.schemas.genomic import GenomicBatchResult
from cava_llm_manager.schemas.review import ConfidenceReviewBatchResult


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, payload: str | dict):
        self._payload = payload

    async def post(self, *args, **kwargs):
        if isinstance(self._payload, dict):
            return FakeResponse(self._payload)

        return FakeResponse({"message": {"content": self._payload}})


class SequenceClient:
    def __init__(self, contents: list[str | dict]):
        self._contents = list(contents)

    async def post(self, *args, **kwargs):
        if not self._contents:
            raise AssertionError("No more fake responses configured")
        payload = self._contents.pop(0)
        if isinstance(payload, dict):
            return FakeResponse(payload)
        return FakeResponse({"message": {"content": payload}})


class DummyAsyncClient:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="genomic_variants",
        model_id="llama3-med42-8b:1:q5_k_m",
        system_prompt_id="mutations",
        fewshot_id="mutations",
        return_schema=GenomicBatchResult,
        inject_schema=False,
    )


def make_llama_cpp_pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="genomic_variants_llama_cpp",
        model_id="llama3-med42-8b:1:q5_k_m",
        system_prompt_id="mutations",
        fewshot_id="mutations",
        return_schema=GenomicBatchResult,
        inject_schema=False,
        host_type="llama_cpp",
        host_url="http://cava-llama:8000/completion",
    )


def make_review_pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="confidence_review",
        model_id="llama3-med42-8b:1:q5_k_m",
        system_prompt_id="mutations",
        fewshot_id=None,
        return_schema=ConfidenceReviewBatchResult,
        inject_schema=False,
    )


def make_revision_pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="confidence_revision",
        model_id="llama3-med42-8b:1:q5_k_m",
        system_prompt_id="mutations",
        fewshot_id=None,
        return_schema=None,
        inject_schema=False,
    )


def test_run_pipeline_batch_validates_conformant_output_without_ollama():
    pipeline = make_pipeline()
    items = [
        "EGFR mutation not detected.",
        "No genomic result reported.",
    ]
    client = FakeClient(
        """```json
        {
          "reports": [
            {
              "report_id": "1",
              "tests": {
                "genomic_marker": "EGFR",
                "test_result": "not detected",
                "variant": null,
                "unexpected": "ignored"
              },
              "extra_field": true
            },
            {
              "report_id": 2,
              "tests": []
            }
          ]
        }
        ```"""
    )

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            items,
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=True,
        )
    )

    assert isinstance(result, GenomicBatchResult)
    assert len(result.reports) == 2
    assert result.reports[0].input_text == items[0]
    assert result.reports[0].tests[0].genomic_marker == "EGFR"
    assert result.reports[0].tests[0].test_result.value == "negative"
    assert result.reports[1].input_text == items[1]
    assert result.reports[1].tests == []


def test_run_pipeline_batch_repairs_nonconformant_shape_into_empty_reports():
    pipeline = make_pipeline()
    items = [
        "ALK status pending.",
        "KRAS G12C mutation detected.",
    ]
    client = FakeClient('{"unexpected_root": "value"}')

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            items,
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=True,
        )
    )

    assert isinstance(result, GenomicBatchResult)
    assert [report.report_id for report in result.reports] == [1, 2]
    assert [report.input_text for report in result.reports] == items
    assert all(report.tests == [] for report in result.reports)


def test_run_pipeline_batch_returns_empty_batch_when_invalid_json_is_suppressed():
    pipeline = make_pipeline()
    client = FakeClient("this is not json")

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            ["MET amplification detected."],
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=False,
        )
    )

    assert isinstance(result, GenomicBatchResult)
    assert result.reports == []


def test_run_pipeline_batch_supports_llama_cpp_completion_response():
    pipeline = make_llama_cpp_pipeline()
    client = FakeClient(
        {
            "content": """{
              "reports": [
                {
                  "report_id": 1,
                  "tests": [
                    {
                      "genomic_marker": "EGFR",
                      "test_result": "positive",
                      "variant": null
                    }
                  ]
                }
              ]
            }"""
        }
    )

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            ["EGFR mutation detected."],
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=True,
        )
    )

    assert isinstance(result, GenomicBatchResult)
    assert result.reports[0].tests[0].test_result.value == "positive"


def test_genomic_batch_result_accepts_null_genomic_marker_without_failing_batch():
    result = GenomicBatchResult.model_validate(
        {
            "reports": [
                {
                    "report_id": 1,
                    "input_text": "Mixed genomic findings.",
                    "tests": [
                        {
                            "genomic_marker": None,
                            "test_result": "positive",
                            "variant": "V600E",
                        },
                        {
                            "genomic_marker": "KRAS",
                            "test_result": "not detected",
                            "variant": None,
                        },
                    ],
                }
            ]
        }
    )

    assert len(result.reports) == 1
    assert len(result.reports[0].tests) == 2
    assert result.reports[0].tests[0].genomic_marker == ""
    assert result.reports[0].tests[0].test_result.value == "positive"
    assert result.reports[0].tests[1].genomic_marker == "KRAS"
    assert result.reports[0].tests[1].test_result.value == "negative"


def test_run_pipeline_batch_can_return_confidence_sidecar_with_accept_review():
    pipeline = make_pipeline()
    pipeline.confidence = ConfidenceConfig(
        enabled=True,
        review_pipeline=make_review_pipeline(),
    )
    client = SequenceClient(
        [
            """{
              "reports": [
                {
                  "report_id": 1,
                  "tests": [
                    {
                      "genomic_marker": "EGFR",
                      "test_result": "positive",
                      "variant": null
                    }
                  ]
                }
              ]
            }""",
            """{
              "reviews": [
                {
                  "report_id": 1,
                  "decision": "accept",
                  "confidence": "high",
                  "issues": [],
                  "rationale": "Matches the source report."
                }
              ]
            }""",
        ]
    )

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            ["EGFR mutation detected."],
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=True,
        )
    )

    assert isinstance(result, ConfidenceBatchResult)
    assert isinstance(result.result, GenomicBatchResult)
    assert result.result.reports[0].tests[0].test_result.value == "positive"
    assert result.summary.reviewed_reports == 1
    assert result.summary.accepted_reports == 1
    assert result.summary.unresolved_reports == 0
    assert result.report_confidence[0].initial_review is not None
    assert result.report_confidence[0].revised is False


def test_run_pipeline_batch_can_revise_low_confidence_report():
    pipeline = make_pipeline()
    pipeline.confidence = ConfidenceConfig(
        enabled=True,
        review_pipeline=make_review_pipeline(),
        revision_pipeline=make_revision_pipeline(),
        attempt_revision=True,
    )
    client = SequenceClient(
        [
            """{
              "reports": [
                {
                  "report_id": 1,
                  "tests": [
                    {
                      "genomic_marker": "EGFR",
                      "test_result": "negative",
                      "variant": null
                    }
                  ]
                }
              ]
            }""",
            """{
              "reviews": [
                {
                  "report_id": 1,
                  "decision": "revise",
                  "confidence": "low",
                  "issues": ["The source report states the mutation was detected."],
                  "rationale": "Current extraction contradicts the source report."
                }
              ]
            }""",
            """{
              "report_id": 1,
              "tests": [
                {
                  "genomic_marker": "EGFR",
                  "test_result": "positive",
                  "variant": null
                }
              ]
            }""",
            """{
              "reviews": [
                {
                  "report_id": 1,
                  "decision": "accept",
                  "confidence": "high",
                  "issues": [],
                  "rationale": "The revised extraction now matches the source."
                }
              ]
            }""",
        ]
    )

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            ["EGFR mutation detected."],
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=True,
        )
    )

    assert isinstance(result, ConfidenceBatchResult)
    assert isinstance(result.result, GenomicBatchResult)
    assert result.original_result.reports[0].tests[0].test_result.value == "negative"
    assert result.result.reports[0].tests[0].test_result.value == "positive"
    assert result.summary.revised_reports == 1
    assert result.summary.accepted_reports == 1
    assert result.report_confidence[0].revised is True
    assert result.report_confidence[0].revision_review is not None
    assert result.report_confidence[0].unresolved is False


def test_run_pipeline_batch_preserves_original_when_revision_fails():
    pipeline = make_pipeline()
    pipeline.confidence = ConfidenceConfig(
        enabled=True,
        review_pipeline=make_review_pipeline(),
        revision_pipeline=make_revision_pipeline(),
        attempt_revision=True,
    )
    client = SequenceClient(
        [
            """{
              "reports": [
                {
                  "report_id": 1,
                  "tests": [
                    {
                      "genomic_marker": "EGFR",
                      "test_result": "negative",
                      "variant": null
                    }
                  ]
                }
              ]
            }""",
            """{
              "reviews": [
                {
                  "report_id": 1,
                  "decision": "revise",
                  "confidence": "low",
                  "issues": ["Output contradicts the source report."],
                  "rationale": "The mutation should be positive."
                }
              ]
            }""",
            "this is not json",
        ]
    )

    result = asyncio.run(
        run_pipeline_batch(
            pipeline,
            ["EGFR mutation detected."],
            client,
            validate_output=True,
            attempts=1,
            raise_on_failure=False,
        )
    )

    assert isinstance(result, ConfidenceBatchResult)
    assert isinstance(result.result, GenomicBatchResult)
    assert result.result.reports[0].tests[0].test_result.value == "negative"
    assert result.summary.revised_reports == 0
    assert result.summary.unresolved_reports == 1
    assert result.report_confidence[0].revised is False
    assert result.report_confidence[0].revision_error is not None


def test_run_pipeline_retries_failed_batches_at_smaller_sizes():
    pipeline = make_pipeline()
    calls: list[list[str]] = []

    async def fake_run_pipeline_batch(
        pipeline,
        batch,
        client,
        *,
        validate_output,
        attempts,
        raise_on_failure,
    ):
        calls.append(list(batch))
        if len(batch) > 1:
            raise RuntimeError("batch too large")
        return {"items": list(batch)}

    with patch.object(pipeline_runner_module.httpx, "AsyncClient", DummyAsyncClient):
        with patch.object(
            pipeline_runner_module,
            "run_pipeline_batch",
            fake_run_pipeline_batch,
        ):
            result = asyncio.run(
                run_pipeline(
                    pipeline,
                    ["a", "b", "c", "d"],
                    batch_size=4,
                    raise_on_failure=True,
                    attempts=1,
                )
            )

    assert calls == [["a", "b", "c", "d"], ["a", "b"], ["a"], ["b"], ["c", "d"], ["c"], ["d"]]
    assert result == [
        {"items": ["a"]},
        {"items": ["b"]},
        {"items": ["c"]},
        {"items": ["d"]},
    ]


def test_run_pipeline_raises_once_single_item_batch_still_fails():
    pipeline = make_pipeline()

    async def fake_run_pipeline_batch(
        pipeline,
        batch,
        client,
        *,
        validate_output,
        attempts,
        raise_on_failure,
    ):
        raise RuntimeError(f"failed {batch}")

    with patch.object(pipeline_runner_module.httpx, "AsyncClient", DummyAsyncClient):
        with patch.object(
            pipeline_runner_module,
            "run_pipeline_batch",
            fake_run_pipeline_batch,
        ):
            try:
                asyncio.run(
                    run_pipeline(
                        pipeline,
                        ["a", "b"],
                        batch_size=2,
                        raise_on_failure=True,
                        attempts=1,
                    )
                )
            except RuntimeError as exc:
                assert "failed ['a']" in str(exc)
            else:
                raise AssertionError("Expected RuntimeError for failing single-item batch")
