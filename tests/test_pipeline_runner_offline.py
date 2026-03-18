import asyncio

from cava_llm_manager.pipelines.base import PipelineSpec
from cava_llm_manager.pipelines.pipeline_runner import run_pipeline_batch
from cava_llm_manager.schemas.genomic import GenomicBatchResult


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, content: str):
        self._content = content

    async def post(self, *args, **kwargs):
        return FakeResponse(
            {
                "message": {
                    "content": self._content,
                }
            }
        )


def make_pipeline() -> PipelineSpec:
    return PipelineSpec(
        name="genomic_variants",
        model_id="llama3-med42-8b:1:q5_k_m",
        system_prompt_id="mutations",
        fewshot_id="mutations",
        return_schema=GenomicBatchResult,
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


def test_run_pipeline_batch_returns_none_when_invalid_json_is_suppressed():
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

    assert result is None
