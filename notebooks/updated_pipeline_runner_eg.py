import asyncio
import os
import time
from pathlib import Path

import pandas as pd

ollama_host = "http://cava_ollama:11434"
OLLAMA = f"{ollama_host}/api/chat"
os.environ["OLLAMA_URL"] = OLLAMA

from cava_llm_manager import get_model, get_registry
from cava_llm_manager.logging import configure_logging
from cava_llm_manager.pipelines.base import PipelineSpec
from cava_llm_manager.pipelines.pipeline_runner import run_pipeline
from cava_llm_manager.schemas.genomic import GenomicBatchResult

configure_logging("WARNING")

reg = get_registry()

this_host = {
    "llama3-med42-8b:1:q3_k_m": "llama_med42",
    "llama3-med42-8b:1:q5_k_m": "llama3_med",
    "gemma-3-4b-it::q4_k_m": "gemma-local",
}

MODEL = "llama3-med42-8b:1:q5_k_m"
get_model(MODEL)
reg.set_model_host_label(MODEL, this_host[MODEL])

GENOMIC_PIPELINE = PipelineSpec(
    name="genomic_variants",
    model_id=MODEL,
    system_prompt_id="mutations",
    fewshot_id="mutations",
    return_schema=GenomicBatchResult,
    inject_schema=False,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "all_notes.csv"
RESULT_DIR = ROOT_DIR / "results"


async def main() -> None:
    inputs = pd.read_csv(INPUT_PATH)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    big_batch = 50
    batch_size = 3

    for i in range(0, len(inputs), big_batch):
        reports = [str(s) for s in inputs.iloc[i:i + big_batch].text.unique()]

        results = await run_pipeline(
            GENOMIC_PIPELINE,
            reports,
            batch_size=batch_size,
            validate_output=True,
            raise_on_failure=False,
        )

        elapsed_minutes = (time.time() - t0) / 60
        print(i, elapsed_minutes)

        result_rows: list[dict] = []

        for batch_result in results:
            if batch_result is None:
                continue

            for report in batch_result.reports:
                if not report.tests:
                    result_rows.append(
                        {
                            "report_id": report.report_id,
                            "input_text": report.input_text,
                            "genomic_marker": "",
                            "test_result": "",
                            "variant": None,
                            "enum_errors": [],
                        }
                    )
                    continue

                for test in report.tests:
                    result_rows.append(
                        {
                            "report_id": report.report_id,
                            "input_text": report.input_text,
                            "genomic_marker": test.genomic_marker,
                            "test_result": test.test_result.value,
                            "variant": test.variant,
                            "enum_errors": test.enum_errors,
                        }
                    )

        pd.DataFrame(result_rows).to_csv(
            RESULT_DIR / f"results_out_{i}.csv",
            index=False,
        )


if __name__ == "__main__":
    asyncio.run(main())
