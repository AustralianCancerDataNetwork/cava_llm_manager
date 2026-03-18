from cava_llm_manager.pipelines.base import PipelineSpec
from cava_llm_manager.schemas.genomic import GenomicBatchResult


def test_genomic_pipeline_renders_expected_prompt_sections():
    pipeline = PipelineSpec(
        name="genomic_variants",
        model_id="llama3-med42-8b:1:q5_k_m",
        system_prompt_id="mutations",
        fewshot_id="mutations",
        return_schema=GenomicBatchResult,
        inject_schema=False,
    )

    rendered = pipeline.render_prompt(
        [
            "EGFR mutation not detected.",
            "BRAF V600E mutation detected.",
        ]
    )

    assert pipeline.root_collection_name == "reports"
    assert pipeline.item_label == "Report"
    assert "===== SYSTEM PROMPT =====" in rendered
    assert "===== USER PROMPT =====" in rendered
    assert "Examples:" in rendered
    assert "Report 1:" in rendered
    assert "Report 2:" in rendered
    assert "EGFR mutation not detected." in rendered
    assert "BRAF V600E mutation detected." in rendered
