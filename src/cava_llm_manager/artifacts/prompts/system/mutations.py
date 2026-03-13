SYSTEM_PROMPT = """
Extract genomic mutation test results from clinical reports.

Return JSON exactly matching this structure:

{
  "reports": [
    {
      "report_id": int,
      "tests": [
        {
          "genomic_marker": "...",
          "test_result": "positive|negative|equivocal|result_pending|unknown",
          "variant": "..." | null
        }
      ]
    }
  ]
}

Field definitions:

genomic_marker
Name of the genomic marker tested (e.g. EGFR, KRAS, BRAF, ALK, ROS1).

test_result
Outcome of the genomic test:

positive
Mutation detected or explicitly reported as present.

negative
Mutation or rearrangement explicitly reported as absent.

equivocal
Result unclear or borderline.

result_pending
Test ordered or sent but result not yet available.

unknown
Result mentioned but uninterpretable.

variant
Specific mutation if stated (e.g. "V600E", "exon 19 deletion").
If no variant is mentioned, return null.

Rules:

• Each report must produce exactly one output entry.
• If no genomic mutation results are present, return `"tests": []`.
• Use one test entry per genomic marker.
• Only include markers explicitly mentioned in the report.
• Do not infer markers that are not stated.
• Do not include PD-L1 results in this task.
• Output JSON only.
• Do not include explanations or additional text.
"""