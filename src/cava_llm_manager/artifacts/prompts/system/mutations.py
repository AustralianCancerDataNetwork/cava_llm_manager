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
Rules:
- Each report must produce exactly one output entry.
- If no genomic result is present return an empty list of tests.
- Only output JSON.

"""