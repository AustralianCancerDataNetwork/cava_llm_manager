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
- Wild-type is another way of saying negative.
- PDL1 expression is not a genomic mutation test and should be ignored.
- estimated glomerular filtration rate is different to epidermal growth factor receptor, although they both may use the same abbreviation.
- estimated glomerular filtration rate should not be reported 
- capitalisation (eGFR instead of EGFR) or a numeric / rate based value associated with it can often be used to disambiguate.
- do not infer results that are not explicitly stated in the text.
- Only output JSON.
"""