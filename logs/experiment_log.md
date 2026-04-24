# Experiment Log Template

Date: [YYYY-MM-DD]  
Tester Name: [YOUR NAME]

| Experiment ID | Query | Template Used | Top-K | Alpha | Retrieved Sources | Response Quality (1-5) | Hallucination Observed | Notes |
|---|---|---|---|---|---|---|---|---|
| EXP-001 | Who won in the Ablekuma North constituency? | Strict | 5 | 0.6 | Ghana_Election_Result.csv | 4 | N | Good factual grounding from election rows. |
| EXP-002 | Summarize key priorities in the 2025 budget statement. | Analytical | 6 | 0.6 | budget_2025.pdf | 4 | N | Strong synthesis; minor verbosity. |
| EXP-003 | Compare election patterns and budget priorities. | Conversational | 7 | 0.5 | CSV + PDF | 3 | N | Retrieval mixed sources, some weak links. |
| EXP-004 | Who won the 2030 election? | Strict | 5 | 0.6 | Ghana_Election_Result.csv | 5 | N | Correctly declined unsupported year. |
| EXP-005 | What happened in 2024? | Analytical | 8 | 0.7 | CSV + PDF | 3 | N | Needed clarification, partial context used. |

## Observations

- Add notes on retrieval quality, prompt behavior, and response faithfulness.
- Track whether strict template consistently avoids unsupported claims.

## Findings

- Document patterns across alpha and top-k settings.
- Record which template performs best for factual versus interpretive questions.
