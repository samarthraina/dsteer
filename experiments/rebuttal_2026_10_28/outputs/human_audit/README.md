# Human audit package

Blank. No annotator has been contacted and no human label exists.

- `blinded_samples.jsonl`: 100 items, 12-13 per original condition, salted-hash order
- `annotation_form.csv`: one blank row per item; one copy per annotator
- `rubric.txt`: the frozen harmfulness rubric, verbatim, with a content note
- `restricted/condition_key.json`: unblinding key -- withhold from annotators
- disagreement subset (<=50 items, separate diagnostic): built after both judges have scored
