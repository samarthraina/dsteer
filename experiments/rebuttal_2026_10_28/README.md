# Rebuttal experiments — 29 October 2026

Code and saved evidence for the matched refusal-direction comparison, benign-utility evaluation and independent-judge check. Pair A is Llama 8B; pair B is Mistral 7B. The original judge is Qwen3.5-27B; the independent judge is Microsoft Phi-4 (14B). Model identities, revisions and decoding settings are retained for reproducibility.

## Results

- `outputs/analysis/`: effects, paired confidence intervals, comparator contrasts, judge agreement and utility measures.
- `figures/`: plots from the completed run.
- `outputs/generations/` and `outputs/scores/`: response-level evidence for the new experiments.
- `inputs/recovered/task027/`: original saved responses and scores used for paired comparisons.
- `configs/protocol.json`: immutable experiment protocol; `configs/amendments/` records amendments.
- `manifests/`: panels, model sources and comparator-selection records.

Select rows with `status=complete` when reading analysis tables. Some tables retain `not_run` placeholders alongside completed rows; these are not additional observations.

## Interpretation

All four D-STEER effects remain positive under the independent judge, with 95% confidence intervals excluding zero. D-STEER exceeds the matched additive comparator in all four direct contrasts, but the comparator is weak; this does not establish superiority over the full reference refusal-ablation method. Reference selection yielded no eligible direction on the tested panels. Tuned comparisons and the human audit remain incomplete.

Benign tests reveal over-refusal and invalid-generation costs. Official all-output IFEval scores, validity-conditional scores and validity-adjusted scores must be distinguished. Confidence intervals describe prompt sampling, not training-seed variability.

## Analysis

Run from this directory in a compatible CPU environment using the pinned dependencies in `configs/requirements-cpu-test.txt` (the recorded local environment used Python 3.9; the GPU environment used Python 3.12):

```sh
python -I scripts/analyse.py
```

This regenerates the core analysis tables from saved evidence without a GPU. `scripts/ifeval_variants.py` additionally requires the official checker dependencies and NLTK resources. See `src/ifeval_score.py` for the exact checker revision and deterministic language-detection setting.

GPU scripts and pinned environment files are included for inspection and reconstruction, not as a one-command cloud deployment. Checkpoint weights, activation tensors, environment caches, cloud credentials, machine-control scripts and operational logs are excluded. Full generation requires materializing the pinned endpoints and public inputs; private checkpoint access must be arranged separately. Some recovery and validation utilities refer to the original project layout. Existing tests may require local test-model paths and a `.venv`.

## Invalid calibration scores

`outputs/quarantine/scores_original_calibration_judge_down_2026-10-29/` preserves invalid records from a judge-connectivity failure. They are not scientific abstentions and are excluded from analysis loading paths. The associated calibration decisions were voided; do not interpret the incident as a failed scientific calibration. Calibration generations remain saved. The scorer now probes judge availability and rejects recognized connection failures.

## Sources and integrity

Third-party model names, dataset identifiers, source headers and dependency names are intentionally retained. The frozen primary implementation and official IFEval checkers are under `vendor/`; their upstream licensing and attribution requirements continue to apply. Public-source locations and revisions are in `manifests/public_inputs.json`. `PACKAGE_SHA256.json` inventories the files in this export (excluding itself).
