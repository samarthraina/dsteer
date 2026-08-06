# dsteer

Investigating the extent to which DPO's behavioral effect on language models can be captured by inference-time activation steering.

**Status:** Active development.

## Overview

This repository contains code for:

- Evaluating instruction-tuning (IT) and DPO checkpoint pairs across multiple model families
- Extracting activation steering vectors from IT/DPO model differences
- Inference-time steering experiments characterizing operating range and limits
- Evaluation via G-Eval (multi-metric suite), AQI, and behavioral comparison

## Repository structure

    src/steering/        Importable Python package (core logic)
    scripts/             Entry points (one script per experiment phase)
    configs/             Experiment configurations
    notebooks/           Exploratory work (not production)
    docs/                Methods, decisions, reproducibility notes
    tests/               Sanity checks
    archive/             Prior version (D-STEER v1, for reference only)

## Setup

    git clone https://github.com/samarthraina/dsteer.git
    cd dsteer
    pip install -r requirements.txt
    export HF_TOKEN=your_token_here

## Pipeline

(To be added as scripts are built.)

## License

MIT