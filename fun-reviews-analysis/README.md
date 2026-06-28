# FUN Reviews Analysis

**Part of the [Digital Game Fun Research](../README.md) repository.**

## Study Description

This folder contains the replication material for a study evaluating a game design approach using data from Steam and SteamSpy. The project is structured as a reproducible data analysis pipeline, including data collection, preprocessing, analysis, and derived outputs.

## Current Status

The folder currently contains a fully implemented pipeline, including data gathering, processing, and analysis stages. All steps are organized and executable through a central script.

## Folder Contents

```
project-root/
├── 00_run_pipeline.py
├── pipeline/
│   ├── A_gathering/
│   ├── B_sampling/
│   ├── C_quality_control/
│   ├── D_descriptive_stats/
│   ├── E_analysis/
│   └── Z_utils/
├── data/
├── analysis_inputs/
└── README.md
```

### File Guide

- `00_run_pipeline.py`
  - Main entry point for the pipeline.
  - Orchestrates all stages sequentially from data collection to analysis.

- `pipeline/`
  - Contains all pipeline stages organized by function:

  - `A_gathering/`
    - Data collection from SteamSpy API (game metadata).
    - Data collection from Steam Web API (reviews and genre information).

  - `B_sampling/`
    - Implements review sampling procedures.
    - Supports both dictionary-based and LLM-based approaches.

  - `C_quality_control/`
    - Performs filtering, validation, and consistency checks on sampled data.

  - `D_descriptive_stats/`
    - Generates summary statistics and descriptive metrics.

  - `E_analysis/`
    - Runs core analytical models, scoring, embeddings, and clustering.

  - `Z_utils/`
    - Shared utility functions used across pipeline stages.

- `data/`
  - Stores raw and processed datasets.
  - Includes intermediate and final outputs produced during execution.

- `analysis_inputs/`
  - Configuration files, dictionaries, and prompts used in analysis.
  - Includes keyword sets and LLM prompt definitions.

## How to Use This Package

1. Run `00_run_pipeline.py` to execute the full pipeline.
2. Data will be collected, processed, and analyzed automatically.
3. Inspect outputs in the `data/` folder.
4. Review configuration settings in `analysis_inputs/` if modification is required.
5. Inspect individual pipeline folders for detailed implementation of each stage.

## Pipeline Summary

### Data Sources

- SteamSpy API for aggregated game metadata.
- Steam Web API for reviews and genre information.
- Data collection implemented in `A_gathering`.

### Selection Procedure

- Construct candidate set from review counts.

- Compute Acceptance Score:

  $$
  \text{Acceptance Score} = \frac{\text{Positive Reviews}}{\text{Total Reviews}} \times \log(\text{Total Reviews})
  $$

- Apply filtering rules:
  - Remove non-game entries.
  - Require valid genre information.
  - Exclude titles with fewer than 50 reviews.
  - Remove top and bottom quartiles by Acceptance Score.
  - Alternate selection from high and low ends.
  - Avoid repeated genres and insufficient English reviews.

### Review Sampling and Processing

- Dictionary-based method:
  - Uses full set of English reviews.
  - Applies normalization steps such as lowercasing and lemmatization.

- LLM-based method:
  - Samples up to 150 reviews per game.
  - Maintains proportional sentiment distribution.
  - Uses deterministic selection.

### Models and Methods

- Dictionary-based scoring:
  - Measures normalized frequency of keywords for:
    - Flow
    - Utility
    - Nostalgia

- LLM-based classification:
  - Assigns scores across categories:
    - Flow
    - Utility
    - Nostalgia
    - None
  - Uses PHI-3 with fixed prompt and deterministic decoding.

- Embeddings:
  - Sentence-BERT (384 dimensions).

- Clustering:
  - MiniBatchKMeans with heuristic cluster selection.

### Metrics and Analysis

- Spearman correlation between FUN indicators and Acceptance Score.
- Comparative summaries between dictionary and LLM methods.
- Mann–Whitney U tests between high- and low-acceptance groups.

### Interpretation of Outputs

- Dictionary scores:
  - Represent normalized keyword frequencies per review.

- LLM scores:
  - Represent averaged semantic probabilities scaled to percentages.

## Dependencies

Required:

- Python 3

Recommended:

- Standard Python scientific libraries (e.g., NumPy, pandas, scikit-learn).
- Access to APIs (SteamSpy and Steam Web API).

## Replicability

- Deterministic sampling and inference settings.
- Fixed random seeds for stochastic components.
- All configuration files stored in `analysis_inputs/`.
- Raw and processed outputs stored in `data/`.

## Archive DOI

[![DOI](https://doi.org/10.17605/OSF.IO/8G2P6)](https://doi.org/10.17605/OSF.IO/8G2P6)

https://osf.io/8g2p6/overview

