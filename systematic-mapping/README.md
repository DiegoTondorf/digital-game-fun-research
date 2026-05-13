# Systematic Mapping Study on Fun in Digital Games

**Part of the [Digital Game Fun Research](../README.md) repository.**

---

## Study Description

This folder contains the replication package for a systematic mapping study investigating the concept of **fun** in digital games. The study surveys the existing literature to identify, classify, and synthesize research on what makes digital games enjoyable, covering definitions, theoretical frameworks, measurement approaches, and empirical findings reported in the literature.

A systematic mapping study (SMS) follows a rigorous, predefined protocol to provide a broad overview of a research area, enabling researchers to identify trends, gaps, and opportunities for future work.

---

## Stable Archived Version

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

> *Replace the placeholder DOI above with the actual Zenodo DOI once the archive has been published.*

---

## Folder Contents

```
systematic-mapping/
├── data/         # Raw and processed data collected during the study
├── scripts/      # Analysis and processing scripts
├── results/      # Output files, figures, and summary tables
└── README.md     # This file
```

### `data/`

Contains all data files collected during the systematic mapping process, including the list of primary studies, extraction forms, and any intermediate datasets produced during screening and selection.

### `scripts/`

Contains all scripts used to process, analyze, and visualize the data. Scripts are provided as-is and should be executed in the order described in the **Reproduction Instructions** section below.

### `results/`

Contains the final output of the analyses, including summary tables, charts, and any other artifacts generated from the scripts. These outputs correspond directly to the figures and tables reported in the published study.

---

## Reproduction Instructions

Follow the steps below to reproduce the results of this study from scratch.

### Prerequisites

1. Ensure the software dependencies listed in the [Dependencies](#dependencies) section are installed.
2. Clone or download this repository (or the archived Zenodo version for a stable snapshot).

### Steps

1. **Prepare the data**
   - Navigate to the `data/` folder and verify that all required input files are present.
   - Consult the data-level `README` (if present) for details on file formats and provenance.

2. **Run the analysis scripts**
   - Navigate to the `scripts/` folder.
   - Execute the scripts in numerical or alphabetical order (e.g., `01_preprocess.R`, `02_analyze.R`, …).
   - Each script includes inline comments describing its purpose and expected inputs/outputs.

3. **Inspect the results**
   - Upon successful execution, output files will be written to the `results/` folder.
   - Compare the generated outputs with the figures and tables in the published manuscript to verify reproducibility.

---

## Dependencies

The following software is required to execute the analysis scripts. Specific version numbers are listed where reproducibility is sensitive to version differences.

| Software | Version | Purpose |
|---|---|---|
| — | — | — |

> *This table will be populated with the actual software names, versions, and purposes once the analysis scripts are finalized and added to the `scripts/` folder.*

---

## Reference to Main Repository

This study is part of the **Digital Game Fun Research** replication repository. For an overview of all available studies, visit the [main README](../README.md).
