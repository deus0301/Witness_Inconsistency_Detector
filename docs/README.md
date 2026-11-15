# Cross-Account Inconsistency Detector

**UML501 — Machine Learning, Group Project**

## Overview

Given multiple free-text accounts (witness statements, reports, tweets) describing the same real-world event, this project automatically detects where those accounts **disagree** — without relying on manually labeled ground-truth narratives, and without trying to determine which account is "correct."

The system does **not**:
- predict who is guilty or at fault
- reconstruct a single canonical narrative of the event
- act as an LLM-based interrogator

Instead, it uses unsupervised learning to cluster claims by sub-event and surface contradictions between accounts in a structured, interpretable way.

## Motivation

Investigators, journalists, and fact-checkers routinely have to reconcile many independent accounts of the same incident. Manually cross-referencing statements for contradictions doesn't scale. This project treats contradiction detection as a machine learning problem: cluster claims about the same sub-event, then score pairs of claims for entailment / contradiction / neutrality.

**Applications:** investigative analysis, journalism, fact-checking, crisis-event verification.

## Approach

1. **Sub-event clustering (unsupervised)** — group individual claims extracted from different accounts by which part/moment of the event they describe, so that only claims about the *same* sub-event are compared.
2. **Contradiction scoring (NLI-based)** — for claims within a cluster, use a Natural Language Inference model to score each pair as entailment, contradiction, or neutral.
3. **Surfacing disagreement** — present clusters where accounts contradict each other in a structured, interpretable format (not a rewritten narrative).

## Datasets

- **MultiNLI** — the only NLI dataset in the project. It underpins the contradiction scorer: a model pretrained on MultiNLI (`cross-encoder/nli-deberta-v3-base`) is used directly, and the MultiNLI validation sets (matched and mismatched) are used to sanity-check it.
- **PHEME** — tweets about real crisis events (Ottawa shooting, Sydney siege, Germanwings crash, Charlie Hebdo shooting); serves as the real-world, multi-person-same-event dataset. Two parts are used: the rumour threads (rumour story, veracity) and the PHEME RTE release (`pheme_rte_datasets_2016/`), which labels 5,413 tweet pairs as entailment / contradiction / unknown and is the main evaluation set. See `data/SCHEMA.md` and `reports/REPORT.md`.

*(An earlier plan to use scraped Famous Trials witness statements plus synthetic scenarios was dropped in favor of the MultiNLI + PHEME route. A later plan to also use a second NLI corpus was dropped as well: MultiNLI is the only NLI dataset.)*

## Team

| Member | Role |
|---|---|
| Yash Aggarwal | Data lead — PHEME / MultiNLI acquisition and schema |
| Divit Malhotra | NLI / contradiction scoring lead |
| Apoorv | Clustering and integration lead |

## Timeline

Target: complete within 2 months.

## Status

Implemented end to end: data layer, NLI scorer, clustering, integration, evaluation, demo and report. See the repository `README.md` for the quick start and `reports/REPORT.md` for results.

## Project Structure

```
.
├── docs/          # project docs (this file, work division)
├── data/          # loaders, cleaning, Claim schema, SCHEMA.md (+ raw/, gitignored)
├── nli_scoring/   # pairwise NLI contradiction scorer (Divit)
├── clustering/    # embeddings, sub-event clustering, cluster labels (Apoorv)
├── pipeline/      # end-to-end integration, disagreement score, reports
├── evaluation/    # experiments, baselines, ablations, figures
├── notebooks/     # exploratory analysis
├── app/           # Streamlit demo
├── reports/       # REPORT.md, slides outline, results, figures
├── tests/         # unit tests
├── configs/       # model names, paths, thresholds
└── run_all.py     # one-command run
```
