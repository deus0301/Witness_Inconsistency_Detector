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

- **SNLI / MultiNLI** — underpins the NLI-based contradiction scorer (alternatively, an off-the-shelf pretrained NLI model may be used directly).
- **PHEME** — annotated tweets about real crisis events (Ottawa shooting, Sydney siege, Germanwings crash, Charlie Hebdo shooting) with entailment/contradiction/unknown labels; serves as the real-world, multi-person-same-event dataset with existing ground truth.

*(An earlier plan to use scraped Famous Trials witness statements plus synthetic scenarios was dropped in favor of the SNLI + PHEME route.)*

## Team

| Member | Role |
|---|---|
| Yash Aggarwal | Data lead — PHEME / SNLI / MultiNLI acquisition and schema |
| Divit Malhotra | NLI / contradiction scoring lead |
| Apoorv | Clustering and integration lead |

## Timeline

Target: complete within 2 months.

## Status

Problem statement and dataset plan finalized. Implementation in progress.

## Project Structure

```
.
├── data/           # SNLI/MultiNLI + PHEME data and preprocessing
├── clustering/      # Sub-event clustering (Apoorv)
├── nli_scoring/      # Contradiction/entailment scoring (Divit)
├── notebooks/        # Exploration and experiments
└── README.md
```

*(Update this structure as the codebase takes shape.)*
