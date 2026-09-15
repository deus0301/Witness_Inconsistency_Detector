# Witness Inconsistency Detector — Work Division & Timeline

**Team:** Yash (Data), Divit (NLI/Scoring), Apoorv (Clustering/Integration)
**Timeline:** 8 weeks

---

## Problem Statement

Automatically detect where multiple free-text accounts of the same event disagree, without relying on manually labeled ground-truth narrative. The goal is not to determine which account is correct — just to surface disagreement in a structured, interpretable way. Applications: investigative analysis, journalism, fact-checking.

**Data plan:** SNLI/MultiNLI to underpin the NLI-based contradiction scorer, and PHEME (annotated tweets about real crisis events — Ottawa shooting, Sydney siege, Germanwings crash, Charlie Hebdo shooting) as the real-world multi-person-same-event dataset with existing ground truth.

---

## The Pipeline

1. **Data ingestion & preprocessing** — clean PHEME/SNLI/MultiNLI, group by event/sub-event, define a shared schema.
2. **Contradiction scoring** — pretrained/fine-tuned NLI model scores pairs of statements as entailment/neutral/contradiction.
3. **Clustering by sub-event + contradiction surfacing** — unsupervised clustering of statements, overlaid with NLI scores to flag disagreeing clusters.
4. **Evaluation & presentation** — validate against PHEME ground truth, build visualizations, write up the report/demo.

---

## Role Assignments

### Yash — Data Lead
- Acquire and clean PHEME + SNLI/MultiNLI
- Structure PHEME data per event, grouped by sub-event
- Build the shared data interface (schema) that Divit and Apoorv consume downstream
- Exploratory analysis: dataset stats, class balance, sanity checks

### Divit — NLI / Contradiction Scoring Lead
- Set up a pretrained NLI model (e.g. RoBERTa/DeBERTa fine-tuned on MNLI from HuggingFace), fine-tune further on SNLI/MultiNLI if time allows
- Build pairwise scoring: for every pair of statements about the same sub-event, output entailment/neutral/contradiction + confidence
- Own model evaluation metrics (accuracy/F1) against PHEME labels

### Apoorv — Clustering & Integration Lead
- Embed statements (sentence-transformers), cluster by sub-event topic (unsupervised — k-means/HDBSCAN or graph-based)
- Combine Divit's pairwise contradiction scores with clusters to surface "this cluster has internal disagreement"
- Own final pipeline integration (glue Yash + Divit's pieces together), visualization (contradiction graphs/heatmaps), and report/demo narrative

**Critical dependency:** the data schema Yash defines. Get Yash and Divit aligned on it early (Week 1) so Divit doesn't build the scorer against a format that changes later.

---

## Timeline (8 Weeks)

| Weeks | Yash | Divit | Apoorv |
|---|---|---|---|
| 1–2 | Get data pipeline running; agree on schema with Divit | Research/test pretrained NLI models on a toy example | Research clustering approaches + sentence embedding options |
| 3–4 | Finalize clean dataset | Build working pairwise NLI scorer on real PHEME data | Build clustering on the same data (can start with dummy scores from Divit) |
| 5–6 | Support integration, help debug data edge cases | Refine scorer, start evaluation against PHEME ground truth | Integrate — wire Divit's scores into clusters |
| 7 | — | Tune thresholds, fix edge cases | Run full pipeline evaluation, polish |
| 8 | — | — | Report, visualizations, demo (buffer for delays) |

---

## Sync Points
- **End of Week 1:** Data schema locked between Yash and Divit.
- **End of Week 4:** Each person has a standalone working component (clean data / scorer / clustering).
- **End of Week 6:** First full end-to-end integration run.
- **Week 8:** Final report and demo.
