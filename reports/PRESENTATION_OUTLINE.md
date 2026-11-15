# Presentation outline (10 slides, about 12 minutes)

Figures referenced are in `reports/figures/`. All numbers come from `reports/results/*.json`.

## Slide 1. Title and one-line pitch
- Cross-Account Inconsistency Detector: find where many accounts of one event disagree.
- UML501 group project: Yash (data), Divit (NLI scoring), Apoorv (clustering and integration).
- One sentence: "We cluster claims by sub-event, then let an NLI model compare claims inside each cluster."

## Slide 2. Problem and what we deliberately do not do
- 2,010 tweets about Charlie Hebdo make about 2 million pairs; nobody can read them.
- Output wanted: a short, attributed list of conflicting statements.
- Not in scope: who is right, who is at fault, a merged narrative, a generative model in the loop.
- Uses: journalism, fact-checking, crisis verification, investigative analysis.

## Slide 3. Data
- MultiNLI: the only NLI training corpus; backs the pretrained scorer and its sanity check.
- PHEME threads: 4 events, 4,524 cleaned source tweets, 1,649 of them in the rumour evaluation set.
- PHEME RTE: 5,413 tweet pairs from the same events labelled entailment / contradiction / unknown (our main test set).
- Figure: `eda_veracity_balance.png`.

## Slide 4. Pipeline
- Diagram: clean -> embed -> cluster -> in-cluster cross-account pairs -> NLI both directions -> disagreement score -> report.
- Models: `all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-base`. Everything config-driven.
- A pair counts only if both NLI directions say contradiction (threshold 0.9).
- Report is extractive: quotes, accounts, probabilities, plus a findings summary (agree / conflict / single source).

## Slide 5. How we evaluate
- Primary: labelled PHEME pairs. Labels come through rumour stories, so they are coarser than sentence-level NLI.
- Secondary: a cluster-level proxy from rumour veracity (a cluster mixing a true and a false story).
- Plus a manual audit of 40 random flagged pairs.
- Say plainly which results rest on which evidence.

## Slide 6. The NLI scorer: strong in-domain, modest on tweets
- MultiNLI validation: 90.0% matched, 90.0% mismatched. Figure: `confusion_mnli_dev_matched_nli-deberta-v3-base.png`.
- Labelled PHEME pairs: 3-class accuracy 0.46 (majority 0.38). Figure: `confusion_pheme_rte.png`.
- Contradiction flags: precision 0.62, recall 0.33, base rate 0.255.
- Direction rule is a dial: max gives 0.45 / 0.57, both-directions gives 0.62 / 0.33.

## Slide 7. Clustering comparison and a surprise
- HDBSCAN + MiniLM matches annotated rumour stories best (ARI 0.44, NMI 0.71). Figure: `clustering_ari.png`.
- But it keeps only 10% of labelled contradictions in the same cluster; agglomerative keeps 55%.
- Why: contradicting tweets belong to different stories about the same thing.
- Lesson: the clustering metric must match the downstream task. Default changed to agglomerative.

## Slide 8. What clustering buys and costs
- Precision on labelled pairs: 0.62 -> 0.67. Unknown pairs wrongly flagged: 9.4% -> 1.2%. Figure: `rte_flag_rates.png`.
- Recall: 0.33 -> 0.13. Clustering can only remove pairs.
- Cost: 38,526 pairs instead of 207,385 (82% fewer); about 8,500 instead of 2 million for Charlie Hebdo. Figure: `pairs_compared.png`.
- Example: Germanwings "distress signal" cluster. Figure: `eval_heatmap_germanwings-crash.png`.

## Slide 9. Honest results and error analysis
- Two in three flagged labelled pairs are real contradictions; most contradictions are missed.
- Manual audit on raw events (earlier configuration): 27.5% genuine, 15% time-driven, 57.5% false alarms.
- Model failures: attribution phrases, headline fragments, entity aliases, numbers counting different things.
- Thresholds barely matter (flat curves). Figure: `threshold_curves.png`.
- Per event: Ottawa precision 0.80 / recall 0.73; Germanwings recall 0.09 (numeric claims).

## Slide 10. Limitations, ethics, next steps, demo
- Limits: low recall, story-level labels, clustering chosen on the test pairs, no notion of time, English source tweets only.
- Ethics: surfaces disagreement, does not determine truth; "accounts agree" is not "true"; human review required.
- Next: fine-tune NLI on the labelled pairs (leave-one-event-out), relevance gate, timestamps.
- Live demo: `streamlit run app/streamlit_app.py` (paste three witness accounts, or pick Germanwings).

## Backup slides
- Cluster-level proxy table and baselines (precision 0.43 vs 0.275 base rate).
- Aggregation ablation (max vs mean vs min).
- Runtime table and reproduction commands.
