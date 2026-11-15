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
- MultiNLI: the only NLI dataset; backs the pretrained scorer and its sanity check.
- PHEME: 4 events, 4,524 cleaned source tweets, 1,649 of them in the rumour evaluation set.
- What we found in the files: rumour story, veracity, turnaround flag. No stance labels, no contradiction labels.
- Figure: `eda_veracity_balance.png`.

## Slide 4. Pipeline
- Diagram: clean -> embed -> cluster -> in-cluster cross-account pairs -> NLI both directions -> disagreement score -> report.
- Models: `all-MiniLM-L6-v2`, `cross-encoder/nli-deberta-v3-base`. Everything config-driven.
- A pair counts only if both NLI directions say contradiction (threshold 0.9).
- Report is extractive: quotes, accounts, probabilities.

## Slide 5. How we evaluate without gold labels
- Pair proxy: debunk vs rumour = contradiction-like; same true story = entailment-like; different stories = neutral-like.
- Cluster proxy: a cluster mixing a true and a false story is disagreement-positive.
- Say plainly: these are proxies, and slide 9 shows where they fail.
- Plus a manual audit of 40 random flagged pairs.

## Slide 6. The NLI scorer: strong in-domain, weak on tweets
- MultiNLI validation: 90.0% matched, 90.0% mismatched. Figure: `confusion_mnli_dev_matched_nli-deberta-v3-base.png`.
- PHEME proxy pairs: accuracy 0.33. Figure: `confusion_pheme_proxy_nli-deberta-v3-base.png`.
- Key observation: 38% of unrelated-story pairs get contradiction >= 0.9 in one direction; 12% in both.
- This is why clustering first is necessary, not optional.

## Slide 7. Clustering comparison
- Table or figure `clustering_ari.png`: HDBSCAN + MiniLM best (ARI 0.42, NMI 0.71).
- Silhouette is low everywhere (0.16 to 0.21): sub-events of one event overlap in embedding space.
- The bigger mpnet model is not better.
- Figure: `eval_clusters_germanwings-crash.png`.

## Slide 8. What clustering buys
- 9,698 pairs instead of 207,385 (95% fewer). Figure: `pairs_compared.png`.
- Flagged pairs about the same story: 27.1% vs 2.1%. Figure: `pair_precision.png`.
- Cost: about half of rumour-versus-debunk pairs are split across clusters.
- Example: Germanwings "distress signal" cluster. Figure: `eval_heatmap_germanwings-crash.png`.

## Slide 9. Honest results and error analysis
- Cluster-level vs proxy: precision 0.51, recall 0.78, F1 0.62; flag-everything gets F1 0.57, random 0.48.
- Manual audit: 27.5% genuine, 15% time-driven, 57.5% false alarms.
- Proxy failures: "still alive" vs "has died" counted as a false positive; two debunks counted as a missed contradiction.
- Model failures: attribution phrases, headline fragments, entity aliases, numbers counting different things.
- Thresholds barely matter (flat curves). Figure: `threshold_curves.png`.

## Slide 10. Limitations, ethics, next steps, demo
- Limits: no gold labels, over-flagging (about half of clusters), no notion of time, English source tweets only.
- Ethics: surfaces disagreement, does not determine truth; inconsistency is not evidence of lying; human review required.
- Next: relevance gate before NLI, timestamps, a few hundred human pair labels, in-domain fine-tuning.
- Live demo: `streamlit run app/streamlit_app.py` (paste three witness accounts, or pick Germanwings).

## Backup slides
- Per-event table (Germanwings F1 0.75, Ottawa 0.49) and why.
- Aggregation ablation (max vs mean vs min).
- Runtime table and reproduction commands.
