# Project brief: Cross-Account Inconsistency Detector

Repo root: `D:\Projects\Witness_Inconsistency_Detector` (branch `feature/nli-finetune`). UML501 ML group project, 3 people, ~2-month timeline.

## Goal and claim
Given several free-text accounts of one event (witness statements, reports, tweets), find where accounts disagree: cluster claims by sub-event (unsupervised), score cross-account claim pairs inside each cluster with NLI, and surface clusters whose accounts contradict, with quotes and attribution. Does NOT decide who is right. No generative LLM in the detection loop.

Implicit claims made in README/REPORT:
1. The NLI scorer flags labelled contradictions with ~0.62 precision (recall 0.33) on PHEME RTE.
2. Clustering is a precision filter and cost saver (82% fewer pairs; unknown-pair false flags 9.4% -> 1.2%).
3. Fine-tuning NLI on PHEME RTE (leave-one-event-out) doubles contradiction recall (0.31 -> 0.64) and triples pipeline recall (0.12 -> 0.36).
4. Cluster-level detection beats flag-all / random baselines on a veracity proxy (F1 0.568 vs 0.431 / 0.357).
5. Target use case is witness statements; evaluation domain is tweets.

## Data
- MultiNLI dev matched/mismatched (sanity check of the off-the-shelf scorer). `data/multinli.py`. MultiNLI train used for replay (3,000 examples) during fine-tuning.
- PHEME rumour threads v2 (4 main events: ottawashooting, sydneysiege, charliehebdo, germanwings-crash), source tweets only, English, >=4 tokens, exact dedup. `data/pheme.py`, `data/cleaning.py`. Raw data in `data/raw/` (gitignored, may not be present).
- PHEME RTE 2016 (in repo, `pheme_rte_datasets_2016/`): 5,413 tweet pairs labelled entailment/unknown/contradiction (1,989 / 2,046 / 1,378). Labels are assigned via the rumour stories the tweets belong to (story-level, not sentence-level). Provided leave-one-event-out train/test XMLs. `data/pheme_rte.py`.
- Veracity proxy: a cluster mixing a true and a false rumour story = disagreement-positive. `data/proxy_labels.py`. 102 clusters, 28 positive under default config.
- Splits: fine-tuning uses leave-one-event-out over 4 events; validation = one whole other training event (cyclic next in `finetune.folds`); "leakage removed from train: 0" is reported. Zero-shot pipeline config (clustering algorithm, threshold 0.9, direction rule) was chosen on the same labelled pairs that are reported, without a held-out split (admitted in README "Known limitations").

## Architecture (MUST BE KEPT)
data clean/filter/dedup -> sentence embeddings (MiniLM default; mpnet comparison) -> clustering (agglomerative default, distance_threshold 0.45; hdbscan/kmeans/graph alternatives; noise -> nearest above 0.6 sim) -> cross-account pairs per cluster (max 400 sampled) -> NLI cross-encoder `cross-encoder/nli-deberta-v3-base` both directions, aggregated (min_contradiction default) -> disagreement score per cluster (threshold 0.9, >=1 contradicting pair, score = 0.5*fraction + 0.5*max) -> JSON/Markdown report + findings summary (agree/conflict/unique) -> Streamlit app.
Optional: fine-tuned NLI checkpoint (`training/finetune_nli.py`), weighted loss, both directions, MNLI replay, early stop on contradiction AP on validation event, operating point (rule + threshold) chosen on validation event.

## Metrics and current results
All in `reports/RESULTS_TABLES.md` (generated from `reports/results/*.json`), discussed in `reports/REPORT.md` (sections 6.1-6.10, 8 Limitations).
- MNLI dev acc 0.900 / 0.900.
- PHEME RTE NLI-only, min rule @0.9: contradiction P 0.619 R 0.329; AP 0.532. Per-event P/R varies widely (ottawa 0.80/0.73; germanwings 0.55/0.09).
- Pipeline on labelled pairs (agglomerative): P 0.672 R 0.132; only 55.1% of labelled contradictions land in the same cluster (HDBSCAN 9.9%).
- Cluster-level vs veracity proxy: P 0.434 R 0.821 F1 0.568; threshold sweep 0.5-0.99 barely changes cluster flags (54 -> 45).
- Fine-tune LOEO mean of 4 folds: zero-shot default P 0.584 R 0.308 F1 0.372 AP 0.504 -> fine-tuned best-F1 P 0.629 R 0.635 F1 0.607 AP 0.612. Per fold: germanwings fine-tuned flags 38.6% of entailment pairs; ottawa fine-tuned AP 0.791 < zero-shot 0.852. Several val-chosen thresholds hit grid edge 0.05. Single seed. Pipeline recall mean 0.118 -> 0.357.
- Manual audit of 40 flagged pairs (older HDBSCAN config): 27.5% genuine, 15% time-driven, 57.5% false alarms. `reports/ERROR_ANALYSIS.md`.
- No confidence intervals, single seed, 4 events.

## Constraints
- Keep the existing NLI architecture (cross-encoder NLI over in-cluster cross-account pairs). Recommendations must be additions or modifications around/inside it, not a rewrite. No generative LLM in the detection loop (an LLM *baseline for evaluation* is a judgement call; flag it as such).
- Focus requested by user: evaluation validity, and what to add around the NLI stage (pre-filters, post-filters, calibration, claim decomposition, temporal handling, etc.).
- Compute: one RTX 4050 laptop GPU; full run ~20 min cold, fine-tune ~10 min per fold.

## Paths
- Code: `data/`, `nli_scoring/scorer.py`, `clustering/`, `pipeline/{run,disagreement,findings,report,config}.py`, `evaluation/{experiments,make_tables,plots}.py`, `training/finetune_nli.py`, `app/streamlit_app.py`, `run_all.py`
- Config: `configs/default.yaml`, `configs/sample.yaml`
- Results: `reports/results/*.json`, `reports/RESULTS_TABLES.md`, `reports/REPORT.md`, `reports/ERROR_ANALYSIS.md`
- Data: `pheme_rte_datasets_2016/pheme_rte_datasets_2016/{per_event_splits,test_training_splits}/*.xml`, `data/SCHEMA.md`
- Example input: `examples/witness_accounts.txt`
- Tests: `tests/`
