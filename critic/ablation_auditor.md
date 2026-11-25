# Baseline and ablation audit: Cross-Account Inconsistency Detector

Scope: `evaluation/experiments.py`, `training/finetune_nli.py`, `pipeline/disagreement.py`, `configs/default.yaml`, `reports/results/*.json`, `reports/RESULTS_TABLES.md`, `reports/REPORT.md` §5, §6.x and §8, and `reports/ERROR_ANALYSIS.md`. No repo file was changed except this one.

New numbers in this audit come from two scripts, now in critic/scripts/
`critic/scripts/baselines.py` and `critic/scripts/noise.py`. They were run with `.venv\Scripts\python`. They load the RTE pairs through the project's own loader `data/pheme_rte.load_pheme_rte` (5,413 pairs). Numbers marked "computed" come from these scripts. Every other number is read from `reports/results/*.json`.

Labels:
- **OBSERVED**: read in code or results, or computed from repo data.
- **GENERAL**: general ML practice.
- **UNVERIFIED**: not checked.

---

## Findings

[B-1] The effective sample size is tiny: 85 label groups and only 20 distinct contradictions
Severity: critical
Confidence: high
Basis: OBSERVED (computed)
Evidence:
- Every RTE label is a deterministic function of the pair's (event, rumour-story pair). All 85 story-pair groups are label-pure (85 of 85). Entailment pairs always name 1 story, and contradiction and unknown pairs always name 2: (entailment,1)=1989, (neutral,2)=2046, (contradiction,2)=1378.
- Contradiction story-pairs per event: ottawa 5, sydney 7, charliehebdo 7, germanwings **1**.
- In germanwings, all 257 contradiction pairs are one story pair: "(Up to) 150 people perished (144 passengers, 6 crew)" against "148 people onboard". The whole event has 5 groups.
- 533 distinct tweets, about 20.3 pairs each.
- On the pooled keyword baseline (F1 0.476), a pair-level bootstrap gives a 95% CI of [0.453, 0.498] (width 0.045). A bootstrap that resamples story-pair groups gives [0.203, 0.666] (width 0.463), about 10x wider.
- REPORT §6.10 says "Folds are small (1,100 to 1,936 test pairs)". The real unit is 1 to 7 contradiction stories per fold.
Why it matters:
- Pair-level P/R/F1 values are not independent observations. Any Wilson or binomial CI on pairs (for example NLI-only P 454/734, Wilson [0.583, 0.653]) is far too narrow.
- The germanwings fold result, which drives most of the fine-tuning headline, measures whether the model handles **one** numeric conflict.
- The charliehebdo fold's validation event is germanwings, so its operating point was tuned on that same single contradiction story.
Fix:
- Report every pair-level metric with a cluster (story-pair-group) bootstrap CI, and state the number of contradiction story-pairs next to every pair count.
- Report micro-pooled LOEO results with a group bootstrap, as well as the unweighted fold mean.
- Treat the germanwings fold as anecdotal in the text.

[B-2] A zero-training keyword heuristic matches or beats the zero-shot NLI scorer, and "flag everything" beats the zero-shot default
Severity: critical
Confidence: high
Basis: OBSERVED (computed; flag-all and the NLI numbers come from results JSON)
Evidence:
- The heuristic "anycue" flags a pair when any of three cues fires:
  - negation/denial-word mismatch (not, no, never, denied, false, hoax, fake, rumour, unconfirmed, ...);
  - number-set mismatch (both texts contain numbers and the sets differ);
  - one of 13 hand-picked antonym pairs (dead/alive, killed/injured, arrested/escaped, ...).
- The lexicon was written once, untuned, before seeing these scores. I did know REPORT §6.2's remark that germanwings conflicts are numeric.

LOEO test events, contradiction F1:

| Held-out | Flag-all | anycue | Number-mismatch only | Zero-shot default (min, 0.9) | Zero-shot val-tuned | Fine-tuned (best F1) |
|---|---|---|---|---|---|---|
| germanwings | 0.362 | 0.518 | 0.587 | 0.159 | 0.338 | 0.620 |
| sydneysiege | 0.414 | 0.451 | 0.467 | 0.273 | 0.324 | 0.465 |
| charliehebdo | 0.361 | 0.569 | 0.554 | 0.294 | 0.447 | 0.651 |
| ottawashooting | 0.510 | 0.286 | 0.026 | 0.761 | 0.688 | 0.692 |
| **Mean** | **0.412** | **0.456** | 0.408 | **0.372** | 0.449 | 0.607 |

- Pooled over all 5,413 pairs:
  - anycue: P 0.467, R 0.485, F1 0.476.
  - NLI min rule at 0.9: F1 0.430 (`RESULTS_TABLES.md:31`).
  - NLI max rule: F1 0.502.
  - Flag-all: F1 0.406.
- The labelled-pair pipeline (agglomerative): F1 0.221 (`RESULTS_TABLES.md:51`). That is below flag-all.
- The only trivial pair-level baselines the project reports are majority-class 3-class accuracy (`experiments.py:261`) and contradiction prevalence. Neither gives the reader an F1 to compare against. Trivial baselines exist only at cluster level (`experiments.py:459-470`).
- TF-IDF cosine and Jaccard used as the contradiction score have no signal at all (pooled AP 0.222 to 0.254 against prevalence 0.255). With the threshold tuned on the validation event, they reach mean test F1 0.403 and 0.393, which is flag-all level.
Why it matters:
- The claim "NLI flags labelled contradictions with ~0.62 precision" (BRIEF claim 1) has no reference point.
- At the shipped operating point, the zero-shot NLI scorer is worse in F1 than flagging every pair, and worse than three regexes. Its advantage is precision at low recall. That is a valid design choice, but it should be shown against a baseline at the same flag rate.
- On germanwings, where fine-tuning shows its largest gain, number-mismatch alone gets 0.587 against fine-tuned 0.620.
Fix:
- Add a pair-level baseline block to `eval_rte` and to `run_fold`: flag-all; random at the system's flag rate; negation, number and antonym cues and their union; TF-IDF/MiniLM cosine with a val-tuned threshold.
- Report each baseline's precision at the NLI system's flag rate, and P@k.
- Use a cue as a **feature** next to NLI, for example a number-mismatch post-filter or booster (see the ablation table).

[B-3] A story-structure "oracle" cue scores as well as the fine-tuned model, so the fine-tuning gain may be story and topic learning
Severity: critical
Confidence: medium
Basis: OBSERVED (computed). The causal reading is inference.
Evidence:
- Baseline: flag a pair if it spans two stories (read from the `rumours` attribute) **and** its TF-IDF cosine is at least t, with t tuned on the same cyclic validation event as `finetune_nli.validation_event`.
  - LOEO test F1: gw 0.614, syd 0.529, ott 0.770, chebdo 0.635. Mean **0.637**, against fine-tuned 0.607.
  - AP of the score two-stories × cosine: mean 0.685 (0.637 / 0.620 / 0.770 / 0.714), against fine-tuned 0.612.
- Lexical similarity alone separates the classes well once story structure is known:
  - Among non-entailment pairs, cosine gives AP 0.675 for contradiction against unknown (prevalence 0.402).
  - Cosine gives AP 0.686 for entailment against the rest (prevalence 0.367).
- The fine-tuned confusion matrices support this reading. Germanwings fine-tuned (`finetune_results.json`, finetuned_best_f1): the entailment row is [269, 13, 177], so 177 of 459 same-story pairs are predicted contradiction, and no contradiction pair is predicted neutral.
- REPORT §6.10 caveat 1 and §8.12 raise this risk but do not test it.
Why it matters: the RTE task reduces to two steps: (a) are these two different stories? (b) are the two stories on the same topic? A cross-encoder can learn both from surface features. The headline "fine-tuning doubles contradiction recall" may measure that the model learned the dataset's labelling procedure, not sentence-level contradiction. This matters directly for witness statements, which have no story structure.
Fix:
- Hand-label a small sentence-level subset, used for evaluation only. For example, 10 pairs from each of the 85 story-pair groups is about 850 pairs; two annotators; report kappa.
- Report zero-shot and fine-tuned results on that subset.
- Report the story-oracle and keyword baselines in the fine-tuning table.
- Add a within-story test: entailment-labelled pairs whose texts actually conflict (as in ERROR_ANALYSIS FP1).

[B-4] The fine-tuning headline is measured against an unfair baseline; against the fair one, the pipeline gain is not supported
Severity: high
Confidence: high
Basis: OBSERVED (computed from `finetune_results.json`)
Evidence:
- REPORT §6.10 says "doubles contradiction recall (0.31 to 0.64)" and "Through the pipeline, recall triples (0.12 to 0.36)". Both compare against zero-shot at a fixed 0.9 threshold. The fine-tuned model gets a rule and threshold tuned on validation.
- The same file has the fair row, zero-shot val-tuned best F1, scored with the identical `choose_operating_point` procedure (`finetune_nli.py:427, 458-469`). So the procedure is equal. Only the headline uses the unfair row.
- Paired fold deltas, n=4, t with 3 degrees of freedom:

| Comparison | Per-fold deltas (gw, syd, chebdo, ott) | Mean | 95% t-CI |
|---|---|---|---|
| Contradiction F1, fine-tuned − zero-shot default | +0.461, +0.192, +0.357, −0.069 | +0.235 | [−0.132, +0.603] |
| Contradiction F1, fine-tuned − zero-shot val-tuned | +0.282, +0.141, +0.204, +0.004 | +0.158 | [−0.029, +0.345] |
| Contradiction F1, fine-tuned − anycue heuristic | +0.102, +0.014, +0.082, +0.406 | +0.151 | [−0.126, +0.428] |
| AP, fine-tuned − zero-shot | +0.089, +0.041, +0.365, −0.061 | +0.109 | [−0.181, +0.398] |
| Pipeline recall, fine-tuned − zero-shot default | +0.578, +0.094, +0.294, −0.010 | +0.239 | [−0.173, +0.651] |
| **Pipeline recall, fine-tuned − zero-shot val-tuned** | **+0.423, +0.072, −0.080, −0.108** | **+0.077** | **[−0.312, +0.465]** |

- Against the val-tuned zero-shot model, pipeline recall gets **worse** in 2 of 4 folds: chebdo 0.569 → 0.489 and ott 0.286 → 0.178 (`RESULTS_TABLES.md:206-213`). Mean pipeline recall is 0.281 → 0.357, not 0.118 → 0.357.
- The germanwings fold, a single contradiction story (B-1), supplies +0.423 of the pipeline-recall gain.
- These fold-level CIs ignore within-fold noise and seed noise (B-6), so they are optimistic.
Why it matters: BRIEF claim 3 ("triples pipeline recall") is not supported by the project's own fair comparison. The contradiction-F1 gain against the fair baseline is plausible (4 of 4 folds are non-negative; a one-sided sign test gives p = 0.0625) but not established.
Fix:
- Make zero-shot val-tuned the reference row in every headline.
- Report fold deltas with CIs, and a micro-pooled group-bootstrap delta. Do this on per-pair predictions: save the per-pair probabilities to `finetune_results.json` so paired tests become possible.
- Drop "triples".

[B-5] The operating point is chosen on one validation event, and it is unstable and sits on the grid edge
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- The grid is 0.05 to 0.99 (`configs/default.yaml:105`).
- Chosen thresholds on the grid floor or near it:
  - fine-tuned: syd 0.05, chebdo 0.07, ott 0.06 / 0.05;
  - zero-shot val-tuned: chebdo 0.05, ott 0.05.
- In the ottawa fold, zero-shot recall_at_precision is `feasible: false` on validation charliehebdo, so it falls back to best-F1.
- Validation-to-test transfer is poor:
  - gw fold, fine-tuned: val P 0.608 → test P 0.467 (the target is ≥ 0.55);
  - chebdo fold, zero-shot: chosen on germanwings as max@0.05, test P 0.329, unknown-pair flag rate 64.5%.
- The thresholds are meaningless as levels: the fine-tuned best-F1 thresholds range from 0.05 (syd) to 0.95 (gw). REPORT §6.10 admits this.
- The demo model takes the *median* of these values (`finetune_nli.py:611`), giving 0.50. The demo then overrides the fold-chosen rule with `min_contradiction` (`default.yaml:78`), which was chosen after looking at one example.
Why it matters: an operating point fit on one event, with 1 to 7 contradiction stories, carries event-specific noise into test. "Recall at P ≥ 0.55" is not a guarantee on new data. Thresholds at the grid floor mean the search wanted to go lower, which is argmax-like behaviour for a saturated model.
Fix (inside the existing architecture):
- (a) Validate on **both** non-test training events: pool them, or do an inner leave-one-event-out loop. Report the chosen point's spread.
- (b) Temperature-scale the logits on validation before thresholding, so thresholds are comparable across folds and models. Report ECE and a reliability diagram.
- (c) Extend the grid below 0.05, or threshold on logit margin or rank.
- (d) Report test precision against the 0.55 target per fold.

[B-6] Single seed; seed variance is unknown and probably not small next to the gains
Severity: high
Confidence: high
Basis: OBSERVED (seed 42, `default.yaml:3`; REPORT §6.10 caveat 2) and GENERAL (variance size)
Evidence:
- Each fold was trained once.
- Training loss falls from about 0.25-0.39 to 0.03-0.05 at epoch 2.
- Validation AP drops after epoch 1 in 3 of 4 folds: gw 0.551 → 0.477, chebdo 0.552 → 0.437, ott 0.783 → 0.753 (`finetune_results.json` history).
- The best epoch is 1 in 3 of 4 folds, so the reported model is a single early-stopped snapshot.
- Fine-tuning a cross-encoder on about 3k pairs commonly varies by several F1 points across seeds (GENERAL).
- REPORT §6.5 itself notes run-to-run variation for UMAP, HDBSCAN and k-means. There is also no repeated-seed run for clustering or for `max_pairs` sampling (`disagreement.py:26-27`).
Why it matters: fold deltas such as +0.004 (ott) or +0.014 (syd against the heuristic) are inside plausible seed noise.
Fix: run 3 to 5 seeds per fold. That is about 10 min × 4 folds × 4 extra seeds, roughly 2.7 h on the RTX 4050. Report mean ± sd per fold, and seed-and-fold bootstrap CIs for the headline deltas.

[B-7] The cluster-level veracity-proxy comparison is not valid across algorithms and only weakly informative within one
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:

Cluster counts and proxy prevalence differ a lot between configurations (`pipeline_results.json`):

| Configuration | Clusters | Prevalence | F1 | Flag-all F1 |
|---|---|---|---|---|
| kmeans/MiniLM | 30 | 0.70 | 0.875 | 0.824 |
| graph/MiniLM | 40 | 0.625 | 0.820 | 0.769 |
| hdbscan/MiniLM | 172 | 0.395 | 0.616 | 0.566 |
| agglomerative/MiniLM | 102 | 0.275 | 0.568 | 0.431 |

- Each configuration is scored on its own clusters against its own labels. F1 against a self-defined positive class with 70% prevalence is not comparable with one at 27%. REPORT §6.6 says this, but the table and BRIEF claim 4 still rank on it.
- The flag rule is "≥ 1 pair at or above the threshold" (`disagreement.py:55`, `min_contradicting_pairs: 1`). The threshold sweep from 0.5 to 0.99 moves flagged clusters only 54 → 45 (`RESULTS_TABLES.md:112`). The mean and min direction rules give identical cluster results (53/102, F1 0.568). Cluster flags are therefore close to insensitive to the NLI score and are probably driven by how many pairs a cluster has (UNVERIFIED: cluster sizes are not saved in the results).
- No size-only or pair-count baseline exists (`experiments.py:459-470` has random, flag-all and flag-none only).
- Within the default configuration, against flag-all (computed, bootstrap over 102 clusters): detector F1 95% CI [0.427, 0.690]; flag-all [0.314, 0.532]; paired difference +0.136, CI [0.041, 0.228]. Fisher exact test on flagged × positive: OR 6.7, p = 0.0003. The gain is real at the cluster level, but clusters are not iid (4 events, 15 to 34 clusters each), and the per-event comparison includes germanwings with 15 clusters and F1 0.857.
- The proxy itself is weak: ERROR_ANALYSIS FN1-FN5 and FN9 are all "proxy is wrong".
Why it matters: BRIEF claim 4 rests on a metric that may be measuring cluster size. The algorithm ranking (kmeans 0.875 > graph 0.820 > ...) is an artefact of base rate, and HDBSCAN versus agglomerative is not comparable either.
Fix:
- Add a "flag the top-k clusters by number of cross-account pairs" baseline at the system's flag rate, and AP of size as a ranking score.
- Compare algorithms on a **fixed reference partition**: score each algorithm's flags projected onto the annotated rumour-story groups, or onto the RTE labelled pairs as §6.4 does. Do not use each algorithm's own clusters.
- Report AP with ranking by `disagreement_score`; today AP ranks by `max_contradiction` (`experiments.py:342`).
- Demote cluster-level proxy results to secondary, which REPORT partly does already.

[B-8] The default clustering and direction rule were selected on the evaluation pairs, then reused in the fine-tuning "held-out" pipeline view
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- REPORT §5 lines 205-208: agglomerative and `min_contradiction` were "chosen on the labelled pairs themselves". `finetune.pipeline_eval_algorithm: agglomerative` (`default.yaml:107`) is then used for the fold pipeline numbers (`finetune_nli.py:507`).
- So clustering was chosen with every event's labelled pairs, including each fold's test event.
- `distance_threshold: 0.45` and `noise_assign_min_sim: 0.6` were "set once and not tuned". That means no sweep exists, not that they were selected fairly.
Why it matters: the pipeline P/R numbers in §6.4 and §6.10 are optimistic, by an unknown amount. Only 4 algorithms were compared, so the selection bias is probably modest (GENERAL).
Fix: inside each LOEO fold, choose the clustering algorithm and distance_threshold on the training or validation events, by same-cluster contradiction rate at a matched pair budget, and apply that choice to the test event.

[B-9] Missing ablations: most components have no with/without result
Severity: high
Confidence: high
Basis: OBSERVED
Evidence (component: status):

| Component | Status | Evidence |
|---|---|---|
| Cleaning, min_tokens, language filter | **No ablation** | — |
| Dedup | **No ablation** | The labelled-pair view forces `dedup=False` (`experiments.py:170`); the deployed pipeline uses `dedup: true` (`default.yaml:26`), so the two views differ and neither is compared with the other. |
| Embedding choice | Partial | Clustering quality and cluster-proxy cover MiniLM and mpnet. The labelled-pair pipeline view (§6.4) and the fine-tune pipeline view use MiniLM only (`experiments.py:769`, `finetune_nli.py:508`). |
| Clustering algorithm | Exists (§6.4) | Test-selected (B-8). |
| distance_threshold | **No sweep** | — |
| Noise assignment (nearest / keep, 0.6) | **No ablation** | — |
| cross_account_only | **No ablation** | The labelled-pair view does not apply it at all (`experiments.py:187-192`). |
| Direction rule | Exists | §6.2, §6.6, the validation comparison. |
| Contradiction threshold | Exists | Sweep (cluster-level), val-tuned (fine-tune). No pair-level PR curve in the tables. |
| max_pairs_per_cluster = 400 | **No ablation** | The labelled-pair view ignores the cap (same-cluster test only, `experiments.py:192`); the all-pairs comparison removes it (`experiments.py:482`). The deployed sampling is never evaluated. REPORT §6.7 says agglomerative produces "a few very large clusters". |
| score_weight_fraction 0.5 | **Never evaluated** | `cluster_level_metrics` uses `flagged` for P/R and `max_contradiction` for AP (`experiments.py:339-342`). The weighted score only orders the report. |
| min_contradicting_pairs 1 | **No ablation** | — |
| Fine-tuning: weighted loss | **No ablation** | `imbalance` options exist (`finetune_nli.py:245-258`) but were not run. |
| Fine-tuning: both_directions | **No ablation** | — |
| Fine-tuning: MNLI replay 3000 | **No ablation** | No MNLI dev accuracy after fine-tuning, so forgetting, which replay is meant to prevent, is unmeasured. |
| Fine-tuning: early stopping | Implicit only | Epoch history exists; no fixed-epoch comparison on test. |
| Fine-tuning: validation scheme | Partial | The random pair split was tried and dropped (val AP 1.0); not reported as a row. |
| Findings summary (agree/conflict/unique) | **Not evaluated** | REPORT §8.9. |

Why it matters: of about 17 components, 4 have a with/without result. Several defaults (cap, weight, min_pairs, cross-account) decide what the user sees but have never been measured.
Fix: see the Ablation Table to Run below.

[B-10] Strong simple baselines and model-swap checks are missing
Severity: medium
Confidence: high
Basis: OBSERVED (`default.yaml:31`; REPORT §5 lines 211-212 "Not run: ... roberta-large-mnli") and GENERAL
Evidence:
- One NLI checkpoint only. The `--compare-nli` path exists (`experiments.py:744-754`) but was never run.
- No larger NLI model (for example deberta-v3-large trained on MNLI/FEVER/ANLI, or roberta-large-mnli). This means you cannot tell how much of the 0.46 3-class accuracy and 0.33 recall is the checkpoint and how much is the task.
- No calibrated zero-shot baseline (temperature scaling on the validation event).
- At pair level, `max_contradiction` has higher F1 (0.502) and higher recall than the shipped `min` (F1 0.430). The choice is a stated preference for precision, but there is no precision-matched comparison.
- LLM-only baseline: none. This is a **judgement call**. The brief forbids an LLM in the detection loop, but an evaluation-only comparison (zero-shot LLM judging "same event and cannot both be true" on the same LOEO test pairs, or on the sentence-level subset from B-3) would calibrate how far the NLI system is from a strong reference. Flag it as non-deployable if run.
Why it matters: without a larger NLI model and a calibrated baseline, "fine-tuning helps" cannot be separated from "any stronger or better-calibrated scorer helps".
Fix:
- Run `--compare-nli` with roberta-large-mnli. The 6 GB GPU fits inference at fp16 with batch about 16 (UNVERIFIED).
- Add a deberta-v3-large NLI zero-shot row in the LOEO table, scored with the same val-tuned operating-point procedure.
- Optional: an LLM evaluation-only row on a 300-500 pair subset.

[B-11] The labelled-pair "pipeline" view does not measure the deployed pipeline
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence: `pipeline_on_labelled_pairs` (`experiments.py:162-209`) counts a pair as flagged if both tweets share a cluster and P(contradiction) ≥ thr. Three things differ from the deployed pipeline:
- it clusters without dedup (line 170);
- it applies no `cross_account_only` filter;
- it applies no `max_pairs_per_cluster` sampling.

The deployed `run_pipeline` does all three (`pipeline/run.py:151-156`).
Why it matters: the reported pipeline recall (0.132; 0.118 → 0.357 for fine-tuning) is an upper bound for the deployed tool. In large agglomerative clusters, the 400-pair cap randomly drops scored pairs.
Fix: compute the labelled-pair view through `run_pipeline` with the deployed config. Report the fraction of labelled contradictions that are (a) co-clustered, (b) cross-account, (c) sampled under the cap, (d) flagged. That is a recall-loss waterfall.

[B-12] Cost against benefit: clustering halves F1 at pair level; its precision gain is within noise
Severity: medium
Confidence: medium
Basis: OBSERVED (computed Wilson CIs, which are optimistic per B-1)
Evidence:

On the 5,230 mapped labelled pairs:

| | Flagged | Precision (Wilson 95% CI) | Recall (Wilson 95% CI) | F1 |
|---|---|---|---|---|
| NLI only | 727 | 0.618 [0.582, 0.652] | 0.333 | 0.433 |
| Agglomerative pipeline | 265 | 0.672 [0.613, 0.725] | 0.132 [0.115, 0.151] | 0.221 |

- The precision CIs overlap even under the iid assumption. Under story-group resampling they would overlap much more.
- The pipeline's F1 is below flag-all (0.406).
- Only 55.1% of contradictions are co-clustered (`RESULTS_TABLES.md:51`). That caps recall regardless of the NLI model, and REPORT §6.10 agrees ("Clustering now limits recall more than the NLI model").
- The stronger case for clustering is realism, not these labelled pairs:
  - same-story rate of flagged pairs: 2.1% → 8.1%;
  - flagged pairs: 28,735 → 3,910;
  - pairs scored: 81.7% fewer (`RESULTS_TABLES.md:121-125`).
  - But debunk pairs kept together fall to 73 of 145, and debunk pairs flagged go 36 → 25.
- The "unknown false flags 9.4% → 1.2%" result is close to built in. Unknown pairs are by construction two *unrelated* stories (B-3), so any topic clustering separates them.
- Cost: all-pairs NLI on 1,283 claims took 641 s cold (REPORT §7). That bound is real for large events, but it is a compute argument, not an accuracy one.
Why it matters: each stage needs to justify itself on an equal budget. Today, clustering's only measured accuracy benefit is the noisy precision gain plus the proxy same-story rate, at a 60% relative recall loss.
Fix:
- Compare clustering against a simpler candidate filter at **matched pair budget**: for example cosine top-k neighbours per claim, a cosine ≥ τ gate, or an entity-overlap gate (REPORT §10 already proposes this). Plot co-retained contradictions against pairs scored.
- Add "neighbouring clusters" pairing as a recall-recovery variant.
- Keep the clustering stage as the architecture requires, but measure its curve.

[B-13] The error analysis is small, coarse, single-annotator and on a stale config; there is no counted taxonomy for RTE or the fine-tuned model
Severity: medium
Confidence: high
Basis: OBSERVED (REPORT §6.9 lines 438-457; `ERROR_ANALYSIS.md` §1-§3)
Evidence:
- 40 random flagged pairs, from the *earlier HDBSCAN* configuration, judged by one annotator (the AI that built the pipeline).
- Three buckets: genuine 11 (27.5%, Wilson [0.161, 0.428]), time-driven 6, not a contradiction 23.
- 18 hand-picked FP/FN examples with commentary, uncounted.
- No error analysis on RTE false positives or false negatives. None on the fine-tuned model, whose germanwings entailment flag rate is 38.6%. None on what clustering drops.
- The recurring causes named in the text (attribution, headline fragments, aliases, numbers counting different things) are never counted.
Why it matters: without counts per type you cannot tell which add-on around NLI (attribution stripping, number normalisation, time gate, alias resolution) is worth building, or measure whether it worked.
Fix: two annotators each label 50 FP and 50 FN for each of zero-shot RTE, fine-tuned RTE and the deployed pipeline. Report kappa and use this taxonomy (multi-label allowed):
- **FP types:**
  - attribution/source difference ("AFP says" vs "CTV reports");
  - different referent/entity (two officials, two locations);
  - entity alias not resolved;
  - numeric, compatible counts (142 passengers + 6 crew vs 150 people);
  - temporal update (state changed);
  - non-propositional fragment (headline, caption, tribute);
  - hedged or reported speech treated as an assertion;
  - same-story elaboration (label says entailment, texts differ in details);
  - label error (story-level label wrong at sentence level).
- **FN types:**
  - numeric conflict missed;
  - entity alias;
  - implicit or world-knowledge contradiction;
  - negation scope;
  - one direction only (lost to the min rule);
  - pair dropped by clustering;
  - pair dropped by cross-account filter, dedup or cap;
  - label error.

Report each type's share and the per-type recall or precision before and after each added module.

[B-14] Reporting details that bias the summary
Severity: low
Confidence: high
Basis: OBSERVED
Evidence:
- The fold mean is unweighted (`finetune_nli.py:552-560`) although test sizes range from 1,100 to 1,936 and contradiction story counts from 1 to 7.
- The "recall at P ≥ 0.55" mode delivers P 0.464 on gw test and 0.392 for zero-shot on syd test, and REPORT does not flag the target miss per fold.
- MNLI dev accuracy (0.900) is reported for the zero-shot model only, not after fine-tuning.
- The demo model has no held-out evaluation (REPORT §6.10 last caveat).
Why it matters: these are small, but together they tilt the summary favourably.
Fix:
- Report micro-pooled and macro results.
- Mark target misses per fold.
- Add MNLI dev accuracy for each fold checkpoint, which is cheap: about 20k pairs at 550 pairs/s.

---

## Ablation Table to Run

All rows use the LOEO folds. Operating points are chosen on validation events only, and B-5 recommends pooling both non-test training events.

Metric columns:
- **Contr. P / R / F1 / AP**: mean ± fold sd, plus a story-pair-group bootstrap CI; always state the number of contradiction stories behind the metric.
- **Ent. flag**: entailment-pair flag rate (story-cue symptom).
- **Co-clust. contr.**: % of labelled contradictions co-clustered (and surviving the cap and cross-account filter where applicable).
- **Pipe P / R**: deployed pipeline on labelled pairs (B-11).
- **Cluster AP vs size baseline**: cluster-level AP against the veracity proxy, minus the size baseline's AP.
- **Pairs scored**: cost.
- **Seeds**: number of seeds.

"✓" means the metric applies to the row; "—" means it does not. The order is how much each row would teach.

| # | Variant (vs current default) | Contr. P/R/F1/AP | Ent. flag | Co-clust. contr. | Pipe P/R | Cluster AP vs size baseline | Pairs scored | Seeds | Teaches |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Trivial pair baselines**: flag-all; random@system rate; negation / number / antonym / union cue; TF-IDF and MiniLM cosine val-tuned. Partly computed above: union mean F1 0.456, flag-all 0.412. | ✓ | ✓ | — | ✓ | — | 0 | 1 | Whether NLI beats regexes; sets the floor for every claim (B-2) |
| 2 | **Story-structure oracle** (two stories × cosine) and **sentence-level relabelled subset** (~850 pairs, 2 annotators): zero-shot vs fine-tuned vs oracle | ✓ | ✓ | — | — | — | 0 | 1 | Whether fine-tuning learns contradiction or story cues (B-3); computed oracle mean F1 0.637 |
| 3 | **Fine-tuned × 5 seeds** vs **zero-shot val-tuned**, paired per-pair predictions saved | ✓ | ✓ | — | ✓ | — | — | 5 | Whether the +0.158 F1 and +0.077 pipeline-recall deltas survive noise (B-4, B-6) |
| 4 | **Operating point**: single val event vs pooled 2-event val vs inner LOEO; ± temperature scaling (report ECE); grid extended below 0.05 | ✓ | ✓ | — | ✓ | — | — | 3 | Threshold stability; makes "P ≥ 0.55" a real guarantee (B-5) |
| 5 | **Candidate filter at matched budget**: agglomerative distance_threshold {0.35, 0.45, 0.55, 0.65, 0.75}; cosine top-k {5, 10, 20}; cosine ≥ τ; + neighbouring-cluster pairs; no clustering; clustering chosen inside the fold (B-8) | ✓ (pipeline) | — | ✓ | ✓ | ✓ | ✓ | 1 | The recall ceiling (55% now) against cost; whether clustering earns its place (B-12) |
| 6 | **Deployed-pipeline recall waterfall**: co-clustered → cross-account → survives cap {100, 400, ∞} → flagged; ± dedup; noise {keep, nearest@0.5/0.6/0.7} | — | — | ✓ | ✓ | ✓ | ✓ | 3 (sampling) | Where recall is lost in the real tool (B-11) |
| 7 | **Fine-tuning knock-outs** (one at a time): − weighted loss (none / sampler); − both_directions; − MNLI replay (0 / 3000 / 10000) + MNLI dev accuracy after; fixed 1 epoch vs early stopping; lr 1e-5 | ✓ | ✓ | — | ✓ | — | — | 3 | Which recipe parts matter; whether replay prevents forgetting (B-9) |
| 8 | **Stronger / other NLI, zero-shot**: roberta-large-mnli; deberta-v3-large (MNLI/FEVER/ANLI); same val-tuned operating point | ✓ | ✓ | — | ✓ | ✓ | — | 1 | Checkpoint dependence; whether fine-tuning beats simply using a bigger model (B-10) |
| 9 | **NLI + cue post-/pre-filters** (architecture kept): number-mismatch booster; attribution-phrase stripping before NLI; same-entity gate; timestamp-order "update vs dispute" tag; require agreement of NLI and cue | ✓ | ✓ | — | ✓ | ✓ | — | 1 | Which add-ons around NLI fix the taxonomy's top error types (B-13) |
| 10 | **Cluster score and flag rule**: score_weight_fraction {0, 0.25, 0.5, 0.75, 1}; min_contradicting_pairs {1, 2, 3, fraction ≥ 0.05}; **size-only baseline**; AP ranking by disagreement_score | — | — | — | — | ✓ | — | 1 | Whether cluster flags reflect NLI or cluster size (B-7) |
| 11 | **Embedding** MiniLM vs mpnet in the labelled-pair and fold pipeline views | — | — | ✓ | ✓ | ✓ | ✓ | 1 | Whether the 55% co-clustering limit is the embedder or the algorithm |
| 12 | **LLM-as-judge, evaluation only** (judgement call; not in the detection loop) on the sentence-level subset and 300-500 LOEO test pairs | ✓ | ✓ | — | — | — | — | 1 | Distance to a strong reference; checks the RTE labels |

Minimum set before making any headline claim: rows 1, 2, 3 and 6. Rows 1 and 2 are CPU-only and take minutes. Row 3 is about 3 GPU-hours. Row 6 needs only cached NLI scores.
