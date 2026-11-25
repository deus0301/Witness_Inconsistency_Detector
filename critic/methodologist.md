# Methodologist review: evaluation validity

Scope: leakage, split design, labels, metrics, statistics, and claims against evidence. Every number
below was read from the repo files or recomputed by me from them. My recomputations used the
PHEME RTE XMLs plus the per-pair NLI scores in the SQLite caches under `outputs/cache/`
(zero-shot cache and the four fold caches, whose run ids match `finetune_results.json`). They
reproduce every reported per-fold P/R/AP in `finetune_results.json` to 3 decimals. The scratch
scripts are outside the repo (in `critic/scripts/`: `leak.py`, `boot.py`, `calib.py`, `lex.py`).
The results JSONs contain no per-pair predictions.

---

[M-1] The effective sample size is about 21 contradiction story-pairs, not 1,378 pairs, and the fine-tuning gain is not significant at that level
Severity: critical
Confidence: high
Basis: OBSERVED
Evidence:
- Labels are a deterministic function of the pair's story set. Grouping the 5,413 pairs by their `rumours` attribute gives 85 story-groups and 0 groups with mixed labels.
- There are only 21 contradiction groups across the 4 events (Ottawa 6, Sydney 7, Charlie Hebdo 7, Germanwings 1).
- Germanwings has 61 distinct tweets, 4 stories and one contradiction group: all 257 contradictions are "(Up to) 150 people perished" vs "There were 148 people onboard". Its reported recall of 0.922 is the hit rate on that single story pair.
- In Ottawa, one group holds 300 of 377 contradictions (80%). In Charlie Hebdo, one group holds 256 of 427 (60%).
- Each tweet appears in 16.6 to 38.1 pairs on average (Germanwings max 49).
- Bootstrap, resampling pairs: mean-over-folds ΔF1 (fine-tuned best-F1 minus zero-shot val-tuned) is [0.132, 0.182].
- Bootstrap, resampling story-groups within each event: the same ΔF1 is [0.002, 0.278] and ΔAP vs zero-shot is [-0.001, 0.299]. The Germanwings recall CI is [0.000, 0.922].
- Treating the 4 events as the unit (t, df=3) on the per-fold differences in `finetune_results.json`:
  - ΔF1 vs zero-shot val-tuned: 0.282, 0.141, 0.204, 0.004, giving roughly [-0.03, 0.35].
  - ΔAP: 0.089, 0.041, 0.365, -0.061, giving roughly [-0.18, 0.40].
  - ΔRecall vs default: 0.829, 0.230, 0.342, -0.093, giving roughly [-0.28, 0.93].
- Within events there are also near-duplicate tweets: Charlie Hebdo has 18 tweet pairs with token Jaccard ≥ 0.8 and 6 duplicate normalised texts.
Why it matters: Pair-level counts make the per-fold numbers look precise when they measure a handful of story pairs. Pair-bootstrap CIs (the natural thing to add) would be too narrow by a large factor. At the event level, none of the claimed fine-tuning gains is distinguishable from zero.
Fix: Report CIs from a cluster bootstrap over story-groups (or stories) within each event, plus a per-event table instead of only the unweighted mean. Report the number of contradiction story-groups next to every recall figure. Save per-pair predictions (pair id, fwd/back probabilities, story key) to `reports/results/` so CIs can be regenerated. Phrase claim 3 as "improved on 3 of 4 events" unless it survives a story-level or event-level test. Add a fifth event if one can be labelled.

[M-2] "Doubles recall (0.31 -> 0.64)" compares operating points, not models, and the mean is carried by one story pair
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- The baseline for claim 3 is `zero_shot_default` (min_contradiction, 0.9), a precision-oriented point. The fine-tuned side is the validation-tuned best-F1 point (`finetune_results.json`, `settings`).
- The like-for-like row, zero-shot validation-tuned best F1, has mean recall 0.547 and F1 0.449 (REPORT.md:491). Against it, recall goes 0.547 -> 0.635 (+0.09, not x2).
- The per-fold recall gains are 0.093->0.922 (GW), 0.177->0.407, 0.234->0.576, and 0.727->0.634 (Ottawa gets worse).
- Without Germanwings, mean recall vs default is 0.379 -> 0.539, and fine-tuned R 0.539 is below zero-shot val-tuned R 0.630 (my computation from the JSON).
- Pipeline recall 0.118 -> 0.357 is also carried by GW (0.033 -> 0.611). Without GW it is 0.147 -> 0.273.
- Mean pipeline precision does not rise: 0.634 -> 0.623 (`mean_over_folds.pipeline`).
- Micro-pooled over all 5,413 test pairs: zero-shot default P 0.619 R 0.329, zero-shot val-tuned P 0.424 R 0.588, fine-tuned P 0.613 R 0.618.
Why it matters: The headline ratio mostly comes from moving the threshold and from one Germanwings story pair, not from a general gain in contradiction detection.
Fix: Make the primary comparison threshold-free (AP and a PR curve per event) plus matched-precision recall (recall at the precision the zero-shot default reaches on validation). Report leave-one-event-out sensitivity of the mean. Drop "doubles/triples" unless it holds against the validation-tuned zero-shot point.

[M-3] Fine-tuning mainly learned "same topic vs unrelated story", which clustering already provides, and not contradiction vs agreement
Severity: high
Confidence: medium
Basis: OBSERVED (numbers) / UNVERIFIED (interpretation)
Evidence:
- There is no textual leakage to explain the gain. Between LOEO train and test files I measured 0 shared tweet ids, 0 exact normalised-text matches, 0 test tweets with a train near-duplicate (Jaccard ≥ 0.6) and 0 shared story strings. The risk lies in the label structure.
- Contradiction and entailment pairs are both on-topic. "Unknown" pairs are two unrelated stories (data/pheme_rte.py:9-14).
- AP with the subsets split out, zero-shot default -> fine-tuned:

  | Event | Contradiction vs unknown | Contradiction vs entailment |
  |---|---|---|
  | GW | 0.516 -> 0.791 | 0.661 -> **0.580** |
  | CH | 0.367 -> 0.743 | 0.861 -> 0.909 |
  | Syd | 0.473 -> 0.512 | 0.700 -> 0.755 |
  | Ott | 0.974 -> 0.956 | 0.864 -> **0.810** |

- On GW, the fine-tuned model flags 36% of pairs inside the "148 onboard" story and 56% inside the "150 perished" story (entailment groups). Overall entailment flag rate is 0.386.
- Pipeline recall gains are smaller than NLI-only recall gains in every fold that improved, because clustering has already removed unrelated pairs (GW NLI R +0.83 vs pipeline +0.58; CH +0.34 vs +0.29; Syd +0.23 vs +0.09; Ott -0.09 vs -0.01).
Why it matters: Most of the gain is in separating contradictions from unrelated pairs. Inside a sub-event cluster, which is where the NLI stage runs, that separation is already largely done. The step that matters there, contradiction vs agreement, improved on 2 events and got worse on 2.
Fix: Report fine-tuning results on same-cluster pairs only, and separately as contradiction-vs-entailment AP. Add a "story-relatedness" baseline: a classifier on embedding cosine only, trained under LOEO. For reference, a lexical LR baseline I ran (TF-IDF cosine, Jaccard, number and negation mismatch, LOEO) gets mean AP 0.356. Evaluate on a small set relabelled sentence by sentence to see whether the fine-tuned model's flags hold at sentence level.

[M-4] The zero-shot configuration, clustering choice and demo settings were all selected on the data that is later reported as test
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- The min_contradiction rule and agglomerative clustering were "chosen after seeing results" on the 5,413 labelled pairs (REPORT.md:205-208, 300). Those pairs are exactly the union of the four LOEO test files (pair counts 1100/1214/1936/1163 match). Claims 1 and 2 are therefore in-sample, and `zero_shot_default` in every fold is test-informed.
- The validation scheme was chosen after seeing the first fold's test scores (REPORT.md:552-554).
- The demo rule `finetune.aggregation_override: min_contradiction` (configs/default.yaml:78) was chosen "after looking at one example" (REPORT.md:578). That example is the only witness-style input, `examples/witness_accounts.txt`, 390 bytes. The choice also used `direction_rules_on_validation`, whose four validation events are all 5,413 labelled pairs.
- The demo threshold 0.5 is the median of the two max-rule fold choices 0.95 and 0.05. No fold chose 0.5 (`all_events_model.operating_point.modes.best_f1.fold_choices`).
- The demo model (trained on all events, 1 epoch) has no evaluation at all.
Why it matters: No untouched data remains for the configuration that ships (demo model, min rule, threshold 0.5) or for the headline zero-shot numbers. The reported figures are upper bounds.
Fix: Freeze the configuration and report the shipped configuration's numbers as "selected in-sample". Add a nested LOEO: choose the rule, threshold and clustering algorithm on the 3 non-test events in each fold, and report the test event. Hold out a witness-style test set that is never used for choices like the override.

[M-5] The witness-statement claim has no evaluation in the target domain
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- All labelled evaluation is on tweets (BRIEF claim 5).
- The only witness-style input is a 3-account, 390-byte invented file (`examples/witness_accounts.txt`). It was used to tune the demo rule (M-4).
- REPORT.md:616-617 admits witness-style input "has had a small made-up example and unit tests only".
- Known failure on that file: "stayed at the scene" vs "got out and called an ambulance" is flagged by both models (REPORT.md:621-625, limitation 11).
- Tweets differ from testimonies in the ways that matter here. Tweets are headline fragments and attributions (ERROR_ANALYSIS FP6-FP9); testimonies are multi-sentence first-person narratives with hedging, time references and vantage points.
Why it matters: Nothing measured supports precision or recall on witness statements. Domain shift is the main risk named in the NLI reference.
Fix: Build a small labelled witness-statement test set before any claim about that use: at least 30 to 50 scenarios, with claim pairs labelled by two annotators and agreement reported. Report pair-level and cluster-level P/R there, with scenario-level bootstrap. Keep it untouched by any tuning.

[M-6] Thresholds are uninformative: probabilities are saturated and uncalibrated, val-chosen points sit at the grid floor and do not transfer
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- Fine-tuned best-F1 thresholds are 0.95, 0.05, 0.07 and 0.06, and recall-at-precision thresholds include 0.05 twice. The grid starts at 0.05 (configs/default.yaml:105), so the optimum may lie below the search range. Zero-shot val-tuned picks 0.05 on CH and Ottawa too.
- Share of test contradiction probabilities strictly between 0.05 and 0.95: fine-tuned GW 0.9%, Syd 4.6%, CH 6.7%, Ott 15.3%.
- ECE for the contradiction probability on test: zero-shot 0.13 to 0.23, fine-tuned 0.12 to 0.25.
- Validation to test transfer fails:
  - GW fine-tuned P falls from 0.608 on validation to 0.467 on test, below the 0.55 target.
  - Sydney F1 falls from 0.784 to 0.465.
- At cluster level, the 0.5 to 0.99 sweep moves flags only from 54 to 45 (REPORT.md:384-388).
Why it matters: "Threshold 0.9" and "operating point chosen on validation" suggest control that does not exist. Precision on a new event cannot be predicted from validation, and the user-facing threshold slider is mostly inert.
Fix: Extend the grid below 0.05 or select on the argmax label directly. Add post-hoc calibration (temperature scaling fitted on the validation event) and report ECE and reliability plots per event. Report the spread of validation-to-test precision across folds as the expected error of the operating point.

[M-7] The cluster-level veracity proxy does not measure contradiction and is confounded with cluster size; the baseline comparison omits a size baseline
Severity: medium
Confidence: medium
Basis: OBSERVED (definitions, counts) / GENERAL (size confound) / UNVERIFIED (magnitude)
Evidence:
- Positive means a cluster mixes a true-story and a false-story tweet, or a debunk and its rumour (data/proxy_labels.py:104-109). Two such tweets need not conflict.
- A cluster is flagged if at least 1 of up to 400 pairs reaches the threshold (configs/default.yaml:55-58). Both the label and the flag become more likely as clusters get larger.
- REPORT admits that "the cluster proxy rewards large clusters" (REPORT.md:141). MiniLM k-means scores F1 0.875 at 70% prevalence (REPORT.md:346) because its clusters are coarse.
- Only 28 positives in 102 clusters. My bootstrap from the reported counts (TP 23, FP 30, FN 5, TN 44, clusters assumed independent): detector F1 [0.426, 0.690]; F1 minus flag-all [0.040, 0.228].
- Per-cluster records (size, label, score) are not saved in `pipeline_results.json`, so a size-only baseline cannot be computed from the artifacts.
Why it matters: "Beats flag-all/random" (claim 4) may reflect the detector flagging larger clusters, not finding disagreement.
Fix: Save per-cluster records. Add baselines that flag the top-k clusters by size, and by number of distinct stories, at the detector's flag rate. Report AP of disagreement score vs proxy restricted to size-matched strata. Present claim 4 as proxy-only evidence.

[M-8] "Clustering is a precision filter" rests on a gain inside overlapping CIs, measured on the set used to choose the clustering
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- Pipeline precision is 0.672 (178/265, Wilson [0.613, 0.725]) vs NLI-only 0.618 (449/727, [0.582, 0.652]) (REPORT.md:283-287). The pipeline set is a subset of the NLI-only set.
- Agglomerative was picked on these pairs (M-4).
- The drop in unknown-pair flags from 9.4% to 1.2% is partly by construction: unknown pairs are two unrelated stories, and topic clustering separates them. The same clustering also drops 44.9% of labelled contradictions (only 55.1% co-clustered).
- In the fine-tune folds, going from NLI-only to pipeline moves mean precision from 0.584 to 0.634 (zero-shot default) and from 0.629 to 0.623 (fine-tuned). There is no consistent gain.
- Section 6.7 uses "flagged pairs about the same rumour story" (2.1% vs 8.1%) as the precision measure. But labelled contradictions are between two different stories (data/pheme_rte.py:9-12), so this measure does not track contradiction.
Why it matters: Claim 2's precision half is not established. Only the cost half (81.7% fewer pairs) is directly supported.
Fix: Report the precision difference with a paired story-group bootstrap. Select the clustering algorithm and threshold inside each LOEO fold. For section 6.7, replace the same-story rate with RTE-labelled precision on the mappable subset.

[M-9] Pair-level precision comes from a curated, contradiction-rich set; the only deployment-like precision estimate is small, single-annotator and from the builder
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- The RTE base rate is 25.5% contradiction and only 38% unknown (REPORT.md:230, 302). In real events almost all pairs are unrelated (REPORT.md:302-304).
- The audit found 11/40 genuine (Wilson [0.161, 0.428]) and 17/40 genuine or time-driven ([0.285, 0.578]).
- The audit was judged by "the AI assistant that built the pipeline" (ERROR_ANALYSIS.md:21-22), on the old HDBSCAN config that is no longer the default (ERROR_ANALYSIS.md:3-7).
- There is no inter-annotator agreement for the audit or for the RTE labels.
Why it matters: Claim 1's "0.62 precision" does not hold for deployment. The audit points to between 0.16 and 0.58 on raw events, depending on whether time-driven conflicts count, and it is not independent.
Fix: Repeat the audit on the current default (and the fine-tuned demo model) with two human annotators blind to model scores, at least 100 flagged pairs, and report kappa. Report precision@k per event alongside pair-level P on RTE.

[M-10] The 3-class accuracy gain (0.46 -> 0.72) mostly measures learning the dataset's story-level "entailment" convention
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- Argmax entailment recall on test rises from 0.088-0.146 (zero-shot) to 0.586-0.835 (fine-tuned), on all 4 folds.
- RTE "entailment" means "same story", even when the tweets add different details (REPORT.md:119-120, 243-246).
- Some labelled contradictions are competing claims, not logical opposites ("Parliament Hill lockdown" vs "University of Ottawa lockdown", REPORT.md:121-122).
- Some unknown pairs are compatible, for example GW "(Up to) 150 people perished" vs "There are no survivors", 99 pairs labelled unknown.
Why it matters: Accuracy and macro-F1 gains do not show better inference. They show agreement with an annotation convention that differs from the witness-statement use.
Fix: Drop 3-class accuracy and macro-F1 from the headline. Report contradiction-vs-entailment AP and the story-group-level results from M-1. Label a sample of 200 RTE pairs at sentence level to estimate the label noise of each class.

[M-11] Single seed, very fast overfitting, and early stopping and operating point chosen on the same single validation event
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- Training history: loss drops from about 0.25-0.39 to 0.03-0.05 by epoch 2, and validation AP falls after epoch 1 in 3 of 4 folds (GW 0.551 -> 0.477, CH 0.552 -> 0.437, Ott 0.783 -> 0.753). The best epoch is 1 in 3 folds.
- Seed is 42 everywhere (configs/default.yaml:3), with one run per fold.
- The same validation event drives epoch choice, rule and threshold (training/finetune_nli.py:312-316, 454). Validation events have very different label mixes: Ottawa validation is 15% unknown, Charlie Hebdo validation is 45% unknown.
Why it matters: Run-to-run variance is unknown and may be the same size as the fold differences. Validation optimism is compounded across three selections.
Fix: Run at least 3 to 5 seeds per fold and report mean ± sd. Fix the epoch count (for example 1) across folds instead of early-stopping on the event that also selects the threshold, or split the validation role across the two non-test training events.

[M-12] Many configurations compared with no correction, and headline rows picked after the fact
Severity: low
Confidence: medium
Basis: OBSERVED
Evidence:
- Comparisons span 8 embedder × clustering configs (REPORT.md:343-350), 4 direction rules, a 9-point threshold grid, 5 settings × 4 folds, and 2 operating modes.
- Headline claims use the most favourable contrasts: the default zero-shot point for claim 3, flag-all and random for claim 4.
- Unweighted fold means are reported. Micro-pooled results differ, for example fine-tuned R 0.618 vs mean 0.635.
Why it matters: With this many options and 4 events, some differences will look meaningful by chance.
Fix: Pre-declare one primary metric (contradiction AP, story-level bootstrap) and one primary comparison (fine-tuned vs zero-shot, both val-tuned). Label everything else exploratory, and report both pooled and per-event figures.

---

## Claim vs evidence

| # | Claim (one sentence) | What the evidence supports |
|---|---|---|
| 1 | The NLI scorer flags labelled contradictions with about 0.62 precision (recall 0.33) on PHEME RTE. | In-sample (rule chosen on these pairs) on a curated set with a 25.5% base rate. Per-event precision ranges from 0.40 to 0.80. Deployment precision from the audit is about 0.16 to 0.43 genuine, from a non-independent annotator (M-4, M-9). |
| 2 | Clustering is a precision filter and cost saver. | Cost: supported (81.7% fewer pairs). Precision: 0.62 -> 0.67 with overlapping CIs, the clustering was chosen on the same pairs, the fine-tune folds show no consistent pipeline precision gain, and 45% of contradictions are lost (M-8). |
| 3 | Fine-tuning doubles contradiction recall and triples pipeline recall on held-out events. | Recall rises against a precision-oriented baseline. Against the validation-tuned zero-shot point it rises 0.547 -> 0.635. The gain is driven by one Germanwings story pair, Ottawa gets worse, story-level and event-level CIs include about 0, and the gain sits mainly in contradiction-vs-unrelated separation (M-1, M-2, M-3). |
| 4 | Cluster-level detection beats flag-all and random baselines on a veracity proxy. | It beats them on a proxy that does not measure contradiction and is confounded with cluster size. No size baseline exists. Under a cluster-independence bootstrap the margin over flag-all is [0.04, 0.23] F1 (M-7). |
| 5 | The target use is witness statements; evaluation is on tweets. | No witness-statement evaluation exists. The only witness example was used for tuning (M-4, M-5). |
