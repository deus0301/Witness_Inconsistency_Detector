# ML Project Review: Cross-Account Inconsistency Detector

Inputs: `critic/BRIEF.md`, then three isolated reviewers (`critic/methodologist.md` M-*, `critic/adversary.md` A-*, `critic/ablation_auditor.md` B-*). This file adds the developer change list (D-*) and the synthesis. Constraint honoured throughout: the cross-encoder NLI over in-cluster cross-account pairs stays; every change is an addition around it or an evaluation change.

## Claim and verdict

The project claims that clustering + NLI finds cross-account contradictions with ~0.62 precision, that clustering is a precision filter, that fine-tuning on PHEME RTE doubles recall / triples pipeline recall, and that this transfers to witness statements.

The evidence supports the **cost** claim (81.7% fewer pairs) and a **plausible but unproven** fine-tuning F1 gain. Everything else is weaker than stated. The labelled set holds only 21 independent contradiction story-pairs (checked: 85 story groups, 0 with mixed labels; germanwings has 1). A three-regex heuristic beats the zero-shot default, and a story-structure cue matches the fine-tuned model. The shipped configuration was chosen on the test data. Nothing has been measured on witness-style text, and probes on that text show near-certain false contradictions (score 1.000) for first-person pairs from different witnesses.

## Top priorities (do these first)

1. **Re-base the evaluation on honest units and fair references** (D-1, D-2, D-3). Save per-pair predictions. Report a story-group bootstrap CI and the number of contradiction stories next to every pair metric. Make "zero-shot, val-tuned" the reference row. Add the pair-level trivial baselines (flag-all, cue union, story oracle) and a cluster-size baseline. Addresses M-1, M-2, M-7, B-1, B-2, B-3, B-4, B-7. *Effect:* honest headlines. Expect "doubles / triples" to become "+0.16 F1, CI including ~0" (B-4). *Effort:* ~1 day, CPU only, from the cached scores in `outputs/cache/`.
2. **Build a witness-domain test set and a typed challenge suite, and never tune on them** (D-4). Addresses M-5, M-4, A-20, B-13. *Effect:* the first number that speaks to the actual use case. *Effort:* 1–2 days for 30–50 scenarios with 2 annotators; the challenge suite is ~2 hours using the adversary's ~64 probes as fixtures.
3. **Fix the pre-NLI text layer** (D-5, D-6, D-7). Use an abbreviation-aware splitter with fragment merging, ground the speaker ("I" → account id), guard against truncation, and put `max_length` into the cache key. Addresses A-1, A-7, A-10, A-12, A-13. *Effect:* removes a whole class of score-1.000 false alarms and lost claims. *Effort:* ~0.5–1 day.
4. **Add typed post-NLI checks that label rather than silently drop** (D-8). Cover numbers and units, attribution / non-observation / hedging, same-referent, and temporal updates. Addresses A-2, A-3, A-6, A-8, A-9, A-16, B-13. *Effect:* precision on witness text, and conflicts in the report gain an interpretable type. *Effort:* 1–2 days.
5. **Add a recall path around clustering, and a recall waterfall** (D-9, D-10). Use cross-cluster nearest-neighbour candidates, prioritised sampling under the 400-pair cap, and "never compared" separated from "not contradicted". Addresses A-17, A-18, B-11, B-12, M-8. *Effect:* recovers some of the 45% of contradictions lost to clustering, and shows where deployed recall goes. *Effort:* ~1 day.

## Findings by severity

Merged and deduplicated; the strongest evidence is kept. Original IDs are in brackets.

### Critical

**[C1] The effective sample is 21 contradiction story-pairs; pair-level CIs are about 10× too narrow** (M-1, B-1)
Confidence high · OBSERVED. I verified this directly: the label is a deterministic function of the `rumours` attribute. There are 85 groups, 0 mixed; contradiction groups per event are 7 / 1 / 6 / 7 (chebdo / gw / ottawa / ssiege). Germanwings recall measures one conflict, "150 perished" vs "148 onboard". A story-group bootstrap widens the keyword-baseline F1 CI from width 0.045 to 0.463 (B-1). It puts the fine-tune ΔF1 at [0.002, 0.278] and ΔAP at [−0.001, 0.299] (M-1).
*Fix:* a story-group bootstrap within each event; report per-event results and the number of contradiction groups; treat germanwings as anecdotal (D-1).

**[C2] Trivial baselines match or beat the zero-shot scorer, and a story oracle matches the fine-tuned one** (B-2, B-3, M-3)
Confidence high (numbers) / medium (interpretation) · OBSERVED.
- LOEO mean contradiction F1: flag-all 0.412, untuned negation / number / antonym cue union 0.456, zero-shot default 0.372, fine-tuned 0.607.
- "Different stories ∧ TF-IDF cosine ≥ t" reaches mean F1 0.637 and AP 0.685, against fine-tuned 0.607 / 0.612.
- Contradiction-vs-entailment AP falls after fine-tuning on gw (0.661 → 0.580) and ottawa (0.864 → 0.810). So the gain sits mostly in separating contradictions from *unrelated* pairs, which clustering already does.

*Fix:* report these baselines in every table (D-2), and report fine-tuning on same-cluster pairs and as contradiction-vs-entailment AP.

**[C3] First-person and different-referent pairs score contradiction ≈1.000** (A-1, A-9, A-19)
Confidence high · OBSERVED (real scorer, zero-shot and fine-tuned).
- "I was inside the shop when it happened." vs "I was outside on the pavement when it happened." scores 1.000.
- "The cyclist was badly injured." vs "The pedestrian was not hurt at all." scores 1.000.

Every first-person cross-account pair involves two different speakers, so this hits the target domain directly. Fine-tuning does not fix it (A-14).
*Fix:* speaker grounding before NLI (D-6), plus a same-referent check after NLI (D-8).

**[C4] The sentence splitter breaks on abbreviations; claims are lost and false agreement is reported** (A-12)
Confidence high · OBSERVED. I reproduced it: `split_sentences("It was approx. 10 p.m. when the car hit Dr. Patel on St. James Street.")` returns `['It was approx.', '10 p.m. when the car hit Dr.', 'Patel on St.', 'James Street.']` (`data/cleaning.py:63`). In a pipeline run, the "10 p.m. vs 6 p.m." conflict was never compared, and "It was approx." was reported as CORROBORATED.
*Fix:* D-5.

### High

**[H1] The fine-tuning headline compares operating points, not models** (M-2, B-4)
OBSERVED. Recall "0.31 → 0.64" uses the fixed 0.9 zero-shot point. Against the zero-shot point tuned the same way: recall 0.547 → 0.635; pipeline recall 0.281 → 0.357, worse in 2 of 4 folds, and germanwings alone supplies +0.42 of the gain. Without germanwings, fine-tuned recall (0.539) is *below* zero-shot val-tuned (0.630).
*Fix:* use zero-shot val-tuned as the reference row; drop "doubles / triples".

**[H2] The shipped configuration was selected on the test data** (M-4, B-8)
OBSERVED.
- The min rule and agglomerative clustering were chosen on the 5,413 pairs, which are exactly the union of the LOEO test files (REPORT.md:205-208).
- `pipeline_eval_algorithm: agglomerative` then reuses that choice inside the folds.
- The demo `aggregation_override: min_contradiction` was chosen on `examples/witness_accounts.txt`.
- The demo threshold 0.5 is a median that no fold chose.
- The all-events demo model has no evaluation.

*Fix:* nested LOEO for the rule, threshold and clustering (D-11); label shipped numbers "selected in-sample".

**[H3] There is no evaluation in the witness-statement domain** (M-5)
OBSERVED. The only witness input is 390 bytes and was used for tuning. It also yields a false conflict, "stayed at the scene" vs "got out and called an ambulance" scoring 0.996 (A-11).
*Fix:* D-4.

**[H4] Operating points are unstable, saturated and uncalibrated** (M-6, B-5, A-4)
OBSERVED.
- Chosen thresholds of 0.05 / 0.06 / 0.07 sit on the grid floor.
- Only 0.9–15% of fine-tuned test probabilities fall in (0.05, 0.95).
- ECE is 0.12–0.25.
- gw validation P 0.608 → test 0.467, below the 0.55 target.
- The demo threshold was chosen for the max rule but is used with the min rule.

*Fix:* temperature scaling on validation, a grid below 0.05 or a logit margin, and validation pooled over both non-test events (D-11).

**[H5] Non-observation, reported speech and temporal updates are scored as contradiction** (A-2, A-8, A-16)
OBSERVED.
- "The man had a gun." vs "I didn't see a gun." scores 0.999.
- "alive when I got there" vs "dead when the ambulance arrived" scores 1.000.
- "The driver told me he had stopped" vs "did not stop" scores 1.000.

*Fix:* D-8 tags.

**[H6] True contradictions are clustered apart and never compared, and the findings report them as "not contradicted"** (A-17, B-12, M-8)
OBSERVED.
- Only 55.1% of labelled contradictions are co-clustered.
- "The driver stayed at the scene." vs "He was gone before anyone could get his number plate." scores 1.000 when scored directly, but both claims end up unclustered.
- Clustering's precision gain is 0.618 → 0.672 with overlapping Wilson CIs, and it does not repeat in the folds (0.629 → 0.623 for fine-tuned).

*Fix:* D-9, D-10.

**[H7] Truncation and a stale cache** (A-10)
OBSERVED. A run-on statement stays one claim. At `max_length: 128` an identical final clause scores neutral 0.997; at 512 it scores entailment 0.996. The NLI cache key ignores `max_length` (`nli_scoring/scorer.py:153`), so changing it on a warm cache silently reuses old scores.
*Fix:* D-7.

**[H8] Single seed, early overfitting, and one validation event for three decisions** (M-11, B-6)
OBSERVED. Validation AP falls after epoch 1 in 3 of 4 folds. Seed is 42 only. The same event chooses the epoch, the rule and the threshold.
*Fix:* 3–5 seeds per fold, ~3 GPU-hours (D-12).

**[H9] The cluster-level proxy is confounded with cluster size** (M-7, B-7)
OBSERVED for the definitions; UNVERIFIED for the magnitude, because per-cluster sizes are not saved. k-means "wins" at 70% prevalence, and the threshold sweep 0.5–0.99 moves flags only from 54 to 45.
*Fix:* save per-cluster records, add a top-k-by-size baseline at the same flag rate, and rank AP by `disagreement_score` (D-3).

**[H10] Most components have no ablation** (B-9)
OBSERVED: about 4 of 17 components have a with/without result. Missing ones include `distance_threshold`, the noise strategy, `cross_account_only`, the 400-pair cap, weighted loss, `both_directions`, and MNLI replay (no MNLI accuracy after fine-tuning, so forgetting is unmeasured).

### Medium

- **[Md1] Deployment precision is unknown.** The audit is 40 pairs on the old HDBSCAN config, judged by the AI that built it; 11/40 genuine, Wilson [0.16, 0.43] (M-9, B-13). *Fix:* 2 human annotators, ≥100 flags on the current default and on the fine-tuned model, with kappa, using the FP/FN taxonomy in B-13.
- **[Md2] The labelled-pair "pipeline" view is not the deployed pipeline.** It runs without dedup, without the cross-account filter and without the cap (`evaluation/experiments.py:162-209` vs `pipeline/run.py:151-156`), so 0.132 recall is an upper bound (B-11). *Fix:* D-10.
- **[Md3] The 3-class accuracy gain (0.46 → 0.72) mostly measures learning the "same story = entailment" convention** (M-10). *Fix:* drop it from the headline.
- **[Md4] Numeric tolerance and units fail.** "about 30 mph" vs "about 50 km/h" is a 0.96 false flag; "about 20" vs "25" scores 0.449, next to the 0.5 threshold (A-6). *Fix:* D-8 numeric checker.
- **[Md5] Account parsing and within-account checks.** A witness's second paragraph becomes `account_2`, so a self-correction is treated as cross-account; within-account inconsistency is never checked (A-13).
- **[Md6] The pair cap is coarse.** 30 clusters hit the 400-pair cap; the largest is sampled at under 0.5% (A-18). *Fix:* prioritised sampling (D-9).
- **[Md7] Strong simple baselines were not run.** `roberta-large-mnli` is configured but not run, and there is no deberta-v3-large (B-10).
- **[Md8] Hedging is inconsistent.** "maybe" kills a flag (0.004) while "I think" keeps it (0.997) (A-3). This is handled by the D-8 hedge tag.

## Developer change list (D-*)

Each change can be evaluated on its own. All are additions or in-place edits; none replaces the NLI stage.

| # | Change | Addresses | Concrete edit | Expected effect / metric | Effort | Risk |
|---|---|---|---|---|---|---|
| D-1 | Per-pair prediction dump + story-group bootstrap | C1, H1, M-12 | In `training/finetune_nli.py` and `evaluation/experiments.py` (`eval_rte`), write `reports/results/pair_predictions_<run>.parquet` (pair id, event, `stories`, label, p_fwd, p_back). New `evaluation/bootstrap.py`: resample `stories` groups within each event (B=2000) and give CIs for P/R/F1/AP and paired deltas. `make_tables.py` prints CI and number of contradiction groups next to every metric, plus micro-pooled and per-event figures. | CIs about 10× wider; headline deltas restated | 4 h | low |
| D-2 | Pair-level baseline block | C2 | `evaluation/baselines_pairs.py`: flag-all, random at the system flag rate, negation / number / antonym cues and their union, TF-IDF cosine (val-tuned), story oracle (diagnostic only, not deployable). Call it from `eval_rte` and `run_fold`; add rows to the fine-tune tables. The adversary's and auditor's scratch scripts (`baselines.py`) are a starting point. | Shows whether NLI beats regexes at matched flag rate (P@rate, AP) | 3 h | low |
| D-3 | Cluster-level size baseline + per-cluster records | H9 | In `pipeline_results.json`, save per cluster: size, n_pairs_scored, n_stories, proxy label, max_contradiction, disagreement_score. Add `top_k_by_size` and `top_k_by_n_stories` at the detector's flag rate; compute AP from `disagreement_score` (`experiments.py:342`). | Tests whether claim 4 is a size effect | 2 h | low |
| D-4 | Witness test set + challenge suite | H3, C3, A-20 | `tests/challenge/witness_pairs.jsonl` holds the ~64 adversary probes, typed (first_person, referent, negation_nonobs, hedge, update, numeric_unit, attribution, coref, truncation, outside_knowledge), with expected label. `tests/test_challenge.py` runs under the `@pytest.mark.real_model` marker and reports pass rate per type. Separately, `data/witness_eval/` holds 30–50 scenarios (3–5 accounts each, hand-written or adapted), with claim pairs labelled by 2 annotators and kappa. Freeze it; never use it for any selection. | First witness-domain P/R; per-type pass rates track D-5 to D-8 | 2 h + 1–2 days | annotation effort |
| D-5 | Abbreviation-aware splitter + fragment merge | C4 | `data/cleaning.py:63`: protect a known-abbreviation list (approx., Dr., St., Mr., Mrs., a.m., p.m., e.g., i.e., No.) and decimals before `_SENT_SPLIT`. After splitting, merge any piece with fewer than `min_tokens` tokens, or with no verb (cheap check: no token from a small verb lexicon, or spaCy if already installed), into its neighbour. Unit tests with the A-12 strings. | Removes the lost-time and false-CORROBORATED class; challenge "truncation / fragment" pass rate | 2 h | low |
| D-6 | Speaker grounding | C3, A-7 | In `pipeline/run.py`, before scoring, rewrite first-person pronouns per claim: I / me / my → "<account_id>" / "<account_id>'s" (and we → "<account_id>'s group"). Keep the original text for the report. Config flag `nli.ground_speaker: true`. Optionally resolve he / she / they to the last named entity in the same account (rule-based; fastcoref if allowed). | First-person false flags drop sharply; check on challenge "first_person" and RTE (should be neutral on tweets) | 3 h | phrasing can shift scores; measure on RTE too |
| D-7 | Truncation guard + cache key | H7 | In `nli_scoring/scorer.py`, include `max_length` and model id in the `text_hash` key. Count tokens per pair: above `max_length`, split into clauses on `; , and then but` and score the max over sub-pairs, or raise `max_length` to 256. Log the truncation rate. | No stale scores; long-claim recall | 2 h | latency on long inputs |
| D-8 | Typed post-NLI checks (label, don't drop) | H5, Md4, Md8, C3 | New `pipeline/pair_checks.py`, run on pairs at or above the threshold. Each check adds a `type` and an `adjusted` flag; the score itself is untouched. **numeric:** extract numbers and units (pint or a small table), compatible within ±15% after unit conversion → `approx_equal`; incompatible → `numeric_conflict`, which can *boost* below-threshold pairs. **attribution:** regex for "told me / said / according to / reported" → `reported_speech`. **non-observation / hedge:** "didn't see / don't remember / maybe / I think" → `epistemic`. **same-referent:** head-noun / NER subject mismatch ("cyclist" vs "pedestrian") → `different_referent`. **update:** temporal markers ("when I got there" vs "when the ambulance arrived", or timestamps more than Δ apart) → `possible_update`. Reports group conflicts by type and rank `different_referent` / `epistemic` last. | Precision on the witness set and challenge suite; on RTE, check that the number booster gains on gw | 1–2 days | rule brittleness; evaluate each check alone |
| D-9 | Recall path around clustering | H6, Md6 | In `pipeline/run.py`, add `candidates.cross_cluster_knn: 5`: for each claim, also pair it with its top-k cosine neighbours from *other* accounts outside its cluster, and pair unclustered claims with their nearest neighbours. Replace random sampling above `max_pairs_per_cluster` with ordering by cosine (or entity overlap) and taking the top 400. | Co-clustered + candidate coverage of labelled contradictions rises from 55%; pairs-scored cost reported | 4 h | more pairs; cap it |
| D-10 | Recall waterfall + "never compared" | Md2, H6 | `evaluation/experiments.py`: push labelled pairs through the *deployed* `run_pipeline` and count survivors at each stage: co-clustered → cross-account → survives dedup → survives cap → flagged. In `pipeline/findings.py`, split "not contradicted" into "compared, no conflict" and "never compared with any other account". | Shows where deployed recall is lost; honest summary | 3 h | low |
| D-11 | Nested selection + calibration | H2, H4 | Inside each LOEO fold, choose clustering algorithm / `distance_threshold`, direction rule and threshold on the 3 non-test events (pooled validation over both non-test training events for fine-tuning). Fit a temperature on validation logits; report ECE and reliability plots. Extend `fine_grid` below 0.05 or threshold on logit margin. Re-evaluate the demo config, or label it in-sample. | Honest test numbers for the shipped config; stable thresholds | 1 day | small results shift |
| D-12 | Seeds + fine-tune knock-outs | H8, H10 | Seeds {13, 42, 7} × 4 folds; knock-outs: `imbalance: none`, `both_directions: false`, `mnli_replay: 0`, fixed 1 epoch. Add MNLI dev accuracy per checkpoint. | Mean ± sd; which recipe parts matter; forgetting | ~3–6 GPU-h | compute |

## Disagreements between reviewers

- **Number of contradiction groups:** B-1 says 20 (ottawa 5), M-1 says 21 (ottawa 6). I recounted with the project loader: **21** (chebdo 7, gw 1, ottawa 6, ssiege 7). The conclusion is the same either way.
- **Whether the fine-tuning gain is real:**
  - B-4 calls the F1 gain "plausible, not established": 4 of 4 folds are non-negative; sign test p = 0.0625.
  - M-1 says it is "not distinguishable from zero" at story or event level.
  - A-14 adds that it does not fix the witness failure modes and flips some correct verdicts.

  **Resolution:** report it as "improved F1 on 3 of 4 held-out events, CI includes ~0; not tested on witness text". Both quantitative views fit that wording.
- **Clustering's value:** B-12 and M-8 say its precision gain is within noise. The project says it is a precision filter. **Resolution:** keep clustering (architecture constraint, and the cost case is solid) but present it as a cost filter that loses recall, and add D-9 to recover recall.
- **LLM baseline:** B-10 and B-12 list one as evaluation-only and a judgement call. **Resolution:** optional and lowest priority. It must never enter the detection loop (project constraint). It is useful only as a ceiling reference on the D-4 witness set.

## Experiments to run

Ordered by how much each would teach. Rows 1–4 are the minimum before any headline claim.

| # | Variant | Contr. P/R/F1/AP (group-bootstrap CI) | Ent. flag rate | Co-clustered / candidate coverage | Deployed pipeline P/R | Witness-set P/R | Pairs scored | Seeds |
|---|---|---|---|---|---|---|---|---|
| 1 | Pair baselines vs zero-shot val-tuned vs fine-tuned (D-1, D-2) | ✓ | ✓ | — | — | — | 0 | 1 |
| 2 | Story oracle, and fine-tuned on same-cluster pairs / contradiction-vs-entailment AP | ✓ | ✓ | — | — | — | 0 | 1 |
| 3 | Witness test set + challenge suite, zero-shot vs fine-tuned, before D-5 to D-8 (D-4) | — | — | — | — | ✓ | — | 1 |
| 4 | Recall waterfall through `run_pipeline` (D-10) | — | — | ✓ | ✓ | — | ✓ | 1 |
| 5 | + D-5 splitter, + D-6 grounding, + D-7 truncation, each alone then together | ✓ | ✓ | — | ✓ | ✓ | — | 1 |
| 6 | + D-8 checks, each alone (numeric, attribution, epistemic, referent, update) | ✓ | ✓ | — | ✓ | ✓ | — | 1 |
| 7 | + D-9 cross-cluster kNN (k = 0, 3, 5, 10) and prioritised capping | ✓ | — | ✓ | ✓ | ✓ | ✓ | 1 |
| 8 | Nested LOEO selection + temperature scaling; ECE (D-11) | ✓ | ✓ | — | ✓ | — | — | 1 |
| 9 | 3 seeds × folds; fine-tune knock-outs; MNLI accuracy after (D-12) | ✓ | ✓ | — | ✓ | ✓ | — | 3 |
| 10 | Cluster-level: size baseline, rank by `disagreement_score`, `min_contradicting_pairs` {1, 2, 3} (D-3) | — | — | — | — | — | — | 1 |
| 11 | Stronger zero-shot NLI: roberta-large-mnli, deberta-v3-large | ✓ | ✓ | — | ✓ | ✓ | — | 1 |

Stress tests from the adversary to include in the D-4 suite, with expected outcomes:
- first-person different speakers (expect neutral; currently 1.000)
- different referents (neutral; currently 1.000)
- non-observation, "didn't see" (neutral or epistemic; currently 0.999)
- temporal update (`possible_update`; currently 1.000)
- abbreviation splitting (one claim, time kept)
- unit-equivalent numbers (no flag; currently 0.96)
- reported speech (`reported_speech`; currently 1.000)
- clustered-apart conflict (compared via D-9)
- run-on statement over 128 tokens (scored per clause)

## Unverified items

- **H9 magnitude of the size confound.** Per-cluster sizes are not saved. Cheapest check: D-3, then re-run only the `pipeline` stage on cached scores.
- **A-* error rates.** The adversary tested each pattern with 1–5 pairs, so these show failure modes, not rates. Cheapest check: D-4 challenge suite with ≥10 pairs per type.
- **M-3 interpretation** (fine-tuning learned story relatedness). Cheapest check: label ~200 RTE pairs sentence by sentence and score both models.
- **D-6 side effects on tweets.** Grounding may shift RTE scores. Cheapest check: re-run `eval_rte` with the flag on and off.

## Notes on method

The three reviewers ran as isolated parallel subagents, each given only the brief and the repo. The adversary and the methodologist ran real models and cached scores, offline. I did the developer and synthesis passes myself, with direct checks of the group counts (C1) and the splitter (C4). Reviewer analysis scripts (bootstrap, baselines, calibration, leakage, adversarial probes) are copied to `critic/scripts/`.
