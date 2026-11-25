# Adversary report: Cross-Account Inconsistency Detector

Scope: red-team inputs against the pipeline as it exists on `feature/nli-finetune` (cleaning -> `split_sentences` -> MiniLM -> agglomerative clustering -> cross-account in-cluster pairs -> `cross-encoder/nli-deberta-v3-base` both directions, `min_contradiction` -> threshold 0.9 -> findings summary).

## How the outputs were produced

The venv and models were already cached (`.venv`, HF cache for `cross-encoder/nli-deberta-v3-base` and `all-MiniLM-L6-v2`, `models/finetuned/nli-deberta-v3-base-pheme-all`). I imported the repo modules and ran them offline (`HF_HUB_OFFLINE=1`) from three scratch scripts outside the repo, with the SQLite and embedding caches turned off. No repo file was modified. Scripts:

- `critic/scripts/probe.py` (49 pairs, zero-shot and fine-tuned; raw output in `probe_out.json` in the same folder)
- `critic/scripts/probe2.py` (control pairs, plus full `run_pipeline` on four account sets using `small_input_overrides`, which is the `--input` path)
- `critic/scripts/probe3.py` (truncation at max_length 128 vs 512)

Notation: `ZS` is the zero-shot model and `FT` is the fine-tuned `pheme-all` checkpoint. `min C` is the contradiction probability after the `min_contradiction` rule. `ab C` / `ba C` are the two directions. A pair is flagged when `min C >= 0.9` in the default (zero-shot) config. The fine-tuned demo uses min rule with threshold 0.5 (see A-4). OBSERVED means I ran it. UNVERIFIED means I predicted it and did not run it. GENERAL means it follows from the code or from known NLI behaviour.

---

[A-1] Different witnesses' "I" is read as one person, so ordinary first-person statements become certain contradictions
Severity: critical
Confidence: high
Basis: OBSERVED
Evidence:
- SPA1 "I was inside the shop when it happened." / "I was outside on the pavement when it happened." gave ZS min C 1.000 and FT 1.000.
- SPA4 "I was standing at the bus stop." / "I was sitting in my car at the lights." gave ZS 1.000 and FT 1.000.
- SPA5 "My car was parked outside the bank." / "My car was in the garage all day." gave ZS 1.000 and FT 1.000.
- SPA2 "I heard two shots." / "I heard three shots." gave ZS 1.000 and FT 1.000. Two witnesses can hear different counts from different places.
- SPA3 "I didn't see anything." / "I saw the car hit the cyclist." gave ZS 1.000 and FT 1.000.
Why it matters: Witness statements are mostly first person. The pairing is cross-account only, so every first-person pair is by construction about two different speakers. The MNLI "same scene, same referent" assumption makes the model read the two "I"s as one person. These pairs get the highest possible confidence, so the threshold cannot remove them, and fine-tuning on tweets does not change them. Any two witnesses who say where they were standing will produce a "conflict" whenever their sentences cluster together. The SPA4 sentences are near-templates of each other, so they will.
Fix: Add a pre-NLI speaker-grounding pass that rewrites first-person pronouns into the account id ("I" -> "Witness A", "my car" -> "Witness A's car") for the text sent to NLI, and keep the raw text for display. Add a post-filter that marks a pair "different speakers' own positions or perceptions" when both claims have a first-person subject and a perception or location predicate. Report those pairs as unique observations, not conflicts. Add SPA1 to SPA5 to a perturbation test suite where the expected label is not-contradiction.

[A-2] "I didn't see X" is scored as a denial of X
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- NEG2 "The man had a gun." / "I didn't see a gun." gave ZS min C 0.999 and FT 1.000.
- SPA3 (above) gave 1.000.
- NEG3 "The driver drove off." / "I didn't see the driver leave." gave ab C 0.998 and ba C 0.013, so min C was 0.013. Only the min rule saved this one, and the rule is not stable (see A-4).
Why it matters: Non-observation is not contradiction, and it is the most common hedge in testimony. The report would quote it as a conflict with high confidence. The project's own reference notes ("'Not mentioned' is not a contradiction") list this exact failure.
Fix: Add a rule-based claim-type tagger before NLI that detects negated perception verbs (see, hear, notice, remember plus a negation) and tags the claim as `non_observation`. Exclude `non_observation` pairs from the contradiction count, or show them in a separate "one witness did not observe what another reports" list. Add perturbation tests "X" vs "I didn't see X" for 10 to 20 predicates.

[A-3] Hedging changes the verdict erratically, and the output carries no hedge signal
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence (second sentence "The light was green." in every case):
- HEDGE0 "The light was red." gave ZS 1.000.
- HEDGE1 "I'm fairly sure the light was red." gave ZS 0.991.
- HEDGE3 "I think the light was red." gave ZS 0.997.
- HEDGE2 "Maybe the light was red." gave ZS 0.004 (ab 0.998, ba 0.004). The flag disappeared.
- NEG4 "I think the light was red, but I'm not completely sure." gave ZS 0.021 (ab 0.999, ba 0.021) but FT 0.918.
- HEDGE4 "I think the light was red." / "I think the light was green." gave ZS 1.000.
Why it matters: A hedged conflict is still a real discrepancy for an investigator, but it should be labelled as hedged. In the current pipeline, the choice of hedge word decides whether the pair is reported at all ("maybe" removes it, "I think" keeps it at full confidence). The zero-shot and fine-tuned models disagree on the long hedge. The disagreement score and the report have no field that says "one side is hedged".
Fix: Add a hedge detector (a cue lexicon: think, maybe, believe, not sure, I guess, possibly, might). Strip the hedge before NLI so the propositional content is scored, then attach `hedged: true` and a down-weight to the pair in `disagreement_stats` and the findings. Add hedge-insertion perturbation tests where the expected behaviour is that the core verdict does not change and the hedge flag is set.

[A-4] Correctness on asymmetric pairs depends on the min rule, and the fine-tuned operating point was chosen for the max rule
Severity: high
Confidence: high
Basis: OBSERVED
Evidence (zero-shot ab C / ba C for pairs that should not be flagged):
- NEG3 "The driver drove off." / "I didn't see the driver leave.": 0.998 / 0.013
- NEG7 "I never said the car was speeding." / "The car was speeding.": 0.100 / 1.000
- TMP2 "At 10:05 the fire was small." / "By 10:30 the whole building was on fire.": 0.005 / 0.997
- TIME1 "The shop closed at 6 pm." / "I left the shop at 7 pm.": 0.005 / 0.985
- COR2 "The second one didn't." / "Both cars stopped at the light.": 0.999 / 0.298. This one is a true contradiction that min removes.
- `models/finetuned/nli-deberta-v3-base-pheme-all/operating_point.json`: `best_f1` is `{"aggregation": "max_contradiction", "threshold": 0.5}` with fold choices `[max 0.95], [max 0.05], [mean 0.07], [mean 0.06]`. `configs/default.yaml` overrides the rule to `min_contradiction` (`aggregation_override`).
Why it matters: One direction of NLI routinely gives 0.98 or more on compatible pairs. The min rule hides this, but at a cost: it kills true contradictions where one direction is uncertain (COR2, HEDGE2, NEG4 zero-shot). Under any `max_contradiction` configuration (the evaluation ablation, or a user who removes the override), the first four pairs become 0.98 to 1.0 false flags. The deployed fine-tuned threshold was selected under a different rule than the one it is used with. The saved 0.5 is also not the median of the four fold thresholds (0.065), so where it came from should be checked.
Fix: Keep both directions, but record the two directional scores in the report and add an "asymmetric" label when |ab - ba| > 0.5. Show those pairs as "one statement may be more specific or time-shifted" instead of silently dropping or flagging them. Re-select the fine-tuned threshold under the rule that is actually deployed, on the validation events.

[A-5] Paraphrase instability in the fine-tuned model
Severity: medium
Confidence: medium
Basis: OBSERVED
Evidence (first sentence vs second sentence, then ZS and FT min C):
- PAR1a "The blue car ran the red light." / "The light was green when the blue car went through.": ZS 0.998, FT 0.995
- PAR1b "The blue car went through on a red signal." / "The blue car crossed while its light was still green.": ZS 0.999, FT 0.993
- PAR1c "The blue car ignored the red light." / "The signal was green for the blue car.": ZS 0.933, FT 0.020 (FT neutral 0.98)
- IRR2 "The driver got out and called an ambulance." / "The driver got out.": ZS entailment 0.998, FT neutral 0.999
- PAR3 voice change: E 0.998 to 1.0 for both models. That is stable.
Why it matters: The same red-vs-green dispute disappears under one paraphrase with the fine-tuned model. IRR2 matters for the findings summary: corroboration needs entailment >= 0.8, so under FT a statement and its shorter restatement no longer count as "accounts agree". Fine-tuning on tweet pairs labelled at story level appears to have moved the entailment and neutral boundary. This is n=1 per pattern, so medium confidence.
Fix: Add a paraphrase-invariance test set (3 to 5 paraphrases per contradiction and per entailment, witness style). Report the label-flip rate for ZS and FT as a gate before shipping a checkpoint. Optionally score a canonicalised paraphrase-normalised form (for example after A-12 decomposition) alongside the raw text and take agreement.

[A-6] Minimal edits that should flip the label do flip, but different surface forms of compatible details also read as contradictions
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- These flip as they should, at about 1.0: MIN1 "Three men got out of the van." / "Two men got out of the van."; MIN3 "turned left onto Main Street" / "turned right"; MIN4 "John was driving the car." / "Mike was driving the car."; MIN5 "ran north" / "ran south".
- MIN8 "The car was going about 30 mph." / "The car was going about 50 km/h." (compatible: 30 mph is about 48 km/h) gave ZS 0.959 and FT 0.975. False flag.
- MIN7 "There were about 20 people at the bus stop." / "There were 25 people at the bus stop." gave ZS 0.034 (ab 0.945) and FT 0.449, against the FT threshold 0.5. Whether it gets flagged depends on the checkpoint.
- MIN2 "It happened at about 9 pm." / "It happened at about 9:15 pm." gave neutral (C 0.001). Correct.
- MIN6 "black jacket" / "dark jacket" gave E 0.989. Correct.
Why it matters: The model flags on token mismatch, not on value mismatch. Witnesses mix units and approximations ("about 20", "a couple of dozen", "50-ish"). Each such pair is reported as a conflict at more than 0.95, with no sign that the values overlap.
Fix: Add a numeric and unit normaliser before NLI (number words to digits, unit conversion to a canonical unit, "about/around/roughly" to a tolerance interval). Add a post-filter: when the only differing span in a flagged pair is a quantity and the intervals overlap, downgrade it to "approximately consistent". Add MIN7 and MIN8 style unit and approximation perturbations to the test suite.

[A-7] Coreference and ellipsis: unresolved pronouns get high-confidence contradictions, elided predicates lose real ones, and the cleaner drops short answers
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- COR1 "He pushed her to the ground." / "The man in the grey coat never touched anyone." gave ZS 1.000 and FT 1.000. "He" may be a different man (the account says "a man in a red hoodie" in an earlier sentence that is not sent to NLI).
- COR3c "The driver did." / "Nobody saw the crash." gave ZS 0.994. The fragment's verb phrase is lost, yet the pair gets near-certainty. COR3 "The driver did." / "Nobody called the police." gave 0.999, which is right only by accident.
- COR2 "The second one didn't." / "Both cars stopped at the light." gave a true contradiction ab 0.999, but min C 0.298. Missed.
- Cleaning: `split_sentences("The first car stopped. The second one didn't. She did.")` keeps "The second one didn't." and drops "She did." (2 tokens is below `min_tokens=3` in `accounts_to_claims`).
Why it matters: Every sentence is scored without its account context. Pronoun sentences inherit the "same referent" prior and get flagged at 1.0. Elliptical sentences, which are common in Q&A-style interview transcripts, are either dropped or scored as fragments.
Fix: Add a coreference pre-pass per account (for example fastcoref or spaCy coreferee, run once per account) that substitutes antecedents before splitting into claims. Add a post-filter that refuses to flag a pair when either claim's subject is an unresolved pronoun or the claim has no main verb (a dependency-parse check). Log those claims as "not comparable" instead of dropping them silently.

[A-8] Updates and time-indexed states are reported as disputes at full confidence
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- TMP1 "The man was still inside the shop." / "The man had already left the shop." gave ZS 1.000 and FT 1.000.
- TMP3 "The victim was alive when I got there." / "The victim was dead when the ambulance arrived." gave 1.000 and 1.000. Both anchors are explicit, and both statements can be true.
- TMP4 "The school is in lockdown." / "The school is no longer in lockdown." gave 1.000 and 1.000.
- TMP2 explicit clock times "At 10:05 ... small" / "By 10:30 ... whole building" gave min 0.005. Saved only by the min rule (ba 0.997).
- `ERROR_ANALYSIS.md`: 6 of 40 audited flags were time-driven.
Why it matters: Witness statements describe sequences ("when I got there", "after the ambulance came"). The scorer never sees timestamps or temporal anchors. The findings caveat says time is not modelled, but each pair is still listed under "Where the accounts conflict".
Fix: Add a temporal-anchor extractor (regex plus dependency patterns for "when/after/before/by/until + clause", clock times, "still/already/no longer/now"). Add a post-filter that relabels a flagged pair as "possible update" when the two claims carry different anchors, or when either contains a state-change marker (still, already, no longer, later, then). For PHEME, also use tweet timestamps. Add TMP1 to TMP4 to the test suite with the expected label "update".

[A-9] Entity confusion: any two subjects in the same frame are treated as the same entity
Severity: critical
Confidence: high
Basis: OBSERVED
Evidence:
- ENT1 "The tall man hit the shopkeeper." / "The short man hit the shopkeeper." gave ZS 1.000 and FT 1.000. Compatible if two men were present.
- ENT4 "The cyclist was badly injured." / "The pedestrian was not hurt at all." gave ZS 1.000 and FT 0.999. These are different people.
- ENT2 "Charb was killed in the attack." / "Stephane Charbonnier survived the attack." gave 1.000. This looks like alias resolution, but the control ENT2c "Charb was killed in the attack." / "Jean Cabut survived the attack." (a different person) also gave 1.000, and ENT2d "Charb was killed..." / "Stephane Charbonnier died..." gave E 0.998. The model flags any killed-vs-survived pair, whoever the subject is. FN6 in `ERROR_ANALYSIS.md` (alias missed) and this result (alias "caught") are the same blindness.
- ENT3 role swap "The officer fired first." / "The suspect fired first." gave 1.000. Correct.
Why it matters: Multi-person incidents (several attackers, victims, bystanders) are exactly where witness accounts are compared. The system cannot tell "a different person" from "a different claim about the same person", and it reports both at 1.0. This is the largest false-alarm group in the audit ("different people, different places"), and it reproduces on witness-style text.
Fix: Add a same-referent gate before counting a contradiction: extract the subject noun phrase heads and named entities of both claims (spaCy). Require a match, a resolvable alias (a small per-case alias table, or string-similarity matching against names), or a role noun shared with the cluster's central claim. When subjects differ, label the pair "about different entities" and do not count it. Add an entity-swap control set (pairs like ENT2c) to the tests, so that "alias handled" cannot pass by accident.

[A-10] Truncation at max_length 128 silently drops the end of long witness sentences
Severity: high
Confidence: high
Basis: OBSERVED
Evidence:
- `split_sentences` only splits on `[.!?]` followed by an uppercase letter or digit, so a run-on spoken account ("... and then he got back into the car and drove off towards the bridge.") stays one claim. In the `long_and_splitting` pipeline run, Witness A's whole 110-word statement became one claim.
- The pair (that statement, "The driver drove off towards the bridge.") is 137 tokens. At 128 the truncated input ends "...and then he got back into[SEP]", so the last clause is gone.
- A_stay = the same statement ending "...stayed with the cyclist until the police arrived." vs B "The driver stayed with the cyclist until the police arrived." gave neutral 0.997 at max_length 128 and entailment 0.996 at 512.
- A_off ("...drove off towards the bridge.") vs "The driver drove off towards the bridge." gave neutral 0.984 at 128 and entailment 0.997 at 512.
- LNG3: two 304-token statements that differ only in the final clause (drove off vs stayed) gave entailment 0.998 at 128 (ZS and FT). The contradiction was fully hidden.
- `text_hash(a, b)` (the cache key) does not include `max_length`, so changing `nli.max_length` on a warm cache reuses scores computed at the old length.
Why it matters: Witnesses tend to state the outcome (what the driver did) at the end of a long narrative sentence. That is exactly the part truncation removes. There is no warning or counter. The result is a confident entailment or neutral, so the claim can even be reported as corroborated or single-source.
Fix: Add a clause-level decomposition step before NLI (split on ", and then", "; ", ", and" plus a verb, "then"), and cap each claim's length. Add a pre-check that counts pairs whose tokenised length exceeds `max_length`, and either re-scores them at 512 or splits them, logging the count in the report. Put `max_length` (and the model revision) into the cache key.

[A-11] The shipped example produces a false conflict, and multi-claim sentences are scored and reported as one unit
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence (`examples/witness_accounts.txt` through `run_pipeline` with `small_input_overrides`):
- "The driver stayed at the scene." (A) / "The driver got out and called an ambulance." (C) gave C 0.996, flagged. These are compatible: getting out and calling an ambulance is staying.
- "The driver did not stay at the scene, he drove off." (B) / "The driver got out and called an ambulance." (C) gave 1.000. This is defensible.
- LNG1 "The driver got out, checked on the cyclist and then drove off." / "The driver stayed at the scene." gave 0.999. The flag is correct, but the agreeing clauses ("got out", "checked on the cyclist") are now part of a "contested" claim, so `build_findings` cannot list them as corroborated.
- LNG2 ordering "First he shouted at her, then he hit her." / "He hit her first, and only then started shouting." gave 0.999. Correct.
Why it matters: The demo input that users and markers see first already contains a false alarm, reported at 0.996. Because claims are sentences, one disputed clause marks the whole sentence as contested and removes its agreeing parts from "What the accounts agree on".
Fix: Use the same clause decomposition as A-10, so contested status attaches to the clause. Add the example file's expected pair labels as a regression test (driver stayed vs called ambulance: not-contradiction).

[A-12] Sentence splitting and the min-token filter destroy times and places, and the leftover fragments get "corroborated"
Severity: critical
Confidence: high
Basis: OBSERVED
Evidence:
- `split_sentences("It was approx. 10 p.m. The man ran.")` returns `['It was approx.', '10 p.m.', 'The man ran.']`. "10 p.m." has 2 tokens and is dropped.
- `split_sentences("We met at St. James Park. ...")` returns `['We met at St.', 'James Park.', ...]`. `"Dr. Smith arrived at 9."` returns `['Dr.', 'Smith arrived at 9.']`.
- Pipeline run `long_and_splitting`: Witness C "It was approx. 10 p.m. The van was parked on St. James Street. Dr. Patel arrived first." vs Witness D "It was around 6 p.m. The van was parked on Kings Road. Dr. Patel arrived last." The claims became "It was approx.", "The van was parked on St.", "Patel arrived first.". The pair 10 p.m. vs 6 p.m. was never scored. "It was approx." vs "It was around 6 p.m." gave C 0.000, and the findings summary lists "It was approx." under CORROBORATED, restated by Witness D.
- "The van was parked on St." vs "The van was parked on Kings Road." gave 0.995. Flagged, but the quoted evidence has lost the street name.
Why it matters: Times, street names and titles are the details witness comparison is for. The splitter, written for tweets, removes them without warning, and the findings then assert agreement on a meaningless fragment. This is a silent high-confidence error in the user-facing summary.
Fix: Replace the regex boundary rule with an abbreviation-aware splitter (spaCy sentencizer or pysbd), or add an abbreviation list (approx., St., Dr., Mr., Mrs., a.m., p.m., No., e.g.) to `_SENT_SPLIT`. After splitting, merge any fragment below `min_tokens` into the previous sentence instead of dropping it. Add a post-filter on the findings that never reports a claim without a main verb as corroborated. Add these strings as unit tests in `tests/`.

[A-13] Account parsing: one witness's multi-paragraph statement becomes several "accounts", and within-account inconsistency is never checked
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- `parse_accounts_text("Witness A: The light was red when the car went through.\n\nOn reflection, the light was green when the car went through.\n\nWitness B: ...")` gives accounts `Witness A`, `account_2`, `Witness B`. Witness A's self-correction becomes a separate account and is paired cross-account. In this run it scored C 0.035, so the true conflict was missed as well.
- `"Witness A (aged 34): The driver stayed."` is not recognised as a name (the regex rejects parentheses). It becomes `account_1`, and the header text is kept inside the claim.
- `"Note: the witness was drunk."` becomes an account called `Note`.
- `cross_account_only: true` means a single witness who says "the light was red" and later "the light was green" is never compared with themselves.
Why it matters: Real statements are multi-paragraph, carry metadata in their headers, and change between interviews. Changes within one statement are one of the most useful signals for investigators, and the pipeline either misses them or attributes them to a phantom second witness.
Fix: Read structured JSON input as the primary form for witness data (already supported), and make the text parser start a new account only at a recognised "Name:" header, appending other paragraphs to the current account. Add an optional within-account pass (same NLI, same clusters, pairs with the same `account_id`) reported in its own section, "statement changes within one account".

[A-14] Distribution shift: fine-tuning on tweets does not fix the witness-specific failure classes
Severity: high
Confidence: medium
Basis: OBSERVED
Evidence:
- FT min C was still 1.000 (or 0.999) on SPA1 to SPA5, TMP1, TMP3, TMP4, ENT1, ENT4, COR1, ATT1 and NEG2.
- FT changed the verdict only on: ATT2 "Police said the suspect was arrested." / "The shopkeeper told me the suspect was arrested." (ZS 0.987 false flag -> FT 0.059), NEG4 (ZS 0.021 -> FT 0.918), PAR1c (0.933 -> 0.020), OUT3b and OUT4 (true knowledge-based contradictions, now neutral), COR3c (0.994 -> 0.001), and IRR2 (entailment -> neutral).
- DS1 "I'm not 100% sure but I believe the van was white, or maybe light grey." / "The van was dark blue." gave ZS 0.990 and FT 0.999. This is a real hedged conflict, flagged with no hedge flag (A-3).
Why it matters: The reported FT gain (recall 0.31 -> 0.64) is measured on PHEME tweets. On the target domain, the dominant false-alarm classes (first person, updates, different entities) do not move, and some correct behaviours flip. The README's witness-statement framing is not supported by any evaluation on first-person text.
Fix: Build a witness-style challenge set from this report's pairs plus 100 to 200 more (expected label per pair, typed by phenomenon). Report per-type accuracy for ZS and FT as a separate table. Keep it as a CheckList-style perturbation suite in `tests/` (it can run with the real model behind a marker), not as training data.

[A-15] Contradictions that need outside knowledge are missed, or caught for surface reasons
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- OUT1b "The attack happened on a Sunday." / "The attack happened on 14 March 2023." (a Tuesday) gave ZS ab 0.999, ba 0.007, min 0.007, and FT 0.016. Missed.
- OUT1 with the weekday written out ("... on Tuesday 14 March 2023.") gave 1.000. So the detection is lexical (Sunday vs Tuesday), not calendar reasoning.
- OUT2 "It was pitch dark when the car crashed." / "The crash happened at 1 pm in July in London." gave neutral 0.996 (ZS) and 0.999 (FT). Missed.
- OUT3b "The van crossed Tower Bridge into Southwark." / "The van never left the north side of the Thames." gave ZS 0.859 (below 0.9) and FT 0.032. Missed.
- OUT4 "He got off the train at Waterloo." / "He was never south of the river that day." gave ZS 0.968 (flagged) and FT 0.010.
Why it matters: Calendar, daylight and geography conflicts are common in testimony and cannot come from sentence-pair NLI. The system gives no sign that such claims were out of scope, and the fine-tuned model drops the few it caught.
Fix: Inside the constraint, add normalisers before NLI that make the implicit explicit: a date normaliser that appends the weekday to absolute dates ("14 March 2023 (Tuesday)"), and optionally sunrise/sunset for a given date and place. Otherwise, document the class as out of scope in the findings caveats and in the challenge set.

[A-16] Reported speech and different sources are read as conflicting claims
Severity: medium
Confidence: high
Basis: OBSERVED
Evidence:
- ATT1 "The driver told me he had stopped at the light." / "The driver did not stop at the light." gave ZS 1.000 and FT 1.000. Account A reports what the driver said; it does not assert that he stopped.
- ATT2 "Police said the suspect was arrested." / "The shopkeeper told me the suspect was arrested." gave ZS 0.987. A false flag (same fact, different sources) at the default threshold. FT 0.059.
- The same pattern appears in `ERROR_ANALYSIS.md` FP6 and FP7 (AFP vs CTV attributions) on tweets.
Why it matters: Witness statements are full of reported speech ("he told me", "the officer said"). The default zero-shot config reproduces the attribution false alarm already documented on tweets.
Fix: Add an attribution stripper and tagger before NLI ("X said/told me/according to X" -> keep the embedded proposition and attach `source=X`). Score the embedded propositions. In the post-filter, label pairs whose only difference is the source as agreement, and pairs where one side is reported speech as "reported claim vs witness's own claim".

[A-17] Pipeline-level miss: true contradictions phrased differently are clustered apart, never reach NLI, and are reported as "not contradicted"
Severity: critical
Confidence: high
Basis: OBSERVED
Evidence (pipeline run `clustered_apart`, the `--input` path with `distance_threshold 0.6` and `noise_assign_min_sim 0.4`):
- Witness A: "The driver stayed at the scene. It was raining heavily. The cyclist was wearing a helmet. I was waiting at the bus stop."
- Witness B: "He was gone before anyone could get his number plate. The tarmac was bone dry. The lad on the bike had nothing on his head. I was waiting at the bus stop too."
- Result: "The driver stayed at the scene.", "It was raining heavily.", "He was gone before anyone could get his number plate." and "The tarmac was bone dry." were all labelled -1 (unclustered) and never paired.
- Scored directly, the driver pair is a certain contradiction (PAR2b, which uses "The driver remained where the crash happened." / "He was gone...": ZS 1.000).
- The findings summary lists all four under "Said by one account only (not confirmed, not contradicted)". The helmet pair (sim 0.45) did cluster and was flagged at 0.999.
- Even in the shipped example, "It was raining heavily." / "The road was dry." has cosine 0.477, just above the 0.4 merge cut.
- At corpus level, `REPORT`/`BRIEF` give only 55.1% of labelled contradictions landing in one cluster (HDBSCAN 9.9%).
Why it matters: Contradictions are often paraphrased denials with little word overlap: "stayed" vs "was gone", "raining" vs "dry tarmac". A MiniLM topic embedding separates them. The output then makes an affirmatively wrong statement ("not contradicted") instead of "not compared". The caveat in `findings.py` is generic and does not say which claims were never compared.
Fix: Keep clustering, but add a cluster-independent candidate retrieval pass: for each claim, take the top-k (for example 5) nearest cross-account claims by cosine regardless of cluster, plus pairs that share a subject entity (A-9) or a slot type (time, place, vehicle). Score those with the same NLI. Measure this filter's recall on the PHEME RTE contradictions (the cluster-only filter is 55%). In findings, split "single_source" into "compared, no conflict" and "never compared", and list unclustered claims explicitly.

[A-18] The 400-pair cap leaves large clusters almost unsampled, and the flag depends on the seed
Severity: medium
Confidence: high
Basis: OBSERVED (cluster sizes) / GENERAL (coverage arithmetic)
Evidence:
- `outputs/reports/*_report.json`: 30 clusters across the 4 events hit the 400 cap. The largest are charliehebdo 405 claims / 150 accounts, sydneysiege 219 / 92, ottawashooting 125 / 65, and germanwings 131 / 89.
- A 405-claim cluster has up to C(405,2) = 81,810 pairs, so 400 sampled is under 0.5% coverage. For 219 claims (23,871 pairs) it is about 1.7%.
- `cluster_pairs` samples uniformly with `seed + cid`, so a different seed or a re-ordered cluster id gives a different contradiction set.
Why it matters: The biggest clusters are the main sub-events, where disputes concentrate. A single contradicting pair can be dropped by random sampling. Cluster recall in the 6.x results partly reflects the sampling seed.
Fix: Replace uniform sampling with prioritised sampling inside the cap: cross-account pairs ranked by similarity plus entity or slot overlap first, and random for the rest. Alternatively, sub-cluster large clusters before pairing. Report the coverage fraction per cluster in the report. Run the evaluation with 3 seeds to show the variance.

[A-19] Silent high-confidence failure classes (summary)
Severity: high
Confidence: high
Basis: OBSERVED
Evidence: These classes produced min C >= 0.98 on compatible pairs, with nothing in the output to mark them:
- Different speakers' first-person positions and perceptions (A-1)
- "I didn't see X" vs "X" (A-2)
- Pronoun-subject sentences (COR1) and verb-less fragments (COR3c) (A-7)
- State updates with explicit time anchors (TMP3) (A-8)
- Different entities in the same frame (ENT1, ENT4, ENT2c) (A-9)
- Reported speech vs assertion (ATT1), and different sources in zero-shot (ATT2) (A-16)
- Unit and approximation differences (MIN8) (A-6)

The following produced confident non-flags that hide real conflicts or assert false agreement:
- Truncated long claims, entailment 0.998 (LNG3) (A-10)
- Splitter fragments reported as corroborated ("It was approx.") (A-12)
- Claims clustered apart, reported "not contradicted" (A-17)

Also, the fine-tuned demo threshold 0.5 is used with a direction rule it was not selected for (A-4).

Why it matters: The contradiction probability that users see is around 1.0 for real and spurious conflicts alike. The threshold sweep result (0.5 -> 0.99 changes cluster flags only from 54 to 45) is consistent with this: the score is saturated and carries almost no ranking information within flagged pairs.
Fix: Do not try to fix this with a threshold. Add the typed post-filter layer from A-1, A-2, A-7, A-8, A-9 and A-16 (speaker, non-observation, unresolved-referent, update, different-entity and attribution labels). Report each flagged pair with its type label, so a reader sees "possible update" or "different person" instead of a bare 1.000. Calibrate on the witness challenge set (A-14) with temperature scaling of the existing cross-encoder logits, so that the probability shown is meaningful.

[A-20] Proposed perturbation suite (additions only, NLI unchanged)
Severity: low
Confidence: high
Basis: GENERAL
Evidence: Every pair above is a ready fixture, with its expected label, and they already run against the real scorer in the scratch scripts listed at the top. Suggested groups, each with 10 to 20 items:
- negation (expected C)
- non-observation (expected not-C)
- hedge insertion (core label unchanged, hedge flag set)
- paraphrase (label unchanged)
- number, direction or name edit (expected C)
- unit or approximation (expected not-C)
- first-person different speakers (expected not-C)
- update with anchors (expected "update")
- entity swap control (expected not-C)
- reported speech (expected not-C)
- long-claim truncation (expected label preserved at 128)
- splitter abbreviations (expected claim text preserved)
- clustered-apart paraphrased denial (expected pair scored)
Why it matters: None of these behaviours is covered by `tests/`, which use `MockNLIScorer`, a lexical heuristic that would pass several of them for the wrong reasons. Without a suite, a checkpoint or threshold change can regress any class silently, as A-5 shows for FT.
Fix: Add `tests/challenge/witness_pairs.jsonl` and a pytest marker (`-m realmodel`) that runs the real cross-encoder. Fail if any class's accuracy drops below its recorded baseline, and print a per-class table into `reports/`.
