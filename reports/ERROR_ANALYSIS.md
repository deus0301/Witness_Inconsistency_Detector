# Error analysis

**Configuration analysed:** `all-MiniLM-L6-v2` embeddings, **HDBSCAN** clustering,
`cross-encoder/nli-deberta-v3-base`, both-directions (`min_contradiction`) rule, threshold 0.9,
rumour source tweets of the four main events. This was the default when the analysis was done.
The default clustering has since changed to agglomerative (see section 6.4 of the report), and the
analysis was not repeated. The examples come from `reports/results/error_candidates_hdbscan.json`;
the texts are the cleaned tweets. Most findings concern the NLI model, which did not change.

"False positive" and "false negative" below are relative to the **veracity proxy**, not to the
PHEME RTE labelled pairs, which arrived later and are the better reference.

Two things are examined:

1. **Errors against the proxy labels** (18 examples below). "False positive" and "false negative"
   are relative to the PHEME proxy, and a large share of them turn out to be errors of the proxy,
   not of the detector.
2. **A manual audit of 40 randomly sampled flagged pairs**, which is the only direct estimate of
   precision in this project.

The commentary and the audit judgements were made by one reader (the AI assistant that built the
pipeline), not by independent annotators. Treat them as an informed first pass.

## 1. False positives against the proxy

Proxy rule: two non-turnaround tweets of the same **true** rumour story should agree. 124 flagged
pairs break that rule.

| # | Event | Claim A | Claim B | P(contr.) | Commentary |
|---|---|---|---|---|---|
| FP1 | Ottawa | Seeing reports that soldier shot at War Memorial is still alive. Hang in there, brother. | BREAKING: confirms the soldier shot this morning has died, a Parliamentary Guard has also been wounded. | 1.000 | **Real disagreement, proxy is wrong.** Both tweets are filed under the true story "The soldier shot at War Memorial has died", but one reports him alive. The detector is right to surface it. |
| FP2 | Ottawa | CXN: 1 gunman shot dead in Parliament Hill attack, soldier shot in Ottawa. No report soldier dead. | BREAKING UPDATE: Canadian soldier injured at Parliament Hill shooting dies | 0.997 | **Real, time-driven.** An early "no report of death" against a later confirmation. The accounts do conflict; the cause is that the situation changed. |
| FP3 | Ottawa | University of Ottawa is in lockdown, classes are cancelled | The University of Ottawa is no longer in lockdown. | 1.000 | **Real, time-driven.** A state change. Without timestamps in the scoring step the system cannot tell an update from a dispute. |
| FP4 | Sydney | Uber reaches 4.0x surge pricing as Sydney faces hostage lockdown | Uber backtracks, offers free rides after prices were gouged for people fleeing | 1.000 | **Time-driven.** Policy reversal reported by different accounts. Four of the 14 top false positives are this one story. |
| FP5 | Sydney | Uber Is Allegedly Charging Passengers $100 Minimum During Sydney Siege | Uber Offers Free Rides During Sydney Hostage Crisis After Surge Pricing | 1.000 | Same pattern as FP4. The second tweet even contains the first claim ("after surge pricing"); the model reads "charging" against "free" as contradiction. |
| FP6 | Ottawa | Canada's Minister of Employment Jason Kenney says the soldier shot in Ottawa has died | Statement from Hamilton Mayor Bob Bratina on death of soldier in Ottawa Shooting | 1.000 | **Model error.** The two tweets agree on the fact and differ only in who is speaking. The model treats different named sources as incompatible. |
| FP7 | Ottawa | AFP: an official says a Canadian soldier has died after being shot | BREAKING: CTV News reports that a solider wounded in a shooting at the National War Memorial has died. | 0.999 | **Model error**, same cause as FP6: different attributions (AFP, CTV) of the same fact. Attribution phrases are the most common source of spurious contradictions. |
| FP8 | Sydney | Prime Minister Tony Abbott's statement on the Sydney Siege. | Grand Mufti of Australia statement condemning Sydney Siege | 1.000 | **Model error.** Two headlines about different people's statements. Nothing is asserted that could conflict. Fragments without a verb phrase are scored unreliably. |
| FP9 | Ottawa | Proudly at his post Nathan Cirillo. This nation thanks you and your family | Heartbreaking Photo Shows Nathan Cirillo Moments Before Shooting | 1.000 | **Model error.** A tribute and a photo caption. Both are about the same person; neither makes a checkable claim about the other. |

Pattern: of these nine, five (FP1 to FP5) are genuine conflicts between accounts that the proxy
cannot see, because veracity is recorded per story and not per tweet. Four are model errors caused
by attribution differences and by headline fragments.

## 2. False negatives against the proxy

Proxy rule: a turnaround (debunking) tweet of a **false** story contradicts every non-turnaround
tweet of the same story. 153 such pairs scored below the threshold.

| # | Event | Debunking tweet | Other tweet of the same false story | P(contr.) | Same cluster | Commentary |
|---|---|---|---|---|---|---|
| FN1 | Ottawa | Police now say there were two shooting incidents in Ottawa: one at the war memorial, the other on Parliament Hill. | Police have clarified that there were two shootings in Ottawa today, not three | 0.000 | yes | **Proxy is wrong.** Both tweets debunk the "three shootings" story; only one carries the turnaround flag. The model correctly scores entailment (1.00). |
| FN2 | Germanwings | reporter James André at French Civil Aviation Authority HQ: "Germanwings plane did NOT report distress signal", no mayday. | BREAKING Germanwings crew did not send distress signal: aviation official | 0.000 | yes | **Proxy is wrong**, same reason. Entailment 0.99. |
| FN3 | Germanwings | Germanwings CEO: Plane victims included 72 German citizens, 35 Spaniards, 2 Americans. | Germanwings crash victims include 72 German citizens, 35 Spaniards, 2 Americans, per CEO | 0.000 | yes | **Proxy is wrong.** Near-duplicates filed under the false story "67 German nationals were onboard". |
| FN4 | Charlie Hebdo | Charlie Hebdo famous cartoonists Charb and Cabu shot dead according to reports | Stéphane Charbonnier and Jean Cabut died for freedom of press. | 0.000 | yes | **Proxy is wrong.** Both say Charb died; the false story is "Charbonnier was critically injured". |
| FN5 | Sydney | No confirmation airspace has been shut down yet in Sydney but planes are avoiding the area. | CORRECTION: We reported earlier Sydney air space was shut down. That is not correct. | 0.000 | yes | **Proxy is wrong.** Two corrections of the same rumour. |
| FN6 | Charlie Hebdo | "The cartoonists Charb & cabu are dead." | Cartoonist Stephane Charbonnier was critically injured in the attack (LeFigaro) | 0.000 | yes | **Real miss.** "Dead" against "critically injured" is a true conflict. The model does not link "Charb" to "Stephane Charbonnier", so it sees two different people. Entity aliases are a blind spot. |
| FN7 | Germanwings | 150 people were on board plane that crashed in France today, Germanwings CEO says. | 142 passengers on board GermanWings plane which has crashed in French Alps, according to AFP | 0.001 | yes | **Arguably a real miss, arguably correct.** 142 passengers plus 6 crew is 150 people, so the statements are compatible if read carefully. PHEME labels the 142 story false. Numeric claims need a notion of what is being counted. |
| FN8 | Ottawa | Rideau Centre general manager tell there was no shooting inside the mall. But people can't leave or enter. | BREAKING: CTV Ottawa confirms there were 3 separate shootings. One on Parliament Hill, one at National War Memorial, one near Rideau Centre | 0.008 | no | **Real miss, two causes.** Clustering separated the pair, so it was never compared in the pipeline; and "inside the mall" against "near Rideau Centre" is not a strict contradiction for the model. |
| FN9 | Ottawa | Police now say there were two shooting incidents in Ottawa | These are the areas where the Ottawa Shootings have been reported. | 0.000 | no | **Proxy is wrong.** The second tweet is a map caption with no count in it; nothing to contradict. |

Pattern: six of the nine "misses" are pairs of tweets that agree. Many tweets inside a false story
are themselves denials, but only the first one has `is_turnaround = 1`, so the proxy pairs two
debunks and calls it a contradiction. The real misses come from entity aliases (FN6), from numbers
that count different things (FN7) and from clustering splitting a rumour from its denial (FN8). In
the default configuration only 50% of debunk pairs end up in the same cluster.

## 3. Manual audit of 40 random flagged pairs

Sample: 40 pairs drawn at random (seed 42) from the 882 pairs flagged at P(contradiction) >= 0.9
in the default configuration. Each pair was read and put in one of three groups.

| Judgement | Pairs | Share | Typical example |
|---|---|---|---|
| Genuine disagreement about the same fact | 11 | 27.5% | "An ISIS flag is being displayed in the window" vs "The flag in the window is not an Islamic State flag"; "three shooting incidents" vs "two shootings, not three"; "Tribute on Banksy's instagram account" vs "This cartoon is not by Banksy"; "10 people confirmed dead" vs "11 dead" |
| Time-driven conflict (the situation changed between the two tweets) | 6 | 15.0% | "Shooter still on loose" vs "Attacker shot dead in Parliament"; "Hostages remain inside" vs "the gunman has released three hostages"; Uber surge pricing vs free rides |
| Not a contradiction | 23 | 57.5% | "soldier ... has died" vs "both gunman and soldier at war memorial are dead"; "University of Ottawa is on lockdown" vs "lockdown at the Rideau Centre has been lifted"; two statements by different officials |

So roughly 4 in 10 flagged pairs (17 of 40) point at a place where accounts really differ, and
about 6 in 10 are false alarms. With 40 pairs the uncertainty is wide: a 95% interval for the
"genuine or time-driven" share is roughly 27% to 59%.

Where the 23 false alarms come from:

- **Compatible statements about related but different things** (about half): different places
  ("University of Ottawa" vs "Rideau Centre"), different people, a count of hostages held vs a count
  of hostages escaped.
- **Same fact with extra detail or a different source**: "soldier has died" vs "gunman and soldier
  are dead".
- **Headlines and captions without a real proposition**: "Prime Minister's statement on the siege".

## 4. What this suggests

1. The NLI model, trained on MultiNLI sentence pairs, treats "about a different thing" as
   "contradicts" far too often on tweets. A relevance check before scoring (do both claims mention
   the same entity and predicate?) would remove the largest group of false alarms.
2. Time matters. Six of 40 flagged pairs are updates. Showing timestamps next to each pair, or
   flagging "later tweet supersedes earlier one", would let a reader separate disputes from updates.
3. Stripping attribution phrases ("AFP:", "CTV News reports that", "according to") before scoring
   should remove a recurring class of model errors.
4. The veracity proxy underestimates the detector in both directions: it calls real conflicts
   false positives and calls agreeing debunks false negatives. Any future evaluation needs
   pair-level human labels; a few hundred would be enough to replace the proxy.
