# Test cases for the demo app

Paste one block into the app's text box ("Paste accounts"), press **Detect disagreement**, and
compare with the expected result. A blank line separates the two accounts.

The "Today" columns are real results from `python -m evaluation.witness_probes --suite extended`
(zero-shot model at threshold 0.9; fine-tuned model at its saved threshold 0.5), with every
text-layer switch on. The number is the contradiction probability of the pair; "not compared"
means the two sentences were put in different clusters, so the model never saw them together.
FAIL rows are known weaknesses, not typos in the cases.

To re-run all of them in one go:

```powershell
.venv\Scripts\python -m evaluation.witness_probes --suite extended --tag mine
```

## A. Real disagreements: should be flagged

| # | Case | Today (zero-shot) | Today (fine-tuned) |
|---|---|---|---|
| A1 | Left vs right | PASS 0.999 | PASS 1.000 |
| A2 | Jacket colour | PASS 1.000 | PASS 1.000 |
| A3 | Order of events | PASS 1.000 | PASS 1.000 |
| A4 | Rain vs dry road | PASS 1.000 | PASS 1.000 |
| A5 | Two men vs one person | PASS 1.000 | **FAIL 0.116** |
| A6 | Morning vs evening | PASS 0.999 | PASS 1.000 |
| A7 | Hedged ("I think") but still a conflict | PASS 0.997 | PASS 1.000 |
| A8 | Conflict hidden behind a pronoun ("He was carrying...") | **FAIL not compared** | **FAIL not compared** |
| A9 | Same fact, different wording ("stayed" vs "was gone") | PASS 1.000 | PASS 1.000 |

**A1**
```
Witness A: The car turned left onto Mill Road.

Witness B: The car turned right onto Mill Road.
```

**A2**
```
Witness A: The man who ran off wore a red jacket.

Witness B: The man who ran off wore a blue jacket.
```

**A3**
```
Witness A: The car hit the cyclist before the bus arrived.

Witness B: The bus arrived before the car hit the cyclist.
```

**A4**
```
Witness A: It was raining heavily at the time.

Witness B: The road was completely dry at the time.
```

**A5**
```
Witness A: Two men got out of the car.

Witness B: Only one person was in the car.
```

**A6**
```
Witness A: It happened at about 9:15 in the morning.

Witness B: It happened late in the evening.
```

**A7**
```
Witness A: I think the car was red.

Witness B: The car was blue.
```

**A8** (why it fails: "He was carrying a black bag" does not say who "he" is, so it is not
clustered with "The man in the grey hoodie had nothing in his hands")
```
Witness A: A man in a grey hoodie ran towards the station. He was carrying a black bag.

Witness B: The man in the grey hoodie had nothing in his hands.
```

**A9**
```
Witness A: The driver stayed at the scene.

Witness B: He was gone before anyone could get his number plate.
```

## B. Compatible accounts: should NOT be flagged

| # | Case | Today (zero-shot) | Today (fine-tuned) |
|---|---|---|---|
| B1 | Paraphrase | PASS 0.000 | PASS 0.000 |
| B2 | 9:15 vs "just after 9" | PASS 0.001 | PASS 0.003 |
| B3 | About 20 vs around 25 people | PASS 0.824 (close) | **FAIL 0.995** |
| B4 | 30 mph vs 50 km/h (the same speed) | **FAIL 0.987** | **FAIL 0.992** |
| B5 | Had a gun vs "I didn't see a gun" | **FAIL 0.999** | **FAIL 1.000** |
| B6 | "He told me he stopped" vs "he did not stop" | **FAIL 1.000** | **FAIL 1.000** |
| B7 | Alive on arrival vs dead later (time passed) | **FAIL 1.000** | **FAIL 1.000** |
| B8 | Cyclist injured vs pedestrian unhurt (different people) | **FAIL 1.000** | **FAIL 0.999** |
| B9 | "We" in two accounts = two groups | PASS 0.095 | PASS 0.013 |
| B10 | One account just adds detail | PASS 0.000 | PASS 0.000 |

B3 to B8 are what the next planned step (checks after the NLI stage: numbers and units,
"I didn't see", "he told me", time passing, different people) is meant to fix.

**B1**
```
Witness A: The van was white.

Witness B: It was a white van.
```

**B2**
```
Witness A: It happened at about 9:15 a.m.

Witness B: It happened just after 9 a.m.
```

**B3**
```
Witness A: About 20 people were waiting at the stop.

Witness B: Around 25 people were waiting at the stop.
```

**B4**
```
Witness A: The car was doing about 30 mph.

Witness B: The car was doing about 50 km/h.
```

**B5**
```
Witness A: The man had a gun in his right hand.

Witness B: I didn't see a gun.
```

**B6**
```
Witness A: The driver told me he had stopped at the crossing.

Witness B: The driver did not stop at the crossing.
```

**B7**
```
Witness A: The man was alive when I got there.

Witness B: The man was dead when the ambulance arrived.
```

**B8**
```
Witness A: The cyclist was badly injured.

Witness B: The pedestrian was not hurt at all.
```

**B9**
```
Witness A: We were sitting at the window table.

Witness B: We were standing outside the cafe.
```

**B10**
```
Witness A: A car hit the cyclist.

Witness B: A blue hatchback hit the cyclist at the junction.
```

## C. Known gap: one account contradicting itself

The system only compares sentences from **different** accounts, so this is never flagged
(both models: not compared).

```
Witness A: The light was red when the car went through. Later on, the light was definitely green when the car went through.

Witness B: The car went through the junction quickly.
```

## D. Cases for the three text-layer switches

Run each one twice, with the named switch on and then off, and open "Claims as split, and the
text the NLI model sees" to see the difference.

**D1. Speaker grounding** (on: not flagged; off: flagged at 1.00, a false alarm)
```
Witness A: I was inside the shop when it happened.

Witness B: I was outside on the pavement when it happened.
```

**D2. Speaker grounding, the "I heard" rule** (should be flagged either way)
```
Witness A: I heard two shots from the alley.

Witness B: I heard three shots from the alley.
```

**D3. Abbreviation-aware splitter** (on: three whole sentences per account. Off: Witness C's
text is cut into "It was approx.", "10 p.m. when it started.", "The van was parked on St." and
"Patel arrived first." The street name and the "Dr." are lost, and "It was approx." is a
meaningless claim. Both conflicts are still flagged either way.)
```
Witness C: It was approx. 10 p.m. when it started. The van was parked on St. James Street. Dr. Patel arrived first.

Witness D: It was around 6 p.m. when it started. The van was parked on Kings Road. Dr. Patel arrived last.
```

**D4. Clause splitting** (on: "drove off" vs "stayed next to the cyclist" is flagged; off: the
model only sees the identical opening of both accounts and calls them entailment)
```
Witness A: I was walking along the canal path at the time and I heard a loud bang which I first thought was a firework, and I turned around towards the junction near the bakery and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was lying on the ground near the kerb with his bike under him, and a woman in a yellow coat ran over to help him and somebody else was shouting for an ambulance, and the driver of the blue car got out and looked at the bike for a few seconds, and then he got back into the car and drove off towards the bridge without stopping.

Witness B: I was walking along the canal path at the time and I heard a loud bang which I first thought was a firework, and I turned around towards the junction near the bakery and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was lying on the ground near the kerb with his bike under him, and a woman in a yellow coat ran over to help him and somebody else was shouting for an ambulance, and the driver of the blue car got out and looked at the bike for a few seconds, and then he stayed next to the cyclist until the police arrived.
```

**D5. Still failing with grounding on** (two different cars, flagged at 1.00)
```
Witness A: My car was parked outside the bank.

Witness B: My car was in the garage all day.
```
