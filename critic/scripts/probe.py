import os, sys, json
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
REPO=r"D:\Projects\Witness_Inconsistency_Detector"
sys.path.insert(0, REPO)
import numpy as np
from nli_scoring.scorer import NLIScorer, aggregate_directions
from pipeline.config import load_config, resolve_model

long_a = ("I was walking my dog along the canal path at the time and I remember it was quite busy because "
 "the market had just closed and there were lots of people carrying bags and talking, and I heard a loud "
 "bang which I first thought was a firework or a car backfiring, and I turned around towards the junction "
 "near the bakery and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was "
 "lying on the ground near the kerb with his bike under him, and a woman in a yellow coat ran over to help "
 "him and somebody else was shouting for an ambulance, and the driver of the blue car got out and looked "
 "at the bike for a few seconds, and then he got back into the car and drove off towards the bridge.")
long_b = ("I was walking my dog along the canal path at the time and I remember it was quite busy because "
 "the market had just closed and there were lots of people carrying bags and talking, and I heard a loud "
 "bang which I first thought was a firework or a car backfiring, and I turned around towards the junction "
 "near the bakery and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was "
 "lying on the ground near the kerb with his bike under him, and a woman in a yellow coat ran over to help "
 "him and somebody else was shouting for an ambulance, and the driver of the blue car got out and looked "
 "at the bike for a few seconds, and then he stayed with the cyclist until the police arrived.")

PAIRS = {
 # negation / hedging
 "NEG1 plain negation": ("The driver stayed at the scene.", "The driver did not stay at the scene."),
 "NEG2 non-observation vs assertion": ("The man had a gun.", "I didn't see a gun."),
 "NEG3 non-observation of leaving": ("The driver drove off.", "I didn't see the driver leave."),
 "NEG4 hedged contradiction": ("I think the light was red, but I'm not completely sure.", "The light was green."),
 "NEG5 double negation (agree)": ("It is not true that the man was unarmed.", "The man was armed."),
 "NEG6 double negation (contra)": ("It is not true that the man was unarmed.", "The man was unarmed."),
 "NEG7 denial of saying": ("I never said the car was speeding.", "The car was speeding."),
 # paraphrase invariance
 "PAR1a": ("The blue car ran the red light.", "The light was green when the blue car went through."),
 "PAR1b": ("The blue car went through on a red signal.", "The blue car crossed while its light was still green."),
 "PAR1c": ("The blue car ignored the red light.", "The signal was green for the blue car."),
 "PAR2a": ("The driver stayed at the scene.", "The driver drove off."),
 "PAR2b": ("The driver remained where the crash happened.", "He was gone before anyone could get his number plate."),
 "PAR3 voice (should be E)": ("The cyclist was hit by the blue car.", "The blue car hit the cyclist."),
 # minimal edits
 "MIN1 count": ("Three men got out of the van.", "Two men got out of the van."),
 "MIN2 approx time": ("It happened at about 9 pm.", "It happened at about 9:15 pm."),
 "MIN3 direction": ("The car turned left onto Main Street.", "The car turned right onto Main Street."),
 "MIN4 name": ("John was driving the car.", "Mike was driving the car."),
 "MIN5 compass": ("The man ran north towards the station.", "The man ran south towards the station."),
 "MIN6 near-synonym colour (compatible)": ("He was wearing a black jacket.", "He was wearing a dark jacket."),
 "MIN7 approx count": ("There were about 20 people at the bus stop.", "There were 25 people at the bus stop."),
 "MIN8 unit conversion (compatible)": ("The car was going about 30 mph.", "The car was going about 50 km/h."),
 # coreference / ellipsis
 "COR1 pronoun, different referent": ("He pushed her to the ground.", "The man in the grey coat never touched anyone."),
 "COR2 ellipsis": ("The second one didn't.", "Both cars stopped at the light."),
 "COR3 elided subject": ("The driver did.", "Nobody called the police."),
 # temporal / update / spatial
 "TMP1 update": ("The man was still inside the shop.", "The man had already left the shop."),
 "TMP2 time-indexed compatible": ("At 10:05 the fire was small.", "By 10:30 the whole building was on fire."),
 "TMP3 alive then dead": ("The victim was alive when I got there.", "The victim was dead when the ambulance arrived."),
 "TMP4 lockdown lifted": ("The school is in lockdown.", "The school is no longer in lockdown."),
 "SPA1 first person different speakers": ("I was inside the shop when it happened.", "I was outside on the pavement when it happened."),
 "SPA2 first person perception": ("I heard two shots.", "I heard three shots."),
 "SPA3 saw nothing vs saw": ("I didn't see anything.", "I saw the car hit the cyclist."),
 # entity confusion
 "ENT1 two men": ("The tall man hit the shopkeeper.", "The short man hit the shopkeeper."),
 "ENT2 alias": ("Charb was killed in the attack.", "Stephane Charbonnier survived the attack."),
 "ENT3 role swap": ("The officer fired first.", "The suspect fired first."),
 "ENT4 different victims": ("The cyclist was badly injured.", "The pedestrian was not hurt at all."),
 # multi-claim / ordering / truncation
 "LNG1 mixed clause": ("The driver got out, checked on the cyclist and then drove off.", "The driver stayed at the scene."),
 "LNG2 ordering": ("First he shouted at her, then he hit her.", "He hit her first, and only then started shouting."),
 "LNG3 long, contradiction at end (truncation)": (long_a, long_b),
 "LNG4 long vs short": (long_a, "The driver stayed with the cyclist until the police arrived."),
 # distribution shift
 "DS1 hedged first person": ("I'm not 100% sure but I believe the van was white, or maybe light grey.", "The van was dark blue."),
 "DS2 tweet-style vs statement": ("BREAKING: driver flees scene after hitting cyclist", "I watched the driver wait beside the cyclist until the ambulance came."),
 # outside context
 "OUT1 calendar": ("The attack happened on a Sunday.", "The attack happened on Tuesday 14 March 2023."),
 "OUT2 daylight": ("It was pitch dark when the car crashed.", "The crash happened at 1 pm in July in London."),
 "OUT3 geography": ("The van drove south over Tower Bridge into Southwark.", "The van was heading north into the City the whole time."),
 # attribution / reported speech
 "ATT1 reported speech": ("The driver told me he had stopped at the light.", "The driver did not stop at the light."),
 "ATT2 different sources same fact": ("Police said the suspect was arrested.", "The shopkeeper told me the suspect was arrested."),
 # irrelevant pair (different things)
 "IRR1 different topic, same frame": ("It was raining heavily.", "The road was dry."),
 "IRR2 omission": ("The driver got out and called an ambulance.", "The driver got out."),
}

def run(model_dir, label):
    s = NLIScorer(model_dir, batch_size=16, max_length=128, cache_path=None, aggregation="min_contradiction", verbose=True)
    keys = list(PAIRS)
    pairs = [PAIRS[k] for k in keys]
    both = s.score_directional(pairs + [(b, a) for a, b in pairs])
    n = len(pairs)
    ab, ba = both[:n], both[n:]
    mn = aggregate_directions(ab, ba, "min_contradiction")
    out = {}
    for i, k in enumerate(keys):
        out[k] = {"ab": [round(float(x),3) for x in ab[i]], "ba": [round(float(x),3) for x in ba[i]],
                  "min": [round(float(x),3) for x in mn[i]]}
    # token lengths
    tok = s.tokenizer
    out["_LNG3_len"] = len(tok(long_a, long_b)["input_ids"])
    out["_LNG4_len"] = len(tok(long_a, "The driver stayed with the cyclist until the police arrived.")["input_ids"])
    return out

cfg = load_config()
res = {}
res["zeroshot"] = run(resolve_model("cross-encoder/nli-deberta-v3-base", cfg), "zs")
res["finetuned_all"] = run(REPO + r"\models\finetuned\nli-deberta-v3-base-pheme-all", "ft")
json.dump(res, open(sys.argv[1], "w"), indent=1)
for k in PAIRS:
    z, f = res["zeroshot"][k], res["finetuned_all"][k]
    print(f"{k:45s} ZS min E/N/C={z['min']}  (ab C={z['ab'][2]}, ba C={z['ba'][2]}) | FT min={f['min']}")
print(res["zeroshot"]["_LNG3_len"], res["zeroshot"]["_LNG4_len"])
