import os, sys, json, copy
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
REPO=r"D:\Projects\Witness_Inconsistency_Detector"
sys.path.insert(0, REPO)
import numpy as np
from nli_scoring.scorer import NLIScorer, aggregate_directions
from clustering.embed import Embedder
from pipeline.config import load_config, resolve_model
from pipeline.run import accounts_to_claims, parse_accounts_text, small_input_overrides, run_pipeline, read_accounts_file

PAIRS = {
 "ENT2c alias control (different person)": ("Charb was killed in the attack.", "Jean Cabut survived the attack."),
 "ENT2d alias, agreeing": ("Charb was killed in the attack.", "Stephane Charbonnier died in the attack."),
 "COR3c elided subject control": ("The driver did.", "Nobody saw the crash."),
 "OUT1b pure calendar": ("The attack happened on a Sunday.", "The attack happened on 14 March 2023."),
 "OUT3b pure geography": ("The van crossed Tower Bridge into Southwark.", "The van never left the north side of the Thames."),
 "OUT4 pure knowledge (station)": ("He got off the train at Waterloo.", "He was never south of the river that day."),
 "HEDGE0 unhedged": ("The light was red.", "The light was green."),
 "HEDGE1 fairly sure": ("I'm fairly sure the light was red.", "The light was green."),
 "HEDGE2 maybe": ("Maybe the light was red.", "The light was green."),
 "HEDGE3 I think": ("I think the light was red.", "The light was green."),
 "HEDGE4 both hedged": ("I think the light was red.", "I think the light was green."),
 "SPA4 first-person agreement-ish": ("I was standing at the bus stop.", "I was sitting in my car at the lights."),
 "SPA5 my car": ("My car was parked outside the bank.", "My car was in the garage all day."),
 "TIME1 absolute times differ": ("The shop closed at 6 pm.", "I left the shop at 7 pm."),
 "TIME2 time split by cleaner": ("It was approx.", "The van left at 10 p.m."),
}

cfg = load_config()
zs = NLIScorer(resolve_model("cross-encoder/nli-deberta-v3-base", cfg), batch_size=16, max_length=128,
               cache_path=None, aggregation="min_contradiction", verbose=False)
ft = NLIScorer(REPO + r"\models\finetuned\nli-deberta-v3-base-pheme-all", batch_size=16, max_length=128,
               cache_path=None, aggregation="min_contradiction", verbose=False)
res = {}
for name, s in (("zs", zs), ("ft", ft)):
    keys = list(PAIRS); pairs = [PAIRS[k] for k in keys]
    both = s.score_directional(pairs + [(b, a) for a, b in pairs]); n = len(pairs)
    mn = aggregate_directions(both[:n], both[n:], "min_contradiction")
    for i, k in enumerate(keys):
        res.setdefault(k, {})[name] = {"abC": round(float(both[i,2]),3), "baC": round(float(both[n+i,2]),3),
                                       "min": [round(float(x),3) for x in mn[i]]}
for k, v in res.items():
    print(f"{k:42s} ZS min={v['zs']['min']} (abC={v['zs']['abC']} baC={v['zs']['baC']}) | FT min={v['ft']['min']}")

# tokenizer consistency base vs finetuned
txts = ["The driver didn't stop at St. James's Park at 10:30pm!", "I'm not 100% sure but I believe the van was white."]
for t in txts:
    a = zs.tokenizer(t)["input_ids"]; b = ft.tokenizer(t)["input_ids"]
    print("TOK same:", a == b, len(a), len(b))
    if a != b: print(zs.tokenizer.convert_ids_to_tokens(a)); print(ft.tokenizer.convert_ids_to_tokens(b))

emb = Embedder(resolve_model(cfg["models"]["embedding"], cfg), cache_dir=None)
class E2:
    name = "minilm"
    def encode(self, t): return emb.encode(t)

def pipe(accounts, label):
    c = load_config()
    claims = accounts_to_claims(accounts, event=label)
    small_input_overrides(c, len(claims))
    r = run_pipeline(claims, c, embedder=E2(), scorer=zs, name=label, verbose=True)
    print(f"\n=== {label} ===")
    for cl, lab in zip(r.claims, r.labels): print(f"  [{lab:2d}] {cl.account_id}: {cl.text}")
    E = r.embeddings
    for (i, j), p in zip(r.pairs, r.probs):
        print(f"  pair {r.claims[i].account_id}|{r.claims[j].account_id}: C={p[2]:.3f} sim={float(E[i]@E[j]):.2f} :: {r.claims[i].text!r} vs {r.claims[j].text!r}")
    f = r.report["findings"]
    print("  flagged clusters:", r.report["summary"]["n_flagged_clusters"], "counts:", f["counts"])
    for e in f["corroborated"]: print("  CORROBORATED:", e["statement"], e["restated_by"])
    for e in f["single_source"]: print("  SINGLE (not contradicted):", e["account_id"], e["text"])
    return r, claims

r, cl = pipe(read_accounts_file(REPO + r"\examples\witness_accounts.txt"), "example_file")
# pairwise sims for missed pairs in example
E = r.embeddings
idx = {c.text: k for k, c in enumerate(r.claims)}
for a, b in [("It was raining heavily.", "The road was dry."),
             ("The driver stayed at the scene.", "The driver did not stay at the scene, he drove off."),
             ("The blue car ran the red light and hit the cyclist.", "The light was green when the blue car went through.")]:
    if a in idx and b in idx: print("  sim", a, "|", b, round(float(E[idx[a]] @ E[idx[b]]), 3))

apart = [
 {"account_id": "Witness A", "text": "The driver stayed at the scene. It was raining heavily. The cyclist was wearing a helmet. I was waiting at the bus stop."},
 {"account_id": "Witness B", "text": "He was gone before anyone could get his number plate. The tarmac was bone dry. The lad on the bike had nothing on his head. I was waiting at the bus stop too."},
]
pipe(apart, "clustered_apart")

multi = [
 {"account_id": "Witness A", "text": "I was walking my dog along the canal path at the time and I heard a loud bang which I first thought was a firework, and I turned around towards the junction near the bakery and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was lying on the ground near the kerb with his bike under him, and a woman in a yellow coat ran over to help him and somebody else was shouting for an ambulance, and the driver of the blue car got out and looked at the bike for a few seconds, and then he got back into the car and drove off towards the bridge."},
 {"account_id": "Witness B", "text": "The driver stayed with the cyclist until the police arrived."},
 {"account_id": "Witness C", "text": "It was approx. 10 p.m. The van was parked on St. James Street. Dr. Patel arrived first."},
 {"account_id": "Witness D", "text": "It was around 6 p.m. The van was parked on Kings Road. Dr. Patel arrived last."},
]
pipe(multi, "long_and_splitting")

self_acc = parse_accounts_text("Witness A: The light was red when the car went through.\n\nOn reflection, the light was green when the car went through.\n\nWitness B: The car stopped at the light.")
print("\nparsed:", self_acc)
pipe(self_acc, "multi_paragraph")
