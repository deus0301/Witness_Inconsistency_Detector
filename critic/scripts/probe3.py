import os, sys
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.path.insert(0, r"D:\Projects\Witness_Inconsistency_Detector")
from nli_scoring.scorer import NLIScorer, aggregate_directions
s = NLIScorer("cross-encoder/nli-deberta-v3-base", cache_path=None, aggregation="min_contradiction", verbose=False)
base = ("I was walking my dog along the canal path at the time and I heard a loud bang which I first thought was a firework, and I turned around towards the junction near the bakery and the pharmacy and I saw that the blue car had hit the cyclist, and the cyclist was lying on the ground near the kerb with his bike under him, and a woman in a yellow coat ran over to help him and somebody else was shouting for an ambulance, and the driver of the blue car got out and looked at the bike for a few seconds, and then he ")
A_off = base + "got back into the car and drove off towards the bridge."
A_stay = base + "stayed with the cyclist until the police arrived."
B = "The driver stayed with the cyclist until the police arrived."
B2 = "The driver drove off towards the bridge."
enc = s.tokenizer(A_off, B, truncation=True, max_length=128)
print("full len", len(s.tokenizer(A_off, B)["input_ids"]))
print("TRUNCATED INPUT:", s.tokenizer.decode(enc["input_ids"])[-250:])
pairs = [(A_off, B), (A_stay, B), (A_off, B2), (A_stay, B2)]
p = s.score_pairs(pairs)
for (a, b), r in zip(pairs, p): print(a[-40:], "|", b, "->", [round(float(x),3) for x in r])
# 128->512 control
s2 = NLIScorer("cross-encoder/nli-deberta-v3-base", cache_path=None, aggregation="min_contradiction", verbose=False, max_length=512)
p2 = s2.score_pairs(pairs)
for (a, b), r in zip(pairs, p2): print("[512]", a[-40:], "|", b, "->", [round(float(x),3) for x in r])
