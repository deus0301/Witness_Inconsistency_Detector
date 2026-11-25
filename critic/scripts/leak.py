import sys, re, itertools
from collections import Counter, defaultdict
sys.path.insert(0, r"D:\Projects\Witness_Inconsistency_Detector")
from data.pheme_rte import load_pheme_rte, load_rte_split, EVENT_FILES, SPLIT_KEYS, load_rte_file
from pathlib import Path

ROOT = Path(r"D:\Projects\Witness_Inconsistency_Detector\pheme_rte_datasets_2016\pheme_rte_datasets_2016")
PER = ROOT / "per_event_splits"; SPL = ROOT / "test_training_splits"

def norm(t):
    t = t.lower()
    t = re.sub(r"@\w+", " ", t); t = re.sub(r"#", " ", t)
    t = re.sub(r"^(rt|breaking|update|breaking news|just in)\b[:\s-]*", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return " ".join(t.split())

def toks(t): return set(norm(t).split())
def jac(a, b): return len(a & b) / max(1, len(a | b))

ev = {e: load_pheme_rte(PER, e) for e in EVENT_FILES}
print("pairs per event", {e: len(v) for e, v in ev.items()})
texts = {e: {} for e in ev}
for e, ps in ev.items():
    for p in ps:
        texts[e][p.tweet_a] = p.premise; texts[e][p.tweet_b] = p.hypothesis
print("distinct tweets per event", {e: len(v) for e, v in texts.items()})
# tweets per pair reuse
for e, ps in ev.items():
    c = Counter([p.tweet_a for p in ps] + [p.tweet_b for p in ps])
    print(e, "pairs per tweet mean %.1f max %d" % (sum(c.values()) / len(c), max(c.values())))

# cross-event tweet id overlap and text overlap
ids = {e: set(v) for e, v in texts.items()}
for a, b in itertools.combinations(ev, 2):
    print("id overlap", a, b, len(ids[a] & ids[b]))
ntexts = {e: {tid: norm(t) for tid, t in v.items()} for e, v in texts.items()}
for a, b in itertools.combinations(ev, 2):
    sa, sb = set(ntexts[a].values()), set(ntexts[b].values())
    print("exact-norm text overlap", a, b, len(sa & sb))
# near dup across events: max jaccard
for a, b in itertools.combinations(ev, 2):
    ta = {k: toks(t) for k, t in texts[a].items()}; tb = {k: toks(t) for k, t in texts[b].items()}
    hits = []
    for ka, xa in ta.items():
        for kb, xb in tb.items():
            j = jac(xa, xb)
            if j >= 0.6: hits.append((j, texts[a][ka], texts[b][kb]))
    hits.sort(reverse=True)
    print("near-dup (jac>=0.6) cross-event", a, b, len(hits), hits[:2])

# within-event near duplicates
for e in ev:
    t = {k: toks(x) for k, x in texts[e].items()}
    ks = list(t); n = 0; exact = Counter(ntexts[e].values())
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            if jac(t[ks[i]], t[ks[j]]) >= 0.8: n += 1
    print(e, "within-event tweet pairs jac>=0.8:", n, "; duplicate normalised texts:", sum(v - 1 for v in exact.values() if v > 1))

# LOEO split files vs per-event files
for held in SPLIT_KEYS:
    tr = load_rte_split(SPL, held, "train"); te = load_rte_split(SPL, held, "test")
    tr_ev = Counter(p.event for p in tr)
    te_ids = {p.tweet_a for p in te} | {p.tweet_b for p in te}
    tr_ids = {p.tweet_a for p in tr} | {p.tweet_b for p in tr}
    te_txt = {norm(p.premise) for p in te} | {norm(p.hypothesis) for p in te}
    tr_txt = {norm(p.premise) for p in tr} | {norm(p.hypothesis) for p in tr}
    te_tok = [toks(x) for x in {p.premise for p in te} | {p.hypothesis for p in te}]
    tr_tok = [toks(x) for x in {p.premise for p in tr} | {p.hypothesis for p in tr}]
    near = sum(1 for a in te_tok if any(jac(a, b) >= 0.6 for b in tr_tok))
    # story strings shared
    te_st = {s.strip() for p in te for s in re.split(r";\s*Story \d+:", p.stories)}
    tr_st = {s.strip() for p in tr for s in re.split(r";\s*Story \d+:", p.stories)}
    print(f"LOEO leave-{held}: train {len(tr)} events {dict(tr_ev)}; test {len(te)} events {dict(Counter(p.event for p in te))}; "
          f"tweet-id overlap {len(te_ids & tr_ids)}; exact-norm text overlap {len(te_txt & tr_txt)}; "
          f"test tweets with near-dup (jac>=.6) in train {near}/{len(te_tok)}; shared story strings {len(te_st & tr_st)}")

# story structure: label purity per story-pair, number of distinct story pairs
def story_split(s):
    parts = re.split(r";\s*(?=Story \d+:)", s)
    return tuple(sorted(re.sub(r"^Story \d+:\s*", "", x).strip() for x in parts))

for e, ps in ev.items():
    g = defaultdict(Counter)
    for p in ps: g[story_split(p.stories)][p.label] += 1
    impure = sum(1 for c in g.values() if len(c) > 1)
    lab_groups = Counter(c.most_common(1)[0][0] for c in g.values())
    contra_groups = [sum(c.values()) for c in g.values() if c.most_common(1)[0][0] == "contradiction"]
    stories = {s for k in g for s in k}
    print(f"{e}: {len(stories)} stories, {len(g)} story-groups; groups by label {dict(lab_groups)}; impure groups {impure}; "
          f"contradiction pairs per contra-group: n={len(contra_groups)} sizes {sorted(contra_groups, reverse=True)}")

# Same tweet labelled differently? Does the same story pair appear with both orders?
# Does story identity predict: for each tweet, its story; entailment iff same story?
for e, ps in ev.items():
    same = Counter((len(story_split(p.stories)) == 1, p.label) for p in ps)
    print(e, "one-story vs label", dict(same))
