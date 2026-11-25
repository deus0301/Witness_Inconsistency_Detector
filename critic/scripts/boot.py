import sys, re, json, sqlite3
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, r"D:\Projects\Witness_Inconsistency_Detector")
from pathlib import Path
from data.pheme_rte import load_rte_split
from pipeline.config import text_hash
from nli_scoring.scorer import aggregate_directions
from sklearn.metrics import average_precision_score

R = Path(r"D:\Projects\Witness_Inconsistency_Detector")
SPL = R / "pheme_rte_datasets_2016/pheme_rte_datasets_2016/test_training_splits"
CACHE = R / "outputs/cache"
res = json.load(open(R / "reports/results/finetune_results.json", encoding="utf-8"))

def load_cache(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    d = {k: (e, n, c) for k, e, n, c in con.execute("select key,e,n,c from scores")}
    con.close(); return d

def scores(cache, pairs):
    f = np.array([cache[text_hash(p.premise, p.hypothesis)] for p in pairs], dtype=float)
    b = np.array([cache[text_hash(p.hypothesis, p.premise)] for p in pairs], dtype=float)
    return f, b

def story_key(s):
    parts = re.split(r";\s*(?=Story \d+:)", s)
    return tuple(sorted(re.sub(r"^Story \d+:\s*", "", x).strip() for x in parts))

def prf(c, flag):
    tp = (c & flag).sum(); fp = (~c & flag).sum(); fn = (c & ~flag).sum()
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)

zs_cache = load_cache(CACHE / "nli_cross-encoder_nli-deberta-v3-base.sqlite")
keys = {"germanwings-crash": "gw", "sydneysiege": "syd", "charliehebdo": "chebdo", "ottawashooting": "ott"}
rng = np.random.default_rng(0)
B = 2000
fold_data = {}
for ev, fold in res["folds"].items():
    test = load_rte_split(SPL, ev, "test")
    assert len(test) == fold["n_test_pairs"], (len(test), fold["n_test_pairs"])
    ft_cache = load_cache(CACHE / f"nli_finetuned_nli-deberta-v3-base-pheme-leave-{keys[ev]}_{fold['run_id']}.sqlite")
    sc = {"zero_shot": scores(zs_cache, test), "finetuned": scores(ft_cache, test)}
    truth = np.array([p.label for p in test]); c = truth == "contradiction"
    groups = np.array([hash(story_key(p.stories)) for p in test])
    S = {}
    for k, s in fold["settings"].items():
        probs = aggregate_directions(*sc[s["model"]], s["aggregation"])[:, 2]
        S[k] = (probs, s["threshold"])
        p, r, f1 = prf(c, probs >= s["threshold"])
        rep = fold["test"][k]["contradiction_at_threshold"]
        ap = average_precision_score(c, probs)
        print(f"{ev:18s} {k:34s} recomputed P{p:.3f} R{r:.3f} F1{f1:.3f} AP{ap:.3f} | reported P{rep['precision']:.3f} R{rep['recall']:.3f} AP{fold['test'][k]['contradiction_average_precision']:.3f}")
    # discrimination subsets for finetuned and zero-shot
    for k in ("zero_shot_default", "finetuned_best_f1"):
        probs = S[k][0]
        m1 = truth != "entailment"; m2 = truth != "neutral"
        print(f"   {k}: AP contra-vs-neutral {average_precision_score(c[m1], probs[m1]):.3f}  AP contra-vs-entail {average_precision_score(c[m2], probs[m2]):.3f}  base rates {c[m1].mean():.3f}/{c[m2].mean():.3f}")
    # per contradiction story-group recall for finetuned
    for k in ("zero_shot_default", "finetuned_best_f1"):
        probs, thr = S[k]
        g = defaultdict(list)
        for gi, ci, fl in zip(groups, c, probs >= thr):
            if ci: g[gi].append(fl)
        print(f"   {k} recall per contradiction story-group (size:recall):", sorted([(len(v), round(float(np.mean(v)), 2)) for v in g.values()], reverse=True))
    # flag rate per entailment story-group (finetuned)
    probs, thr = S["finetuned_best_f1"]
    g = defaultdict(list)
    for gi, t, fl in zip(groups, truth, probs >= thr):
        if t == "entailment": g[gi].append(fl)
    print("   finetuned entailment flag rate per story-group (size:rate):", sorted([(len(v), round(float(np.mean(v)), 2)) for v in g.values()], reverse=True))
    fold_data[ev] = (c, groups, S)

def metrics(c, S, idx):
    out = {}
    for k, (probs, thr) in S.items():
        cc = c[idx]; pp = probs[idx]
        p, r, f1 = prf(cc, pp >= thr)
        ap = average_precision_score(cc, pp) if 0 < cc.sum() < len(cc) else np.nan
        out[k] = (p, r, f1, ap)
    return out

def ci(a):
    a = np.array(a); a = a[~np.isnan(a)]
    return f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"

KS = ["zero_shot_default", "zero_shot_val_best_f1", "finetuned_best_f1"]
for mode in ("pair", "story-group"):
    print("\n==== bootstrap resampling unit:", mode)
    pooled = defaultdict(list)  # mean over folds per replicate
    for ev, (c, groups, S) in fold_data.items():
        reps = defaultdict(list)
        ug = np.unique(groups); members = {g: np.where(groups == g)[0] for g in ug}
        for b in range(B):
            if mode == "pair":
                idx = rng.integers(0, len(c), len(c))
            else:
                gs = rng.choice(ug, len(ug), replace=True)
                idx = np.concatenate([members[g] for g in gs])
            m = metrics(c, S, idx)
            for k in KS:
                reps[k].append(m[k])
            reps["dF1"].append(m["finetuned_best_f1"][2] - m["zero_shot_val_best_f1"][2])
            reps["dAP"].append(m["finetuned_best_f1"][3] - m["zero_shot_default"][3])
            reps["dR_default"].append(m["finetuned_best_f1"][1] - m["zero_shot_default"][1])
        fold_data[ev] = (c, groups, S, reps) if len(fold_data[ev]) == 3 else (c, groups, S, reps)
        line = []
        for k in KS:
            a = np.array(reps[k])
            line.append(f"{k}: R{ci(a[:,1])} F1{ci(a[:,2])} AP{ci(a[:,3])}")
        print(f"{ev:18s} n_groups={len(ug)} | " + " | ".join(line))
        print(f"{'':18s} dF1(ft-zs_valtuned) {ci(reps['dF1'])}  dAP(ft-zs) {ci(reps['dAP'])}  dRecall(ft-zs_default) {ci(reps['dR_default'])}")
        fold_data[ev] = (c, groups, S)
        for k in KS:
            pooled[k].append(np.array(reps[k]))
        pooled["dF1"].append(np.array(reps["dF1"])); pooled["dAP"].append(np.array(reps["dAP"])); pooled["dR"].append(np.array(reps["dR_default"]))
    print("MEAN over 4 folds (conditional on these 4 events):")
    for k in KS:
        a = np.mean(np.stack(pooled[k]), axis=0)
        print(f"  {k}: P{ci(a[:,0])} R{ci(a[:,1])} F1{ci(a[:,2])} AP{ci(a[:,3])}")
    for k in ("dF1", "dAP", "dR"):
        a = np.nanmean(np.stack(pooled[k]), axis=0)
        print(f"  {k}: {ci(a)}  P(<=0)={np.mean(a<=0):.3f}")

# leave-one-fold-out sensitivity on mean
print("\nmean recall / F1 excluding each fold (finetuned_best_f1 vs zero_shot_default vs zs_val_best_f1):")
for ex in res["folds"]:
    rest = [f for e, f in res["folds"].items() if e != ex]
    fmt = lambda k, m: np.mean([f["test"][k]["contradiction_at_threshold"][m] for f in rest])
    print(f"  without {ex}: R ft {fmt('finetuned_best_f1','recall'):.3f} zs_def {fmt('zero_shot_default','recall'):.3f} zs_val {fmt('zero_shot_val_best_f1','recall'):.3f} | F1 ft {fmt('finetuned_best_f1','f1'):.3f} zs_val {fmt('zero_shot_val_best_f1','f1'):.3f} | AP ft {np.mean([f['test']['finetuned_best_f1']['contradiction_average_precision'] for f in rest]):.3f} zs {np.mean([f['test']['zero_shot_default']['contradiction_average_precision'] for f in rest]):.3f}")
# pooled (micro) metrics across folds
print("\nmicro-pooled across folds (all 5,413 test pairs):")
for k in ["zero_shot_default", "zero_shot_val_best_f1", "finetuned_best_f1"]:
    cs = np.concatenate([fd[0] for fd in fold_data.values()])
    fl = np.concatenate([fd[2][k][0] >= fd[2][k][1] for fd in fold_data.values()])
    print(" ", k, "P%.3f R%.3f F1%.3f" % prf(cs, fl))
