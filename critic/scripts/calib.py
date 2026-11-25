import sys, json
import numpy as np
sys.path.insert(0, r"D:\Projects\Witness_Inconsistency_Detector")
exec(open("boot.py").read().split("zs_cache = ")[0])
zs_cache = load_cache(CACHE / "nli_cross-encoder_nli-deberta-v3-base.sqlite")
keys = {"germanwings-crash": "gw", "sydneysiege": "syd", "charliehebdo": "chebdo", "ottawashooting": "ott"}
def ece(y, p, n=10):
    bins = np.linspace(0, 1, n + 1); e = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo)
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e
for ev, fold in res["folds"].items():
    test = load_rte_split(SPL, ev, "test")
    ft = load_cache(CACHE / f"nli_finetuned_nli-deberta-v3-base-pheme-leave-{keys[ev]}_{fold['run_id']}.sqlite")
    truth = np.array([p.label for p in test]); y = (truth == "contradiction").astype(float)
    for name, cache, agg in (("zs", zs_cache, "min_contradiction"), ("ft", ft, fold["settings"]["finetuned_best_f1"]["aggregation"])):
        pr = aggregate_directions(*scores(cache, test), agg)
        c = pr[:, 2]
        mid = ((c > 0.05) & (c < 0.95)).mean()
        am = np.array(["entailment", "neutral", "contradiction"])[pr.argmax(1)]
        er = (am[truth == "entailment"] == "entailment").mean()
        print(f"{ev:18s} {name} {agg:17s} frac contra-prob in (0.05,0.95): {mid:.3f}; ECE(contra) {ece(y, c):.3f}; entailment recall(argmax) {er:.3f}; acc {(am == truth).mean():.3f}")
