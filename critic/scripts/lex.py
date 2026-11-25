import sys, re
import numpy as np
sys.path.insert(0, r"D:\Projects\Witness_Inconsistency_Detector")
from pathlib import Path
from data.pheme_rte import load_pheme_rte, EVENT_FILES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

PER = Path(r"D:\Projects\Witness_Inconsistency_Detector\pheme_rte_datasets_2016\pheme_rte_datasets_2016\per_event_splits")
ev = {e: load_pheme_rte(PER, e) for e in EVENT_FILES}
NEG = {"not", "no", "never", "denies", "denied", "false", "fake", "hoax", "untrue", "isn't", "wasn't", "n't"}
def feats(ps, vec):
    A = vec.transform([p.premise for p in ps]); B = vec.transform([p.hypothesis for p in ps])
    cos = np.asarray(A.multiply(B).sum(1)).ravel()
    out = []
    for p, c in zip(ps, cos):
        a, b = set(p.premise.lower().split()), set(p.hypothesis.lower().split())
        na = set(re.findall(r"\d+", p.premise)); nb = set(re.findall(r"\d+", p.hypothesis))
        out.append([c, len(a & b) / max(1, len(a | b)), float(bool(na) and bool(nb) and na != nb),
                    float(bool(a & NEG) != bool(b & NEG)), abs(len(a) - len(b))])
    return np.array(out)
vec = TfidfVectorizer().fit([t for ps in ev.values() for p in ps for t in (p.premise, p.hypothesis)])  # unsupervised, text only
aps = []
for held in ev:
    tr = [p for e, ps in ev.items() if e != held for p in ps]; te = ev[held]
    Xtr, Xte = feats(tr, vec), feats(te, vec)
    ytr = np.array([p.label == "contradiction" for p in tr]); yte = np.array([p.label == "contradiction" for p in te])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    s = clf.predict_proba(Xte)[:, 1]
    lab = np.array([p.label for p in te]); m = lab != "entailment"
    ap = average_precision_score(yte, s); aps.append(ap)
    print(f"{held:18s} lexical-LR AP {ap:.3f} (base rate {yte.mean():.3f}); contra-vs-neutral AP {average_precision_score(yte[m], s[m]):.3f}; tfidf-cos alone AP {average_precision_score(yte, Xte[:,0]):.3f}")
print("mean lexical-LR AP", round(float(np.mean(aps)), 3))
