import sys, re, collections
import numpy as np
sys.path.insert(0, r"D:/Projects/Witness_Inconsistency_Detector")
from data.pheme_rte import load_pheme_rte, EVENT_FILES
from sklearn.metrics import average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
root = r"D:/Projects/Witness_Inconsistency_Detector/pheme_rte_datasets_2016/pheme_rte_datasets_2016/per_event_splits"
ev = {e: load_pheme_rte(root, e) for e in EVENT_FILES}
allp = [p for e in ev for p in ev[e]]
print("n", len(allp), collections.Counter(p.label for p in allp))
# story structure
nst = collections.Counter((p.label, len([s for s in p.stories.split(";") if s.strip()])) for p in allp)
print("label x n_stories", nst)
# story-pair purity
sp = collections.defaultdict(collections.Counter)
for p in allp:
    key = frozenset(s.strip() for s in p.stories.split(";"))
    sp[(p.event, key)][p.label] += 1
pure = sum(1 for c in sp.values() if len(c) == 1)
print("story-pair groups", len(sp), "pure", pure)
# tweet frequency
tw = collections.Counter([p.tweet_a for p in allp] + [p.tweet_b for p in allp])
print("distinct tweets", len(tw), "mean pairs per tweet", np.mean(list(tw.values())))

NEG = re.compile(r"\b(not|no|never|none|nobody|deny|denies|denied|false|fake|hoax|untrue|debunk\w*|rumou?r\w*|unconfirmed|n't)\b|n't", re.I)
NUM = re.compile(r"\b\d+\b")
ANT = [("dead","alive"),("killed","injured"),("killed","wounded"),("arrested","escaped"),("arrested","at large"),
       ("confirmed","denied"),("confirmed","unconfirmed"),("one","two"),("released","held"),("true","false"),
       ("shot dead","on the loose"),("captured","on the loose"),("inside","outside")]
def neg(t): return bool(NEG.search(t))
def nums(t): return set(NUM.findall(t))
def toks(t): return set(re.findall(r"[a-z0-9']+", t.lower()))
def ant(a,b):
    a,b=a.lower(),b.lower()
    return any((x in a and y in b and y not in a) or (y in a and x in b and x not in a) for x,y in ANT)
tf = TfidfVectorizer(min_df=1).fit([p.premise for p in allp]+[p.hypothesis for p in allp])
def feats(ps):
    A = tf.transform([p.premise for p in ps]); B = tf.transform([p.hypothesis for p in ps])
    cos = np.asarray(A.multiply(B).sum(1)).ravel()
    jac = np.array([len(toks(p.premise)&toks(p.hypothesis))/max(1,len(toks(p.premise)|toks(p.hypothesis))) for p in ps])
    negm = np.array([neg(p.premise)!=neg(p.hypothesis) for p in ps])
    numm = np.array([bool(nums(p.premise)) and bool(nums(p.hypothesis)) and nums(p.premise)!=nums(p.hypothesis) for p in ps])
    antm = np.array([ant(p.premise,p.hypothesis) for p in ps])
    return dict(cos=cos, jac=jac, negm=negm, numm=numm, antm=antm, anycue=negm|numm|antm)
def prf(y, f):
    tp=int((y&f).sum()); fp=int((~y&f).sum()); fn=int((y&~f).sum())
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    return P,R,(2*P*R/(P+R) if P+R else 0), int(f.sum())
y = np.array([p.label=="contradiction" for p in allp])
F = feats(allp)
print("\nPOOLED (5413), prevalence %.3f, flag-all F1 %.3f" % (y.mean(), 2*y.mean()/(1+y.mean())))
for k in ("negm","numm","antm","anycue"):
    print(k, "P=%.3f R=%.3f F1=%.3f n=%d" % prf(y, F[k]))
for k in ("cos","jac"):
    print(k, "AP(high=contra)=%.3f AP(low=contra)=%.3f" % (average_precision_score(y,F[k]), average_precision_score(y,-F[k])))
    best = max(((prf(y, F[k]>=t)+(t,'>=')) for t in np.quantile(F[k], np.linspace(0.02,0.98,49))), key=lambda r:r[2])
    best2 = max(((prf(y, F[k]<=t)+(t,'<=')) for t in np.quantile(F[k], np.linspace(0.02,0.98,49))), key=lambda r:r[2])
    print("   oracle best-F1 >=:", ["%.3f"%x if isinstance(x,float) else x for x in best], " <=:", ["%.3f"%x if isinstance(x,float) else x for x in best2])
# 3-class: entailment vs not with cosine (story structure leak)
ye = np.array([p.label=="entailment" for p in allp])
print("entailment AP from tfidf cos %.3f (prev %.3f)" % (average_precision_score(ye,F["cos"]), ye.mean()))
yn = np.array([p.label=="neutral" for p in allp])
print("unknown AP from -cos %.3f (prev %.3f)" % (average_precision_score(yn,-F["cos"]), yn.mean()))
# cos among non-entailment: contra vs unknown
m = ~ye
print("contra vs unknown (non-ent pairs) AP cos high %.3f low %.3f prev %.3f" % (average_precision_score(y[m],F["cos"][m]),average_precision_score(y[m],-F["cos"][m]), y[m].mean()))

# LOEO with val event (same cyclic scheme), tune direction+threshold of cos on val event, report on test
folds = ["germanwings-crash","sydneysiege","ottawashooting","charliehebdo"]
print("\nLOEO (val = next event cyclic); per-event features")
res = collections.defaultdict(list)
for i,h in enumerate(folds):
    v = folds[(i+1)%4]
    tp_, vp_ = ev[h], ev[v]
    yt = np.array([p.label=="contradiction" for p in tp_]); yv = np.array([p.label=="contradiction" for p in vp_])
    Ft, Fv = feats(tp_), feats(vp_)
    out = {"flag_all": prf(yt, np.ones_like(yt))}
    for k in ("negm","numm","antm","anycue"):
        out[k] = prf(yt, Ft[k])
    for k in ("cos","jac"):
        cands=[]
        for t in np.quantile(Fv[k], np.linspace(0.02,0.98,49)):
            cands.append((prf(yv,Fv[k]>=t)[2],'>=',t)); cands.append((prf(yv,Fv[k]<=t)[2],'<=',t))
        f1,d,t = max(cands)
        pred = Ft[k]>=t if d=='>=' else Ft[k]<=t
        out[k+"_valtuned"] = prf(yt,pred)
        out[k+"_AP_test_low"] = average_precision_score(yt,-Ft[k]); out[k+"_AP_test_high"] = average_precision_score(yt,Ft[k])
    print(h, "prev %.3f" % yt.mean())
    for k,vv in out.items():
        print("   ", k, ["%.3f"%x if isinstance(x,float) else x for x in (vv if isinstance(vv,tuple) else (vv,))])
        res[k].append(vv[2] if isinstance(vv,tuple) else vv)
print("\nMEAN over folds:")
for k,v in res.items(): print("  ", k, "%.3f" % np.mean(v))
