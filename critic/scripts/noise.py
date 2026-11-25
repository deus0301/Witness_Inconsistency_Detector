import sys, re, collections, math
import numpy as np
sys.path.insert(0, r"D:/Projects/Witness_Inconsistency_Detector")
sys.path.insert(0, r"critic/scripts")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import baselines as B
from scipy import stats
rng = np.random.default_rng(0)
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(c-h,3), round(c+h,3))
print("Wilson CIs")
print(" NLI-only min pooled P", 454, 734, wilson(454,734), " R 453/1378", wilson(453,1378))
print(" labelled-pipeline agglo P", round(.6717*265), 265, wilson(178,265), " R", 178, 1348, wilson(178,1348))
print(" NLI-only (mapped) P", round(.6176*727), 727, wilson(449,727))
print(" cluster-level P 23/53", wilson(23,53), " R 23/28", wilson(23,28))
print(" audit genuine+time 17/40", wilson(17,40), " genuine 11/40", wilson(11,40))
print(" unknown flag rate pipeline 1.2% vs 9.4% of 2046")
# cluster-level bootstrap: TP23 FP30 FN5 TN44
cells = np.array([23,30,5,44]); n=cells.sum()
y = np.array([1]*23+[0]*30+[1]*5+[0]*44,bool); f=np.array([1]*23+[1]*30+[0]*5+[0]*44,bool)
def f1(y,f):
    tp=(y&f).sum(); fp=(~y&f).sum(); fn=(y&~f).sum(); return 2*tp/(2*tp+fp+fn) if tp else 0.
ds=[];dets=[];fa=[]
for _ in range(10000):
    i=rng.integers(0,n,n); a=f1(y[i],f[i]); b=f1(y[i],np.ones(n,bool)); ds.append(a-b); dets.append(a); fa.append(b)
print("cluster-level F1 detector 95%% CI %.3f-%.3f; flag-all %.3f-%.3f; diff %.3f [%.3f, %.3f]; P(diff<=0)=%.3f" % (*np.percentile(dets,[2.5,97.5]),*np.percentile(fa,[2.5,97.5]),np.mean(ds),*np.percentile(ds,[2.5,97.5]),np.mean(np.array(ds)<=0)))
# Fisher exact: flagged vs positive
print(" Fisher exact flagged x proxy-positive", stats.fisher_exact([[23,30],[5,44]]))
# fine-tune fold deltas
zs_def = dict(gw=.159,syd=.273,ch=.294,ott=.761); zs_val=dict(gw=.338,syd=.324,ch=.447,ott=.688)
ft = dict(gw=.620,syd=.465,ch=.651,ott=.692); anyc=dict(gw=.518,syd=.451,ch=.569,ott=.286); fa=dict(gw=.362,syd=.414,ch=.361,ott=.510)
ap_z=dict(gw=.414,syd=.402,ch=.347,ott=.852); ap_f=dict(gw=.503,syd=.443,ch=.712,ott=.791)
pr_zd=dict(gw=.033,syd=.057,ch=.195,ott=.188); pr_zv=dict(gw=.188,syd=.079,ch=.569,ott=.286); pr_f=dict(gw=.611,syd=.151,ch=.489,ott=.178)
def tci(a,b,name):
    d=np.array([b[k]-a[k] for k in a]); m=d.mean(); s=d.std(ddof=1); h=stats.t.ppf(.975,3)*s/2
    print(f" {name}: deltas {np.round(d,3).tolist()} mean {m:.3f} sd {s:.3f} 95% t-CI [{m-h:.3f}, {m+h:.3f}]")
print("Fold-level paired deltas (n=4, df=3)")
tci(zs_def,ft,"F1 ft - zs_default"); tci(zs_val,ft,"F1 ft - zs_valtuned"); tci(anyc,ft,"F1 ft - anycue heuristic")
tci(fa,zs_def,"F1 zs_default - flag_all"); tci(fa,zs_val,"F1 zs_val - flag_all"); tci(fa,anyc,"F1 anycue - flag_all")
tci(ap_z,ap_f,"AP ft - zs"); tci(pr_zd,pr_f,"pipeline R ft - zs_default"); tci(pr_zv,pr_f,"pipeline R ft - zs_valtuned")
# group structure / design effect using anycue on pooled RTE
allp=B.allp; F=B.F; yy=B.y
g = np.array([hash((p.event, frozenset(s.strip() for s in p.stories.split(';')))) for p in allp])
ug = np.unique(g); print("story-pair groups", len(ug), {e: len({gg for gg,p in zip(g,allp) if p.event==e}) for e in B.EVENT_FILES})
pred = F["anycue"]
def prf1(y,f):
    tp=(y&f).sum(); fp=(~y&f).sum(); fn=(y&~f).sum(); return 2*tp/(2*tp+fp+fn)
idx_by = {u: np.where(g==u)[0] for u in ug}
pb=[];gb=[]
for _ in range(3000):
    i=rng.integers(0,len(yy),len(yy)); pb.append(prf1(yy[i],pred[i]))
    gs=rng.choice(ug,len(ug)); j=np.concatenate([idx_by[u] for u in gs]); gb.append(prf1(yy[j],pred[j]))
print("anycue pooled F1 %.3f: pair-bootstrap 95%% [%.3f,%.3f] width %.3f | story-pair-group bootstrap [%.3f,%.3f] width %.3f" % (prf1(yy,pred),*np.percentile(pb,[2.5,97.5]),np.ptp(np.percentile(pb,[2.5,97.5])),*np.percentile(gb,[2.5,97.5]),np.ptp(np.percentile(gb,[2.5,97.5]))))
# oracle story cue: two stories AND tfidf cos >= t (t tuned on val event); LOEO
folds=["germanwings-crash","sydneysiege","ottawashooting","charliehebdo"]
out=[]
for i,h in enumerate(folds):
    v=folds[(i+1)%4]
    def fz(ps):
        ff=B.feats(ps); two=np.array([len(p.stories.split(';'))==2 for p in ps]); yv=np.array([p.label=='contradiction' for p in ps]); return ff['cos'],two,yv
    cv,twv,yv=fz(B.ev[v]); ct,twt,yt=fz(B.ev[h])
    best=max(((prf1(yv,twv&(cv>=t)),t) for t in np.quantile(cv,np.linspace(.02,.98,49))))
    r=prf1(yt,twt&(ct>=best[1])); out.append(r)
    from sklearn.metrics import average_precision_score
    print(f" oracle 'two stories & cos>=t' {h}: val t={best[1]:.3f} test F1 {r:.3f}; AP(two*cos) {average_precision_score(yt, twt*ct):.3f}")
print(" oracle mean F1 %.3f" % np.mean(out))
