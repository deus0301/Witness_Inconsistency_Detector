"""Interactive demo.  Start with:  streamlit run app/streamlit_app.py

Paste several accounts of one event (or pick a PHEME event) and see the sub-event
clusters and the claim pairs that contradict each other. The app only surfaces
disagreement; it never says which account is right.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from clustering.cluster import ALGORITHMS  # noqa: E402
from clustering.embed import build_embedder  # noqa: E402
from data.pheme import list_events, load_event_from_config  # noqa: E402
from nli_scoring.scorer import build_scorer  # noqa: E402
from pipeline.config import load_config, resolve_path  # noqa: E402
from pipeline.report import render_markdown  # noqa: E402
from pipeline.run import (accounts_to_claims, parse_accounts_text, run_pipeline,  # noqa: E402
                          small_input_overrides)

EXAMPLE = """Witness A: The blue car ran the red light and hit the cyclist. The driver stayed at the scene. It was raining heavily.

Witness B: The light was green when the blue car went through. The cyclist came out of a side street. The driver did not stay at the scene, he drove off.

Witness C: A blue car hit a cyclist at the junction. The driver got out and called an ambulance. The road was dry."""

st.set_page_config(page_title="Cross-Account Inconsistency Detector", layout="wide")


@st.cache_resource(show_spinner="Loading models (first run downloads them)...")
def load_models(mock: bool):
    cfg = load_config()
    cfg["models"]["mock"] = mock
    return build_embedder(cfg), build_scorer(cfg)


st.title("Cross-Account Inconsistency Detector")
st.caption("Clusters claims by sub-event, then scores claim pairs inside each cluster with an NLI "
           "model. It shows where accounts disagree. It does not decide who is right.")

cfg = load_config()
with st.sidebar:
    st.header("Settings")
    mock = st.checkbox("Use mock models (no download, low quality)", value=False)
    algorithm = st.selectbox("Clustering algorithm", ALGORITHMS,
                             index=ALGORITHMS.index(cfg["clustering"]["algorithm"]),
                             help="Pasted text works best with agglomerative.")
    threshold = st.slider("Contradiction threshold", 0.5, 0.99,
                          float(cfg["disagreement"]["contradiction_threshold"]), 0.01)
    st.markdown(f"Embeddings: `{cfg['models']['embedding']}`  \nNLI: `{cfg['models']['nli']}`")

source = st.radio("Input", ["Paste accounts", "PHEME event"], horizontal=True)
claims = []
name = "custom"
if source == "Paste accounts":
    raw = st.text_area("One account per paragraph (blank line between accounts). "
                       "Optionally start a paragraph with a name and a colon.", EXAMPLE, height=260)
    claims = accounts_to_claims(parse_accounts_text(raw), event="custom")
    # short inputs: looser clusters so related sentences are actually compared
    cut = st.sidebar.slider("Agglomerative distance threshold", 0.2, 0.9, 0.6, 0.05)
    small_input_overrides(cfg, len(claims), cut)
else:
    try:
        events = list_events(resolve_path(cfg, "pheme_v2"))
    except FileNotFoundError:
        events = []
        st.error("PHEME not found under data/raw/pheme_v2 (see README for the expected layout).")
    if events:
        main = [e for e in cfg["events"]["main"] if e in events]
        name = st.selectbox("Event", main + [e for e in events if e not in main])
        n_max = st.slider("Maximum number of source tweets", 50, 1500, 300, 50)
        rumours_only = st.checkbox("Only threads annotated as rumours", value=True)
        claims = load_event_from_config(cfg, name, rumours_only=rumours_only, max_claims=n_max)

st.write(f"**{len(claims)} claims** from **{len({c.account_id for c in claims})} accounts**")

if st.button("Detect disagreement", type="primary", disabled=len(claims) < 2):
    cfg["models"]["mock"] = mock
    cfg["disagreement"]["contradiction_threshold"] = threshold
    embedder, scorer = load_models(mock)
    with st.spinner("Clustering and scoring claim pairs..."):
        result = run_pipeline(claims, cfg, embedder, scorer, algorithm, name=name, verbose=False)
    report = result.report
    s = report["summary"]
    cols = st.columns(4)
    cols[0].metric("Sub-event clusters", s["n_clusters"])
    cols[1].metric("Flagged clusters", s["n_flagged_clusters"])
    cols[2].metric("Pairs scored", s["n_pairs_scored"])
    cols[3].metric("Pairs if all compared", s["n_pairs_all_pairs"])

    st.subheader("Clusters with internal disagreement")
    flagged = [c for c in report["clusters"] if c["flagged"]]
    if not flagged:
        st.info("No cluster reached the contradiction threshold. Try lowering it.")
    for cluster in flagged:
        title = (f"{', '.join(cluster['label_terms']) or 'cluster ' + str(cluster['cluster_id'])} "
                 f"- {cluster['n_contradicting_pairs']} contradicting pairs, "
                 f"score {cluster['disagreement_score']:.2f}")
        with st.expander(title, expanded=len(flagged) <= 5):
            st.markdown(f"Most central claim: *{cluster['central_claim']}*  \n"
                        f"{cluster['size']} claims from {cluster['n_accounts']} accounts.")
            st.dataframe(pd.DataFrame([
                {"Account A": p["claim_a"]["account_id"], "Claim A": p["claim_a"]["text"],
                 "Account B": p["claim_b"]["account_id"], "Claim B": p["claim_b"]["text"],
                 "P(contradiction)": round(p["p_contradiction"], 3)}
                for p in cluster["contradicting_pairs"]]), hide_index=True, width="stretch")

    st.subheader("All clusters")
    st.dataframe(pd.DataFrame([
        {"Cluster": c["cluster_id"], "Label": ", ".join(c["label_terms"]), "Claims": c["size"],
         "Accounts": c["n_accounts"], "Pairs scored": c["n_pairs_scored"],
         "Contradicting pairs": c["n_contradicting_pairs"],
         "Disagreement score": round(c["disagreement_score"], 3), "Flagged": c["flagged"]}
        for c in report["clusters"]]), hide_index=True, width="stretch")

    by_cluster = {cid: [] for cid in sorted(set(result.labels.tolist()))}
    for claim, label in zip(result.claims, result.labels.tolist()):
        by_cluster[label].append({"Account": claim.account_id, "Claim": claim.text})
    with st.expander("Claims per cluster"):
        for cid, rows in by_cluster.items():
            st.markdown(f"**{'Unclustered' if cid < 0 else f'Cluster {cid}'}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.download_button("Download report (Markdown)", render_markdown(report),
                       file_name=f"{name}_disagreement_report.md")
    st.caption(report["disclaimer"])
