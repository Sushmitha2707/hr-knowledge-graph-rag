"""Compare ranking methods on labelled job/candidate pairs.
Metrics: Precision@3 (label>=1 counts as relevant) and MRR. Run: python evaluate.py"""
import json
import numpy as np
from kg import load_graph, HR, ROOT
from retrieval import (cand_ids_texts, tfidf_scores, graph_score, minmax, VectorIndex)

K = 3


def metrics(ranked, labels):
    rel = [labels.get(c, 0) >= 1 for c in ranked]
    p_at_k = sum(rel[:K]) / K
    mrr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
    return p_at_k, mrr


def main():
    g = load_graph()
    ids, texts = cand_ids_texts(g)
    labels = json.load(open(ROOT / "data" / "labels.json"))
    try:
        vindex = VectorIndex(texts)
    except Exception as e:  # not installed / model not downloadable
        print(f"[embeddings skipped: {type(e).__name__}]"); vindex = None

    results = {}
    for jid, jlabels in labels.items():
        job = HR[jid]; jtext = str(g.value(job, HR.description))
        tf = tfidf_scores(jtext, texts)
        gs = np.array([graph_score(g, job, HR[c]) for c in ids])
        methods = {"TF-IDF": tf, "Graph only": gs,
                   "TF-IDF + Graph": 0.5 * minmax(tf) + 0.5 * minmax(gs)}
        if vindex is not None:
            em = vindex.scores(jtext)
            methods["Embeddings"] = em
            methods["Embeddings + Graph"] = 0.5 * minmax(em) + 0.5 * minmax(gs)
        for name, s in methods.items():
            ranked = [ids[i] for i in np.argsort(-s)]
            results.setdefault(name, []).append(metrics(ranked, jlabels))

    print(f"\n{'Method':<22}{'P@'+str(K):>8}{'MRR':>8}   (n={len(labels)} jobs)")
    for name, rows in results.items():
        p, m = np.mean(rows, axis=0)
        print(f"{name:<22}{p:>8.2f}{m:>8.2f}")


if __name__ == "__main__":
    main()
