"""Ranking methods: TF-IDF baseline, embeddings (+ vector index), graph score, hybrids."""
import numpy as np
from rdflib import RDF
from rdflib.namespace import SKOS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from kg import HR


def cand_ids_texts(g):
    ids = sorted((str(c).split("#")[1] for c in g.subjects(RDF.type, HR.Candidate)),
                 key=lambda x: int(x[1:]))
    return ids, [str(g.value(HR[i], HR.resumeText)) for i in ids]


def tfidf_scores(job_text, cand_texts):
    v = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X = v.fit_transform([job_text] + cand_texts)
    return cosine_similarity(X[0], X[1:]).ravel()


class VectorIndex:
    """Sentence-transformer embeddings in a FAISS index (numpy fallback)."""
    def __init__(self, texts, model="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.E = self.model.encode(texts, normalize_embeddings=True).astype("float32")
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.E.shape[1]); self.index.add(self.E)
        except ImportError:
            self.index = None

    def scores(self, query):
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        if self.index is not None:
            s, i = self.index.search(q, len(self.E))
            out = np.zeros(len(self.E)); out[i[0]] = s[0]
            return out
        return (self.E @ q[0])


def ancestors(g, skill):
    return set(g.transitive_objects(skill, SKOS.broader)) - {skill}


def graph_score(g, job, cand):
    """Average, over required skills, of the best match the candidate offers:
    exact 1.0 | candidate skill is narrower (e.g. PyTorch for 'deep learning') 0.7 |
    sibling under same parent 0.4 | candidate only knows the broader area 0.3."""
    req = set(g.objects(job, HR.requiresSkill)); have = set(g.objects(cand, HR.hasSkill))
    if not req:
        return 0.0
    total = 0.0
    for r in req:
        r_parents = set(g.objects(r, SKOS.broader)); best = 0.0
        for c in have:
            if c == r: w = 1.0
            elif r in ancestors(g, c): w = 0.7
            elif r_parents & set(g.objects(c, SKOS.broader)): w = 0.4
            elif c in ancestors(g, r): w = 0.3
            else: w = 0.0
            best = max(best, w)
        total += best
    return total / len(req)


def explain(g, job, cand):
    """Human-readable graph facts: how each required skill is (or isn't) covered."""
    facts = []
    for r in sorted(g.objects(job, HR.requiresSkill), key=str):
        rl = g.value(r, SKOS.prefLabel); how = "NOT covered"
        for c in g.objects(cand, HR.hasSkill):
            cl = g.value(c, SKOS.prefLabel)
            if c == r: how = f"exact match ({cl})"; break
            if r in ancestors(g, c): how = f"covered via narrower skill {cl}"
        facts.append(f"{rl}: {how}")
    return facts


def minmax(x):
    x = np.asarray(x, float); rng = x.max() - x.min()
    return (x - x.min()) / rng if rng else np.zeros_like(x)
