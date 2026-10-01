"""Grounded explanation: retrieve resume text + graph facts, ask Gemini why a candidate fits.
Usage: python rag.py J1      (set GEMINI_API_KEY to call Gemini; otherwise prints the prompt)"""
import os, sys
import numpy as np
from kg import load_graph, HR
from retrieval import cand_ids_texts, tfidf_scores, graph_score, explain, minmax


def build_prompt(g, job, cid):
    cand = HR[cid]
    facts = "\n".join(f"- {f}" for f in explain(g, job, cand))
    return (f"Job: {g.value(job, HR.name)}\nJob description: {g.value(job, HR.description)}\n\n"
            f"Candidate resume: {g.value(cand, HR.resumeText)}\n\n"
            f"Knowledge-graph facts (use ONLY these and the resume):\n{facts}\n\n"
            "In 3 sentences, explain how well this candidate fits the job. "
            "Cite the graph facts you rely on and state any gaps. Do not invent experience.")


def main(jid="J1"):
    g = load_graph(); job = HR[jid]
    ids, texts = cand_ids_texts(g)
    tf = tfidf_scores(str(g.value(job, HR.description)), texts)
    gs = np.array([graph_score(g, job, HR[c]) for c in ids])
    best = ids[int(np.argmax(0.5 * minmax(tf) + 0.5 * minmax(gs)))]
    prompt = build_prompt(g, job, best)
    print(f"Top candidate for {jid}: {best}\n")
    if os.getenv("GEMINI_API_KEY"):
        from google import genai
        client = genai.Client()
        print(client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text)
    else:
        print("[GEMINI_API_KEY not set - showing the grounded prompt instead]\n")
        print(prompt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "J1")
