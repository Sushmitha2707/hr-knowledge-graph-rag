# HR Knowledge Graph + RAG

Ground candidate-to-job matching in an **ontology and knowledge graph** instead of keyword overlap alone.
Extends my TF-IDF resume screener (`AI_Resume_Screening_Bot`) with a semantic layer.

## Why
Keyword/TF-IDF matching misses that a candidate who knows *PyTorch* satisfies a job asking for *deep learning*.
A skills taxonomy (SKOS) plus RDF/OWL schema encodes that knowledge explicitly, SPARQL queries it, and
embeddings in a vector index add semantic retrieval. An LLM then explains each match using retrieved graph facts.

## Architecture
```
resumes + JDs ──► skill extraction (ontology labels) ──► RDF graph (rdflib)
                                                           │
   SPARQL queries ◄────────────────────────────────────────┤
   TF-IDF / embeddings (FAISS) ──► hybrid ranking ◄── graph score (skos:broader hierarchy)
                                          │
                              RAG explanation (Gemini, grounded in graph facts)
```

## Files
| File | Purpose |
|---|---|
| `ontology/hr_ontology.ttl` | OWL schema (Candidate, Job, Employer, Skill, properties) + SKOS skill taxonomy |
| `kg.py` | Builds the graph from `data/` and extracts skills with the ontology's own labels |
| `queries.py` | SPARQL: all-required-skills, skill gaps, hierarchy coverage (`skos:broader*`), adjacent skills |
| `retrieval.py` | TF-IDF baseline, sentence-transformer embeddings + FAISS index, graph score, hybrid |
| `evaluate.py` | Precision@3 and MRR for each method on labelled pairs |
| `rag.py` | Retrieves graph facts + resume text and asks Gemini for a grounded explanation |

## Run
```bash
pip install -r requirements.txt
python kg.py          # build graph
python queries.py     # SPARQL demos
python evaluate.py    # compare TF-IDF vs embeddings vs graph vs hybrids
python rag.py J1      # grounded explanation (set GEMINI_API_KEY to call Gemini)
```

## IMPORTANT: replace the sample data before quoting any number
`data/` ships with 12 made-up candidates, 4 jobs and labels so the code runs end to end.
They are a smoke test only, so **do not report results from them**. For a real result:
1. Collect 25-30 resume/JD pairs (public sample resumes, anonymised classmates' resumes with permission, real JDs).
2. Label each pair 0 (not relevant) / 1 (partial) / 2 (strong) in `data/labels.json`, ideally with a second person labelling independently.
3. Extend the skills in `ontology/hr_ontology.ttl` to cover your data.
4. Run `python evaluate.py` and report P@3/MRR for the real set, including the sample size and the fact that labels are manual.

## Limitations (say these in interviews)
- Skill extraction is dictionary matching; an NER or LLM extractor would generalise better.
- Hybrid weights (0.5/0.5) and hierarchy weights (1.0/0.7/0.4/0.3) are hand-set, not tuned.
- Small evaluation set; results show a trend, not statistical significance.
