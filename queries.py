"""SPARQL queries over the HR knowledge graph. Run: python queries.py"""
from kg import load_graph, HR
from rdflib import URIRef

PREFIX = """PREFIX hr: <http://example.org/hr#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

# 1. Candidates holding EVERY skill the job requires (exact match only)
ALL_REQUIRED = PREFIX + """
SELECT ?name WHERE {
  ?cand a hr:Candidate ; hr:name ?name .
  FILTER NOT EXISTS {
    ?job hr:requiresSkill ?s .
    FILTER NOT EXISTS { ?cand hr:hasSkill ?s }
  }
}"""

# 2. Skills a job requires that a candidate lacks (skill gap)
GAPS = PREFIX + """
SELECT ?label WHERE {
  ?job hr:requiresSkill ?s . ?s skos:prefLabel ?label .
  FILTER NOT EXISTS { ?cand hr:hasSkill ?s }
}"""

# 3. Candidates covering required skills through the hierarchy:
#    candidate skill ?c is the required skill or a narrower kind of it
#    (e.g. job needs 'deep learning', candidate knows PyTorch)
COVERAGE = PREFIX + """
SELECT ?name (COUNT(DISTINCT ?s) AS ?covered) WHERE {
  ?job hr:requiresSkill ?s .
  ?cand a hr:Candidate ; hr:name ?name ; hr:hasSkill ?c .
  ?c skos:broader* ?s .
} GROUP BY ?name ORDER BY DESC(?covered)"""

# 4. Skills adjacent to a given skill (siblings under the same parent)
SIBLINGS = PREFIX + """
SELECT ?label WHERE {
  ?skill skos:broader ?p . ?other skos:broader ?p ; skos:prefLabel ?label .
  FILTER (?other != ?skill)
}"""


def show(title, rows):
    print(f"\n== {title}")
    rows = list(rows)
    if not rows:
        print("   (none)")
    for r in rows:
        print("  ", " | ".join(str(x) for x in r))


if __name__ == "__main__":
    g = load_graph()
    job = HR["J1"]; cand = HR["C7"]
    show("Candidates with ALL skills required by J4",
         g.query(ALL_REQUIRED, initBindings={"job": HR["J4"]}))
    show("Skills J1 requires that C7 (Gautam) lacks",
         g.query(GAPS, initBindings={"job": job, "cand": cand}))
    show("J2 (ML Engineer): candidates by required skills covered via hierarchy",
         g.query(COVERAGE, initBindings={"job": HR["J2"]}))
    show("Skills adjacent to PyTorch",
         g.query(SIBLINGS, initBindings={"skill": HR["PyTorch"]}))
