"""Build the HR knowledge graph: ontology + candidates + jobs -> RDF graph."""
import json, re
from pathlib import Path
from rdflib import Graph, Namespace, RDF, Literal
from rdflib.namespace import SKOS

HR = Namespace("http://example.org/hr#")
ROOT = Path(__file__).parent


def skill_labels(g):
    """{skill URI: [prefLabel + altLabels]}"""
    out = {}
    for s in g.subjects(RDF.type, HR.Skill):
        out[s] = [str(o) for p in (SKOS.prefLabel, SKOS.altLabel) for o in g.objects(s, p)]
    return out


def extract_skills(g, text):
    """Dictionary-based skill extraction using the ontology's own labels.
    Short labels (<=4 chars, e.g. SQL, RAG, OWL) are matched case-sensitively
    to avoid false hits."""
    found = set()
    for skill, labels in skill_labels(g).items():
        for label in labels:
            flags = 0 if len(label) <= 4 else re.IGNORECASE
            pattern = r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])"
            if re.search(pattern, text, flags):
                found.add(skill)
                break
    return found


def build_graph(save=True):
    g = Graph()
    g.parse(ROOT / "ontology" / "hr_ontology.ttl")
    g.bind("hr", HR); g.bind("skos", SKOS)

    for c in json.load(open(ROOT / "data" / "candidates.json")):
        u = HR[c["id"]]
        g.add((u, RDF.type, HR.Candidate))
        g.add((u, HR.name, Literal(c["name"])))
        g.add((u, HR.resumeText, Literal(c["text"])))
        for s in extract_skills(g, c["text"]):
            g.add((u, HR.hasSkill, s))

    for j in json.load(open(ROOT / "data" / "jobs.json")):
        u = HR[j["id"]]
        emp = HR[re.sub(r"\W+", "", j["employer"])]
        g.add((u, RDF.type, HR.Job))
        g.add((u, HR.name, Literal(j["title"])))
        g.add((u, HR.description, Literal(j["text"])))
        g.add((emp, RDF.type, HR.Employer)); g.add((emp, HR.name, Literal(j["employer"])))
        g.add((u, HR.postedBy, emp))
        for s in extract_skills(g, j["text"]):
            g.add((u, HR.requiresSkill, s))

    if save:
        g.serialize(ROOT / "ontology" / "hr_graph.ttl", format="turtle")
    return g


def load_graph():
    path = ROOT / "ontology" / "hr_graph.ttl"
    if not path.exists():
        return build_graph()
    g = Graph(); g.parse(path); g.bind("hr", HR); g.bind("skos", SKOS)
    return g


if __name__ == "__main__":
    g = build_graph()
    print(f"Graph built: {len(g)} triples")
    print("Saved to ontology/hr_graph.ttl")
