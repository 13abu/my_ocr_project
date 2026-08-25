"""
IRP correspondence network -- OFFICER layer only
---------------------------------------------------
Institutional correspondence structure: who reports to whom. This is a
DIFFERENT question from subject_network.py and produces a different
result -- the native Residency Agents (Abd al-Latif, Razuqi) outranking
every British officer in raw connection count, a finding about
administrative hierarchy, not about grief or ontology. No subject
nodes, no tiers, no response-status here; this is the officer-layer
finding on its own, not buried under 43 subject nodes.

Install if needed:
    pip install networkx matplotlib --break-system-packages
"""
import csv
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import re

INPUT_TSV = Path("./spreadsheet.tsv")
OUTPUT_DIR = Path("./network")
OUTPUT_DIR.mkdir(exist_ok=True)
PNG_PATH = OUTPUT_DIR / "officer_network.png"


def norm(name):
    return name.strip()


def short_label(name: str) -> str:
    """Strip a trailing parenthetical role/institution for display only."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def build_graph(rows):
    """Officer nodes only (authors AND recipients), 'correspondence'
    edges only. No subject layer at all."""
    G = nx.MultiDiGraph()
    for r in rows:
        author = norm(r.get('Author/Officer', ''))
        recipient = norm(r.get('Recipient', ''))
        if not author or not recipient or author == recipient:
            continue
        G.add_node(author)
        G.add_node(recipient)
        G.add_edge(author, recipient, doc=r.get('Folio', ''), date=r.get('Date', ''))
    return G


def print_report(G):
    # Undirected degree for ranking -- who's most connected overall,
    # regardless of whether they're usually the author or the recipient.
    UG = G.to_undirected()
    degrees = sorted(UG.degree(), key=lambda kv: -kv[1])

    print(f"Officer/institution nodes: {G.number_of_nodes()}")
    print(f"Total correspondence edges: {G.number_of_edges()}")
    print()
    print("=" * 60)
    print("OFFICER RANKING BY CONNECTIONS (undirected degree)")
    print("=" * 60)
    for i, (name, deg) in enumerate(degrees[:20], start=1):
        print(f"  #{i:<3} {deg:<4} {short_label(name)}")
    return degrees


def export_data(G):
    nx.write_graphml(G, OUTPUT_DIR / 'officer_network.graphml')
    print(f"\nExported officer_network.graphml to {OUTPUT_DIR}/")


def plot_network(G, degrees, output_path: Path):
    UG = G.to_undirected()
    fig, ax = plt.subplots(figsize=(18, 14))

    pos = nx.spring_layout(UG, k=0.7, seed=42, iterations=100)

    degree_map = dict(degrees)
    node_sizes = [200 + degree_map.get(n, 0) * 120 for n in UG.nodes()]

    # Highlight the two native Residency Agents at the center of the
    # established finding -- everyone else stays a uniform grey so the
    # contrast is visually obvious, not just legible from the ranking list.
    HIGHLIGHT = {"Abd al-Latif bin Abd al-Rahman", "Abd al-Razzaq Razuqi (Residency Agent, Sharjah)"}
    node_colors = ["#B8860B" if n in HIGHLIGHT else "#6A8CAF" for n in UG.nodes()]

    nx.draw_networkx_nodes(UG, pos, node_size=node_sizes, node_color=node_colors,
                            alpha=0.85, ax=ax, linewidths=0)
    nx.draw_networkx_edges(UG, pos, edge_color="#B0B0B0", alpha=0.5, width=0.8, ax=ax)

    # Label everyone with degree >= 3 -- below that, labels just add noise
    # at this node count without adding information.
    labels = {n: short_label(n) for n in UG.nodes() if degree_map.get(n, 0) >= 3}
    nx.draw_networkx_labels(UG, pos, labels=labels, font_size=8, font_color="#1A1A1A", ax=ax)

    ax.set_title("IOR/R/15 Officer Correspondence Network\n"
                  "(node size = connections; gold = native Residency Agents)",
                  fontsize=14, fontweight="bold")
    ax.axis("off")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved officer network plot to {output_path}")
    plt.close(fig)


def main():
    if not INPUT_TSV.exists():
        raise SystemExit(f"Could not find {INPUT_TSV}.")
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    G = build_graph(rows)
    degrees = print_report(G)
    export_data(G)
    plot_network(G, degrees, PNG_PATH)


if __name__ == "__main__":
    main()