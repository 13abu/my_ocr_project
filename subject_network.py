"""
IRP correspondence network -- SUBJECT layer only
--------------------------------------------------
This is Instrument 3 proper: which bodies generate relational responses,
which produce only administrative endpoints. Every subject gets a node,
tiered by ontological status, bordered by response_status. Authors appear
only as small unlabeled anchor points for the 'concerns' edge -- this
script does NOT draw officer-to-officer correspondence at all. That's a
different question, answered by officer_network.py instead. Splitting
these was a deliberate fix: the combined version buried the actual
finding (0/21 Enslaved/manumitted subjects ever reply) under 55 officer
nodes and their routine institutional routing, which is a different
result entirely (see officer_network.py).

Classification (TIER / RESPONSE_STATUS) was done by direct reading of
each case's dated author->recipient chain, not inferred automatically.

Install if needed:
    pip install networkx matplotlib --break-system-packages
"""
import csv
import json
import math
import re
from pathlib import Path
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

INPUT_TSV = Path("./spreadsheet.tsv")
OUTPUT_DIR = Path("./network")
OUTPUT_DIR.mkdir(exist_ok=True)
PNG_PATH = OUTPUT_DIR / "subject_network.png"

# --- tier scheme (confirmed) ---
TIER = {
    "Queen Victoria": "1. Sovereign",
    "Shaikh Isa": "2. Gulf ruler",
    "Shaikh Humeid bin Abdullah": "2. Gulf ruler",
    "Shaikh Jabir": "2. Gulf ruler",
    "Shaikh Salim": "2. Gulf ruler",
    "John Gordon Lorimer": "2. Gulf ruler (reverse case)",
    "Father Gabriel Fleming": "3. European/British subject",
    "Mohammed bin Haji": "4a. BAPCO worker",
    "Nassir bin Awadh": "4a. BAPCO worker",
    "A. Rahim Ali Baz": "4a. BAPCO worker",
    "Salim bin Saeed": "4a. BAPCO worker",
    "Amer bin Saad": "4a. BAPCO worker",
    "Hassan bin Abdulla": "4a. BAPCO worker",
    "Abdullah bin Nassir al Awadh": "4a. BAPCO worker",
    "Abbas bin Habib": "4a. BAPCO worker",
    "Ali bin Ahmed": "4a. BAPCO worker",
    "Alwiyah bint Shabib": "4b. Ordinary subject abroad",
    "Mohammed Saleh al Bahraini": "4b. Ordinary subject abroad",
    "Sadiga bint Haj Muhammad": "4b. Ordinary subject abroad",
    "S'adiyah bint Sheber": "4b. Ordinary subject abroad",
    "Haj Mahdi": "4b. Ordinary subject abroad",
    "Almas of Suwahil": "5. Enslaved/manumitted",
    "Maryam bint Mabrook": "5. Enslaved/manumitted",
    "Ghazul bint Suwahil": "5. Enslaved/manumitted",
    "Salim bin Bial": "5. Enslaved/manumitted",
    "Faraj bin Dadkhuda": "5. Enslaved/manumitted",
    "Alya bint Firoz": "5. Enslaved/manumitted",
    "Mubarak bin Bukhit": "5. Enslaved/manumitted",
    "Firoz bin Anbar": "5. Enslaved/manumitted",
    "Zahra bint Ghuloom": "5. Enslaved/manumitted",
    "Nasir bin Bilal": "5. Enslaved/manumitted",
    "Salim bin Ali": "5. Enslaved/manumitted",
    "Anbar": "5. Enslaved/manumitted",
    "Sa'id bin Zayed": "5. Enslaved/manumitted",
    "Mubarak bin Yaruh": "5. Enslaved/manumitted",
    "Farhan bin Salmin": "5. Enslaved/manumitted",
    "Marzuq bin Sanqur": "5. Enslaved/manumitted",
    "Khazur bint Abdullah": "5. Enslaved/manumitted",
    "Almas bin Kamin": "5. Enslaved/manumitted",
    "Thani bin Sa'ad": "5. Enslaved/manumitted",
    "Salihah bint Abdur Rahman": "5. Enslaved/manumitted",
    "Nafa'ah bint Sorur": "5. Enslaved/manumitted",  # FLAGGED: not yet verified against the folio image
    "Dr. Steele": "comparator: intra-colonial dispute",
    "Gray, Mackenzie & Co.": "comparator: intra-colonial dispute",
    "estate of a manumitted person": "liminal: 4/5 boundary",
}

# response_status principle: "response" requires the SUBJECT to be the one
# writing back, in their own name -- not a third party (a Ruler, a master)
# negotiating custody/ransom/status about them. Isa->Hamad and Fleming->Pelly
# qualify because the subject's own successor/friend replies as himself.
# Ghazul/Anbar/Nasir do NOT qualify even though a Ruler writes back in each
# case: the Ruler is negotiating custody terms about the subject, and the
# subject never enters the correspondence under their own name.
RESPONSE_STATUS = {
    "Queen Victoria": "endpoint",
    "Shaikh Isa": "response",
    "Shaikh Humeid bin Abdullah": "endpoint",
    "Shaikh Jabir": "endpoint",
    "Shaikh Salim": "endpoint",
    "John Gordon Lorimer": "narrated",
    "Father Gabriel Fleming": "response",
    "Mohammed bin Haji": "endpoint",
    "Nassir bin Awadh": "endpoint",
    "A. Rahim Ali Baz": "endpoint",
    "Salim bin Saeed": "endpoint",
    "Amer bin Saad": "endpoint",
    "Hassan bin Abdulla": "endpoint",
    "Abdullah bin Nassir al Awadh": "endpoint",
    "Abbas bin Habib": "endpoint",
    "Ali bin Ahmed": "endpoint",
    "Alwiyah bint Shabib": "endpoint",
    "Mohammed Saleh al Bahraini": "endpoint",
    "Sadiga bint Haj Muhammad": "endpoint",
    "S'adiyah bint Sheber": "endpoint",
    "Haj Mahdi": "endpoint",
    "Almas of Suwahil": "endpoint",
    "Maryam bint Mabrook": "endpoint",
    "Ghazul bint Suwahil": "endpoint",
    "Salim bin Bial": "endpoint",
    "Faraj bin Dadkhuda": "endpoint",
    "Alya bint Firoz": "endpoint",
    "Mubarak bin Bukhit": "endpoint",
    "Firoz bin Anbar": "endpoint",
    "Zahra bint Ghuloom": "endpoint",
    "Nasir bin Bilal": "endpoint",
    "Salim bin Ali": "endpoint",
    "Anbar": "endpoint",
    "Sa'id bin Zayed": "endpoint",
    "Mubarak bin Yaruh": "endpoint",
    "Farhan bin Salmin": "endpoint",
    "Marzuq bin Sanqur": "endpoint",
    "Khazur bint Abdullah": "endpoint",
    "Almas bin Kamin": "endpoint",
    "Thani bin Sa'ad": "endpoint",
    "Salihah bint Abdur Rahman": "endpoint",
    "Nafa'ah bint Sorur": "endpoint",
    "Dr. Steele": "response",
    "Gray, Mackenzie & Co.": "response",
    "estate of a manumitted person": "endpoint",
}


def norm(name):
    return name.strip()


def build_graph(rows):
    """Subject nodes + AUTHOR nodes only. No recipient nodes, no
    correspondence edges -- that's officer_network.py's job."""
    G = nx.MultiDiGraph()
    for r in rows:
        subj = r['Subject'].split(';')[0].strip()
        author = norm(r['Author/Officer'])
        if not subj or subj not in TIER:
            continue
        G.add_node(subj, kind='subject', tier=TIER[subj], status=RESPONSE_STATUS[subj])
        G.add_node(author, kind='author')
        G.add_edge(author, subj, kind='concerns', doc=r['Folio'], date=r['Date'])
    return G


def print_report(G):
    subj_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'subject']
    author_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'author']

    print(f"Subject nodes: {len(subj_nodes)}")
    print(f"Author nodes: {len(author_nodes)}")
    print(f"Total 'concerns' edges: {G.number_of_edges()}")
    print()

    by_tier = defaultdict(lambda: defaultdict(int))
    for n in subj_nodes:
        d = G.nodes[n]
        by_tier[d['tier']][d['status']] += 1

    print(f"{'Tier':<32}{'response':<10}{'narrated':<10}{'endpoint':<10}{'total':<6}")
    for tier in sorted(by_tier.keys()):
        counts = by_tier[tier]
        total = sum(counts.values())
        print(f"{tier:<32}{counts.get('response',0):<10}{counts.get('narrated',0):<10}{counts.get('endpoint',0):<10}{total:<6}")


def export_data(G):
    data = nx.node_link_data(G, edges="edges")
    with open(OUTPUT_DIR / 'subject_network.json', 'w') as f:
        json.dump(data, f, indent=2)
    nx.write_graphml(G, OUTPUT_DIR / 'subject_network.graphml')
    print(f"\nExported subject_network.json and subject_network.graphml to {OUTPUT_DIR}/")


TIER_COLORS = {
    "1. Sovereign": "#8B0000",
    "2. Gulf ruler": "#B8860B",
    "2. Gulf ruler (reverse case)": "#DAA520",
    "3. European/British subject": "#4682B4",
    "4a. BAPCO worker": "#2E8B57",
    "4b. Ordinary subject abroad": "#6B8E23",
    "5. Enslaved/manumitted": "#8B008B",
    "comparator: intra-colonial dispute": "#708090",
    "liminal: 4/5 boundary": "#A9A9A9",
}

STATUS_BORDER = {
    "response": "#00C853",
    "narrated": "#FF9800",
    "endpoint": "#D32F2F",
}


def compute_layout(G):
    """Plain force-directed layout, no ring constraint. Tier is a
    categorical variable, not a continuous one -- color already encodes
    it unambiguously, so forcing nodes onto rings was adding a second,
    more fragile encoding of the same thing, and fighting the solver's
    natural clustering in the process. Letting the graph settle freely
    means subjects who share authors end up visually close together on
    their own, which is arguably more informative than a fixed ring."""
    UG = G.to_undirected()
    return nx.spring_layout(UG, k=0.9, seed=42, iterations=150)


def short_label(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def plot_network(G, output_path: Path):
    subj_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'subject']
    author_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'author']

    fig, ax = plt.subplots(figsize=(20, 16))

    pos = compute_layout(G)

    # Author nodes: small, pale, mostly unlabeled anchor points -- this is
    # deliberately NOT a competing network. They exist only so the
    # 'concerns' edge has somewhere to start from.
    author_degrees = dict(G.degree(author_nodes))
    nx.draw_networkx_nodes(
        G, pos, nodelist=author_nodes, node_color="#D5D5D5",
        node_size=[80 + author_degrees.get(n, 0) * 15 for n in author_nodes],
        alpha=0.5, ax=ax, linewidths=0,
    )

    subj_colors = [TIER_COLORS.get(G.nodes[n]['tier'], "#000000") for n in subj_nodes]
    subj_borders = [STATUS_BORDER.get(G.nodes[n]['status'], "#000000") for n in subj_nodes]
    nx.draw_networkx_nodes(
        G, pos, nodelist=subj_nodes, node_color=subj_colors,
        node_size=800, alpha=0.95, ax=ax,
        edgecolors=subj_borders, linewidths=3,
    )

    nx.draw_networkx_edges(
        G, pos, edgelist=list(G.edges()), edge_color="#6A0DAD",
        alpha=0.55, arrows=True, arrowsize=6, width=1.1, ax=ax,
        connectionstyle="arc3,rad=0.05",
    )

    # Labels: subjects always labeled. Authors labeled only if they cover
    # 3+ subjects, since a name on every faint grey dot would recreate the
    # exact clutter this split was meant to fix.
    subj_labels = {n: short_label(n) for n in subj_nodes}
    nx.draw_networkx_labels(G, pos, labels=subj_labels, font_size=9,
                             font_color="#1A1A1A", ax=ax)
    author_labels = {n: short_label(n) for n in author_nodes if author_degrees.get(n, 0) >= 3}
    nx.draw_networkx_labels(G, pos, labels=author_labels, font_size=7,
                             font_color="#555555", ax=ax)

    tier_handles = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
    status_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=c, markeredgewidth=3, markersize=12, label=s)
        for s, c in STATUS_BORDER.items()
    ]
    author_handle = [mpatches.Patch(color="#D5D5D5", label="author (anchor only, not a network)")]

    leg1 = ax.legend(handles=tier_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      title="Subject tier (fill color)", fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=status_handles + author_handle, loc="upper left",
              bbox_to_anchor=(1.01, 0.5), title="Response status (border) / node kind",
              fontsize=8, title_fontsize=9)

    ax.set_aspect("equal")
    ax.relim()
    ax.autoscale_view()
    ax.margins(0.08)
    ax.set_title("IOR/R/15 Subject Network\n(who gets written about, and whether they ever reply — color = tier, layout = actual correspondence structure)",
                  fontsize=14, fontweight="bold")
    ax.axis("off")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved subject network plot to {output_path}")
    plt.close(fig)


def main():
    if not INPUT_TSV.exists():
        raise SystemExit(f"Could not find {INPUT_TSV}.")
    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    G = build_graph(rows)
    print_report(G)
    export_data(G)
    plot_network(G, PNG_PATH)


if __name__ == "__main__":
    main()