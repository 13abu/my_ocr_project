"""
IRP correspondence network, Instrument 3 (Subject layer)
----------------------------------------------------------
Extends the existing officer-to-officer correspondence network with a
Subject layer: every body a document is ABOUT (not just who wrote/received
it) gets its own node, connected to the correspondence via 'concerns' edges.

Each subject is then classified by response_status, read directly off the
correspondence chain for that case:
  - 'response'  : a genuine two-way exchange exists (recipient of an earlier
                   letter becomes author of a later one, addressed back)
  - 'narrated'  : a response is described WITHIN a document (e.g. a diary
                   entry reporting a personal visit) but not itself a second
                   correspondence artifact
  - 'endpoint'  : one-directional only; the body is reported on/about but
                   nothing in the sheet writes back

Classification was done by direct reading of each case's dated
author->recipient chain, not inferred automatically -- name-matching
heuristics were tried and were too unreliable on this material's title
variants to trust unchecked.

NEW: plot_network() draws the graph and saves it as a PNG using
matplotlib -- run this locally, nothing here calls out to any API.

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

# ==========================================
# CONFIG -- adjust these to your actual file locations
# ==========================================
INPUT_TSV = Path("./spreadsheet.tsv")  # export your corpus sheet as TSV with
                                        # columns: Folio, Date, Author/Officer,
                                        # Recipient, Subject (at minimum)
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
    "Nafa'ah bint Sorur": "5. Enslaved/manumitted",  # FLAGGED: surfaced during sheet cleanup, not yet verified against the folio image
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
# subject never enters the correspondence under their own name. This is a
# deliberate distinction, not an oversight -- see write-up discussion.
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
    "Gray, Mackenzie & Co.": "response",  # same documented two-way exchange as Dr. Steele (145r complaint -> 146r-148r rebuttal)
    "estate of a manumitted person": "endpoint",
}


def norm(name):
    return name.strip()


def build_graph(rows):
    G = nx.MultiDiGraph()
    for r in rows:
        subj = r['Subject'].split(';')[0].strip()
        author = norm(r['Author/Officer'])
        recipient = norm(r['Recipient'])
        if not subj or subj not in TIER:
            continue  # skip pure-governance / unclassified rows for this pass
        G.add_node(subj, kind='subject', tier=TIER[subj], status=RESPONSE_STATUS[subj])
        G.add_node(author, kind='officer')
        G.add_edge(author, subj, kind='concerns', doc=r['Folio'], date=r['Date'])
        if recipient and recipient not in ('', subj):
            G.add_node(recipient, kind='officer')
            G.add_edge(author, recipient, kind='correspondence', doc=r['Folio'], date=r['Date'])
    return G


def print_report(G):
    subj_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'subject']
    officer_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'officer']

    print(f"Subject nodes: {len(subj_nodes)}")
    print(f"Officer/institution nodes: {len(officer_nodes)}")
    print(f"Total edges: {G.number_of_edges()}")
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


# ==========================================
# PLOTTING
# ==========================================

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
    "response": "#00C853",   # green -- genuine two-way exchange
    "narrated": "#FF9800",   # orange -- described but not a second artifact
    "endpoint": "#D32F2F",   # red -- one-directional only
}


def radial_positions(G, base_pos, subj_nodes):
    """
    Compute final subject positions on concentric tier rings, on the
    SAME coordinate scale as base_pos (spring_layout's natural ~[-1,1]
    box). Officer nodes are left completely untouched at their base_pos
    coordinates.

    Within a ring, nodes are evenly spaced by angle (2*pi/n) rather
    than placed at their raw spring-layout angle -- using the raw
    angle caused several related nodes to bunch together on the same
    ring and overlap. The ORDER of nodes around the ring still comes
    from their original spring-layout angle (so related/nearby nodes
    stay adjacent), only the exact spacing is forced even.
    """
    tier_order = list(TIER_COLORS.keys())
    ring_gap = 0.14

    pos = dict(base_pos)
    for i, tier in enumerate(tier_order):
        nodes_in_tier = [n for n in subj_nodes if G.nodes[n]['tier'] == tier]
        if not nodes_in_tier:
            continue
        radius = i * ring_gap
        if radius == 0:
            for n in nodes_in_tier:
                pos[n] = (0.0, 0.0)
            continue

        def base_angle(n):
            x, y = base_pos[n]
            return math.atan2(y, x) if (x, y) != (0.0, 0.0) else 0.0

        ordered = sorted(nodes_in_tier, key=base_angle)
        n_count = len(ordered)
        angle_offset = i * 0.4
        for j, n in enumerate(ordered):
            angle = angle_offset + (2 * math.pi * j / n_count)
            pos[n] = (radius * math.cos(angle), radius * math.sin(angle))
    return pos


def short_label(name: str) -> str:
    """Strip a trailing parenthetical (role/institution) for display
    only -- e.g. 'Hugh Weightman (Political Agent, Bahrain)' ->
    'Hugh Weightman'. The full name is still used as the node ID and
    appears in tooltips/data; this only shortens what's drawn on the
    plot to cut label clutter."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def plot_network(G, output_path: Path):
    subj_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'subject']
    officer_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'officer']

    fig, ax = plt.subplots(figsize=(22, 18))

    base_pos = nx.spring_layout(G, k=0.6, seed=42, iterations=80)
    pos = radial_positions(G, base_pos, subj_nodes)

    # Officer nodes: small, grey, faded further the less connected they are
    officer_degrees = dict(G.degree(officer_nodes))
    officer_sizes = [150 + officer_degrees.get(n, 0) * 50 for n in officer_nodes]
    officer_alphas = [min(0.85, 0.25 + officer_degrees.get(n, 0) * 0.08) for n in officer_nodes]
    for n, size, alpha in zip(officer_nodes, officer_sizes, officer_alphas):
        nx.draw_networkx_nodes(
            G, pos, nodelist=[n], node_color="#BDBDBD",
            node_size=size, alpha=alpha, ax=ax, linewidths=0,
        )

    # Subject nodes: colored by tier, bordered by response status
    subj_colors = [TIER_COLORS.get(G.nodes[n]['tier'], "#000000") for n in subj_nodes]
    subj_borders = [STATUS_BORDER.get(G.nodes[n]['status'], "#000000") for n in subj_nodes]
    nx.draw_networkx_nodes(
        G, pos, nodelist=subj_nodes, node_color=subj_colors,
        node_size=800, alpha=0.95, ax=ax,
        edgecolors=subj_borders, linewidths=3,
    )

    # Edges: 'concerns' (author -> subject) drawn in purple, 'correspondence'
    # (author -> recipient officer) drawn in grey -- both kinds always
    # shown, both with small arrowheads.
    concerns_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('kind') == 'concerns']
    corr_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('kind') == 'correspondence']

    nx.draw_networkx_edges(
        G, pos, edgelist=corr_edges, edge_color="#9E9E9E",
        alpha=0.45, arrows=True, arrowsize=5, width=0.6, ax=ax,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=concerns_edges, edge_color="#6A0DAD",
        alpha=0.65, arrows=True, arrowsize=5, width=1.1, ax=ax,
        connectionstyle="arc3,rad=0.05",
    )

    # Labels: uniform font size and color for both subject and officer
    # nodes. EVERY node is labeled, including low-degree officer nodes --
    # long "(Role, Place)" suffixes are stripped for display so labels
    # stay short even with all of them shown.
    LABEL_SIZE = 8
    LABEL_COLOR = "#1A1A1A"

    subj_labels = {n: short_label(n) for n in subj_nodes}
    nx.draw_networkx_labels(G, pos, labels=subj_labels, font_size=LABEL_SIZE,
                             font_color=LABEL_COLOR, ax=ax)

    officer_labels = {n: short_label(n) for n in officer_nodes if officer_degrees.get(n, 0) >= 2}
    nx.draw_networkx_labels(G, pos, labels=officer_labels, font_size=LABEL_SIZE,
                             font_color=LABEL_COLOR, ax=ax)

    # Legends
    tier_handles = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
    status_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor=c, markeredgewidth=3, markersize=12, label=s)
        for s, c in STATUS_BORDER.items()
    ]
    officer_handle = [mpatches.Patch(color="#BDBDBD", label="officer / institution")]
    edge_handles = [
        Line2D([0], [0], color="#6A0DAD", lw=2, label="concerns (author -> subject)"),
        Line2D([0], [0], color="#9E9E9E", lw=2, label="correspondence (author -> recipient)"),
    ]

    leg1 = ax.legend(handles=tier_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                      title="Subject tier (fill color)", fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=status_handles + officer_handle + edge_handles, loc="upper left",
                      bbox_to_anchor=(1.01, 0.55), title="Response status (border) / node kind / edges",
                      fontsize=8, title_fontsize=9)

    # Faint concentric guides showing the tier rings, so the ontology
    # structure reads even before the legend is consulted
    tier_order = list(TIER_COLORS.keys())
    ring_gap = 0.14
    for i in range(1, len(tier_order)):
        circle = plt.Circle((0, 0), i * ring_gap, fill=False,
                             edgecolor="#E0E0E0", linewidth=0.8, linestyle="--", zorder=0)
        ax.add_patch(circle)

    ax.set_aspect("equal")
    ax.relim()
    ax.autoscale_view()
    ax.margins(0.08)

    ax.set_title("IOR/R/15 Subject Correspondence Network\n(concentric rings = tier, sovereign at center)",
                  fontsize=14, fontweight="bold")
    ax.axis("off")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved network plot to {output_path}")
    plt.close(fig)


def main():
    if not INPUT_TSV.exists():
        raise SystemExit(
            f"Could not find {INPUT_TSV}. Export your corpus sheet as a "
            f"tab-separated file with at least Folio, Date, Author/Officer, "
            f"Recipient, Subject columns, and update INPUT_TSV at the top "
            f"of this script if the filename differs."
        )

    with open(INPUT_TSV) as f:
        rows = list(csv.DictReader(f, delimiter='\t'))

    G = build_graph(rows)
    print_report(G)
    export_data(G)
    plot_network(G, PNG_PATH)


if __name__ == "__main__":
    main()