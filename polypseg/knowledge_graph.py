"""Knowledge graph linking datasets, models, architectural components, experiments and metrics.

Built with networkx (in-process, no server). `to_cypher()` exports the same graph as Cypher
statements so it can be loaded into Neo4j unchanged when a Neo4j instance is available.
"""

import math

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

MODEL_COMPONENTS = {
    "Basic U-Net": ["Encoder-decoder", "Skip connections"],
    "Attention U-Net": ["Encoder-decoder", "Skip connections", "Attention gates"],
    "U-Net++": ["Encoder-decoder", "Nested dense skip connections"],
    "DeepLabV3+": ["Encoder-decoder", "Atrous convolution", "ASPP"],
}
DATASET_FACTS = {
    "SUN Database": {"annotation": "Bounding boxes", "modality": "Colonoscopy video frames"},
    "LDPolypVideo": {"annotation": "Bounding boxes", "modality": "Colonoscopy video frames"},
}
NODE_COLOURS = {
    "Model": "#4f46e5", "Dataset": "#0ea5e9", "Experiment": "#9ca3af",
    "Component": "#10b981", "Fact": "#f59e0b", "Metric": "#ef4444",
}


def _base_model(name: str) -> str:
    return name.split(" (")[0]


def build_graph(results: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for metric in ["Dice", "IoU", "Accuracy"]:
        G.add_node(f"metric:{metric}", label=metric, kind="Metric")

    for _, r in results.iterrows():
        model, ds = str(r["model"]), str(r["dataset"])
        m_id, d_id = f"model:{model}", f"dataset:{ds}"
        e_id = f"exp:{model}|{ds}"
        G.add_node(m_id, label=model, kind="Model")
        G.add_node(d_id, label=ds, kind="Dataset")
        G.add_node(e_id, label=f"{model} on {ds}", kind="Experiment",
                   dice=float(r["final_dice"]), iou=float(r["final_iou"]),
                   accuracy=float(r["final_accuracy"]), samples=int(r.get("samples_used", 0) or 0))
        G.add_edge(m_id, e_id, rel="RAN")
        G.add_edge(e_id, d_id, rel="ON_DATASET")
        G.add_edge(e_id, "metric:Dice", rel="SCORED", value=float(r["final_dice"]))
        G.add_edge(e_id, "metric:IoU", rel="SCORED", value=float(r["final_iou"]))
        G.add_edge(e_id, "metric:Accuracy", rel="SCORED", value=float(r["final_accuracy"]))

        for comp in MODEL_COMPONENTS.get(_base_model(model), []):
            G.add_node(f"comp:{comp}", label=comp, kind="Component")
            if not G.has_edge(m_id, f"comp:{comp}"):
                G.add_edge(m_id, f"comp:{comp}", rel="USES")
        for key, val in DATASET_FACTS.get(ds, {}).items():
            G.add_node(f"fact:{val}", label=val, kind="Fact")
            if not G.has_edge(d_id, f"fact:{val}"):
                G.add_edge(d_id, f"fact:{val}", rel=key.upper())

    # Derived insight edges
    for ds, g in results.groupby("dataset"):
        best = g.loc[g["final_dice"].idxmax()]
        G.add_edge(f"model:{best['model']}", f"dataset:{ds}", rel="BEST_ON", value=float(best["final_dice"]))
    return G


def graph_summary(G):
    kinds = pd.Series([d["kind"] for _, d in G.nodes(data=True)]).value_counts().to_dict()
    rels = pd.Series([d["rel"] for *_, d in G.edges(data=True)]).value_counts().to_dict()
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "node_kinds": kinds, "relations": rels}


def neighbours(G, node_id):
    """Outgoing + incoming relations of a node as a DataFrame."""
    rows = []
    for _, v, d in G.out_edges(node_id, data=True):
        rows.append({"direction": "->", "relation": d["rel"], "node": G.nodes[v]["label"],
                     "kind": G.nodes[v]["kind"], "value": d.get("value")})
    for u, _, d in G.in_edges(node_id, data=True):
        rows.append({"direction": "<-", "relation": d["rel"], "node": G.nodes[u]["label"],
                     "kind": G.nodes[u]["kind"], "value": d.get("value")})
    return pd.DataFrame(rows)


def models_sharing_component(G, component):
    cid = f"comp:{component}"
    return sorted({G.nodes[u]["label"] for u, _, d in G.in_edges(cid, data=True) if d["rel"] == "USES"})


def connection_path(G, a, b):
    """Shortest undirected path between two node ids, as labels."""
    try:
        path = nx.shortest_path(G.to_undirected(as_view=True), a, b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return [G.nodes[n]["label"] for n in path]


def plot_graph(G, highlight=None, show_kinds=None, height=650, seed=7):
    show_kinds = set(show_kinds or NODE_COLOURS)
    H = G.subgraph([n for n, d in G.nodes(data=True) if d["kind"] in show_kinds])
    pos = nx.spring_layout(nx.Graph(H), seed=seed, k=1.4 / math.sqrt(max(1, H.number_of_nodes())))

    ex, ey, mx, my, mtext = [], [], [], [], []
    for u, v, d in H.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]
        mx.append((x0 + x1) / 2)
        my.append((y0 + y1) / 2)
        mtext.append(d["rel"] + (f" {d['value']:.3f}" if d.get("value") is not None else ""))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#cbd5e1", width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", marker=dict(size=4, color="#cbd5e1"),
                             text=mtext, hoverinfo="text", showlegend=False))
    for kind, colour in NODE_COLOURS.items():
        nodes = [n for n, d in H.nodes(data=True) if d["kind"] == kind]
        if not nodes:
            continue
        hover = []
        for n in nodes:
            d = H.nodes[n]
            extra = "".join(f"<br>{k}: {v:.4f}" for k, v in d.items() if isinstance(v, float))
            hover.append(f"<b>{d['label']}</b><br>{kind}{extra}")
        size = [26 if n == highlight else (18 if kind in ("Model", "Dataset") else 11) for n in nodes]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text" if kind != "Experiment" else "markers",
            text=[H.nodes[n]["label"] for n in nodes], textposition="top center",
            hovertext=hover, hoverinfo="text", name=kind,
            marker=dict(size=size, color=colour, line=dict(width=2 if kind != "Experiment" else 0, color="#ffffff")),
        ))
    fig.update_layout(height=height, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(orientation="h"))
    return fig


def _cy(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def to_cypher(G) -> str:
    """Cypher script that recreates the graph in Neo4j (MERGE, so it is idempotent)."""
    lines = []
    for n, d in G.nodes(data=True):
        props = ", ".join(f"{k}: {v!r}" if isinstance(v, (int, float)) else f"{k}: '{_cy(str(v))}'"
                          for k, v in d.items() if k != "kind")
        lines.append(f"MERGE (:{d['kind']} {{id: '{_cy(n)}', {props}}});")
    for u, v, d in G.edges(data=True):
        val = f" {{value: {d['value']}}}" if d.get("value") is not None else ""
        lines.append(f"MATCH (a {{id: '{_cy(u)}'}}), (b {{id: '{_cy(v)}'}}) MERGE (a)-[:{d['rel']}{val}]->(b);")
    return "\n".join(lines)
