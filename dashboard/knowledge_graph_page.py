"""Knowledge graph explorer page."""

import streamlit as st

from polypseg import knowledge_graph as KG


def render(ctx):
    st.subheader("Knowledge Graph")
    st.caption("Links models, architectural components, datasets, experiments and metrics. "
               "Built in-process with networkx and exportable to Neo4j as Cypher.")
    G = KG.build_graph(ctx["filtered_df"])
    s = KG.graph_summary(G)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Nodes", s["nodes"])
    k2.metric("Edges", s["edges"])
    k3.metric("Models", s["node_kinds"].get("Model", 0))
    k4.metric("Components", s["node_kinds"].get("Component", 0))

    labels = {d["label"] + f"  [{d['kind']}]": n for n, d in G.nodes(data=True) if d["kind"] != "Experiment"}
    c1, c2 = st.columns([1, 3])
    with c1:
        kinds = st.multiselect("Show node types", list(KG.NODE_COLOURS), default=list(KG.NODE_COLOURS))
        focus = st.selectbox("Focus node", sorted(labels))
    with c2:
        st.plotly_chart(KG.plot_graph(G, highlight=labels[focus], show_kinds=kinds), width="stretch")

    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(f"**Relations of {focus.split('  [')[0]}**")
        nb = KG.neighbours(G, labels[focus])
        st.dataframe(nb, hide_index=True, width="stretch")
    with q2:
        st.markdown("**How are two nodes connected?**")
        a = st.selectbox("From", sorted(labels), key="kg_a")
        b = st.selectbox("To", sorted(labels), index=min(1, len(labels) - 1), key="kg_b")
        path = KG.connection_path(G, labels[a], labels[b])
        st.write(" → ".join(path) if path else "No path.")
    with q3:
        st.markdown("**Models sharing a component**")
        comps = sorted(d["label"] for _, d in G.nodes(data=True) if d["kind"] == "Component")
        if comps:
            comp = st.selectbox("Component", comps)
            st.write(", ".join(KG.models_sharing_component(G, comp)))

    st.download_button("⬇️ Export to Neo4j (Cypher)", KG.to_cypher(G).encode(), "polypseg_graph.cypher", "text/plain")
    st.caption("Load in Neo4j Browser or with `cypher-shell -f polypseg_graph.cypher`.")
