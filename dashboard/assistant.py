"""RAG chatbot page."""

import os

import streamlit as st

from polypseg import config as C
from polypseg import rag
from polypseg.train import PIPELINE_RESULTS_CSV

NO_LLM = "None (retrieval only, offline)"
SUGGESTIONS = [
    "Which model performs best on LDPolypVideo?",
    "Compare Attention U-Net and U-Net++.",
    "Why is accuracy high even for weak models?",
    "Which experiment had unstable training?",
    "What hyper-parameters were used?",
    "What future work is planned?",
]


@st.cache_resource(show_spinner="Indexing project knowledge...")
def _retriever(include_pipeline: bool, stamp: float):
    extra = [PIPELINE_RESULTS_CSV, C.LEADERBOARD_CSV] if include_pipeline else [C.LEADERBOARD_CSV]
    docs = rag.build_corpus(extra)
    return rag.Retriever(docs), len(docs)


def _stamp():
    paths = [C.RESULTS_CSV, PIPELINE_RESULTS_CSV, C.LEADERBOARD_CSV, C.CHECKPOINT_DIR]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def render(ctx):
    st.subheader("AI Research Assistant (RAG)")
    retriever, n_docs = _retriever(ctx.get("include_pipeline", False), _stamp())

    with st.expander("⚙️ LLM settings", expanded="chat" not in st.session_state):
        c1, c2 = st.columns(2)
        provider = c1.selectbox("Provider", [NO_LLM] + list(rag.PROVIDERS), key="rag_provider")
        if provider != NO_LLM:
            p = rag.PROVIDERS[provider]
            model = c2.text_input("Model", p["model"], key=f"rag_model_{provider}")
            if p["env"]:
                rag.load_dotenv()
                env_found = bool(os.environ.get(p["env"]))
                key = st.text_input(f"API key ({'found in environment' if env_found else 'not set'})", type="password",
                                    key=f"rag_key_{provider}", help=f"Or set {p['env']} in a .env file. Kept only in this session.")
                st.caption(f"Free key: {p['signup']}")
            else:
                key = None
                st.caption(f"Runs locally: install Ollama ({p['signup']}) and run `ollama pull {p['model']}`.")
            st.caption("Questions and the retrieved project passages are sent to the selected provider.")
        else:
            model, key = None, None
            st.caption("Answers are the top retrieved passages. Pick a provider for natural-language answers.")
        st.caption(f"Knowledge base: {n_docs} passages (results, rankings, models, datasets, metrics, README, CompleteIdea, checkpoints).")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    cols = st.columns(3)
    clicked = None
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 3].button(q, key=f"sugg_{i}", width="stretch"):
            clicked = q

    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    for s in turn["sources"]:
                        st.markdown(f"**[{s['n']}]** `{s['source']}` (score {s['score']:.3f})  \n{s['text']}")

    question = st.chat_input("Ask about datasets, models, metrics or results...") or clicked
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        history = [{"role": t["role"], "content": t["content"]} for t in st.session_state.chat]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = rag.answer(question, retriever, None if provider == NO_LLM else provider, model, key or None, history)
                except Exception as exc:
                    st.error(str(exc))
                    res = rag.answer(question, retriever)
            if provider != NO_LLM and res["mode"] == "extractive":
                st.caption("No API key available, so showing retrieved passages.")
            st.markdown(res["answer"])
            with st.expander("Sources"):
                for s in res["sources"]:
                    st.markdown(f"**[{s['n']}]** `{s['source']}` (score {s['score']:.3f})  \n{s['text']}")
        st.session_state.chat += [{"role": "user", "content": question},
                                  {"role": "assistant", "content": res["answer"], "sources": res["sources"]}]

    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()
