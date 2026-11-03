import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.generation.chain import FinConRAGPipeline
from src.ingestion.run_ingestion import run_ingestion_pipeline

st.set_page_config(
    page_title="FinCon & Regulatory RAG Assistant",
    page_icon="💼",
    layout="wide"
)

st.markdown("""<style>
.stApp{background:#FFFFFF}
[data-testid="stSidebar"]{background:#EAF7F6;border-right:2px solid #22B9AE}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:#2B2F35}
.stApp p,.stApp label,.stApp li,.stApp span{color:#2B2F35}
.stButton>button{background:#22B9AE;color:#fff;border:none;border-radius:8px}
.stButton>button:hover{background:#A44251;color:#fff}
.stTextInput input,.stTextArea textarea,[data-baseweb="select"]>div{background:#fff!important;color:#2B2F35!important;border:1px solid #5DB5D7!important}
[data-testid="stAlert"]{background:#FBF3DC;border-left:4px solid #EAC56A}
[data-testid="stExpander"]{background:#F3FAFD;border:1px solid #5DB5D7}
code{background:#FBF3DC!important;color:#A44251!important}
</style>""", unsafe_allow_html=True)

st.title(" FinCon & Financial Regulatory Intelligence Assistant")
st.markdown(
    "**Production Hybrid RAG** • Hierarchical Parent-Child Indexing • Cross-Encoder Reranker • Citation Enforcement"
)

st.sidebar.header("⚙️ Configuration & Pipeline")
top_k = st.sidebar.slider("Top-K Rerank Results", min_value=1, max_value=8, value=3)

if st.sidebar.button("🔄 Run Document Ingestion"):
    with st.spinner("Parsing documents & building Hybrid Indexes..."):
        try:
            run_ingestion_pipeline()
            st.sidebar.success("Ingestion and indexing complete!")
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 2026 Financial Standards Queries")
sample_queries = [
    "What is the mandatory timeline for Form 8-K disclosure under SEC Item 1.05 for material cybersecurity incidents?",
    "What are the differences between Scope 1, Scope 2, and Scope 3 emissions under IFRS S2 in 2026?",
    "What are the eligibility criteria and turnover thresholds for EU CSRD Wave 2 compliance in 2026?",
    "What is the Basel III Endgame capital output floor percentage for risk-weighted assets?",
    "What confidence level is required for Expected Shortfall under the FRTB market risk framework?"
]

selected_sample = st.sidebar.selectbox("Choose a 2026 sample query:", [""] + sample_queries)

@st.cache_resource
def get_pipeline():
    return FinConRAGPipeline()

pipeline = get_pipeline()

user_query = st.text_input("Enter your Financial Controlling / Accounting Question:", value=selected_sample if selected_sample else "")

if st.button("🔍 Search & Analyze", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Executing Hybrid Retrieval, Reranking & Synthesis..."):
            result = pipeline.run(query=user_query, top_k_rerank=top_k)

            st.markdown("### 📝 Financial Controlling Synthesis")
            st.markdown(result["answer"])

            entities = result.get("entities", [])
            if entities:
                ent_badges = "  ".join([f"`🏷️ {e['text']} ({e['label']})`" for e in entities])
                st.markdown(f"**Recognized Financial Entities (spaCy):** {ent_badges}")

            if result.get("expanded_query") and result.get("expanded_query") != user_query:
                st.info(f"🔄 **Pre-Retrieval Query Expansion:** `{result['expanded_query']}`")

            st.markdown("---")
            st.markdown("### 📚 Source Attributions")
            cols = st.columns(len(result["sources"]) if result["sources"] else 1)
            for i, src in enumerate(result["sources"]):
                cols[i % len(cols)].info(f"📄 **{src}**")

            with st.expander("🔍 Deep Dive: Inspect Retrieved Child Chunks & Reranker Scores"):
                for idx, chunk in enumerate(result["retrieved_child_chunks"], start=1):
                    st.markdown(f"#### Rank {idx}: `{chunk['id']}`")
                    st.markdown(f"**Cross-Encoder Rerank Score:** `{chunk.get('rerank_score', 0.0):.4f}` | **RRF Score:** `{chunk.get('rrf_score', 0.0):.4f}`")
                    st.text_area(f"Child Chunk #{idx} Content", chunk["text"], height=100, key=f"child_{idx}")
                    st.markdown(f"**Parent ID:** `{chunk.get('parent_id')}` | **Metadata:** {chunk.get('metadata')}")
                    st.markdown("---")

            with st.expander("📖 Deep Dive: Full Parent Context Sent to LLM"):
                st.code(result["context_used"], language="markdown")
