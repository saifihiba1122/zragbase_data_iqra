import streamlit as st

st.set_page_config(
    page_title="Book RAG Assistant",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_rag():
    import rag
    return rag


st.markdown("""
<style>
.main { background-color: #f5f7fb; }
.block-container { padding-top: 2rem; }
[data-testid="stSidebar"] { background-color: #101827; }
[data-testid="stSidebar"] * { color: white; }
.stChatMessage { border-radius: 12px; }
h1 { color: #155eef; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📚 Book Assistant")
    st.write("RAG-Based Q&A System")
    st.divider()
    st.success("Gemini + FAISS Pipeline")
    st.caption("Query → Embed → Search → LLM → Answer")
    st.divider()
    st.caption("Tech Stack")
    st.write("• Sentence Transformers")
    st.write("• FAISS Vector Database")
    st.write("• Google Gemini LLM")
    st.write("• Streamlit")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Database Book Assistant")
st.write("Ask questions from **Modern Database Management, 10th Edition**.")
st.caption("RAG Pipeline: Query → Embedding → FAISS Search → LLM Generation")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    with st.spinner("Loading RAG system..."):
        rag = load_rag()
    st.success("RAG system ready!")
except Exception as e:
    st.error(f"Failed to load RAG system: {e}")
    st.stop()


def snippet(text, limit=250):
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("📄 View Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{rag.get_source_label(src)}**")
                    st.caption(f"Score: {src['final_score']:.3f} | Distance: {src['distance']:.4f}")
                    st.write(snippet(src["text"]))
                    st.divider()

        if msg.get("timing"):
            st.caption(msg["timing"])

query = st.chat_input("Ask a question about databases...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Processing your question..."):
            try:
                result = rag.pipeline(query)

                if not result["relevant"]:
                    st.toast("⚠️ This question is outside the book's knowledge base!", icon="⚠️")
                    answer = (
                        "**This question is outside the book's knowledge base.**\n\n"
                        "I can only answer questions related to "
                        "*Modern Database Management, 10th Edition*. "
                        "Please ask something about databases, SQL, ER models, "
                        "normalization, or other topics covered in the book."
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                    })

                elif result["llm_error"]:
                    st.warning(f"LLM Error: {result['llm_error']}")
                    answer = "⚠️ Could not generate an answer. Showing retrieved sources instead."
                    st.markdown(answer)

                    with st.expander("📄 Retrieved Sources"):
                        for src in result["chunks"][:5]:
                            st.markdown(f"**{rag.get_source_label(src)}**")
                            st.write(snippet(src["text"]))
                            st.divider()

                    timing = f"Search: {result['timing']['search']:.2f}s | Total: {result['timing']['total']:.2f}s"
                    st.caption(timing)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": result["chunks"][:5],
                        "timing": timing,
                    })

                else:
                    st.markdown(result["answer"])

                    with st.expander("📄 View Sources"):
                        for src in result["chunks"][:5]:
                            st.markdown(f"**{rag.get_source_label(src)}**")
                            st.caption(f"Score: {src['final_score']:.3f} | Distance: {src['distance']:.4f}")
                            st.write(snippet(src["text"]))
                            st.divider()

                    t = result["timing"]
                    timing = (
                        f"Embed: {t['embed']:.2f}s | "
                        f"Search: {t['search']:.2f}s | "
                        f"LLM: {t['llm']:.2f}s | "
                        f"Total: {t['total']:.2f}s"
                    )
                    st.caption(timing)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["chunks"][:5],
                        "timing": timing,
                    })

            except Exception as e:
                st.error(f"Error: {e}")
