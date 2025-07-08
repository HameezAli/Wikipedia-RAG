import os
import shutil
import re
import streamlit as st
import nest_asyncio
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# Apply async patch
nest_asyncio.apply()

# Use your Groq API key directly here
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Storage path
index_dir = 'Wikipedia-RAG'

# Session state setup
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "terms" not in st.session_state:
    st.session_state.terms = []

# 🔧 Build or rebuild the index
def build_index(terms):
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

    docs = WikipediaReader().load_data(pages=terms, auto_suggest=False)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    index.storage_context.persist(persist_dir=index_dir)
    return index

# 🧠 Query Engine
def get_query_engine(api_key, terms):
    os.environ["GROQ_API_KEY"] = api_key
    index = build_index(terms)
    llm = Groq(model="llama-3.1-8b-instant", api_key=api_key) # old model was qwen-qwq-32b
    return index.as_query_engine(llm=llm, similarity_top_k=3)

# 🖥 Main UI
def main():
    st.set_page_config(page_title="Wiki-RAG")
    st.title("📘 Wiki-RAG")

    # Step 1: Get Wikipedia pages
    page_input = st.text_input("📚 Enter Wikipedia page name:")
    if st.button("Load Pages"):
        terms = [term.strip() for term in page_input.split(",") if term.strip()]
        if terms:
            st.session_state.terms = terms
            st.session_state.index_loaded = True
            st.success(f"Loaded pages: {', '.join(terms)}")
        else:
            st.error("Please enter a valid page name.")

    # Step 2: Ask Question
    if st.session_state.index_loaded:
        question = st.text_input("💬 Ask a question about the loaded Wikipedia pages:")

        if st.button("Submit", key="submit_btn") and question:
            with st.spinner("Thinking..."):
                try:
                    qa = get_query_engine(GROQ_API_KEY, st.session_state.terms)
                    response = qa.query(question)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    return

            # Process response
            raw_response = response.response
            think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
            think_block = think_match.group(1).strip() if think_match else "No internal reasoning found."
            cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

            st.subheader("📌 Answer")
            st.write(cleaned_response)

            st.subheader("🧠 Internal Reasoning")
            st.text(f"<think>\n{think_block}\n</think>")

            st.subheader("📚 Retrieved Contexts (Subheadings Only)")
            for src in response.source_nodes:
                content = src.node.get_content()
                subheadings = re.findall(r"^={2,6}\s.*?\s={2,6}$", content, re.MULTILINE)

                if subheadings:
                    for h in subheadings:
                        clean = re.sub(r"=+", "", h).strip()
                        st.markdown(f"- {clean}")
                else:
                    st.markdown("_No subheadings found._")

if __name__ == "__main__":
    main()
