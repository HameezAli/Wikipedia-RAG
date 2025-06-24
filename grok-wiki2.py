import os
import streamlit as st
import nest_asyncio
import re
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

nest_asyncio.apply()
index_dir = 'Wikipedia-RAG'

f1_terms = [
    "Formula One",
    "Formula One car",
    "Formula One engine",
    "Formula One constructors",
    "Formula One drivers",
    "Formula One World Championship",
    "Formula One qualifying",
    "Max Verstappen",
    "Formula One race weekend",
    "List of Formula One circuits",
    "List of Formula One World Drivers' Champions",
    "List of Formula One World Constructors' Champions",
    "Formula One Group",
    "Lewis Hamilton"
]

@st.cache_resource
def get_index():
    if os.path.isdir(index_dir):
        storage = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage, embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"))
    else:
        docs = WikipediaReader().load_data(pages=f1_terms, auto_suggest=False)
        embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
        index.storage_context.persist(persist_dir=index_dir)
        return index

@st.cache_resource
def get_query_engine(groq_api_key: str):
    os.environ["GROQ_API_KEY"] = groq_api_key
    index = get_index()
    llm = Groq(model="qwen-qwq-32b", api_key=groq_api_key)  
    return index.as_query_engine(llm=llm, similarity_top_k=3)

import re

def main():
    st.title('🏁 Wikipedia RAG with Groq')
    groq_api_key = st.text_input("🔑 Enter your GROQ API Key:", type="password")
    question = st.text_input('💬 Ask a question about Formula One:')

    if st.button('Submit', key='submit_btn') and question and groq_api_key:
        with st.spinner('Thinking...'):
            try:
                qa = get_query_engine(groq_api_key)
                response = qa.query(question)
            except Exception as e:
                st.error(f"❌ Failed to get response: {e}")
                return

        raw_response = response.response

        # Extract and remove <think>...</think> block
        think_match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
        think_block = think_match.group(1).strip() if think_match else "No internal reasoning found."
        cleaned_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()

        # 📌 Answer
        st.subheader('📌 Answer')
        st.write(cleaned_response)

        # 🧠 <think> Block
        st.subheader('🧠 Internal Reasoning')
        st.text(f"<think>\n{think_block}\n</think>")

        st.subheader('📚 Retrieved Contexts (Subheadings Only)')
        for src in response.source_nodes:
            content = src.node.get_content()

            # Extract subheadings like "== Heading ==" or "=== Subheading ==="
            subheadings = re.findall(r"^={2,6}\s.*?\s={2,6}$", content, re.MULTILINE)

            if subheadings:
                st.markdown(f"**From: `{src.node.metadata.get('source', 'Unknown Source')}`**")
                for heading in subheadings:
                    # Convert wiki-style headings to readable text (strip =)
                    clean_heading = re.sub(r"=+", "", heading).strip()
                    st.markdown(f"- {clean_heading}")
            else:
                st.markdown(f"**From: `{src.node.metadata.get('source', 'Unknown Source')}`**")
                st.markdown("_No subheadings found._")



if __name__ == '__main__':
    main()
