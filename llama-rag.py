import os
import streamlit as st
from llama_index.llms.ollama import Ollama  # If this fails, try the alternative below:
from llama_index.llms import Ollama
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

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
        return load_index_from_storage(storage)
    else:
        docs = WikipediaReader().load_data(pages=f1_terms, auto_suggest=False)
        embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
        index.storage_context.persist(persist_dir=index_dir)
        return index

@st.cache_resource
def get_query_engine():
    index = get_index()
    llm = Ollama(model="llama3.2", temperature=0.7)  # make sure `ollama run llama3` is working
    return index.as_query_engine(llm=llm, similarity_top_k=3)

def main():
    st.title('Wikipedia RAG Application (Local LLaMA 3.2)')
    question = st.text_input('Ask a question')
    if st.button('Submit') and question:
        with st.spinner('Thinking...'):
            qa = get_query_engine()
            response = qa.query(question)
        st.subheader('Answer')
        st.write(response.response)
        st.subheader('Retrieved Contexts')
        for src in response.source_nodes:
            st.markdown(src.node.get_content())

if __name__ == '__main__':
    main()
