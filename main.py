import os
import streamlit as st # to host the app
from dotenv import load_dotenv # to load api key from .env
from llama_index.llms.openai import OpenAI # LLM 
from llama_index.embeddings.openai import OpenAIEmbedding # embedding model for vector spaces
from llama_index.readers.wikipedia import WikipediaReader # read data frm wikipedia articles
from llama_index.core import VectorStoreIndex,StorageContext,load_index_from_storage # Vector data manipulation
load_dotenv() # Load the api key
index_dir = 'Wikipedia-RAG' # vector storage directory
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
def get_index():  # function to retrive the vector embeddings
    if os.path.isdir(index_dir): # if already exists, load and use it
        storage = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage
    else: # if does not exist create, store and use it
        docs=WikipediaReader().load_data(pages=f1_terms,auto_suggest=False)
        embedding_model = OpenAIEmbedding(model='text-embedding-3-small')
        index = VectorStoreIndex.from_documents(docs,embed_model=embedding_model)
        index.storage_context.persist(persist_dir=index_dir)
        return index

@st.cache_resource
def get_query_engine(): # use LLM to retrieve top 3 articles related to the query
    index=get_index()
    llm=OpenAI(model='gpt-4o-mini',temperature=2) # Temperature means creativity allowance
    return index.as_query_engine(llm=llm,similarity_top_k=3)

# UI Code
def main():
    st.title('Wikipedia RAG Application')
    question = st.text_input('Ask a question')
    if st.button('Submit') and question:
        with st.spinner('Thinking...'):
            qa=get_query_engine()
            response=qa.query(question)
        st.subheader('Answer')
        st.write(response.response)
        st.subheader('Retrieved Contexts') # Retrieving the source articles for analysis
        for src in response.source_nodes:
            st.markdown(src.node.get_content())

if __name__ == '__main__':
    main()
