import streamlit as st
import os
import re

from config import MODEL_PATH, TOP_K
from src.extractor import WebTextExtractor
from src.embedder_chroma import EmbedderChroma
from src.memory_manager import MemoryManager
from src.langchain_qa_chain import LangchainQAChain
from sentence_transformers import CrossEncoder
import time
import random 
def slugify_url(url):

    url = url.strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'[^a-z0-9]+', '_', url)
    return url.strip('_')

def get_chroma_dir_from_urls(url_list):
    slug_parts = [slugify_url(url) for url in sorted(url_list)]
    slug = "__".join(slug_parts)
    random_suffix = f"{random.randint(1000, 9999)}_{int(time.time())}"
    return f"chroma_db_{slug}_{random_suffix}"

st.title("Web-Based QA Tool")

url_input = st.text_input("Enter one URL only:")
query=st.text_input("Ask your question here:")

if st.button("Process and Answer"):
    urls = [url.strip() for url in url_input.split(",") if url.strip()]

    if len(urls) == 0:
        st.warning("⚠️ Please enter a valid URL.")
        st.stop()
    elif len(urls) > 1:
        st.warning("⚠️ Only one URL is allowed. Please enter a single URL.")
        st.stop()

    with st.spinner("🔍 Extracting content from URL..."):
        extractor = WebTextExtractor(urls)
        docs = extractor.extract_all()

    dynamic_chroma_dir = get_chroma_dir_from_urls(urls)
    import os
    st.info(f"📂 Using Chroma vector store directory: `{dynamic_chroma_dir}'")

    with st.spinner("🔗 Creating embeddings and storing in ChromaDB..."):
        embedder = EmbedderChroma(persist_directory=dynamic_chroma_dir)
        embedder.create_vector_store(docs, urls=urls)


    vectorstore = embedder.get_vector_store()

    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})              
    docs = retriever.invoke(query)
    for i, doc in enumerate(docs):
            print(f"[{i+1}] {doc.page_content}...") 
     


    # print(f"Total Documents in ChromaDB: {vectorstore._collection.count()}")
  


    with st.spinner("💬 Chatting with LLM..."):
        # memory = MemoryManager().get_memory()
        qa_chain = LangchainQAChain(MODEL_PATH, retriever)
        answer = qa_chain.run(query)

    st.subheader("🧠 Answer:")
    st.success(answer)
