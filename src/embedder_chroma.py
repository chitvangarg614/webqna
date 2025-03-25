from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbedderChroma:
    def __init__(self, persist_directory):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        

    def create_vector_store(self, docs, urls=None):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=70)
        chunks = text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
        self.vectorstore.persist()

    def get_vector_store(self):
        if self.vectorstore is None:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function
                )
            else:
                raise FileNotFoundError(f"ChromaDB directory not found: {self.persist_directory}")

        return self.vectorstore
