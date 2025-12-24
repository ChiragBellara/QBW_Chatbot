import uuid
import os
from typing import List, Dict, Any
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class HandleIngestionAndRetrieval:
    """Manages document embedding in ChromaDB vector store"""

    def __init__(self, config: Dict, persist_directory: str = "../data/text_files") -> None:
        """
        Initialize a vector store

        :param collection_name: Name of the ChromaDB collection
        :param persist_directory: Directory to store a persistent copy of the data store
        """
        self.config = config
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100)
        self.vector_store = self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            return Chroma(collection_name="information", persist_directory=self.persist_directory,
                          embedding_function=FastEmbedEmbeddings())
        except Exception as ex:
            print(f"Error initializing vector store: {ex}")
            raise

    def load_document(self, file_path):
        """
        Load the document from the given file path

        :param file_path: path of the file to be loaded into the vector store.
        """
        loader = None
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = csv_loader.CSVLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print("Unsupported file type")
            return None
        return loader.load()

    def add_document(self, documents: List[Any] | None):
        """
        Add documents and their embeddings to the vector store

        :param documents: List of documents
        """
        if documents is None:
            print("Failed to add documents.")
            return
        chunks = self.text_splitter.split_documents(documents)
        uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        self.vector_store.add_documents(chunks, id=uuids)
        print(f"Added document to the database.")

    def get_docs_by_similarity(self, query: str):
        """
        Get the documents that match the context of the given query

        :param documents: query to be looked for in the vector database
        """
        return self.vector_store.similarity_search_with_score(
            query=query,
            k=self.config["rag_options"]["results_to_return"],
            score_threshold=self.config["rag_options"]["similarity_threshold"],
        )
