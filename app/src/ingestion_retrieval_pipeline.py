import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagOptions:
    results_to_return: int = 5
    similarity_threshold: Optional[float] = None  # make optional
    use_mmr: bool = True
    mmr_fetch_k: int = 20
    mmr_lambda: float = 0.5


class HandleIngestionAndRetrieval:
    """Manages document embedding in ChromaDB vector store with metadata and deterministic IDs"""

    def __init__(
        self,
        config: Dict,
        persist_directory: str = "../data/knowledge",
        collection_name: str = "knowledge",
        chunk_size: int = 600,
        chunk_overlap: int = 100
    ) -> None:
        """
        Initialize a vector store

        :param config: Dictionary containing the settings for the LLM and RAG pipelines
        :param persist_directory: Directory to store a persistent copy of the data store
        :param collection_name: Name of the ChromaDB collection
        :param chunk_size: Size of text chunks (default 600 for better precision with short FAQ/policy snippets)
        :param chunk_overlap: Overlap between chunks
        """
        self.config = config
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Better chunking defaults for RAG - tuned for clinic documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings
                "! ",    # Exclamation endings
                "? ",    # Question endings
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Spaces
                ""       # Characters
            ]
        )
        self.vector_store = self._initialize_store()

    def _initialize_store(self) -> Chroma:
        """Initialize the Chroma vector store"""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=FastEmbedEmbeddings()
            )
            logger.info(
                f"Initialized vector store at {self.persist_directory}")
            return vector_store
        except Exception as ex:
            logger.error(f"Error initializing vector store: {ex}")
            raise

    def _get_file_type(self, file_path: str) -> str:
        """Determine file type from extension"""
        ext = Path(file_path).suffix.lower()
        type_map = {
            ".pdf": "pdf",
            ".docx": "docx",
            ".csv": "csv",
            ".txt": "txt",
            ".md": "markdown"
        }
        return type_map.get(ext, "unknown")

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a stable, deterministic document ID based on file path"""
        # Use absolute path for consistency
        abs_path = os.path.abspath(file_path)
        # Create a hash of the file path for a stable ID
        doc_id = hashlib.md5(abs_path.encode()).hexdigest()[:16]
        return f"doc_{doc_id}"

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate deterministic chunk ID based on doc_id and chunk_index"""
        return f"{doc_id}_chunk_{chunk_index:06d}"

    def load_document(self, file_path: str, clinic_section: Optional[str] = None) -> List[Document]:
        """
        Load the document from the given file path with metadata

        :param file_path: Path of the file to be loaded into the vector store
        :param clinic_section: Optional section identifier (e.g., "pricing", "policy", "faq")
        :return: List of Document objects with metadata
        """
        loader = None
        file_type = self._get_file_type(file_path)

        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = CSVLoader(file_path)
        elif file_path.endswith(".txt") or file_path.endswith(".md"):
            loader = TextLoader(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path}")
            return []

        try:
            documents = loader.load()
            doc_id = self._generate_doc_id(file_path)
            source_name = Path(file_path).name

            # Add metadata to each document
            for i, doc in enumerate(documents):
                # Preserve existing metadata if any
                if not hasattr(doc, 'metadata') or doc.metadata is None:
                    doc.metadata = {}

                # Add required metadata
                doc.metadata.update({
                    "source": source_name,
                    "doc_id": doc_id,
                    "file_type": file_type,
                    "file_path": file_path
                })

                # Add page number for PDFs (if available)
                if file_type == "pdf" and "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"]
                elif file_type == "pdf":
                    # Try to extract page from metadata or default to 0
                    doc.metadata["page"] = doc.metadata.get("page", 0)

                # Add optional clinic_section
                if clinic_section:
                    doc.metadata["clinic_section"] = clinic_section

            logger.info(
                f"Loaded {len(documents)} pages from {source_name} (doc_id: {doc_id})")
            return documents
        except Exception as ex:
            logger.error(f"Error loading document {file_path}: {ex}")
            return []

    def add_document(
        self,
        documents: Optional[List[Document]],
        clinic_section: Optional[str] = None
    ) -> bool:
        """
        Add documents and their embeddings to the vector store with deterministic IDs

        :param documents: List of Document objects
        :param clinic_section: Optional section identifier
        :return: True if successful, False otherwise
        """
        if documents is None or len(documents) == 0:
            logger.warning(
                "Failed to add documents: empty or None document list")
            return False

        # Ensure all documents have doc_id
        if not documents[0].metadata.get("doc_id"):
            logger.warning(
                "Documents missing doc_id metadata, generating new one")
            # This shouldn't happen if load_document is used, but handle gracefully
            doc_id = self._generate_doc_id(
                documents[0].metadata.get("source", "unknown"))
            for doc in documents:
                doc.metadata["doc_id"] = doc_id

        doc_id = documents[0].metadata["doc_id"]

        # Add clinic_section if provided and not already present
        if clinic_section:
            for doc in documents:
                if "clinic_section" not in doc.metadata:
                    doc.metadata["clinic_section"] = clinic_section

        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")

        # Generate deterministic chunk IDs
        chunk_ids = [
            self._generate_chunk_id(doc_id, i)
            for i in range(len(chunks))
        ]

        # Check if document already exists (re-ingestion prevention)
        existing_ids = self.vector_store.get(ids=chunk_ids)
        if existing_ids and len(existing_ids["ids"]) > 0:
            logger.info(
                f"Document {doc_id} already exists. Deleting old chunks before re-ingestion.")
            self.delete_document_by_id(doc_id)

        try:
            self.vector_store.add_documents(chunks, ids=chunk_ids)
            logger.info(
                f"Successfully added {len(chunks)} chunks to database (doc_id: {doc_id})")
            return True
        except Exception as ex:
            logger.error(f"Error adding documents to vector store: {ex}")
            return False

    def get_docs_by_similarity(self, query: str) -> List[Tuple[Document, float]]:
        """
        Get documents that match the context of the given query using MMR or similarity search

        :param query: Query to be searched in the vector database
        :return: List of tuples (Document, score)
        """
        rag_options = self.config.get("rag_options", {})
        use_mmr = rag_options.get("use_mmr", True)
        results_to_return = rag_options.get("results_to_return", 5)
        similarity_threshold = rag_options.get("similarity_threshold")
        mmr_fetch_k = rag_options.get("mmr_fetch_k", 20)
        mmr_lambda = rag_options.get("mmr_lambda", 0.5)

        try:
            if use_mmr:
                # Use MMR for better diversity
                results = self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=results_to_return,
                    fetch_k=mmr_fetch_k,
                    lambda_mult=mmr_lambda
                )
                # MMR doesn't return scores, so we'll get them separately
                # For MMR, we'll use a placeholder score or fetch with scores
                # Chroma's MMR doesn't directly return scores, so we do a similarity search for scores
                scored_results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=results_to_return
                )
                # Filter by threshold if provided
                if similarity_threshold is not None:
                    scored_results = [
                        (doc, score) for doc, score in scored_results
                        if score <= similarity_threshold  # Lower is more similar in some models
                    ]
                logger.info(
                    f"MMR retrieval returned {len(scored_results)} results")
                return scored_results
            else:
                # Standard similarity search
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=results_to_return,
                    score_threshold=similarity_threshold
                )
                logger.info(
                    f"Similarity search returned {len(results)} results")
                # Log actual scores for debugging
                if results:
                    scores = [score for _, score in results]
                    logger.debug(
                        f"Score range: min={min(scores):.4f}, max={max(scores):.4f}, mean={sum(scores)/len(scores):.4f}")
                return results
        except Exception as ex:
            logger.error(f"Error retrieving documents: {ex}")
            return []

    def delete_document_by_id(self, doc_id: str) -> bool:
        """
        Delete all chunks associated with a document ID

        :param doc_id: Document ID to delete
        :return: True if successful, False otherwise
        """
        try:
            # Get all chunk IDs for this document
            # Since we use deterministic IDs, we can query by pattern
            # Chroma doesn't support pattern matching directly, so we need to get all IDs
            all_data = self.vector_store.get()
            if not all_data or "ids" not in all_data:
                logger.warning(
                    f"No documents found to delete for doc_id: {doc_id}")
                return False

            # Filter IDs that match the doc_id pattern
            chunk_ids_to_delete = [
                chunk_id for chunk_id in all_data["ids"]
                if chunk_id.startswith(f"{doc_id}_chunk_")
            ]

            if not chunk_ids_to_delete:
                logger.warning(f"No chunks found for doc_id: {doc_id}")
                return False

            # Delete the chunks
            self.vector_store.delete(ids=chunk_ids_to_delete)
            logger.info(
                f"Deleted {len(chunk_ids_to_delete)} chunks for doc_id: {doc_id}")
            return True
        except Exception as ex:
            logger.error(f"Error deleting document {doc_id}: {ex}")
            return False

    def update_document(self, file_path: str, clinic_section: Optional[str] = None) -> bool:
        """
        Update a document by re-ingesting it (delete old, add new)

        :param file_path: Path to the document file
        :param clinic_section: Optional section identifier
        :return: True if successful, False otherwise
        """
        logger.info(f"Updating document: {file_path}")
        documents = self.load_document(file_path, clinic_section)
        if not documents:
            return False

        # add_document will handle deletion of existing chunks
        return self.add_document(documents, clinic_section)

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document

        :param doc_id: Document ID
        :return: Dictionary with document metadata or None if not found
        """
        try:
            all_data = self.vector_store.get()
            if not all_data or "ids" not in all_data:
                return None

            # Find first chunk for this doc_id
            chunk_ids = [
                chunk_id for chunk_id in all_data["ids"]
                if chunk_id.startswith(f"{doc_id}_chunk_")
            ]

            if not chunk_ids:
                return None

            # Get metadata from first chunk
            chunk_data = self.vector_store.get(ids=[chunk_ids[0]])
            if chunk_data and "metadatas" in chunk_data and chunk_data["metadatas"]:
                return chunk_data["metadatas"][0]

            return None
        except Exception as ex:
            logger.error(f"Error getting document metadata for {doc_id}: {ex}")
            return None
