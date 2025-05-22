from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore

def create_retriever(file_data,
                     file_name):

    # === Change  text to Langchain Document ===
    docs = Document(page_content = file_data,
                    metadata = {
                        "source": file_name
                    })
    
    documents = [docs]

    # === Creating Chunks ===
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 50
    )

    split_docs = text_splitter.split_documents(documents = documents)

    # === Creating vector DB ===
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma.from_documents(documents = split_docs,
                                      embedding = embeddings)
    
    retriever = vector_db.as_retriever(
        search_type = "similarity",
        search_kwargs = {
            "k": 3
        }
    )

    return retriever, split_docs

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_summary_retriever(
    table_summaries: list,
    text_summaries: list,
    collection_name: str = "summary_store",
):
    """
    Creates a Chroma retriever from table and text summaries using manual .add_texts().

    Args:
        table_summaries (list): List of table-based financial summaries (as strings)
        text_summaries (list): List of text-based financial summaries (as strings)
        collection_name (str): Name of the Chroma collection
        persist_directory (str): Path to persist Chroma DB

    Returns:
        retriever: A retriever interface from Chroma
        documents: List of Document objects created (with metadata)
    """
    summaries = table_summaries + text_summaries

    # Create Document objects with minimal metadata
    documents = [
        Document(page_content=s, metadata={"source": "table" if i < len(table_summaries) else "text"})
        for i, s in enumerate(summaries)
    ]

    # Extract page_content for Chroma text ingestion
    texts = [doc.page_content for doc in documents]

    # Init embeddings and Chroma DB manually
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )

    # Add texts (instead of using from_documents)
    vector_db.add_texts(texts=texts, metadatas=[doc.metadata for doc in documents])

    # Create retriever manually
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever, documents