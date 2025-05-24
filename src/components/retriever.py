from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def create_summary_retriever(
    table_summaries: list,
    text_summaries: list,
    collection_name: str = "summary_store",
):
    """
    Creates a Chroma retriever from table and text summaries,
    splitting each summary into 2 parts to improve retrieval quality.

    Returns:
        retriever: A retriever interface from Chroma
        documents: List of Document objects created (with metadata)
    """

    # Combine summaries and determine their types
    all_summaries = [(s, "table") for s in table_summaries] + [(s, "text") for s in text_summaries]

    # Split each summary into 2 chunks
    splitter = CharacterTextSplitter(
        chunk_size=300,      # adjust depending on average summary length
        chunk_overlap=20     # small overlap to retain context
    )

    documents = []
    for content, source_type in all_summaries:
        split_chunks = splitter.split_text(content)
        for i, chunk in enumerate(split_chunks):  # only take 2 parts per summary
            documents.append(Document(
                page_content=chunk,
                metadata={"source": source_type, "chunk": i + 1}
            ))

    # Extract content and metadata
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Initialize Chroma DB
    embeddings = OpenAIEmbeddings()
    vector_db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )

    # Add to Chroma
    vector_db.add_texts(texts=texts, metadatas=metadatas)

    # Return retriever
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5}
    )

    return retriever, documents