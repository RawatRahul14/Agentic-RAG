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

def create_multi_vector_retriever(vectorstore, text_summaries, texts, table_summaries, tables):
    """
    Creates a MultiVectorRetriever using provided summaries and full content for texts and tables.

    Args:
        vectorstore: The vector store to index summary documents.
        text_summaries (list): Summarized representations of text chunks.
        texts (list): Full text content corresponding to text_summaries.
        table_summaries (list): Summarized representations of table data.
        tables (list): Full HTML/text table content corresponding to table_summaries.

    Returns:
        MultiVectorRetriever: A retriever that maps summary vectors to full content using a document store.
    """
    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "fintech-rag"
    
    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore = vectorstore,
        docstore = store,
        id_key = id_key,
    )
    
    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        """
        Adds summary documents to the vectorstore and full documents to the docstore.

        Args:
            retriever (MultiVectorRetriever): The multi-vector retriever instance.
            doc_summaries (list): Summary-level text chunks.
            doc_contents (list): Full content to be stored and linked.
        """

        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]

        summary_docs = [
            Document(page_content = str(s), metadata = {id_key: doc_ids[i]}) 
            for i, s in enumerate(doc_summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
    # Add texts, tables
    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    
    return retriever

def chroma_retriever(text_summaries,
                     texts,
                     table_summaries,
                     tables):
    
    vectorstore = Chroma(collection_name = "FinTech RAG",
                         embedding_function = OpenAIEmbeddings())
    
    retriever = create_multi_vector_retriever(vectorstore = vectorstore,
                                              text_summaries = text_summaries,
                                              texts = texts,
                                              table_summaries = table_summaries,
                                              tables = tables)
    
    retriever = retriever.as_retriever(
            search_type = "similarity",
            search_kwargs = 
                {
                    "k": 3
                }
        )
    
    return retriever