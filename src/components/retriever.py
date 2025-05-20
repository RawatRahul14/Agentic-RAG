from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

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
        search_type = "mmr",
        search_kwargs = {
            "k": 3
        }
    )

    return retriever, split_docs