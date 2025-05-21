from langchain_core.documents import Document

def build_documents_from_text_and_tables(texts: list[str], tables: list[str], file_name: str) -> list[Document]:
    documents = []

    for i, text in enumerate(texts):
        documents.append(Document(
            page_content=text.strip(),
            metadata={"source": file_name, "type": "text", "section": f"text-{i+1}"}
        ))

    for j, table_html in enumerate(tables):
        documents.append(Document(
            page_content=table_html.strip(),
            metadata={"source": file_name, "type": "table", "section": f"table-{j+1}"}
        ))

    return documents