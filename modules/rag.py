import streamlit as st
from src.utils.upload import get_file

def rag_pipeline_page():
    st.header("Agentic RAG")

    # === File Uploader ====
    uploaded_file = st.sidebar.file_uploader("Upload your document file.",
                                             accept_multiple_files = False,
                                             type = ["pdf", "txt"],
                                             key = "rag_file_uploader")
    
    # === Extract Data ===
    if uploaded_file:
        with st.spinner(text = "Extracting Data..."):
            file_data = get_file(file = uploaded_file)

            # === Cannot extract data ===
            if not file_data.strip():
                st.warning("Could not extract any text from the document. Please try another file.")

            st.success("Data Extracted Successfully.")