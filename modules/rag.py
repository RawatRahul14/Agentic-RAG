import streamlit as st
from src.utils.upload import get_file
from src.components.retriever import create_retriever

def rag_pipeline_page():
    st.header("Agentic RAG")

    # === File Uploader ====
    uploaded_file = st.sidebar.file_uploader("Upload your document file.",
                                             accept_multiple_files = False,
                                             type = ["pdf", "txt"],
                                             key = "rag_file_uploader")
    
    # === Extract Data, Creating Database ===
    if uploaded_file:

        # === Session State for extracted file data ===
        if "file_data" not in st.session_state or st.session_state["file_data"] is None:
            st.session_state["file_data"] = ""
            st.session_state["file_name"] = None
            st.session_state["retriever"] = None
            st.session_state["chunks"] = []

        # === Avoids unnecessary vectordb creation ===
        if uploaded_file.name != st.session_state["file_name"]:
            # === Extracting Data ===
            with st.spinner(text = "Extracting Data..."):
                try:
                    file_data = get_file(file = uploaded_file)

                    # === Cannot extract data ===
                    if not file_data.strip():
                        st.warning("Could not extract any text from the document. Please try another file.")

                    # === Extract data ===
                    else:
                        st.session_state["file_data"] = file_data
                        st.session_state["file_name"] = uploaded_file.name
                        st.success("Data Extracted Successfully.")
                
                except Exception as e:
                    raise e

            # === Creating Retriever ===
            with st.spinner(text = "Creating Database..."):
                try:
                    file_data = st.session_state["file_data"]
                    file_name = st.session_state["file_name"]

                    retriever, chunks = create_retriever(file_data = file_data,
                                                        file_name = file_name)
                    

                    # === Cannot create retriever ===
                    if retriever is None:
                        st.warning("Could not create retriever. Please try another file.")

                    # === Retriever created ===
                    else:
                        st.session_state["retriever"] = retriever                
                        st.session_state["chunks"] = chunks         
                    
                except Exception as e:
                    raise e