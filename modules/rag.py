import streamlit as st
from src.utils.upload import get_file
from src.components.retriever import create_retriever

def rag_pipeline_page():
    st.header("Agentic RAG")

    # === File Uploader ===
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

            with st.sidebar:

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

                        # === Getting the file_data and file_name ===
                        file_data = st.session_state["file_data"]
                        file_name = st.session_state["file_name"]

                        retriever, chunks = create_retriever(file_data = file_data, file_name = file_name)

                        # === Cannot create retriever ===
                        if retriever is None:
                            st.warning("Could not create retriever. Please try another file.")

                        # === Retriever created ===
                        else:
                            st.session_state["retriever"] = retriever
                            st.session_state["chunks"] = chunks
                            st.success("Database Created Successfully.")
                    except Exception as e:
                        raise e

    # === Chat Interface ===
    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # === Show chat history ===
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # === Input box ===
    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # === Placeholder response, replace with your RAG response ===
        response = f"🤖 This is a placeholder response for: **{user_input}**"

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)