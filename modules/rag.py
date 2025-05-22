import streamlit as st
from src.utils.get_data import extract_from_pdf
from src.components.summarizer import get_summary
from src.components.retriever import create_retriever, create_summary_retriever
from langchain_core.messages import HumanMessage
from src.main_graph import build_agentic_rag_graph

def rag_pipeline_page():
    st.header("Agentic RAG")

    # === File Uploader ===
    uploaded_file = st.sidebar.file_uploader(
        "Upload your document file.",
        accept_multiple_files=False,
        type=["pdf", "txt"],
        key="rag_file_uploader"
    )

    # === Extract Data & Create Vector DB ===
    if uploaded_file:
        if "file_data" not in st.session_state or st.session_state["file_data"] is None:
            st.session_state["file_data"] = ""
            st.session_state["file_name"] = None
            st.session_state["retriever"] = None
            st.session_state["chunks"] = []

        if uploaded_file.name != st.session_state["file_name"]:
            with st.sidebar:
                # Extract file content
                with st.spinner(text="Extracting Data..."):
                    try:
                        texts, tables = extract_from_pdf(uploaded_file = uploaded_file)
                        text_summaries, table_summaries = get_summary(texts = texts, tables = tables)
                        st.success("Data extracted successfully Successfully.")
                    except Exception as e:
                        raise e

                with st.spinner(text = "Generating summaries..."):
                    try: 
                        st.session_state["text_summaries"] = text_summaries
                        st.session_state["table_summaries"] = table_summaries
                        st.session_state["file_name"] = uploaded_file.name
                        st.success("Summaries created Successfully.")
                    except Exception as e:
                        raise e

                # Create retriever
                with st.spinner(text="Creating Database..."):
                    try:
                        text_summaries = st.session_state["text_summaries"]
                        table_summaries = st.session_state["table_summaries"]
                        retriever, _ = create_summary_retriever(text_summaries = text_summaries, table_summaries = table_summaries)

                        if retriever is None:
                            st.warning("Could not create retriever. Please try another file.")
                        else:
                            st.session_state["retriever"] = retriever
                            st.success("Database Created Successfully.")
                    except Exception as e:
                        raise e

    # === Initialize Chat History ===
    st.divider()
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # === User Input ===
    user_input = st.chat_input("Ask a question about the document...")

    if user_input:
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # === Initialize Graph Once ===
        if "graph" not in st.session_state:
            st.session_state["graph"] = build_agentic_rag_graph()

        graph = st.session_state["graph"]

        # === Input to LangGraph ===
        input_data = {
            "question": HumanMessage(content = user_input),
            "retriever": st.session_state["retriever"]
        }

        # === Call the graph ===
        result = graph.invoke(input = input_data)

        # === Extract AI's response from state
        response_msg = next(
            (msg for msg in result["messages"][::-1] if msg.type == "ai"),
            None
        )

        response = response_msg.content if response_msg else "⚠️ No answer generated."

        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        retrieved_docs = result.get("documents", [])