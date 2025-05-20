import streamlit as st
from modules.auth_page import login_page, signup_page
from modules.rag import rag_pipeline_page

def main() -> None:
    """
    Entry point of the Streamlit application.
    Handles session state, authentication, and page routing.
    """

    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "page" not in st.session_state:
        st.session_state.page = "login"

    # Authenticated user view
    if st.session_state.authenticated:
        st.sidebar.title("🚀 Navigation")
        page: str = st.sidebar.radio("Go to", ["RAG Pipeline"], key = "nav")

        if page == "RAG Pipeline":
            rag_pipeline_page()

        # Logout
        if st.sidebar.button("🔒 Logout", key = "logout"):
            st.session_state.authenticated = False
            st.session_state.page = "login"
            st.rerun()

    # Unauthenticated user view
    else:
        if st.session_state.page == "login":
            login_page()
        elif st.session_state.page == "signup":
            signup_page()


# ===== Entry Point =====
if __name__ == "__main__":
    main()