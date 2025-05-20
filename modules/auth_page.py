import streamlit as st
from modules.auth import login_user, signup_user

def login_page() -> None:
    """
    Renders the login page and handles login logic.
    """
    st.title("🔐 Login")

    email: str = st.text_input("Email", key = "login_email")
    password: str = st.text_input("Password", type = "password", key = "login_password")

    if st.button("Login", key = "login_button"):
        if not email or not password:
            st.warning("Please enter both email and password.")
        else:
            success: bool = login_user(email, password)
            if success:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid email or password.")

    if st.button("📝 Sign Up", key = "goto_signup"):
        st.session_state.page = "signup"
        st.rerun()


def signup_page() -> None:
    """
    Renders the signup page and handles new user registration.
    """
    st.title("📝 Sign Up")

    new_email: str = st.text_input("Email", key = "signup_email")
    new_password: str = st.text_input("Password", type = "password", key = "signup_password")
    confirm_pw: str = st.text_input("Confirm Password", type = "password", key = "signup_confirm")

    if st.button("Sign Up", key = "signup_button"):
        if not new_email or not new_password or not confirm_pw:
            st.warning("All fields are required.")
        elif new_password != confirm_pw:
            st.error("Passwords do not match.")
        else:
            created: bool = signup_user(new_email, new_password)
            if created:
                st.success("Account created! Please log in.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("Email already exists or creation failed.")

    if st.button("Back to Login", key = "back_to_login"):
        st.session_state.page = "login"
        st.rerun()