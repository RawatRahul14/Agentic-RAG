import bcrypt
from Database.get_collection import collection_link
from datetime import datetime

def signup_user(email: str,
                password: str) -> bool:
    """
    Registers a new user with the given email and password.

    Args:
        email (str): The user's email address.
        password (str): The user's plaintext password.

    Returns:
        bool: True if signup is successful, False if the email already exists or insertion fails.
    """
    users = collection_link()

    # Check for existing user
    if users.find_one({"email": email}):
        return False

    # Hash the password securely
    hashed_pw: bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    user_doc: dict = {
        "email": email,
        "password": hashed_pw,
        "created_at": datetime.strftime()
    }

    try:
        users.insert_one(user_doc)
        return True
    except Exception as e:
        # Optionally log `e` for debugging
        return False


def login_user(email: str, password: str) -> bool:
    """
    Authenticates a user using their email and password.

    Args:
        email (str): The user's email address.
        password (str): The user's plaintext password.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    users = collection_link()
    user = users.find_one({"email": email})

    if not user:
        return False

    return bcrypt.checkpw(password.encode("utf-8"), user["password"])