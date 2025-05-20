from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

def collection_link() -> MongoClient:
    """
    Establishes a connection to the MongoDB collection specified in the environment variables.

    Returns:
        MongoClient.Collection: A reference to the specified MongoDB collection.
    """
    mongo_pass: str = os.getenv("mongo_pass")
    mongo_uri: str = f"mongodb+srv://rahulrawat:{mongo_pass}@cluster0.6h9hn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

    mongo_db: str = os.getenv("MONGODB_DB")
    mongo_collection: str = os.getenv("MONGODB_COLLECTION")

    client: MongoClient = MongoClient(mongo_uri)
    db = client[mongo_db]
    users_collection = db[mongo_collection]

    return users_collection