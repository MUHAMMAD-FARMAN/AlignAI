# mongodb_connection.py
from pymongo import MongoClient
import json

# Load the config.json file
with open('config.json', 'r') as f:
    config = json.load(f)


def get_database():
    CONNECTION_STRING = config["mongodb"]["connection_string"]
    client = MongoClient(CONNECTION_STRING)
    return client['chatbot']  # Replace with your database name
