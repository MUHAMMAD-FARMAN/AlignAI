from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId

app = Flask(__name__)
CORS(app)  # This will allow all domains to access your Flask API

# MongoDB connection
client = MongoClient(
    "mongodb+srv://farman_chat:farman_chat@cluster0.gz7j0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["chatbot"]
users_collection = db["user"]

# '/' route


@app.route('/')
def home():
    return "Hello, World!"


@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    confirm_password = data.get("confirm_password")
    print(data)

    # Password confirmation check
    if password != confirm_password:
        return jsonify({"message": "Passwords do not match!"}), 400

    # Check if user already exists
    if users_collection.find_one({"email": email}):
        return jsonify({"message": "User already exists!"}), 400

    # Insert new user
    hashed_password = generate_password_hash(password)
    users_collection.insert_one(
        {"name": name, "email": email, "password": hashed_password})

    return jsonify({"message": "User created successfully!"}), 201


@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    # Find user by email
    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"message": "User not found!"}), 404

    # Verify password
    if check_password_hash(user["password"], password):
        # Remove sensitive information
        user.pop("password", None)

        # convert ObjectId to string
        user["_id"] = str(user["_id"])
        
        return jsonify({"message": "Login successful!", "user": user}), 200
    else:
        return jsonify({"message": "Incorrect password!"}), 401


if __name__ == '__main__':
    app.run(debug=True)
