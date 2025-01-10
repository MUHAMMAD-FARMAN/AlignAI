# store_exercise.py
from mongodb_connection import get_database
import uuid
from datetime import datetime


def store_exercise_data(user_id, exercise_name, total_reps, reps_data):
    db = get_database()
    exercise_collection = db['chatbot']  # Replace with your collection name

    exercise_data = {
        "user_id": user_id,
        "exercise_name": exercise_name,
        "date": datetime.now(),
        "total_reps": total_reps,
        "reps_data": reps_data,
        "session_id": str(uuid.uuid4())
    }

    exercise_collection.insert_one(exercise_data)
    print(f"Exercise data stored successfully for user {user_id}.")
