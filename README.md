# AlignAI: AI Physiotherapist for Stroke Patients  

AlignAI is an advanced AI-driven physiotherapy solution designed to assist stroke patients in their rehabilitation journey. The system combines cutting-edge machine learning models, real-time exercise monitoring, and a user-friendly chatbot interface to deliver personalized guidance and progress tracking.  

---

## Features  
- **Interactive Chatbot**: Provides answers to stroke rehabilitation-related queries using the LLaMA 3.1 405b model.  
- **Exercise Monitoring**: Tracks patient rehabilitation exercises in real-time using Mediapipe and OpenCV.  
- **Resource Retrieval**: Fetches the latest books and guidelines on stroke recovery from a Qdrant vector database.  

---

## Setup
1. **Clone the repo**:
   Open terminal and run following command:
   ``` bash
   git clone https://github.com/MUHAMMAD-FARMAN/AlignAI
   cd AlignAI
   ```
2. **Install Dependencies**:  
   Navigate to the respective folders and install dependencies:  
   ```bash
   pip install -r chatbot/requirements.txt
   pip install -r backend/requirements.txt
   ```
3.	**Run the Chatbot Module**:
  Navigate to the chatbot folder in your project directory.
  Execute the following command to launch the chatbot interface using Gradio:
  ```bash
  gradio app.py
  ```
  This will start the chatbot application, allowing users to interact with the physiotherapy system.
  
4.	**Run the Backend Module**:
  Navigate to the backend folder in your project directory.
  Start the backend server using Flask with the following command:
  ```bash
  flask run
  ```
  The backend handles API requests and database interactions.
  
5.	**Launch the User Interface**:
  Open the landing_page folder in your project directory.
  Locate the index.html file and open it in your web browser.
_Hurray! The Project is Running_

# Demo Video:

You can watch the demo video here: [AlignAI.mp4](Demo/AlignAI.mp4)
