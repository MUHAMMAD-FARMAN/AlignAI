import gradio as gr
from PIL import Image
import torch
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import mysql.connector
from datetime import datetime

import json

# Load the config.json file
with open('config.json', 'r') as f:
    config = json.load(f)

# Accessing API keys and URLs from config
openai_base_url = config['openai']['base_url']
openai_api_key = config['openai']['api_key']

qdrant_url = config['qdrant']['url']
qdrant_api_key = config['qdrant']['api_key']


client = OpenAI(
    base_url=openai_base_url, api_key=openai_api_key
)

url = qdrant_url
api_key = qdrant_api_key

# Initialize Qdrant client
qdrant_client = QdrantClient(url=url, api_key=api_key)


def get_sentenceTF_embeddings(sentences):
    model = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = []

    for chunk in sentences:
        embeddings.append(model.encode(chunk))
    return embeddings


def Embed_stenteceTF(sentence):
    model = SentenceTransformer('all-MiniLM-L12-v2')
    return model.encode(sentence)


messages = []


def custom_prompt(query: str):
    query_embedding_response = get_sentenceTF_embeddings(query)
    query_embedding = query_embedding_response[0].tolist()

    # Perform similarity search
    results = qdrant_client.search(
        collection_name="chatbot_phyc",
        query_vector=query_embedding,

    )

    # Extract the page content from the results
    source_knowledge = "\n".join([x.payload['text'] for x in results])

    # Create the augmented prompt
    augment_prompt = f"""Using the contexts below, answer the query,and dont mention the context explicitly:

    Additional Knowledge:
    {source_knowledge}

    Query: {query}"""

    return augment_prompt


def model_inference(user_prompt, chat_history):
    query = user_prompt["text"]
    prompt = {"role": "system", "content": custom_prompt(query)}
    messages.append(prompt)

    res = client.chat.completions.create(
        model="meta/llama-3.1-405b-instruct",
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )

    full_response = ""
    for chunk in res:
        if chunk.choices[0].delta.content is not None:
            full_response += chunk.choices[0].delta.content
    return full_response


# Create a chatbot interface
chatbot = gr.Chatbot(
    label="chatbot_phyc",
    avatar_images=[None, None],
    show_copy_button=True,
    likeable=True,
    layout="panel",
    height=400,
)
output = gr.Textbox(label="Prompt")
