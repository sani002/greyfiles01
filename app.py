import os
import json
from datetime import datetime
import streamlit as st
from pymongo import MongoClient
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from neo4j import GraphDatabase
from llama_index.core.node_parser import SentenceSplitter
import re
from dotenv import dotenv_values

# ---- MongoDB Setup ----
MONGODB_URL = "mongodb+srv://smsakeefsani3:DQtEtUakz9fVv6Db@cluster0.bkwpm.mongodb.net/"
MONGODB_DATABASE = "greyfiles"
MONGODB_COLLECTION = "chat_history"

# Create a MongoDB client
mongo_client = MongoClient(MONGODB_URL)
mongo_db = mongo_client[MONGODB_DATABASE]
mongo_collection = mongo_db[MONGODB_COLLECTION]

# Function to save query, response, and feedback to MongoDB
def save_to_mongodb(user_question, response, feedback=None):
    document = {
        "user_question": user_question,
        "response": response,
        "timestamp": datetime.now(),
        "feedback": feedback  # Include feedback if provided
    }
    mongo_collection.insert_one(document)

# Function to save the chat history in real-time
def save_chat_history_real_time(chat_history, file_path):
    try:
        serializable_chat_history = []
        for entry in chat_history:
            serializable_chat_history.append({
                "user": entry["user"],
                "response": str(entry["response"]),
                "feedback": entry["feedback"]
            })

        # Save the serializable chat history to a JSON file
        with open(file_path, "w") as f:
            json.dump(serializable_chat_history, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")

# Streamlit page configuration
st.set_page_config(
    page_title="Grey Files 0.1",
    page_icon="🐦‍⬛",
    layout="wide",
)

# Ensure the chat history folder exists
if "session_file" not in st.session_state:
    session_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a folder if it doesn't exist
    os.makedirs("chat_history", exist_ok=True)

    # Save the file in the chat_history folder
    st.session_state.session_file = os.path.join("chat_history", f"chat_{session_start_time}.json")

# ---- Hide Streamlit Default Elements ----
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {
                    content:'This app provides answers based on documents regarding Bangladesh history.'; 
                    visibility: visible;
                    display: block;
                    position: relative;
                    padding: 5px;
                    top: 2px;
                }
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ---- Neo4j Database Credentials ----
NEO4J_URI = "bolt+s://82c0dc6b.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "yZ5fUk04w5s4zrRHb0P9x2q9gR72miCPbY103DV90Ds"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load secrets
try:
    secrets = dotenv_values(".env")  # for dev env
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except:
    secrets = st.secrets  # for streamlit deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# save the api_key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---- Streamlit App Setup ----
st.title("Grey Files Prototype 0.1")
st.image('https://github.com/sani002/greyfiles01/blob/main/Grey%20Files.png?raw=true')
st.caption("Ask questions regarding historical events, relations, and key dates on Bangladesh. Our database is still maturing. Please be kind. Haha!")

# ---- Recursive Directory Reader and Preprocessing ----
@st.cache_data
def load_and_preprocess_documents():
    reader = SimpleDirectoryReader(input_dir="books")
    all_docs = []
    for docs in reader.iter_data():
        for doc in docs:
            # Preprocess document: convert text to uppercase for consistency
            doc.text = doc.text.upper()
            all_docs.append(doc)
    return all_docs

# ---- Set up Embedding Model and LLM ----
@st.cache_data
def setup_model_and_index(_all_docs):  # The leading underscore is added to avoid caching this argument
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    from llama_index.core import Settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # ---- Semantic Chunking of Documents ----
    text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=300)
    nodes = text_splitter.get_nodes_from_documents(_all_docs, show_progress=True)

    # ---- Index Creation (In-memory) ----
    index = VectorStoreIndex(nodes, llm=llm, embed_model=embed_model)

    return index

# Load documents and create index
all_docs = load_and_preprocess_documents()
index = setup_model_and_index(all_docs)  # Argument now passed as _all_docs in the function

# ---- Prompt Template ----
prompt_template = """
Use the following pieces of information to answer the user's question.
Give all possible answers from all the books regarding that question. If not found in your training books, answer from your pretrained data.
Must mention the book or source and page number with each answer.

Context: {context}
Graph Insights: {graph_insights}
Chat History: {chat_history}
Question: {question}

Answer concisely and provide additional helpful insights if applicable.
"""

context = """You search from every related information from the graph insights!"""

# ---- Graph Query Function ----
# Enhanced function to fetch deeply connected insights from the graph
def get_graph_insights(question, driver):
    with driver.session() as session:
        # Query Neo4j for relevant insights based on the question
        result = session.run(
            """
            // Match nodes that may contain the question keyword in key properties
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($question)  // For entities with 'name' properties
            OR toLower(n.title) CONTAINS toLower($question)  // For Works of Art, Laws, etc.
            OR toLower(n.description) CONTAINS toLower($question)  // For nodes with 'description' properties
            OR toLower(n.value) CONTAINS toLower($question)  // For Dates
            OPTIONAL MATCH (n)-[r]-(related)  // Optional match for related nodes
            RETURN labels(n) AS node_labels,
                   n.name AS name,
                   n.title AS title,
                   n.description AS description,
                   n.value AS value,
                   type(r) AS relationship,
                   collect(related.name) AS related_names,
                   collect(related.title) AS related_titles,
                   collect(related.description) AS related_descriptions
            """,
            question=question
        )

        # Prepare the insights from the query results
        insights = []
        for record in result:
            # Determine the type of node (Person, Event, Date, Location, WorkOfArt, etc.) based on labels
            node_labels = record["node_labels"]
            node_type = "Unknown"
            node_value = None

            # Identify the type of node and relevant property
            if "Person" in node_labels:
                node_type = "Person"
                node_value = record["name"]
            elif "Event" in node_labels:
                node_type = "Event"
                node_value = record["name"]
            elif "Date" in node_labels:
                node_type = "Date"
                node_value = record["value"]
            elif "Location" in node_labels:
                node_type = "Location"
                node_value = record["name"]
            elif "WorkOfArt" in node_labels:
                node_type = "WorkOfArt"
                node_value = record["title"]
            elif "Law" in node_labels:
                node_type = "Law"
                node_value = record["title"]
            elif "Reform" in node_labels:
                node_type = "Reform"
                node_value = record["description"]
            elif "Change" in node_labels:
                node_type = "Change"
                node_value = record["description"]
            elif "Movement" in node_labels:
                node_type = "Movement"
                node_value = record["name"]
            elif "Strike" in node_labels:
                node_type = "Strike"
                node_value = record["name"]
            elif "Crisis" in node_labels:
                node_type = "Crisis"
                node_value = record["name"]
            elif "Rebellion" in node_labels:
                node_type = "Rebellion"
                node_value = record["name"]

            # Format insights based on the type of node
            if node_value:
                insights.append(f"{node_type}: {node_value} (Related: {', '.join(record['related_names'] + record['related_titles'] + record['related_descriptions'])})")

        return insights

# ---- Combined Query Function ----
def combined_query(user_question, index, driver, chat_history):
    # Get the context from the vector store index
    context = index.query(user_question)
    # Get insights from Neo4j
    graph_insights = get_graph_insights(user_question, driver)

    # Create the full prompt
    prompt = prompt_template.format(
        context=context,
        graph_insights="\n".join(graph_insights),
        chat_history="\n".join(f"User: {entry['user']}\nAssistant: {entry['response']}" for entry in chat_history),
        question=user_question
    )

    # Generate the response from the model
    response = index.query(prompt)
    return response

# ---- Chat History Management ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input for questions
user_question = st.chat_input("Ask your question:")

if user_question:
    # Get response from the combined query function
    response = combined_query(user_question, index, driver, st.session_state.chat_history)

    # Append question and response to the chat history
    st.session_state.chat_history.append({
        "user": user_question,
        "response": str(response),
        "feedback": None  # Placeholder for feedback
    })

    # Save to MongoDB without feedback initially
    save_to_mongodb(user_question, response)  # Save query and response to MongoDB

    # Save chat after each message (real-time saving)
    save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)

# Display the chat history in a conversational manner
for idx, chat in enumerate(st.session_state.chat_history):
    if chat["user"] == "User Suggestion":
        # Skip displaying suggestions in the chat UI
        continue

    with st.chat_message("user", avatar="🦉"):
        st.markdown(chat["user"])
    with st.chat_message("assistant", avatar="🐦‍⬛"):
        st.markdown(chat["response"])
        
        # Add Like/Dislike buttons for feedback
        col1, col2 = st.columns([1, 1])
        if chat["feedback"] is None:
            with col1:
                if st.button("Like", key=f"like_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "like"
                    save_to_mongodb(chat["user"], chat["response"], "like")  # Save with feedback
                    save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)
            with col2:
                if st.button("Dislike", key=f"dislike_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "dislike"
                    save_to_mongodb(chat["user"], chat["response"], "dislike")  # Save with feedback
                    save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)
        else:
            # After feedback is given, disable buttons or change their appearance
            with col1:
                st.button("Liked", disabled=True, key=f"liked_{idx}")
            with col2:
                st.button("Disliked", disabled=True, key=f"disliked_{idx}")

# Close the Neo4j driver when done
driver.close()
