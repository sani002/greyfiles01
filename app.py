from dotenv import dotenv_values
import os
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from neo4j import GraphDatabase
from llama_index.core.node_parser import SentenceSplitter
from pymongo import MongoClient  # Added for MongoDB integration
from datetime import datetime

# Streamlit page configuration
st.set_page_config(
    page_title="Grey Files 0.1",
    page_icon="🐦‍⬛",
    layout="wide",
)

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

# ---- MongoDB Atlas Connection ----
MONGO_URI = "mongodb+srv://smsakeefsani3:DQtEtUakz9fVv6Db@cluster0.bkwpm.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["greyfiles_db"]  # Replace with your database name
collection = db["chat_history"]  # Collection for chat history
user_collection = db["user_data"]  # Collection for storing user login data

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

# Save the api_key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Modify this function to save a single entry at a time
def save_chat_history_to_mongodb(entry):
    try:
        # Prepare the entry as a serializable document for MongoDB
        serializable_entry = {
            "user": entry["user"],
            "response": entry["response"],
            "feedback": entry["feedback"],
            "timestamp": entry.get("timestamp", datetime.now().isoformat())  # Use entry timestamp if available
        }
        
        # Insert the single chat message or suggestion into MongoDB
        collection.insert_one(serializable_entry)
    except Exception as e:
        st.error(f"Failed to save chat history: {e}")


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
Give all possible answers from all the books regarding that question. If not found in your training books, answer from your pretained data.
Must mention the book or source and page number with each answer.

Context: {context}
Graph Insights: {graph_insights}
Chat History: {chat_history}
Question: {question}

Answer concisely and provide additional helpful insights if applicable.
"""

context = """You are trained on these books on historical events, relations, and key dates regarding Bangladesh:

Bangladesh: A Legacy of Blood
Author: Anthony Mascarenhas

The Blood Telegram: Nixon, Kissinger, and a Forgotten Genocide
Author: Gary J. Bass

Liberation War Debates in the UK Parliament
Author: UK Parliament

The Cruel Birth of Bangladesh Through the Eyes of America
Author: Adit Mahmood

The Rape of Bangladesh
Author: ANTHONY MASCARENHAS

Pakistan Failure in National Integration
Author: Rounaq Jahan
"""

# ---- Graph Query Function ----
def get_graph_insights(question, driver):
    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($question)
                OR toLower(n.value) CONTAINS toLower($question)
                OR toLower(n.title) CONTAINS toLower($question)
                OPTIONAL MATCH (n)-[r]-(related)
                RETURN labels(n) AS node_labels,
                       n.name AS name,
                       n.value AS value,
                       n.title AS title,
                       type(r) AS relationship,
                       collect(related.name) AS related_names,
                       collect(related.value) AS related_values,
                       collect(related.title) AS related_titles,
                       collect(labels(related)) AS related_labels
                """,
                question=question
            )

            insights = []
            for record in result:
                node_labels = record["node_labels"]
                node_type = "Unknown"
                node_name_or_value = None

                # Identify node type based on labels
                if "Person" in node_labels:
                    node_type = "Person"
                    node_name_or_value = record["name"]
                elif "Event" in node_labels:
                    node_type = "Event"
                    node_name_or_value = record["name"]
                elif "Date" in node_labels:
                    node_type = "Date"
                    node_name_or_value = record["value"]
                elif "Location" in node_labels:
                    node_type = "Location"
                    node_name_or_value = record["name"]
                elif "Organization" in node_labels:
                    node_type = "Organization"
                    node_name_or_value = record["name"]
                elif "WorkOfArt" in node_labels:
                    node_type = "WorkOfArt"
                    node_name_or_value = record["title"]

                related_info = []
                # Collect related nodes' names, values, and titles
                for related_name, related_value, related_title, related_label in zip(
                    record["related_names"],
                    record["related_values"],
                    record["related_titles"],
                    record["related_labels"]
                ):
                    # Determine the type of related node
                    related_type = "Unknown"
                    if "Person" in related_label:
                        related_type = "Person"
                    elif "Event" in related_label:
                        related_type = "Event"
                    elif "Date" in related_label:
                        related_type = "Date"
                    elif "Location" in related_label:
                        related_type = "Location"
                    elif "Organization" in related_label:
                        related_type = "Organization"

                    # Choose the appropriate field (name, value, or title) to represent the related node
                    related_field = related_name or related_value or related_title
                    if related_field:
                        related_info.append(f"{related_type}: {related_field}")

                # Construct the insight based on node and relationships
                if node_name_or_value:
                    insight = f"{node_type}: {node_name_or_value}"
                    if related_info:
                        insight += f" is associated with: {', '.join(related_info)}"
                    if record["relationship"]:
                        insight += f" (Relationship: {record['relationship']})"
                    insights.append(insight)

            return "\n".join(insights) if insights else "No relevant insights found."

    except Exception as e:
        st.error(f"Error fetching graph insights: {e}")
        return "No graph insights available due to an error."

# ---- Combined Query Function with Chat History ----
def combined_query(question, query_engine, driver, chat_history):
    graph_insights = get_graph_insights(question, driver)
    
    # Format the chat history for the prompt
    formatted_chat_history = "\n".join(
        f"User: {entry['user']}\nAssistant: {entry['response']}" for entry in chat_history
    )
    
    query_prompt = prompt_template.format(
        context=context,
        graph_insights=graph_insights,
        chat_history=formatted_chat_history,
        question=question
    )
    
    response = query_engine.query(query_prompt)
    return response

# ---- Session State Initialization ----
if "username" not in st.session_state:
    st.session_state.username = ""
if "form" not in st.session_state:
    st.session_state.form = "login_form"  # Start with login form
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # Track login status

# ---- Helper Functions ----
def toggle_form():
    """Switch between login and signup forms."""
    st.session_state.form = 'signup_form' if st.session_state.form == 'login_form' else 'login_form'

def user_update(name):
    """Update session state with the logged-in username."""
    st.session_state.username = name
    st.session_state.logged_in = True  # Set logged-in status

def user_logout():
    """Clear user session data to log out."""
    st.session_state.username = ""
    st.session_state.logged_in = False
    st.session_state.form = "login_form"  # Reset to login form

@st.cache_data
def validate_user(username, password):
    """Cached function to validate user credentials with the database."""
    return user_collection.find_one({'username': username, 'password': password})

# ---- Login and Signup Interface ----
if not st.session_state.logged_in:
    if st.session_state.form == 'login_form':
        # Centered Login Form
        st.title("Welcome to Grey Files")
        st.subheader("Please sign in")

        login_form = st.form(key='login_form', clear_on_submit=True)
        username = login_form.text_input(label='Username')
        password = login_form.text_input(label='Password', type='password')
        login_button = login_form.form_submit_button(label='Sign In')
        
        # Login button functionality
        if login_button:
            user_data = validate_user(username, password)
            if user_data:
                user_update(username)
                st.success(f"Welcome, {username}!")  # Successful login message
            else:
                st.error("Invalid username or password. Please try again.")

        # Button to switch to Signup form
        st.markdown("Don't have an account?")
        if st.button("Sign up!"):
            toggle_form()

    elif st.session_state.form == 'signup_form':
        # Centered Signup Form
        st.title("Create an Account")
        
        signup_form = st.form(key='signup_form', clear_on_submit=True)
        new_username = signup_form.text_input(label='Username*')
        new_user_email = signup_form.text_input(label='Email Address*')
        new_user_location = signup_form.text_input(label='Location')
        new_user_profession = signup_form.text_input(label='Profession')
        new_user_password = signup_form.text_input(label='Password*', type='password')
        user_password_conf = signup_form.text_input(label='Confirm Password*', type='password')
        signup_button = signup_form.form_submit_button(label='Sign Up')
        
        # Signup button functionality
        if signup_button:
            if '' in [new_username, new_user_email, new_user_password, user_password_conf]:
                st.error('Please fill in all required fields.')
            elif new_user_password != user_password_conf:
                st.error("Passwords do not match.")
            elif user_collection.find_one({'username': new_username}):
                st.error('Username already exists.')
            elif user_collection.find_one({'email': new_user_email}):
                st.error('Email is already registered.')
            else:
                # Add the new user to the database
                user_data = {
                    "username": new_username,
                    "email": new_user_email,
                    "location": new_user_location,
                    "profession": new_user_profession,
                    "password": new_user_password,
                    "created_at": datetime.now()
                }
                user_collection.insert_one(user_data)
                user_update(new_username)
                st.success("Registration successful! You are now logged in.")
        
        # Button to switch back to Login form
        st.markdown("Already have an account?")
        if st.button("Sign in!"):
            toggle_form()

# ---- Main App Content (only for logged-in users) ----
@st.cache_data
if st.session_state.logged_in:
    st.sidebar.title("Account Options")
    if st.sidebar.button("Log Out"):
        user_logout()  # Call function to log out and reset state
        st.success("You have been logged out.")

    st.title("Grey Files Prototype 0.1")
    st.image('https://github.com/sani002/greyfiles01/blob/main/Grey%20Files.png?raw=true')
    st.caption("Ask questions regarding historical events, relations, and key dates on Bangladesh. Our database is still maturing. Please be kind. Haha!")
    # Sidebar for suggestions
    with st.sidebar:
        st.header("Suggestions")
        suggestion = st.text_area("Have a suggestion? Let us know!")
        if st.button("Submit Suggestion"):
            if suggestion:
                # Structure the suggestion entry as a dictionary
                suggestion_entry = {
                    "user": "User Suggestion",
                    "response": suggestion,
                    "feedback": None,
                    "timestamp": datetime.now().isoformat()  # Add timestamp for suggestion
                }
                
                # Add the suggestion to chat history
                st.session_state.chat_history.append(suggestion_entry)
                
                # Save the suggestion directly
                save_chat_history_to_mongodb(suggestion_entry)
                
                st.success("Thank you for your suggestion!")
            else:
                st.warning("Please enter a suggestion before submitting.")



    # Main Chat Interface with Like/Dislike Buttons
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.chat_input("Ask your question:")

    # Process the user's question and save only the latest message
    if user_question:
        response = combined_query(user_question, index.as_query_engine(), driver, st.session_state.chat_history)
        
        # Append question and response to the chat history
        latest_entry = {
            "user": user_question,
            "response": str(response),
            "feedback": None  # Placeholder for feedback
        }
        st.session_state.chat_history.append(latest_entry)
        
        # Save only the latest message (real-time saving of the latest entry)
        save_chat_history_to_mongodb(latest_entry)


    # Display the chat history in a conversational manner (skip suggestions)
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
                        # Save only the updated entry with feedback
                        save_chat_history_to_mongodb(st.session_state.chat_history[idx])
                with col2:
                    if st.button("Dislike", key=f"dislike_{idx}"):
                        st.session_state.chat_history[idx]["feedback"] = "dislike"
                        # Save only the updated entry with feedback
                        save_chat_history_to_mongodb(st.session_state.chat_history[idx])