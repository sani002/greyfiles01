import os
from datetime import datetime
import json
import pymongo
import streamlit as st
from bson.objectid import ObjectId
from dotenv import dotenv_values

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from neo4j import GraphDatabase
from llama_index.core.node_parser import SentenceSplitter

# ---- Streamlit Page Configuration ----
st.set_page_config(
    page_title="Grey Files 0.1",
    page_icon="🐦‍⬛",
    layout="wide",
)

# ---- MongoDB Connection Setup ----
DATABASE_NAME = "greyfiles"
COLLECTION_NAME = "chat_history"

# Initialize MongoDB client
client = pymongo.MongoClient("mongodb+srv://<username>:<password>@cluster0.bkwpm.mongodb.net/<dbname>?retryWrites=true&w=majority&tls=true&tlsAllowInvalidCertificates=true")
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

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

# ---- Function to Insert Chat Data into MongoDB ----
def insert_chat_data(user, response, feedback=None, suggestion=False):
    chat_entry = {
        "_id": str(ObjectId()),  # Generate a unique ObjectId for MongoDB _id field
        "user": "User Suggestion" if suggestion else user,
        "response": response,
        "feedback": feedback,
        "timestamp": datetime.utcnow().isoformat()  # Save timestamp in UTC format
    }
    collection.insert_one(chat_entry)

# ---- Streamlit App Setup ----
st.title("Grey Files Prototype 0.1")
st.image('https://github.com/sani002/greyfiles01/blob/main/Grey%20Files.png?raw=true')
st.caption("Ask questions regarding historical events, relations, and key dates on Bangladesh.")

# Ensure the chat history folder exists
if "session_file" not in st.session_state:
    session_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("chat_history", exist_ok=True)
    st.session_state.session_file = os.path.join("chat_history", f"chat_{session_start_time}.json")

# ---- Load and Preprocess Documents ----
@st.cache_data
def load_and_preprocess_documents():
    reader = SimpleDirectoryReader(input_dir="books")
    all_docs = []
    for docs in reader.iter_data():
        for doc in docs:
            doc.text = doc.text.upper()  # Convert text to uppercase for consistency
            all_docs.append(doc)
    return all_docs

# ---- Set up Embedding Model and LLM ----
@st.cache_data
def setup_model_and_index(_all_docs):
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)
    
    from llama_index.core import Settings
    Settings.llm = llm
    Settings.embed_model = embed_model

    # Semantic chunking
    text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=300)
    nodes = text_splitter.get_nodes_from_documents(_all_docs, show_progress=True)
    index = VectorStoreIndex(nodes, llm=llm, embed_model=embed_model)
    
    return index

# Load documents and create index
all_docs = load_and_preprocess_documents()
index = setup_model_and_index(all_docs)

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

context = "You search from every related information from the graph insights!"

# ---- Graph Query Function ----
import re

def get_graph_insights(question, driver):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($question)
            OR toLower(n.title) CONTAINS toLower($question)
            OR toLower(n.description) CONTAINS toLower($question)
            OR toLower(n.value) CONTAINS toLower($question)
            OPTIONAL MATCH (n)-[r]-(related)
            RETURN labels(n) AS node_labels, n.name AS name,
                   n.title AS title, n.description AS description,
                   n.value AS value, type(r) AS relationship,
                   collect(related.name) AS related_names,
                   collect(related.title) AS related_titles,
                   collect(related.description) AS related_descriptions
            """,
            question=question
        )

        insights = []
        for record in result:
            node_labels = record["node_labels"]
            node_value = record.get("name") or record.get("title") or record.get("description") or record.get("value")
            related_info = [name for name in record["related_names"] if name]
            insight = f"{node_labels[0]}: {node_value} is associated with {', '.join(related_info)}" if related_info else ""
            insights.append(insight)

        return "\n".join(insights) if insights else "No relevant insights found."

# ---- Combined Query Function with Chat History ----
def combined_query(question, query_engine, driver, chat_history):
    graph_insights = get_graph_insights(question, driver)
    
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

# Sidebar for suggestions
with st.sidebar:
    st.header("Suggestions")
    suggestion = st.text_area("Have a suggestion? Let us know!")
    if st.button("Submit Suggestion"):
        if suggestion:
            insert_chat_data("User Suggestion", suggestion, suggestion=True)
            st.success("Thank you for your suggestion!")
        else:
            st.warning("Please enter a suggestion before submitting.")

# Main Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input("Ask your question:")

if user_question:
    response = combined_query(user_question, index.as_query_engine(), driver, st.session_state.chat_history)
    
    st.session_state.chat_history.append({
        "user": user_question,
        "response": str(response),
        "feedback": None
    })
    
    insert_chat_data(user_question, response)

# Display the chat history
for idx, chat in enumerate(st.session_state.chat_history):
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
                    collection.update_one({"_id": chat["_id"]}, {"$set": {"feedback": "like"}})
            with col2:
                if st.button("Dislike", key=f"dislike_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "dislike"
                    collection.update_one({"_id": chat["_id"]}, {"$set": {"feedback": "dislike"}})
        else:
            # Display feedback status
            st.write("Feedback given:", chat["feedback"])
