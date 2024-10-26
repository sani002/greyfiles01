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
import json
from datetime import datetime


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

context = """You search from every related information from the graph insights!"""

# ---- Graph Query Function ----
import re

# Categorize the question based on its type (Who/What/When/Where)
def categorize_question(question):
    question = question.lower()
    if re.search(r"\bwho\b", question):
        return 'Person'
    elif re.search(r"\bwhat\b", question):
        return 'General'
    elif re.search(r"\bwhen\b", question):
        return 'Date'
    elif re.search(r"\bwhere\b", question):
        return 'Location'
    else:
        return 'General'


# Function to fetch detailed insights from the graph
import re

# Categorize the question based on its type (Who/What/When/Where)
def categorize_question(question):
    question = question.lower()
    if re.search(r"\bwho\b", question):
        return 'Person'
    elif re.search(r"\bwhat\b", question):
        return 'General'
    elif re.search(r"\bwhen\b", question):
        return 'Date'
    elif re.search(r"\bwhere\b", question):
        return 'Location'
    else:
        return 'General'

# Enhanced function to fetch deeply connected insights from the graph
def get_graph_insights(question, driver):
    # Categorize the question to guide the query
    category = categorize_question(question)

    with driver.session() as session:
        # Cypher query to fetch nodes and relationships, expanding to 4 levels of depth, all relationships
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($question)
        OR toLower(n.title) CONTAINS toLower($question)
        OR toLower(n.description) CONTAINS toLower($question)
        OR toLower(n.value) CONTAINS toLower($question)

        // Traverse up to 4 levels of relationships to gather deep insights, without limiting by label or relationship type
        CALL apoc.path.expandConfig(n, {
            minLevel: 1,
            maxLevel: 4,
            relationshipFilter: ">",  // Expand along all relationships
            labelFilter: ">Person|>Event|>Location|>WorkOfArt|>ScientificDiscovery|>PoliticalParty|>Technology",
            uniqueness: "NODE_PATH"  // Ensure uniqueness of paths traversed
        }) YIELD path
        WITH path, nodes(path) AS allNodes, relationships(path) AS allRels
        RETURN allNodes, allRels
        """

        # Run the query to retrieve nodes and relationships
        result = session.run(query, question=question)

        # Prepare the insights from the query results
        insights = []
        for record in result:
            nodes = record['allNodes']
            relationships = record['allRels']

            # Track nodes and relationships to avoid duplicates
            seen_nodes = set()
            seen_relationships = set()

            # Process nodes: capture name, title, description, and any other relevant property
            for node in nodes:
                node_labels = list(node.labels)
                node_info = None

                # Handle various node types and gather the most relevant properties for each
                if "Person" in node_labels:
                    node_info = f"Person: {node.get('name', 'Unknown')}"
                elif "Event" in node_labels:
                    node_info = f"Event: {node.get('name', 'Unknown')}"
                elif "Location" in node_labels:
                    node_info = f"Location: {node.get('name', 'Unknown')}"
                elif "WorkOfArt" in node_labels:
                    node_info = f"Work of Art: {node.get('title', 'Unknown')}"
                elif "ScientificDiscovery" in node_labels:
                    node_info = f"Scientific Discovery: {node.get('title', 'Unknown')}"
                elif "PoliticalParty" in node_labels:
                    node_info = f"Political Party: {node.get('name', 'Unknown')}"
                elif "Technology" in node_labels:
                    node_info = f"Technology: {node.get('name', 'Unknown')}"
                elif "Law" in node_labels:
                    node_info = f"Law: {node.get('title', 'Unknown')}"
                elif "Reform" in node_labels:
                    node_info = f"Reform: {node.get('description', 'Unknown')}"
                elif "Change" in node_labels:
                    node_info = f"Change: {node.get('description', 'Unknown')}"
                elif "Crisis" in node_labels:
                    node_info = f"Crisis: {node.get('name', 'Unknown')}"
                elif "Trade" in node_labels:
                    node_info = f"Trade: {node.get('name', 'Unknown')}"
                elif "Empire" in node_labels:
                    node_info = f"Empire: {node.get('name', 'Unknown')}"
                elif "Dynasty" in node_labels:
                    node_info = f"Dynasty: {node.get('name', 'Unknown')}"
                elif "Invasion" in node_labels:
                    node_info = f"Invasion: {node.get('name', 'Unknown')}"
                elif "Colonization" in node_labels:
                    node_info = f"Colonization: {node.get('name', 'Unknown')}"
                elif "Rebellion" in node_labels:
                    node_info = f"Rebellion: {node.get('name', 'Unknown')}"
                elif "Religion" in node_labels:
                    node_info = f"Religion: {node.get('name', 'Unknown')}"
                elif "Art" in node_labels:
                    node_info = f"Art: {node.get('name', 'Unknown')}"
                elif "Architecture" in node_labels:
                    node_info = f"Architecture: {node.get('name', 'Unknown')}"

                if node_info and node_info not in seen_nodes:
                    insights.append(node_info)
                    seen_nodes.add(node_info)

            # Process relationships between nodes, ensuring no duplicate relationships
            for relationship in relationships:
                start_node = relationship.start_node.get('name', relationship.start_node.get('title', 'Unknown'))
                end_node = relationship.end_node.get('name', relationship.end_node.get('title', 'Unknown'))
                rel_type = type(relationship).__name__

                rel_info = f"Relationship: {start_node} --[{rel_type}]--> {end_node}"
                if rel_info not in seen_relationships:
                    insights.append(rel_info)
                    seen_relationships.add(rel_info)

        # Format the results
        formatted_insights = "\n".join(insights) if insights else "No relevant insights found."

        return formatted_insights


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

# Sidebar for suggestions
with st.sidebar:
    st.header("Suggestions")
    suggestion = st.text_area("Have a suggestion? Let us know!")
    if st.button("Submit Suggestion"):
        if suggestion:
            # Add the suggestion to the chat history
            st.session_state.chat_history.append({
                "user": "User Suggestion",
                "response": suggestion,
                "feedback": None
            })
            
            # Save the suggestion in real-time
            save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)
            
            st.success("Thank you for your suggestion!")
        else:
            st.warning("Please enter a suggestion before submitting.")


# Main Chat Interface with Like/Dislike Buttons
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.chat_input("Ask your question:")

if user_question:
    response = combined_query(user_question, index.as_query_engine(), driver, st.session_state.chat_history)
    
    # Append question and response to the chat history
    st.session_state.chat_history.append({
        "user": user_question,
        "response": str(response),
        "feedback": None  # Placeholder for feedback
    })
    
    # Save chat after each message (real-time saving)
    save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)

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
                    save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)
            with col2:
                if st.button("Dislike", key=f"dislike_{idx}"):
                    st.session_state.chat_history[idx]["feedback"] = "dislike"
                    save_chat_history_real_time(st.session_state.chat_history, st.session_state.session_file)
        else:
            # After feedback is given, disable buttons or change their appearance
            with col1:
                st.button("Liked", disabled=True, key=f"liked_{idx}")
            with col2:
                st.button("Disliked", disabled=True, key=f"disliked_{idx}")
