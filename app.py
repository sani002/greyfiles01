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
def get_graph_insights(entity_name, driver):
    with driver.session() as session:
        # Cypher query to match entities without specifying a type, and fetch related entities up to many levels
        cypher_query = f"""
        MATCH (entity)
        WHERE entity.name CONTAINS $entity_name OR entity.description CONTAINS $entity_name
        // First-level relationships
        OPTIONAL MATCH (entity)-[r1]-(related1)
        // Second-level relationships
        OPTIONAL MATCH (related1)-[r2]-(related2)
        // Third-level relationships
        OPTIONAL MATCH (related2)-[r3]-(related3)
        // Fourth-level relationships
        OPTIONAL MATCH (related3)-[r4]-(related4)
        // Fifth-level relationships
        OPTIONAL MATCH (related4)-[r5]-(related5)
        // Sixth-level relationships
        OPTIONAL MATCH (related5)-[r6]-(related6)
        // Seventh-level relationships
        OPTIONAL MATCH (related6)-[r7]-(related7)
        // Eighth-level relationships
        OPTIONAL MATCH (related7)-[r8]-(related8)
        // Ninth-level relationships
        OPTIONAL MATCH (related8)-[r9]-(related9)
        // Tenth-level relationships
        OPTIONAL MATCH (related9)-[r10]-(related10)
        // Eleventh-level relationships
        OPTIONAL MATCH (related10)-[r11]-(related11)
        // Twelfth-level relationships
        OPTIONAL MATCH (related11)-[r12]-(related12)
        // Thirteenth-level relationships
        OPTIONAL MATCH (related12)-[r13]-(related13)
        // Fourteenth-level relationships
        OPTIONAL MATCH (related13)-[r14]-(related14)
        // Fifteenth-level relationships
        OPTIONAL MATCH (related14)-[r15]-(related15)
        RETURN DISTINCT entity, r1, related1, r2, related2, r3, related3, r4, related4, r5, related5, 
                         r6, related6, r7, related7, r8, related8, r9, related9, r10, related10, 
                         r11, related11, r12, related12, r13, related13, r14, related14, r15, related15
        """

        # Execute the query to fetch all connected nodes and relationships
        result = session.run(cypher_query, entity_name=entity_name)

        # Initialize an empty dictionary to hold the results
        insights = {
            "main_entity": {"name": entity_name},
            "relationships": [],
            "related_entities": []
        }

        # Process the result and build the insights data structure
        for record in result:
            # For each level, capture relationships and entities if they exist
            rel1 = record.get('r1')
            related1 = record.get('related1')
            rel2 = record.get('r2')
            related2 = record.get('related2')
            rel3 = record.get('r3')
            related3 = record.get('related3')
            rel4 = record.get('r4')
            related4 = record.get('related4')
            rel5 = record.get('r5')
            related5 = record.get('related5')
            rel6 = record.get('r6')
            related6 = record.get('related6')
            rel7 = record.get('r7')
            related7 = record.get('related7')
            rel8 = record.get('r8')
            related8 = record.get('related8')
            rel9 = record.get('r9')
            related9 = record.get('related9')
            rel10 = record.get('r10')
            related10 = record.get('related10')
            rel11 = record.get('r11')
            related11 = record.get('related11')
            rel12 = record.get('r12')
            related12 = record.get('related12')
            rel13 = record.get('r13')
            related13 = record.get('related13')
            rel14 = record.get('r14')
            related14 = record.get('related14')
            rel15 = record.get('r15')
            related15 = record.get('related15')

            # Capture relationships and entities for all 15 levels
            if related1:
                insights["related_entities"].append({
                    "entity": related1["name"],
                    "relationship": str(rel1.type()) if rel1 else "Unknown"
                })

            if related2:
                insights["related_entities"].append({
                    "entity": related2["name"],
                    "relationship": str(rel2.type()) if rel2 else "Unknown"
                })

            if related3:
                insights["related_entities"].append({
                    "entity": related3["name"],
                    "relationship": str(rel3.type()) if rel3 else "Unknown"
                })

            if related4:
                insights["related_entities"].append({
                    "entity": related4["name"],
                    "relationship": str(rel4.type()) if rel4 else "Unknown"
                })

            if related5:
                insights["related_entities"].append({
                    "entity": related5["name"],
                    "relationship": str(rel5.type()) if rel5 else "Unknown"
                })

            if related6:
                insights["related_entities"].append({
                    "entity": related6["name"],
                    "relationship": str(rel6.type()) if rel6 else "Unknown"
                })

            if related7:
                insights["related_entities"].append({
                    "entity": related7["name"],
                    "relationship": str(rel7.type()) if rel7 else "Unknown"
                })

            if related8:
                insights["related_entities"].append({
                    "entity": related8["name"],
                    "relationship": str(rel8.type()) if rel8 else "Unknown"
                })

            if related9:
                insights["related_entities"].append({
                    "entity": related9["name"],
                    "relationship": str(rel9.type()) if rel9 else "Unknown"
                })

            if related10:
                insights["related_entities"].append({
                    "entity": related10["name"],
                    "relationship": str(rel10.type()) if rel10 else "Unknown"
                })

            if related11:
                insights["related_entities"].append({
                    "entity": related11["name"],
                    "relationship": str(rel11.type()) if rel11 else "Unknown"
                })

            if related12:
                insights["related_entities"].append({
                    "entity": related12["name"],
                    "relationship": str(rel12.type()) if rel12 else "Unknown"
                })

            if related13:
                insights["related_entities"].append({
                    "entity": related13["name"],
                    "relationship": str(rel13.type()) if rel13 else "Unknown"
                })

            if related14:
                insights["related_entities"].append({
                    "entity": related14["name"],
                    "relationship": str(rel14.type()) if rel14 else "Unknown"
                })

            if related15:
                insights["related_entities"].append({
                    "entity": related15["name"],
                    "relationship": str(rel15.type()) if rel15 else "Unknown"
                })

        # Check if insights were found and return them, or provide a default message if no data was retrieved.
        if insights["related_entities"]:
            return insights
        else:
            return "No relevant insights found."




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
