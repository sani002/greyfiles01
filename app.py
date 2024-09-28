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

# ---- Neo4j Database Credentials ----
NEO4J_URI = "bolt+s://206d9625.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "TMRa604NneBDNaTfEH7ZhGqPFBhlHLrarLJQXwc83dg"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# ---- Environment Variables ----
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Streamlit App Setup ----
st.title("Grey Files Prototype 0.1")
st.caption("Ask questions regarding historical events, relations, and key dates on Bangladesh. Our database is still maturing. Please be kind. Haha!")
st.image('https://github.com/sani002/mkpapp/blob/main/Header.png?raw=true')

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

# ---- Recursive Directory Reader and Preprocessing ----
reader = SimpleDirectoryReader(input_dir="books")
all_docs = []
for docs in reader.iter_data():
    for doc in docs:
        # Preprocess document: convert text to uppercase for consistency
        doc.text = doc.text.upper()
        all_docs.append(doc)

# ---- Set up Embedding Model and LLM ----
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Groq(model="llama3-70b-8192", api_key="YOUR_GROQ_API_KEY")

# ---- Semantic Chunking of Documents ----
text_splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=300)
nodes = text_splitter.get_nodes_from_documents(all_docs, show_progress=True)

# ---- Index Creation (In-memory) ----
index = VectorStoreIndex(nodes, llm=llm, embed_model=embed_model)

# ---- Prompt Template ----
prompt_template = """
Use the following pieces of information to answer the user's question.
Give all possible answers from all the books regarding that question. If not found in your training books, answer from your pretained data.
Must mention the book or source and page number with each answer.

Context: {context}
Graph Insights: {graph_insights}
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

# ---- Combined Query Function ----
def combined_query(question, query_engine, driver):
    graph_insights = get_graph_insights(question, driver)
    query_prompt = prompt_template.format(context=context, graph_insights=graph_insights, question=question)
    response = query_engine.query(query_prompt)
    return response

# ---- Main Chat Interface ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Use st.chat_input to mimic a chat interface for user input
user_question = st.chat_input("Ask your question:")

if user_question:
    response = combined_query(user_question, index.as_query_engine(), driver)

    # Append question and response to the chat history
    st.session_state.chat_history.append({"user": user_question, "response": response})

# Display the chat history in a conversational manner
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["user"])
    with st.chat_message("assistant"):
        st.markdown(chat["response"])
