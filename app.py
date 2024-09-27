import os
import streamlit as st
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from neo4j import GraphDatabase
from llama_index.core.node_parser import SentenceSplitter

# ---- Neo4j Database Credentials ----
NEO4J_URI = "bolt+s://dc164147.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "w-rTj118fA1bv-ivSQVx5vIGnrmLx7B-8GTtXYIEZDw"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# ---- Environment Variables ---
os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Data ingestion: load all files from a directory
directory_path = "E:/Grey files/books"
reader = SimpleDirectoryReader(input_dir=directory_path)
documents = reader.load_data()

# Split the documents into nodes
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)

# Set up embedding model and LLM
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

from llama_index.core import Settings
# Create service context
Settings.llm = llm
Settings.embed_model = embed_model
# Create vector store index
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, node_parser=nodes)
vector_index.storage_context.persist(persist_dir="./storage_mini")

# Load the index from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
index = load_index_from_storage(storage_context)

# Set up the QueryEngine from the VectorStoreIndex
query_engine = index.as_query_engine()

# ---- PROMPT TEMPLATE ----
prompt_template = """
Use the following pieces of information to answer the user's question. Give all the possible answers you know from All the books regarding that question.
If you don't know the answer, answer from your previous knowledge but mention the source.
Mention the book and page number from the book with each answer.

Context: {context}
Graph Insights: {graph_insights}
Question: {question}

Answer the question and provide additional helpful information, based on the pieces of information and graph insights, if applicable. Be succinct.

Responses should be properly formatted to be easily read, like this:
General Overview:
Additional informations: (Points with page no and sources)
"""

# Define the context for your prompt
context = "This directory contains multiple documents providing important historical events, personal relations and characteristics and important dates of Bangladesh."

def get_graph_insights(question, driver):
    with driver.session() as session:
        # Query Neo4j for relevant insights based on the question
        result = session.run(
            """
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($question)
            OR toLower(n.value) CONTAINS toLower($question)
            OPTIONAL MATCH (n)-[r:ASSOCIATED_WITH]->(related)
            RETURN labels(n) AS node_labels, n.name AS name, n.value AS value, collect(related.name) AS related_names, collect(related.value) AS related_values
            """,
            question=question
        )

        insights = []
        for record in result:
            node_type = "Person" if "Person" in record["node_labels"] else "Event" if "Event" in record["node_labels"] else "Date"
            node_name_or_value = record["name"] if record["name"] else record["value"]
            related_info = []

            for related_name in record["related_names"]:
                if related_name:
                    related_info.append(related_name)
            for related_value in record["related_values"]:
                if related_value:
                    related_info.append(related_value)

            insight = f"{node_type}: {node_name_or_value}"
            if related_info:
                insight += f" is associated with: {', '.join(related_info)}"
            insights.append(insight)

        return "\n".join(insights) if insights else "No relevant insights found."

# Streamlit app
st.title("Grey Files Chatbot: Prototype 0.1")

# Initialize session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input box
user_question = st.text_input("Ask your question:")

if user_question:
    # Add the user question to the chat history
    st.session_state.history.append({"role": "user", "content": user_question})
    
    # Get insights from Neo4j
    graph_insights = get_graph_insights(user_question, driver)
    
    # Prepare query prompt
    query_prompt = prompt_template.format(context=context, graph_insights=graph_insights, question=user_question)
    
    # Get response from the LLM
    resp = query_engine.query(query_prompt)
    
    # Add the LLM response to the chat history
    st.session_state.history.append({"role": "assistant", "content": resp})

# Display the conversation
for message in st.session_state.history:
    if message["role"] == "user":
        st.write(f"**You**: {message['content']}")
    else:
        st.write(f"**Grey Files**: {message['content']}")
