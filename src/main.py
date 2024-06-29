import os

import streamlit as st
from neo4j import GraphDatabase

from dotenv import load_dotenv
load_dotenv()

from test_neo4j_with_vectorstore import get_person_by_embedding
from embeddings.openai_embeddings import EmbeddingsProcessor

emb_processor = EmbeddingsProcessor()

# Neo4j AuraDB connection details
uri = os.getenv("NEO4J_URI")
password = os.getenv("NEO4J_KEY")
username = 'neo4j'

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# # React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# response = f"Echo: {prompt}"
# # Display assistant response in chat message container
# with st.chat_message("assistant"):
#     st.markdown(response)
# # Add assistant response to chat history
# st.session_state.messages.append({"role": "assistant", "content": response})

if prompt is not None:
    emb = emb_processor.get_embedding(prompt)

    # Connect to Neo4j AuraDB
    driver = GraphDatabase.driver(uri, auth=(username, password))
    emb_processor = EmbeddingsProcessor()
    with driver.session() as session:
        nodes = session.execute_read(get_person_by_embedding, emb, 1)

    # Close the driver connection
    driver.close()    # Display assistant response in chat message container

    with st.chat_message("neo4j"):
        st.markdown(nodes)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": nodes})


    
