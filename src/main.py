import os

import streamlit as st
from streamlit_agraph import agraph
from neo4j import GraphDatabase
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from embeddings.openai_embeddings import EmbeddingsProcessor
from graph.dao import ChunkDAO
from graph.vizualizations import make_graph

from streamlit_agraph import agraph, Node, Edge, Config
from chat.chat_manager import ChatManager

def _process_node(record, node_name):
    return {
        "elem_id": record[node_name].element_id,
        "type": list(record[node_name].labels)[0],
        **dict(record[node_name])
    }

def _process_edge(record, edge_name):
    return {
        "elem_id": record[edge_name].element_id,
        "type": record[edge_name].type,
        **dict(record[edge_name])
    }

def process_results(result):
    return [
        {
            "node1": _process_node(rec, 'n'),
            "node2": _process_node(rec, 'm'),
            "relationship": _process_edge(rec, 'r')
        }
        for rec in result
    ]

def main():
    print('='*50)
    print('REFRESHED')
    st.set_page_config(page_title="Echo Bot", page_icon="🤖", layout="wide", )

    # Neo4j AuraDB connection details
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_KEY")
    username = 'neo4j'

    # Set up OpenAI client
    ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    chat_manager = ChatManager(client=ai_client, model=st.session_state["openai_model"])

    emb_processor = EmbeddingsProcessor()
    chunk_dao = ChunkDAO(uri=uri, username=username, password=password,
                         embeddings_processor=emb_processor)

    st.title("Graph Bot")
    col_chat, col_viz = st.columns([3, 2])
    chat_container = col_chat.container()
    prompt = col_chat.chat_input("Ask a question to your documents")

    if 'nodes' not in st.session_state:
        st.session_state['nodes'] = []  
    if 'relationships' not in st.session_state:
        st.session_state['relationships'] = []

    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize chat window_history
    if "window_history" not in st.session_state:
        st.session_state.window_history = []

    # Display chat messages from history on app rerun
    with col_chat:
        with chat_container:
            st.subheader("Chat")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # React to user input
        print('CREATING CHAT INPUT')
        if prompt:
            print('CHAT INPUT USED')
            # Display user message in chat message container
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # Add user message to chat messages and sliding history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.window_history.append({"role": "user", "content": prompt})

        if prompt is not None:
          
            #TODO: Here should be router
            where_to_go = chat_manager.router(st.session_state.window_history, prompt)

            if where_to_go == "SEARCH":
                # Connect to Neo4j AuraDB
                driver = GraphDatabase.driver(uri, auth=(username, password))
                # semantic query
                data = chunk_dao.get_chunk_by_query(prompt, num_results=3, depth=1)
                nodes = [d['node1'] for d in data if d['node1'] is not None] +\
                    [d['node2'] for d in data if d['node2'] is not None]
                nodes = list({node['elem_id']: node for node in nodes}.values())
                relationships = data
                nodes_str = "\n\n".join([f"--- Chunk {i} ---\n\n" + 
                                        'Node type: ' + node.get('node_type', 'UnknownType') +
                                        '\n\nText: ' + node.get('text', f"{node.get('type')}={node.get('name')}")
                                        for i, node in enumerate(nodes)])
                st.session_state['nodes'] = nodes
                st.session_state['relationships'] = relationships

                # Close the driver connection
                driver.close()    # Display assistant response in chat message container

            elif where_to_go == "CONVERSATION":
                nodes_str = ""
            elif where_to_go == "STOP":
                nodes_str = ""
            elif where_to_go == "ERROR":
                nodes_str = ""

            with chat_container:
                with st.chat_message("assistant"):
                    stream = chat_manager.query_model(messages_history=st.session_state.window_history, prompt=prompt, context=nodes_str)
                    if (type(stream) == str):
                        response = st.write(stream)
                    else:
                        response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.window_history.append({"role": "user", "content": response})

            # MANAGING WINDOW HISTORY
            window_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.window_history] if len(st.session_state.window_history) > 0 else []
            new_window_history = chat_manager.manage_history(window_history)
            # print("HISTORY: ", st.session_state.window_history, "\n")
            st.session_state.window_history = new_window_history
            # print("HISTORY PO ZMIANIE: ", st.session_state.window_history)

            # Add assistant response to chat history
            # st.session_state.messages.append({"role": "assistant", "content": nodes_str})
        else:
            print('prompt is None')

    with col_viz:
        print('creating graph')
        st.subheader("Graph Vizualization")
        graph_data = make_graph(
            st.session_state['nodes'], st.session_state['relationships']
        )
        st.write(f'Got {len(graph_data["nodes"])} nodes and {len(graph_data["edges"])} relationships')

        agraph(**graph_data)


if __name__ == '__main__':
    main()