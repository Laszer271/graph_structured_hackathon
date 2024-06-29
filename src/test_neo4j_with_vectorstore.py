import os

from neo4j import GraphDatabase

from embeddings.openai_embeddings import EmbeddingsProcessor

from dotenv import load_dotenv
load_dotenv()


# Neo4j AuraDB connection details
uri = os.getenv("NEO4J_URI")
password = os.getenv("NEO4J_KEY")
username = 'neo4j'

def create_index_if_not_exists(tx):
    tx.run("""
        CREATE VECTOR INDEX person_embeddings IF NOT EXISTS
        FOR (p:Person)
        ON p.embedding
        OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
        }}
        """
    )

# Function to add a node
def add_person(tx, name, embedding):
    tx.run("CREATE (a:Person {name: $name, embedding: $embedding})", name=name, embedding=embedding)

# Function to remove a node
def remove_person(tx, name):
    tx.run("MATCH (a:Person {name: $name}) DETACH DELETE a", name=name)

# Function to query a node
def get_person_by_embedding(tx, embedding, num_results):
    result = tx.run("""
        CALL db.index.vector.queryNodes(
            'person_embeddings',
            $num_results,
            $embedding
        ) YIELD node
        RETURN node""",
        num_results=num_results,
        embedding=embedding
    )
    return [record["node"] for record in result]


if __name__ == '__main__':
    # Connect to Neo4j AuraDB
    driver = GraphDatabase.driver(uri, auth=(username, password))
    emb_processor = EmbeddingsProcessor()
    with driver.session() as session:
        # Add a node
        name = 'Alice'
        embedding = emb_processor.get_embedding(name)
        session.execute_write(add_person, name, embedding)

        session.execute_write(create_index_if_not_exists)

        # Query the node
        nodes = session.execute_read(get_person_by_embedding, embedding, 1)
        for node in nodes:
            print(node)

        # Remove the added node
        session.execute_write(remove_person, name)
        # should return empty
        nodes = session.execute_read(get_person_by_embedding, embedding, 1)
        print(len(nodes))
        print('SUCCESSFULLY REMOVED NODE' if len(nodes) == 0 else 'FAILED TO REMOVE NODE')

    # Close the driver connection
    driver.close()

