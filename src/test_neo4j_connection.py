import os

from neo4j import GraphDatabase

from dotenv import load_dotenv
load_dotenv()


# Neo4j AuraDB connection details
uri = os.getenv("NEO4J_URI")
password = os.getenv("NEO4J_KEY")
username = 'neo4j'

# Function to add a node
def add_person(tx, name):
    tx.run("CREATE (a:Person {name: $name})", name=name)

# Function to remove a node
def remove_person(tx, name):
    tx.run("MATCH (a:Person {name: $name}) DETACH DELETE a", name=name)

# Function to query a node
def get_person(tx, name):
    result = tx.run("MATCH (a:Person {name: $name}) RETURN a", name=name)
    return [record["a"] for record in result]

if __name__ == '__main__':
    # Connect to Neo4j AuraDB
    driver = GraphDatabase.driver(uri, auth=(username, password))

    with driver.session() as session:
        # Add a node
        session.execute_write(add_person, "Alice")

        # Query the node
        nodes = session.execute_read(get_person, "Alice")
        for node in nodes:
            print(node)

        # Remove the added node
        session.execute_write(remove_person, "Alice")
        # should return empty
        nodes = session.execute_read(get_person, "Alice")
        print(len(nodes))
        print('SUCCESSFULLY REMOVED NODE' if len(nodes) == 0 else 'FAILED TO REMOVE NODE')

    # Close the driver connection
    driver.close()

