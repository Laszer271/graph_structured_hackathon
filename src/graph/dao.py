from neo4j import GraphDatabase

from embeddings.openai_embeddings import EmbeddingsProcessor

from dotenv import load_dotenv
load_dotenv()


class BaseDAO:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

class DocumentDAO(BaseDAO):
    def __init__(self, uri, username, password):
        super().__init__(uri, username, password)

    def add_document(self, name, path):
        with self.driver.session() as session:
            session.run("CREATE (a:Document {name: $name, path: $path})", name=name, path=path)

    def get_document_by_name(self, name):
        with self.driver.session() as session:
            result = session.run("MATCH (a:Document {name: $name}) RETURN a", name=name)
            return [record["a"] for record in result]
        
    