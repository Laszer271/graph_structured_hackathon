from neo4j import GraphDatabase

from embeddings.openai_embeddings import EmbeddingsProcessor
from .schema import DocumentSchema, FolderSchema, ChunkSchema, EntitySchema

from dotenv import load_dotenv
load_dotenv()


class BaseDAO:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()


class FolderDAO(BaseDAO):
    def __init__(self, uri, username, password):
        super().__init__(uri, username, password)

    def _add_folder(self, tx, folder: FolderSchema):
        tx.run("CREATE (a:Folder {name: $name, path: $path})", name=folder.name, path=folder.path)

    def _get_folder_by_name(self, tx, name):
        result = tx.run("MATCH (a:Folder {name: $name}) RETURN a", name=name)
        return [record["a"] for record in result]
    
    def _get_folder_by_path(self, tx, path):
        result = tx.run("MATCH (a:Folder {path: $path}) RETURN a", path=path)
        return [record["a"] for record in result]
    
    def _get_folder_by_name_and_path(self, tx, name, path):
        result = tx.run("MATCH (a:Folder {name: $name, path: $path}) RETURN a",
                        name=name, path=path)
        return [record["a"] for record in result]
    
    def _connect_folder_to_folder(self, tx, folder_name, folder_path,
                                  parent_folder_name, parent_folder_path):
        tx.run("""
            MATCH (f:Folder {name: $folder_name, path: $folder_path})
            MATCH (pf:Folder {name: $parent_folder_name, path: $parent_folder_path})
            CREATE (pf)-[:CONTAINS_FOLDER]->(f)
            """, 
            folder_name=folder_name,
            folder_path=folder_path,
            parent_folder_name=parent_folder_name,
            parent_folder_path=parent_folder_path
        )
    
    def add_folder(self, name, path):
        with self.driver.session() as session:
            session.execute_write(self._add_folder, FolderSchema(name=name, path=path))

    def get_folder_by_name(self, name):
        with self.driver.session() as session:
            result = session.execute_read(self._get_folder_by_name, name)
            return [FolderSchema(**record) for record in result]
        
    def get_folder_by_path(self, path):
        with self.driver.session() as session:
            result = session.execute_read(self._get_folder_by_path, path)
            return [FolderSchema(**record) for record in result]
        
    def get_folder_by_name_and_path(self, name, path):
        with self.driver.session() as session:
            result = session.execute_read(self._get_folder_by_name_and_path, name, path)
            return [FolderSchema(**record) for record in result]
    
    def connect_folder_to_folder(self, folder_name, folder_path,\
                                 parent_folder_name, parent_folder_path):
        with self.driver.session() as session:
            session.write_transaction(self._connect_folder_to_folder,
                                      folder_name,
                                      folder_path,
                                      parent_folder_name,
                                      parent_folder_path)

class DocumentDAO(BaseDAO):
    def __init__(self, uri, username, password):
        super().__init__(uri, username, password)

    def _add_document(self, tx, document: DocumentSchema):
        tx.run("CREATE (a:Document {name: $name, path: $path})", name=document.name, path=document.path)

    def _get_document_by_name(self, tx, name):
        result = tx.run("MATCH (a:Document {name: $name}) RETURN a", name=name)
        return [record["a"] for record in result]
    
    def _get_document_by_path(self, tx, path):
        result = tx.run("MATCH (a:Document {path: $path}) RETURN a", path=path)
        return [record["a"] for record in result]
    
    def _get_document_by_name_and_path(self, tx, name, path):
        result = tx.run("MATCH (a:Document {name: $name, path: $path}) RETURN a",
                        name=name, path=path)
        return [record["a"] for record in result]
    
    def _connect_document_to_folder(self, tx, document_name, document_path,
                                    folder_name, folder_path):
        tx.run("""
            MATCH (d:Document {name: $document_name, path: $document_path})
            MATCH (f:Folder {name: $folder_name, path: $folder_path})
            CREATE (f)-[:CONTAINS_DOCUMENT]->(d)
            """,
            document_name=document_name,
            document_path=document_path,
            folder_name=folder_name,
            folder_path=folder_path
        )

    def add_document(self, name, path):
        with self.driver.session() as session:
            session.execute_write(self._add_document, DocumentSchema(name=name, path=path))

    def get_document_by_name(self, name):
        with self.driver.session() as session:
            result = session.execute_read(self._get_document_by_name, name)
            return [DocumentSchema(**record) for record in result]
        
    def get_document_by_path(self, path):
        with self.driver.session() as session:
            result = session.execute_read(self._get_document_by_path, path)
            return [DocumentSchema(**record) for record in result]
        
    def get_document_by_name_and_path(self, name, path):
        with self.driver.session() as session:
            result = session.execute_read(self._get_document_by_name_and_path, name, path)
            return [DocumentSchema(**record) for record in result]
        
    def connect_document_to_folder(self, document_name, document_path,
                                   folder_name, folder_path):
        with self.driver.session() as session:
            session.write_transaction(self._connect_document_to_folder,
                                      document_name,
                                      document_path,
                                      folder_name,
                                      folder_path)

        
class ChunkDAO(BaseDAO):
    def __init__(self, uri, username, password, embeddings_processor=None):
        super().__init__(uri, username, password)
        self.embeddings_processor = embeddings_processor

    def _add_chunk(self, tx, chunk: ChunkSchema):
        tx.run("CREATE (a:Chunk {id: $id, text: $text, page: $page, chunk_nr: $chunk_nr})",
               id=chunk.id, text=chunk.text, page=chunk.page, chunk_nr=chunk.chunk_nr)

    def _get_chunk_by_id(self, tx, id):
        result = tx.run("MATCH (a:Chunk {id: $id}) RETURN a", id=id)
        return [record["a"] for record in result]
    
    def _get_chunk_by_document(self, tx, document: DocumentSchema):
        result = tx.run("""
            MATCH (a:Chunk)-[:PART_OF]->(d:Document {name: $name, path: $path})
            RETURN a
            """, name=document.name, path=document.path)
        return [record["a"] for record in result]
    
    def _get_chunk_by_document_and_page(self, tx, document: DocumentSchema, page):
        result = tx.run("""
            MATCH (a:Chunk)-[:PART_OF]->(d:Document {name: $name, path: $path})
            WHERE a.page = $page
            RETURN a
            """, name=document.name, path=document.path, page=page)
        return [record["a"] for record in result]
    
    def _get_chunk_by_document_and_chunk_nr(self, tx, document: DocumentSchema, chunk_nr):
        result = tx.run("""
            MATCH (a:Chunk)-[:PART_OF]->(d:Document {name: $name, path: $path})
            WHERE a.chunk_nr = $chunk_nr
            RETURN a
            """, name=document.name, path=document.path, chunk_nr=chunk_nr)
        return [record["a"] for record in result]
    
    def _get_chunk_by_query(self, tx, query, num_results=1):
        embedding = self.embeddings_processor.get_embedding(query)
        result = tx.run("""
            CALL db.index.vector.queryNodes(
                'chunk_embeddings',
                $num_results,
                $embedding
            ) YIELD node RETURN node
            """, num_results=num_results, embedding=embedding)
        return [record["node"] for record in result]
    
    def _connect_chunk_to_document(self, tx, chunk_id, document_name, document_path):
        tx.run("""
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (d:Document {name: $document_name, path: $document_path})
            CREATE (c)-[:CONTAINS_CHUNK]->(d)
            """,
            chunk_id=chunk_id,
            document_name=document_name,
            document_path=document_path
        )

    def _connect_chunk_to_document_by_chunk_nr(self, tx, chunk_nr, document_name, document_path):
        tx.run("""
            MATCH (c:Chunk)-[:PART_OF]->(d:Document {name: $document_name, path: $document_path})
            CREATE (d)-[:CONTAINS_CHUNK]->(c)
            """,
            chunk_nr=chunk_nr,
            document_name=document_name,
            document_path=document_path
        )

    def add_chunk(self, id, text, page, chunk_nr):
        with self.driver.session() as session:
            session.execute_write(self._add_chunk, ChunkSchema(id=id, text=text, page=page, chunk_nr=chunk_nr))

    def get_chunk_by_id(self, id):
        with self.driver.session() as session:
            result = session.execute_read(self._get_chunk_by_id, id)
            return [ChunkSchema(**record) for record in result]
        
    def get_chunk_by_document(self, document):
        with self.driver.session() as session:
            result = session.execute_read(self._get_chunk_by_document, document)
            return [ChunkSchema(**record) for record in result]
        
    def get_chunk_by_document_and_page(self, document, page):
        with self.driver.session() as session:
            result = session.execute_read(self._get_chunk_by_document_and_page, document, page)
            return [ChunkSchema(**record) for record in result]
        
    def get_chunk_by_document_and_chunk_nr(self, document, chunk_nr):
        with self.driver.session() as session:
            result = session.execute_read(self._get_chunk_by_document_and_chunk_nr, document, chunk_nr)
            return [ChunkSchema(**record) for record in result]
        
    def get_chunk_by_query(self, query, num_results=1):
        with self.driver.session() as session:
            result = session.execute_read(self._get_chunk_by_query, query, num_results)
            return [ChunkSchema(**record) for record in result]
        
    def connect_chunk_to_document(self, chunk_id, document_name, document_path):
        with self.driver.session() as session:
            session.write_transaction(self._connect_chunk_to_document, chunk_id, document_name, document_path)

    def connect_chunk_to_document_by_chunk_nr(self, chunk_nr, document_name, document_path):
        with self.driver.session() as session:
            session.write_transaction(self._connect_chunk_to_document_by_chunk_nr, chunk_nr, document_name, document_path)
        

class EntityDAO(BaseDAO):
    def __init__(self, uri, username, password):
        super().__init__(uri, username, password)

    def _add_entity(self, tx, entity: EntitySchema):
        tx.run("CREATE (a:Entity {name: $name, type: $type})", name=entity.name, type=entity.type)

    def _get_entity_by_name(self, tx, name):
        result = tx.run("MATCH (a:Entity {name: $name}) RETURN a", name=name)
        return [record["a"] for record in result]
    
    def _get_entity_by_type(self, tx, type):
        result = tx.run("MATCH (a:Entity {type: $type}) RETURN a", type=type)
        return [record["a"] for record in result]
    
    def _get_entity_by_name_and_type(self, tx, name, type):
        result = tx.run("MATCH (a:Entity {name: $name, type: $type}) RETURN a",
                        name=name, type=type)
        return [record["a"] for record in result]
    
    def _connect_entity_to_chunk(self, tx, entity_name, entity_type, chunk_id):
        tx.run("""
            MATCH (e:Entity {name: $entity_name, type: $entity_type})
            MATCH (c:Chunk {id: $chunk_id})
            CREATE (c)-[:MENTIONS]->(e)
            """,
            entity_name=entity_name,
            entity_type=entity_type,
            chunk_id=chunk_id
        )

    def add_entity(self, name, type):
        with self.driver.session() as session:
            session.execute_write(self._add_entity, EntitySchema(name=name, type=type))

    def get_entity_by_name(self, name):
        with self.driver.session() as session:
            result = session.execute_read(self._get_entity_by_name, name)
            return [EntitySchema(**record) for record in result]
        
    def get_entity_by_type(self, type):
        with self.driver.session() as session:
            result = session.execute_read(self._get_entity_by_type, type)
            return [EntitySchema(**record) for record in result]
        
    def get_entity_by_name_and_type(self, name, type):
        with self.driver.session() as session:
            result = session.execute_read(self._get_entity_by_name_and_type, name, type)
            return [EntitySchema(**record) for record in result]
        
    def connect_entity_to_chunk(self, entity_name, entity_type, chunk_id):
        with self.driver.session() as session:
            session.write_transaction(self._connect_entity_to_chunk, entity_name, entity_type, chunk_id)