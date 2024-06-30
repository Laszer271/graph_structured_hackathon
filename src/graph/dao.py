from typing import List
from neo4j import GraphDatabase

from embeddings.openai_embeddings import EmbeddingsProcessor
from .schema import DocumentSchema, FolderSchema, ChunkSchema, EntitySchema

from dotenv import load_dotenv
load_dotenv()


def _process_node(record):
    rec_dict = dict(record)
    if "embedding" in rec_dict:
        del rec_dict["embedding"]

    return {
        "elem_id": record.element_id,
        "node_type": list(record.labels)[0],
        **rec_dict
    }


def _process_node_with_name(record, record_name):
    rec = record[record_name]
    return _process_node(rec)

def _process_edge(record):
    return {
        "elem_id": record.element_id,
        "edge_type": record.type,
        **dict(record)
    }


class BaseDAO:
    def __init__(self, uri, username, password):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()


class FolderDAO(BaseDAO):
    def __init__(self, uri, username, password):
        super().__init__(uri, username, password)

    def _add_folder(self, tx, folder: FolderSchema):
        tx.run(
            """
            MERGE (a:Folder {name: $name, path: $path})
            ON CREATE SET a.name = $name, a.path = $path
            """,
            name=folder.name, path=folder.path
        )

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
            MERGE (pf)-[:CONTAINS_FOLDER]->(f)
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
        tx.run(
            """
            MERGE (a:Document {name: $name, path: $path})
            ON CREATE SET a.name = $name, a.path = $path
            """,
            name=document.name, path=document.path
        )

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
            MERGE (f)-[:CONTAINS_DOCUMENT]->(d)
            """,
            document_name=document_name,
            document_path=document_path,
            folder_name=folder_name,
            folder_path=folder_path
        )

    def add_document(self, document: DocumentSchema):
        with self.driver.session() as session:
            session.execute_write(self._add_document, document)

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
        self.create_index_if_not_exists()

    @staticmethod
    def _create_index_if_not_exists(tx):
        tx.run("""
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:Chunk)
            ON c.embedding
            OPTIONS {indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
            }}
            """
        )

    def create_index_if_not_exists(self):
        with self.driver.session() as session:
            session.execute_write(ChunkDAO._create_index_if_not_exists)

    def _add_chunk(self, tx, chunk: ChunkSchema):
        if not chunk.embedding:
            chunk.embedding = self.embeddings_processor.get_embedding(chunk.text)
        tx.run(
            """
            MERGE (a:Chunk {id: $id})
            ON CREATE SET a.text = $text, a.page = $page, a.chunk_nr = $chunk_nr, a.embedding = $embedding
            """,
            id=chunk.id,
            text=chunk.text,
            page=chunk.page,
            chunk_nr=chunk.chunk_nr,
            embedding=chunk.embedding
        )

    def _get_chunk_by_id(self, tx, id):
        result = tx.run("MATCH (a:Chunk {id: $id}) RETURN a", id=id)
        return [_process_node_with_name(record, "a") for record in result]
    
    def _get_chunk_by_document(self, tx, document: DocumentSchema):
        result = tx.run("""
            MATCH (d:Document {name: $name, path: $path})-[:CONTAINS_CHUNK]->(a:Chunk)
            RETURN a
            """, name=document.name, path=document.path)
        return [_process_node_with_name(record, "a") for record in result]
    
    def _get_chunk_by_document_and_page(self, tx, document: DocumentSchema, page):
        result = tx.run("""
            MATCH (d:Document {name: $name, path: $path})-[:CONTAINS_CHUNK]->(a:Chunk)
            WHERE a.page = $page
            RETURN a
            """, name=document.name, path=document.path, page=page)
        return [_process_node_with_name(record, "a") for record in result]
    
    def _get_chunk_by_document_and_chunk_nr(self, tx, document: DocumentSchema, chunk_nr):
        result = tx.run("""
            (d:Document {name: $name, path: $path})-[:CONTAINS_CHUNK]->(a:Chunk)
            WHERE a.chunk_nr = $chunk_nr
            RETURN a
            """, name=document.name, path=document.path, chunk_nr=chunk_nr)
        return [_process_node_with_name(record, "a") for record in result]
    
    # def _get_chunk_by_query(self, tx, query, num_results=1, depth=0):
    #     embedding = self.embeddings_processor.get_embedding(query)
        
    #     base_query = """
    #         CALL db.index.vector.queryNodes(
    #             'chunk_embeddings',
    #             $num_results,
    #             $embedding
    #         ) YIELD node
    #         WITH collect(node) AS nodes
    #     """
        
    #     if depth > 0:
    #         assert isinstance(depth, int), 'Unsafe depth value, depth value must be an integer'
    #         base_query += f"""
    #         WITH nodes
    #         UNWIND nodes AS n
    #         MATCH (n)-[r*1..{depth}]-(m)
    #         RETURN collect(distinct n) + collect(distinct m) AS nodes, collect(distinct r) AS relationships
    #         """
    #     else:
    #         base_query += """
    #         RETURN nodes, [] AS relationships
    #         """
        
    #     print(base_query)
        
    #     result = tx.run(base_query, num_results=num_results, embedding=embedding)

    #     nodes = []
    #     relationships = []
    #     for record in result:
    #         print('----- RECORD -----')
    #         print([r['relationships'] for r in record[0]])
    #         raise
    #         nodes.extend([_process_node(r) for r in record['nodes']])
    #         relationships.extend([_process_edge(r) for r in record['relationships']])
    #     return nodes, relationships
        # return [res['nodes'] for res in result], [res['relationships'] for res in result]

    @classmethod
    def _process_relationships(cls, relationships: List):
        data = []
        if len(relationships) > 0:
            for rel in relationships:
                if isinstance(rel, list):
                    data.extend(cls._process_relationships(rel))
                    continue
                nodes = list(rel.nodes)
                node1 = _process_node(nodes[0])
                node2 = _process_node(nodes[1])
                relationship = _process_edge(rel)
                data.append({
                    "node1": node1,
                    "node2": node2,
                    "relationship": relationship
                })
        return data

    @classmethod
    def _process_results(cls, results):
        data = []
        for record in results:
            if isinstance(record, list):
                data.extend([cls._process_results(rec) for rec in record])
                continue
            else:
                node1 = _process_node_with_name(record, 'node')

                relationships = record.get('relationships', [])
                # secondary_nodes = record.get('nodes', [])
                # print(secondary_nodes)
                # raise
                # if len(relationships) != len(secondary_nodes):
                #     raise ValueError('Mismatched number of relationships and secondary nodes')
                data.extend(cls._process_relationships(relationships))
                
        return data

        

    def _get_chunk_by_query(self, tx, query, num_results=1, depth=0):
        embedding = self.embeddings_processor.get_embedding(query)
        
        base_query = """
            CALL db.index.vector.queryNodes(
                'chunk_embeddings',
                $num_results,
                $embedding
            ) YIELD node
            WITH node
        """
        
        if depth > 0:
            assert isinstance(depth, int), 'Unsafe depth value, depth value must be an integer'
            base_query += f"""
            MATCH (node)-[r*1..{depth}]-(m)
            RETURN node, collect(r) as relationships, collect(m) as nodes
            """
        else:
            base_query += """
            RETURN node, null as r, null as m
            """
        
        print(base_query)
        
        result = tx.run(base_query, num_results=num_results, embedding=embedding)
        data = self._process_results(result)
        
        return data

    
    def _connect_chunk_to_document(self, tx, chunk: ChunkSchema, document: DocumentSchema):
        tx.run("""
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (d:Document {name: $document_name, path: $document_path})
            MERGE (d)-[:CONTAINS_CHUNK]->(c)
            """,
            chunk_id=chunk.id,
            document_name=document.name,
            document_path=document.path
        )

    def add_chunk(self, chunk: ChunkSchema):
        with self.driver.session() as session:
            session.execute_write(self._add_chunk, chunk)

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
        
    # def get_chunk_by_query(self, query, num_results=1, depth=0):
    #     with self.driver.session() as session:
    #         result, relationships = session.execute_read(self._get_chunk_by_query, query, num_results, depth)
    #         # print(result)
    #         return result, relationships
    #         # return [ChunkSchema(**record) for record in result], relationships

    def get_chunk_by_query(self, query, num_results=1, depth=0):
        with self.driver.session() as session:
            data = session.execute_read(self._get_chunk_by_query, query, num_results, depth)
            return data
        
    def connect_chunk_to_document(self, chunk: ChunkSchema, document: DocumentSchema):
        with self.driver.session() as session:
            session.write_transaction(self._connect_chunk_to_document, chunk, document)


class EntityDAO(BaseDAO):
    def __init__(self, uri, username, password):
        super().__init__(uri, username, password)

    def _add_entity(self, tx, entity: EntitySchema):
        tx.run(
            """
            MERGE (a:Entity {name: $name, type: $type})
            ON CREATE SET a.name = $name, a.type = $type
            """,
            name=entity.name, type=entity.type
        )

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
    
    def _connect_entity_to_chunk(self, tx, entity: EntitySchema, chunk: ChunkSchema):
        tx.run("""
            MATCH (e:Entity {name: $entity_name, type: $entity_type})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            entity_name=entity.name,
            entity_type=entity.type,
            chunk_id=chunk.id
        )

    def add_entity(self, entity: EntitySchema):
        with self.driver.session() as session:
            session.execute_write(self._add_entity, entity)

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
        
    def connect_entity_to_chunk(self, entity: EntitySchema, chunk: ChunkSchema):
        with self.driver.session() as session:
            session.write_transaction(self._connect_entity_to_chunk, entity, chunk)

