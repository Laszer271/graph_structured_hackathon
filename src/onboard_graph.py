import os

from semantic_router.encoders import OpenAIEncoder
from semantic_chunkers import StatisticalChunker,ConsecutiveChunker

from data_structurer.test_data import arxiv
from graph.dao import FolderDAO, DocumentDAO, ChunkDAO, EntityDAO
from graph.schema import FolderSchema, DocumentSchema, ChunkSchema, EntitySchema
from embeddings.openai_embeddings import EmbeddingsProcessor
from data_structurer.chunker import StatisticalChunk
from data_structurer.entity_search import GLiNEREntitySearcher

from dotenv import load_dotenv
load_dotenv()

def chunk_text(text):
    #Variables
    encoder = OpenAIEncoder(openai_api_key=os.environ["OPENAI_API_KEY"])
    s_chunker = StatisticalChunker(encoder=encoder)

    #Statistical chunker test
    statistical_chunker = StatisticalChunk(chunker=s_chunker)
    statistical_chunks = statistical_chunker.chunk(text)
    return statistical_chunks

def find_entities(text):
    gliner_searcher = GLiNEREntitySearcher()
    return gliner_searcher.search_entities(text)

if __name__ == '__main__':
    # Neo4j AuraDB connection details
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_KEY")
    username = 'neo4j'

    emb_processor = EmbeddingsProcessor()

    folder_dao = FolderDAO(uri=uri, password=password, username=username)
    document_dao = DocumentDAO(uri=uri, password=password, username=username)
    chunk_dao = ChunkDAO(uri=uri, password=password, username=username, emb_processor=emb_processor)
    entity_dao = EntityDAO(uri=uri, password=password, username=username)

    document = DocumentSchema(path="test_data", name="arxiv.txt")
    document_dao.create_document(document)

    chunks = chunk_text(arxiv)
    print(len(chunks))
    chunks = [ChunkSchema(id=i, text=chunk, chunk_nr=i) for i, chunk in enumerate(chunks)]
    for chunk in chunks:
        chunk_dao.create_chunk(chunk, document)
        chunk_dao.connect_chunk_to_document(
            chunk_id=chunk.id,
            document_name=document.name,
            document_path=document.path
        )

        entities = find_entities(chunk.text)
        for entity in entities:
            entity_dao.create_entity(EntitySchema(name=entity["word"], type=entity["entity"]), chunk)
            entity_dao.connect_entity_to_chunk(
                entity_name=entity["word"],
                entity_type=entity["entity"],
                chunk_id=chunk.id
            )
