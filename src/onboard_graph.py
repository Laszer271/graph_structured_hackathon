import os
from tqdm import tqdm
from pathlib import Path

from semantic_router.encoders import OpenAIEncoder
from semantic_chunkers import StatisticalChunker,ConsecutiveChunker
import pandas as pd

from description_generation.describer import OpenAIDescriber
from data_structurer.test_data import arxiv
from graph.dao import FolderDAO, DocumentDAO, ChunkDAO, EntityDAO
from graph.schema import FolderSchema, DocumentSchema, ChunkSchema, EntitySchema
from embeddings.openai_embeddings import EmbeddingsProcessor
from data_structurer.chunker import StatisticalChunk
from data_structurer.entity_search import GLiNEREntitySearcher

from dotenv import load_dotenv
load_dotenv()

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.Series([str(p) for p in path.rglob('*.txt')], name='Path').to_frame()
    print(df.shape)
    df['DocumentName'] = df['Path'].str.split('/').str[2]
    df['DateProcessed'] = df['Path'].str.split('/').str[4]
    df['PageName'] = df['Path'].str.split('/').str[-1]
    df = df.loc[df['PageName'] != 'complete.txt']
    print(df.shape)
    df = df.drop_duplicates(subset=['DocumentName', 'PageName'])
    print(df.shape)
    df['PageId'] = df['PageName'].str.split('.').str[0]
    
    df_images = pd.Series([str(p) for p in path.rglob('*.png')], name='ImagePath').to_frame()
    df_images['DocumentName'] = df_images['ImagePath'].str.split('/').str[2]
    df_images['ImageName'] = df_images['ImagePath'].str.split('/').str[-1]
    df_images['ImageId'] = df_images['ImageName'].str.split('.').str[0]
    df_images = df_images.loc[~df_images['ImageName'].str.contains('.bin')]
    df_images = df_images.loc[~df_images['ImageName'].str.contains('.nrm')]
    df_images = df_images.drop_duplicates(subset=['DocumentName', 'ImageName'])
    
    print(df.shape)
    df = df.merge(df_images, left_on=('DocumentName', 'PageId'), right_on=('DocumentName', 'ImageId'), how='left')
    print(df.shape)
    print(df.isna().sum())
    return df

def describe_images(df: pd.DataFrame, describer, output_path: str):
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    
    descriptions = []
    usages = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Image Description Generation'):
        text_path = row['Path']
        image_path = row['ImagePath']
        try:
            description, usage = describer.describe(text_path, image_path)
            descriptions.append(description)
            usages.append(usage)
        except:
            print(f"Failed to describe image {image_path}")
            break

    df_descriptions = pd.DataFrame(usages)
    df_descriptions['Description'] = descriptions
    df_descriptions.to_csv(output_path, index=False)
    return df_descriptions
        


def chunk_text(text):
    #Variables
    encoder = OpenAIEncoder(openai_api_key=os.environ["OPENAI_API_KEY"])
    s_chunker = StatisticalChunker(encoder=encoder)

    #Statistical chunker test
    statistical_chunker = StatisticalChunk(chunker=s_chunker)
    statistical_chunks = statistical_chunker.chunk(text)
    chunks = [chunk.content for chunk in statistical_chunks]
    return chunks

def find_entities(text):
    gliner_searcher = GLiNEREntitySearcher()
    return gliner_searcher.search_entities(text)

if __name__ == '__main__':
    # Neo4j AuraDB connection details
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_KEY")
    username = 'neo4j'

    data = load_data('../data')
    describer = OpenAIDescriber(openai_api_key=os.getenv('OPENAI_API_KEY'), model='gpt-4o')
    descriptions = describe_images(data, describer, 'desc_backup.csv')
    print(descriptions)
    data = pd.concat([data, descriptions], axis=1)

    emb_processor = EmbeddingsProcessor()

    folder_dao = FolderDAO(uri=uri, password=password, username=username)
    document_dao = DocumentDAO(uri=uri, password=password, username=username)
    chunk_dao = ChunkDAO(uri=uri, password=password, username=username, embeddings_processor=emb_processor)
    entity_dao = EntityDAO(uri=uri, password=password, username=username)

    folder_dao.add_folder(path='', name='data')

    documents = data['DocumentName'].unique()
    # print(documents)
    print(len(data))
    chunk_id = 0
    for doc_cnt, document in enumerate(tqdm(documents)):
        # if doc_cnt == 2:
        #     raise
        df_pages = data.loc[data['DocumentName'] == document]
        print(len(df_pages))
        # raise
        document_data = df_pages.iloc[0]
        document = DocumentSchema(path='data', name=document)
        document_dao.add_document(document)
        document_dao.connect_document_to_folder(
            document_name=document.name, document_path=document.path,
            folder_name='data', folder_path=''
        )
        
        for i, page_row in df_pages.iterrows():
            page_nr = int(page_row['PageId'])
            
            chunks = chunk_text(page_row['Description'])
            chunks_processed = []
            for chunk_nr, chunk in enumerate(chunks):
                chunk = ChunkSchema(id=chunk_id, text=chunk, chunk_nr=chunk_nr, page=page_nr)
                chunk_id += 1
                chunks_processed.append(chunk)
            chunks = chunks_processed
            print(len(chunks))

            for chunk in chunks:
                chunk_dao.add_chunk(chunk)
                chunk_dao.connect_chunk_to_document(
                    chunk=chunk,
                    document=document
                )

                entities = find_entities(chunk.text)
                for entity in entities:
                    entity = EntitySchema(name=entity["text"].upper(), type=entity["label"])
                    if len(entity.name) <= 2:
                        continue # Skip entities with less than 2 characters
                    entity_dao.add_entity(entity)
                    entity_dao.connect_entity_to_chunk(
                        entity,
                        chunk
                    )
