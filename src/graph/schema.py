import os
from typing import List

from pydantic import BaseModel, Field

class BaseSchema(BaseModel):

    class Config:
        from_attributes = True

    def to_dict(self):
        return self.__dict__


class DocumentSchema(BaseSchema):
    # id: str = Field(None, description='Unique Document ID')
    path: str = Field(None, description='Path to the document')
    name: str = Field(None, description='Name of the document')

    def __str__(self):
        return os.path.join(self.path, self.name)

    def __repr__(self):
        return os.path.join(self.path, self.name)
    

class FolderSchema(BaseSchema):
    path: str = Field(None, description='Path to the folder')
    name: str = Field(None, description='Name of the folder')

    def __str__(self):
        return os.path.join(self.path, self.name)
    
    def __repr__(self):
        return os.path.join(self.path, self.name)
    

class ChunkSchema(BaseSchema):
    id: int = Field(None, description='Unique Chunk ID')
    text: str = Field(None, description='Text of the chunk')
    page: int = Field(None, description='Page number of the chunk')
    chunk_nr: int = Field(None, description='Chunk number (in the document)')
    embedding: List[float] = Field(None, description='Embedding of the chunk')


class EntitySchema(BaseSchema):
    name: str = Field(None, description='Name of the entity')
    type: str = Field(None, description='Type of the entity')
