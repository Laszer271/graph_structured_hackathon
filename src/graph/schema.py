import os
from pydantic import BaseModel, Field


class DocumentSchema(BaseModel):
    # id: str = Field(None, description='Unique Document ID')
    path: str = Field(None, description='Path to the document')
    name: str = Field(None, description='Name of the document')

    class Config:
        from_attributes = True

    def __str__(self):
        return os.path.join(self.path, self.name)

    def __repr__(self):
        return os.path.join(self.path, self.name)
    

class FolderSchema(BaseModel):
    path: str = Field(None, description='Path to the folder')
    name: str = Field(None, description='Name of the folder')

    class Config:
        from_attributes = True

    def __str__(self):
        return os.path.join(self.path, self.name)
    
    def __repr__(self):
        return os.path.join(self.path, self.name)
    

class ChunkSchema(BaseModel):
    text: str = Field(None, description='Text of the chunk')
    page: int = Field(None, description='Page number of the chunk')
    chunk_nr: int = Field(None, description='Chunk number (in the document)')


class EntitySchema(BaseModel):
    pass
