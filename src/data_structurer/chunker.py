from semantic_chunkers import StatisticalChunker,ConsecutiveChunker
from semantic_router.encoders import OpenAIEncoder

from abc import ABC, abstractmethod
from src.data_structurer.test_data import arxiv
import os
from dotenv import load_dotenv
load_dotenv()

class Chunker(ABC):
    def __init__(self, chunker):
        self.chunker:StatisticalChunker | ConsecutiveChunker = chunker

    @abstractmethod
    def chunk(self, sentence):
        pass

    @abstractmethod
    def pretty_print(self, chunks):
        pass

class StatisticalChunk(Chunker):
    def __init__(self,chunker):
        super().__init__(chunker)

    def chunk(self, sentence):
        return self.chunker(docs=[sentence])[0]

    def pretty_print(self, chunks):
        self.chunker.print(chunks[0])

class ConsecutiveChunk(Chunker):
    def __init__(self,chunker):
        super().__init__(chunker)

    def chunk(self, sentence):
        return self.chunker(docs=[sentence])

    def pretty_print(self, chunks):
        self.chunker.print(chunks[0])


if __name__ == "__main__":
    #Variables
    encoder = OpenAIEncoder(openai_api_key=os.environ["OPENAI_API_KEY"])
    s_chunker = StatisticalChunker(encoder=encoder)
    c_chunker = ConsecutiveChunker(encoder=encoder)

    #Statistical chunker test
    statistical_chunker = StatisticalChunk(chunker=s_chunker)
    statistical_chunks = statistical_chunker.chunk(arxiv)
    statistical_chunker.pretty_print(statistical_chunks)

    # consecutive_chunker = ConsecutiveChunk(chunker=c_chunker)
