import os
from typing import List

import openai

class EmbeddingsProcessor:
    def __init__(self, model="text-embedding-3-small", get_tokens=False):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.get_tokens = get_tokens

    def get_embedding(self, text: str):
        result = self.client.embeddings.create(input=[text], model=self.model)
        embedding = result.data[0].embedding
        if not self.get_tokens:
            return embedding
        return embedding, result.usage.total_tokens
    
    def get_batch_embeddings(self, texts: List[str]):
        result = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [embedding.embedding for embedding in result.data]
        if not self.get_tokens:
            return embeddings
        return embeddings, result.usage.total_tokens
    
if __name__ == '__main__':
    processor = EmbeddingsProcessor()
    result = processor.get_embedding('Hello, world!')
    print(type(result), len(result))
    result = processor.get_batch_embeddings(['Hello, world!', 'Goodbye, world!'])
    print(type(result), type(result[0]), len(result), len(result[0]))

    processor_with_tokens = EmbeddingsProcessor(get_tokens=True)
    result, tokens = processor_with_tokens.get_embedding('Hello, world!')
    print(type(result), len(result), f'| Tokens: {tokens}')