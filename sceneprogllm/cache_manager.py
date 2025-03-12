import sqlite3
import os
import shutil
import numpy as np
import json
from typing import Optional, Union
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

class CacheManager:
    def __init__(self, name: str, no_cache: bool = False):
        self.name = name
        self.no_cache = no_cache
        self.db_path = f"llm_cache/{self.name}_cache.db"
        if not no_cache:
            self._init_db()

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=os.getenv('OPENAI_API_KEY'))

    def _init_db(self):
        """Initializes the SQLite database."""
        if not os.path.exists("llm_cache"):
            os.makedirs("llm_cache")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                query TEXT PRIMARY KEY,
                response TEXT,
                embedding BLOB
            )
        """)
        self.conn.commit()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates an OpenAI embedding for the given text."""
        response = np.array(self.embeddings.embed_query(text), dtype=np.float32)
        return response

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Computes cosine similarity between a vector and a matrix."""
        vec1 = vec1.reshape(1, -1)  # Ensure query embedding is (1, embedding_dim)
        if len(vec2.shape) == 1:
            vec2 = vec2.reshape(1, -1)  # Ensure stored embeddings are (num_entries, embedding_dim)
        
        dot_product = np.dot(vec2, vec1.T).flatten()  # (num_entries, embedding_dim) @ (embedding_dim, 1) -> (num_entries,)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2, axis=1)
        similarity = np.divide(dot_product, norm1 * norm2, out=np.zeros_like(dot_product), where=norm2 != 0)
        return similarity

    def _find_similar(self, query: str, threshold: float = 0.85) -> Optional[str]:
        """Finds a semantically similar query if available."""
        query_embedding = self._get_embedding(query)  # Shape: (embedding_dim,)
        self.cursor.execute("SELECT query, embedding FROM cache")
        rows = self.cursor.fetchall()
        if not rows:
            return None
        
        stored_queries, stored_embeddings = zip(*rows)
        stored_embeddings = np.array([np.frombuffer(e, dtype=np.float32) for e in stored_embeddings])
        if len(stored_embeddings.shape) == 1:
            stored_embeddings = stored_embeddings.reshape(1, -1)
        
        similarities = self._cosine_similarity(query_embedding, stored_embeddings)
        best_idx = np.argmax(similarities)

        if similarities[best_idx] >= threshold:
            return stored_queries[best_idx]
        return None

    def respond(self, query: str) -> Optional[Union[str, dict, BaseModel]]:
        """Returns a cached response if available."""
        if self.no_cache:
            return None
        self.cursor.execute("SELECT response FROM cache WHERE query = ?", (query,))
        row = self.cursor.fetchone()
        if row:
            return self._deserialize_response(row[0])
        similar_match = self._find_similar(query)
        print(f"Similar match: {similar_match}")
        if similar_match:
            self.cursor.execute("SELECT response FROM cache WHERE query = ?", (similar_match,))
            row = self.cursor.fetchone()
            return self._deserialize_response(row[0]) if row else None
        return None

    def append(self, query: str, response: Union[str, dict, BaseModel]):
        """Stores a new response in the cache."""
        if self.no_cache:
            return
        response_str = self._serialize_response(response)
        embedding = self._get_embedding(query)
        self.cursor.execute("INSERT OR REPLACE INTO cache (query, response, embedding) VALUES (?, ?, ?)", (query, response_str, embedding))
        self.conn.commit()

    def _serialize_response(self, response: Union[str, dict, BaseModel]) -> str:
        """Serializes response to a string for storage."""
        if isinstance(response, BaseModel):
            return response.json()
        elif isinstance(response, dict):
            return json.dumps(response)
        return response

    def _deserialize_response(self, response_str: str) -> Union[str, dict, BaseModel]:
        """Deserializes a stored response back into its original format."""
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            return response_str

    def close(self):
        """Closes the database connection."""
        self.conn.close()

    def clear(self):
        clear_llm_cache()

def clear_llm_cache():
    """Clears the cache directory."""
    if os.path.exists("llm_cache"):
        shutil.rmtree("llm_cache")
