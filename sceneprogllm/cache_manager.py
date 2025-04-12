import os
import shutil
import numpy as np
import json
from typing import Optional, Union
from pydantic import BaseModel

class CacheManager:
    def __init__(self, name: str, no_cache: bool = False):
        self.name = name
        self.no_cache = no_cache
        self.db_path = f"llm_cache/{self.name}_cache.npz"
        if not no_cache:
            self._init_db()

    def _init_db(self):
        if not os.path.exists("llm_cache"):
            os.makedirs("llm_cache")
        
        if os.path.exists(self.db_path):
            # Load existing database
            data = np.load(self.db_path, allow_pickle=True)
            self.db = {key: data[key] for key in data.files}
        else:
            # Create a new database
            self.db = {}
            self.commit()

    def commit(self):
        if not self.no_cache:
            np.savez(self.db_path, **self.db)

    def respond(self, query: str, pydantic_object) -> Optional[Union[str, dict, BaseModel]]:
        if self.no_cache:
            return None
        
        if query in self.db:
            response_str = self.db[query].item()
            response = self._deserialize_response(response_str)
            if pydantic_object:
                try:
                    return pydantic_object(**response)
                except :
                    return response
        else:
            return None

    def append(self, query: str, response: Union[str, dict, BaseModel]):
        if self.no_cache:
            return
        
        response_str = self._serialize_response(response)
        self.db[query] = np.array(response_str, dtype=object)
        self.commit()
        
    def _deserialize_response(self, response_str: str) -> Union[str, dict, BaseModel]:
        """Deserializes a stored response back into its original format."""
        try:
            return json.loads(response_str)
        except json.JSONDecodeError:
            return response_str
        
    def _serialize_response(self, response: Union[str, dict, BaseModel]) -> str:
        """Serializes response to a string for storage."""
        if isinstance(response, BaseModel):
            return response.json()
        elif isinstance(response, dict):
            return json.dumps(response)
        return response
    
    def clear(self):
        clear_llm_cache()

def clear_llm_cache():
    """Clears the cache directory."""
    if os.path.exists("llm_cache"):
        shutil.rmtree("llm_cache")