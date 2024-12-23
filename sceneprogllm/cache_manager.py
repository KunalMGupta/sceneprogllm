import os
import json
import threading
import hashlib

from gptcache import Cache
from gptcache.adapter.api import init_similar_cache

def get_hashed_name(name):
    return hashlib.md5(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    init_similar_cache(cache_obj=cache_obj, data_dir=f"llm_cache/{hashed_llm}")

class CacheManager:
    def __init__(self, name, no_cache=False):
        self.name=name  
        self.no_cache = no_cache
        if self.no_cache:
            self.cache_path = None
        else:
            self.cache_path = self.init()


    def respond(self, query):
        if self.no_cache:
            return None
        import json
        with open(self.cache_path, "r") as f:
            cache = json.load(f)
        if query in cache:
            return cache[query]
        else:
            return None
    
    def append(self, query, response):
        if self.no_cache:
            return None
        import json
        with open(self.cache_path, "r") as f:
            cache = json.load(f)
        cache[query] = response
        with open(self.cache_path, "w") as f:
            json.dump(cache, f)
    
    def init(self):
        import os
        if not os.path.exists("llm_cache"):
            os.makedirs("llm_cache")
        
        CACHE_PATH = f"llm_cache/{self.name}_cache.json"
        if not os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "w") as f:
                f.write("{}")

        return CACHE_PATH
    
def clear_llm_cache():
    import os
    import shutil
    if os.path.exists("llm_cache"):
        shutil.rmtree("llm_cache")
    