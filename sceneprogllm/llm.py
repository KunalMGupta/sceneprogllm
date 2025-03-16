'''
Implementation of the LLM class
Key features
1. support for text, code, json output formats
2. Image input support
3. Caching support 
4. Support for image generation
5. Ensure JSON is generated correctly
'''
import os
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser
from langchain_ollama import ChatOllama
from .cache_manager import CacheManager
from .image_helper import ImageHelper
from .text2img import text2imgSD, text2imgOpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(filename='sceneprogllm.log', encoding='utf-8', level=logging.INFO)

class LLM:
    def __init__(self,
                 name, 
                 system_desc=None, 
                 response_format="text", 
                 json_keys=None,
                 use_cache=True,
                 model_name='gpt-4o',
                 ):
        
        assert response_format in ['text', 'list', 'code', 'json', 'image', 'pydantic','3d'], "Invalid response format, must be one of 'text', 'list', 'code', 'json', 'image', 'pydantic'"
        self.name = name
        self.response_format = response_format

        if self.response_format == "json" and not json_keys:
            logger.warning("json_keys must be provided when response_format is 'json'")

        if self.response_format == "3d":
            logger.warning("3D response is not supported.")
            return

        self.json_keys = json_keys
        self.use_cache = use_cache
        self.model_name = model_name
        self.system_desc = system_desc or "You are a helpful assistant."
        
        if use_cache:
            self.cache = CacheManager(self.name, no_cache=not use_cache)
        self.text2img = text2imgSD if model_name == 'SD' else text2imgOpenAI

        # Configure the response format
        if self.response_format == "json":
            self.response_format_config = {"type": "json_object"}
        else:
            self.response_format_config = {"type": "text"}

        # Initialize the model with the given configuration
        print(model_name)
        if 'gpt' in model_name:
            self.model = ChatOpenAI(
                model=model_name,
                api_key=os.getenv('OPENAI_API_KEY'),
            )
            logger.info(f"Using OpenAI model: {model_name}")
        else:
            self.model = ChatOllama(model=model_name, temperature=0.8)
            logger.info(f"Using Ollama model: {model_name}")
            
        # Initial system message and prompt template
        self.image_helper = ImageHelper(self.system_desc)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_desc),
            ("human", "{input}")
        ])

    def __call__(self, query, image_paths=None, pydantic_object=None):

        if self.response_format == "3d":
            from .textto3d import text_to_3d
            return text_to_3d(query)
        
        if self.response_format == "pydantic":
            assert pydantic_object, "Pydantic object is required for response format 'pydantic'"
            
        if self.response_format == "image":
            return self.text2img(query)
        
        if self.use_cache and not image_paths:
            result = self.cache.respond(query)
            if result:
                return result
            
        # Generates a response from the model based on the query and history.
        self.history = [{"role": "system", "content": self.system_desc}]
        self.history.append({"role": "human", "content": query})
        
        full_prompt = self._get_prompt_with_history()
        
        if self.response_format == "json":
            chain = self.prompt_template | self.model | SimpleJsonOutputParser()
        elif self.response_format == "list":
            chain = self.prompt_template | self.model | CommaSeparatedListOutputParser()
        elif self.response_format == "pydantic":
            parser = PydanticOutputParser(pydantic_object=pydantic_object)
            self.prompt_template = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{input}\n",
                input_variables=["input"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = self.prompt_template | self.model | parser
        else:
            chain = self.prompt_template | self.model | StrOutputParser()
        
        if self.response_format == "json":
            if not self.json_keys:
                json_example = {"key": "value"}
            else:
                json_example = {key: "value" for key in self.json_keys}
            full_prompt += f"""Return a JSON object STRICTLY with the following keys: {self.json_keys}.
            All keys must be present in the response.
            e.g.:
```json
{json.dumps(json_example, indent=4)}
```"""
        

        if self.response_format == "code":
            full_prompt += """Return only python code in Markdown format, e.g.:
```python
....
```"""

        if self.response_format == "list":
            full_prompt += """Return a comma separated list of items, e.g.:
item1, item2, item3
"""
        
        if image_paths is not None:
            self.prompt_template = ChatPromptTemplate.from_messages(
                messages = self.image_helper.prepare_image_prompt_template(image_paths)
            )
            chain = self.prompt_template | self.model | StrOutputParser()
            result = self.image_helper.invoke_image_prompt_template(chain, full_prompt, image_paths)
        else:
            
            result = chain.invoke({"input": full_prompt})
        
        if self.response_format == "code":
            result = self._sanitize_output(result)

        if self.response_format == "json" and self.json_keys:
            for key in self.json_keys:
                if key not in result:
                    query = f""" 
                    For the query: {query}, the following response was generated: {result}. It didn't follow the expected format containing the keys: {self.json_keys}. Please ensure that the response follows the expected format and contains all the keys.
                    """
                    logging.error(query)
                    return None
        
        if self.response_format == "json":
            result = self._adjust_json_types(str(result))

        if self.use_cache and not image_paths:
            self.cache.append(query, result)
        
        return result
    
    def _adjust_json_types(self, json_str):
        """Adjust JSON types for None and boolean values."""
        try:
            json_obj = json.loads(json_str)
            adjusted_json_str = json.dumps(json_obj, default=self._json_type_converter)
            return adjusted_json_str
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON response.")
            return json_str

    def _json_type_converter(self, obj):
        """Convert JSON types to appropriate Python types."""
        if obj is None:
            return None
        if obj == "True":
            return True
        elif obj == "False":
            return False
        elif obj == "None":
            return None

        return str(obj)
    
    def _get_prompt_with_history(self):
        """Constructs the full prompt including chat history."""
        return "".join([f"{msg['role']}: {msg['content']}\n" for msg in self.history])
    
    def _sanitize_output(self, text: str):
        _, after = text.split("```python")
        return after.split("```")[0]
    
    def clear_cache(self):
        if self.use_cache:
            self.cache.clear()
        else:
            logger.error("Cache is not enabled for this LLM instance.")