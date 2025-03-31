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
from pydantic import BaseModel, Json, create_model
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser
from langchain_ollama import ChatOllama

from .cache_manager import CacheManager
from .image_helper import ImageHelper
from .text2img import text2imgSD, text2imgOpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(filename='sceneprogllm.log', encoding='utf-8', level=logging.INFO)

class ListResponse(BaseModel):
    response: list[str]

class DefaultJsonResponse(BaseModel):
    response: str

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
        # sanitize query
        if '{' in query:
            query = query.replace('{', '{{').replace('}', '}}')
        ## sanitize system description
        if '{' in self.system_desc:
            system_desc = system_desc.replace('{', '{{').replace('}', '}}')
        
        if self.response_format == "3d":
            from .textto3d import text_to_3d
            return text_to_3d(query)
        elif self.response_format == "pydantic":
            assert pydantic_object, "Pydantic object is required for response format 'pydantic'"
        elif self.response_format == "image":
            return self.text2img(query)
        elif self.response_format == "list":
            pydantic_object = ListResponse
        elif self.response_format == "json":
            if self.json_keys is not None:
                CustomJSONModel = create_model(
                    'CustomJSONModel',
                    **{key: (str, ...) for key in self.json_keys}
                )
                pydantic_object = CustomJSONModel
            else:
                pydantic_object = DefaultJsonResponse
        
        if self.use_cache and not image_paths:
            result = self.cache.respond(query)
            if result:
                return result
            
        # Generates a response from the model based on the query and history.
        self.history = [{"role": "system", "content": self.system_desc}]
        self.history.append({"role": "human", "content": query})
        
        full_prompt = self._get_prompt_with_history()
        
        if pydantic_object is not None:
            parser = PydanticOutputParser(pydantic_object=pydantic_object)
            self.prompt_template = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{input}\n",
                input_variables=["input"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            chain = self.prompt_template | self.model | parser
        else:
            chain = self.prompt_template | self.model | StrOutputParser()
            
        if self.response_format == "code":
            full_prompt += """Return only python code in Markdown format, e.g.:
```python
....
```"""
        
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

        if self.use_cache and not image_paths:
            self.cache.append(query, result)
        
        if self.response_format == "json":
            result = result.model_dump_json()
        if self.response_format == "list":
            result = result.response
        return result
    
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