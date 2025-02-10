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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, SimpleJsonOutputParser, CommaSeparatedListOutputParser, PydanticOutputParser
from langchain_ollama import ChatOllama
from .cache_manager import CacheManager
from .image_helper import ImageHelper
from .text2img import text2imgSD, text2imgOpenAI

class LLM:
    def __init__(self,
                 name, 
                 system_desc=None, 
                 response_format="text", 
                 fast=False, 
                 num_images=0,
                 image_detail='auto',
                 json_keys=None,
                 use_cache=True,
                 image_generator='SD',
                 use_ollama=False,
                 ollama_model_name='llama3.2-vision',
                 use_local_image=True):
        
        assert response_format in ['text', 'list', 'code', 'json', 'image', 'pydantic'], "Invalid response format, must be one of 'text', 'list', 'code', 'json', 'image', 'pydantic'"
        self.name=name
        self.response_format = response_format
        self.image_detail=image_detail
        self.image_input=True if num_images > 0 else False
        self.json_keys = json_keys
        self.num_images = num_images    
        self.use_ollama = use_ollama

        self.cache = CacheManager(self.name, no_cache=False)
        # if use_cache:
        #     if use_ollama:
        #         self.cache = CacheManager(self.name, no_cache=False)
        #     else:
        #         self.cache = GPTCache(init_gptcache)
        #         set_llm_cache(self.cache)
        # else:
        #     if use_ollama:
        #         self.cache = CacheManager(self.name, no_cache=True)

        self.use_cache = use_cache
        self.text2img = text2imgSD if image_generator == 'SD' else text2imgOpenAI
        self.use_local_image = use_local_image
    
        # Configure the response format
        if self.response_format == "json":
            self.response_format_config = {"type": "json_object"}
        else:
            self.response_format_config = {"type": "text"}

        # Initialize the model with the given configuration
        if self.use_ollama:
            self.model = ChatOllama(model=ollama_model_name, temperature=1)
        else:
            self.model = ChatOpenAI(
                model='gpt-4o' if not fast else "gpt-4o-mini",
                api_key=os.getenv('OPENAI_API_KEY'),
                # model_kwargs={"response_format": self.response_format_config},
            )

        # Initial system message and prompt template
        self.system_desc = system_desc or "You are a helpful assistant."
        
        if self.image_input:
            self.image_helper = ImageHelper(self.system_desc, num_images, image_detail)
            if self.use_local_image:
                self.prompt_template = ChatPromptTemplate.from_messages(
                    messages = self.image_helper.native_prepare_image_prompt_template()
                )
            else:
                self.prompt_template = ChatPromptTemplate.from_messages(
                    messages = self.image_helper.prepare_image_prompt_template()
                )
        else:
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.system_desc),
                ("human", "{input}")
            ])

    def __call__(self, query, image_paths=None, pydantic_object=None):

        if self.response_format == "pydantic":
            assert pydantic_object, "Pydantic object is required for response format 'pydantic'"
            
        if self.response_format == "image":
            return self.text2img(query)
        
        if self.use_cache and not self.image_input:
        # if self.use_cache and self.use_ollama:
            result = self.cache.respond(query)
            if result:
                return result
            # if result and not self.image_input:
            #     return result
            
        """Generates a response from the model based on the query and history."""
        # Append the user query to the history
        self.history = [{"role": "system", "content": self.system_desc}]
        self.history.append({"role": "human", "content": query})
        
        # Prepare the full prompt with chat history
        full_prompt = self._get_prompt_with_history()
        
        # Define the chain based on the response format
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
            full_prompt += """Return only JSON object, e.g.:
```json
{
    "key": "value"
}
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
        # Invoke the model and get the response
        if self.image_input:
            if self.use_local_image:
                result = self.image_helper.native_invoke_image_prompt_template(chain, full_prompt, image_paths)
            else:
                assert len(image_paths) == self.num_images, f"Number of images should be {self.num_images}."
                image_paths = self.image_helper.upload_images_to_s3(image_paths)
                result = self.image_helper.invoke_image_prompt_template(chain, full_prompt, image_paths)
        else:
            result = chain.invoke({"input": full_prompt})
        
        if self.response_format == "code":
            result = self._sanitize_output(result)

        if self.response_format == "json" and self.json_keys:
            # Check if all the keys are present in the response
            for key in self.json_keys:
                if key not in result:
                    query = """ 
                    For the query: {query}, the following response was generated: {response}. It didn't follow the expected format containing the keys: {self.json_keys}. Please ensure that the response follows the expected format and contains all the keys.
                    """
                    result = self(query, image_paths)
        
        # if self.use_ollama:
        if self.use_cache and not self.image_input:
            self.cache.append(query, result)
        # Return the response
        return result
    
    def _get_prompt_with_history(self):
        """Constructs the full prompt including chat history."""
        return "".join([f"{msg['role']}: {msg['content']}\n" for msg in self.history])
    
    def _sanitize_output(self, text: str):
        _, after = text.split("```python")
        return after.split("```")[0]
    
    def clear_cache(self):
        if self.use_cache:
        # if self.use_cache and not self.use_ollama:
            self.cache.clear()
        else:
            print("Cache is not enabled for this LLM instance.")