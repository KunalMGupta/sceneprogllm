import os
from pydantic import BaseModel, create_model
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
load_dotenv()
from .image_helper import ImageHelper
from .text2img import text2imgSD, text2imgOpenAI

from .cache_manager import CacheManager

class ListResponse(BaseModel):
    response: list[str]

class DefaultJsonResponse(BaseModel):
    response: str

class LLM:
    def __init__(self, 
                 name, 
                 system_desc="You are a helpful assistant.",
                 response_format="text",
                 json_keys=None,
                 use_cache=True,
                 model_name='gpt-4o',
                 seed=124,
                 temperature=0.8,
                 ):
        assert response_format in ['text', 'list', 'code', 'json', 'image', 'pydantic'], "Invalid response format, must be one of 'text', 'list', 'code', 'json', 'image', 'pydantic'"
        self.name = name
        self.response_format = response_format
        self.text2img = text2imgOpenAI

        if use_cache:
            self.cache = CacheManager(self.name, no_cache=not use_cache)

        if self.response_format == "json":
            if json_keys is not None:
                self.json_keys = [(key.split(':')[0], key.split(':')[1] if ':' in key else 'str') for key in json_keys]
            else:
                raise ValueError("json_keys must be provided when response_format is 'json'")
        
        self.use_cache = use_cache
        self.model_name = model_name
        self.system_desc = system_desc
        self.temperature = temperature
        self.seed = seed

        if 'gpt' in model_name:
            self.model = ChatOpenAI(model_name=model_name, api_key=os.getenv("OPENAI_API_KEY"),seed=self.seed, temperature=self.temperature)
        elif 'llama' in model_name or 'deepseek' in model_name:
            self.model = ChatOllama(model=model_name, temperature=self.temperature, seed=self.seed)
        elif 'SD' in model_name:
            self.text2img = text2imgSD
        else:
            raise ValueError("Unsupported model name. Please use 'gpt' or 'ollama'.")
        
        self.image_helper = ImageHelper(self.system_desc)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_desc),
            ("human", "{input}")
        ])

        self.msg_header = f"""
"system": "{self.system_desc}",
"""
    def __call__(self, query, image_paths=None, pydantic_object=None):

        # sanitize the query and system description
        query = query.replace('{', '{{').replace('}', '}}')
        self.system_desc = self.system_desc.replace('{', '{{').replace('}', '}}')

        # Early return for image response format
        if self.response_format == "image":
            return self.text2img(query)
        
        if self.response_format == "pydantic":
            assert pydantic_object is not None, "pydantic_object must be provided when response_format is 'pydantic'"
        elif self.response_format == "list":
            pydantic_object = ListResponse
        elif self.response_format == "json":
            if self.json_keys is not None:
                CustomJSONModel = create_model(
                    'CustomJSONModel',
                    **{key: (type, ...) for key, type in self.json_keys}
                )
                pydantic_object = CustomJSONModel
            else:
                pydantic_object = DefaultJsonResponse
        
        if self.use_cache and not image_paths:

            commit = self.compute_commit(query, pydantic_object)
            cached_result = self.cache.respond(commit, pydantic_object)
            if cached_result:
                return cached_result

        full_prompt = self.msg_header + f"""
"human": "{query}",
"""
        if pydantic_object is not None:
            parser = PydanticOutputParser(pydantic_object=pydantic_object)
            self.prompt_template = PromptTemplate(
                template="Answer the user query.\n{format_instructions}\n{input}\n",
                input_variables=["input"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
        else:
            parser = StrOutputParser()
        
        if self.response_format == "code":
            full_prompt += """Return only python code in Markdown format, e.g.:
```python
....
```"""
        if image_paths is not None:
            if pydantic_object is not None:
                format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
            else:
                format_instructions = "Responsed as plain text."

            # Special case of llama3.2-vision as it can take only one image
            if 'llama3.2-vision' in self.model_name:
                assert len(image_paths) == 1, "Only one image is supported for llama3.2-vision"
                from langchain_ollama import OllamaLLM
                from .image_helper import convert_to_base64
                self.model = OllamaLLM(model=self.model_name, temperature=self.temperature, seed=self.seed)
                image_b64 = convert_to_base64(image_paths[0])[0]
                llm_with_image_context = self.model.bind(images=[image_b64])
                result = llm_with_image_context.invoke(full_prompt+"\n"+format_instructions)
                result = parser.parse(result)
            else:
                self.prompt_template = ChatPromptTemplate.from_messages(
                    messages = self.image_helper.prepare_image_prompt_template(image_paths, format_instructions),
                )
                chain = self.prompt_template | self.model | parser
                result = self.image_helper.invoke_image_prompt_template(chain, full_prompt, image_paths)
        else:
            chain = self.prompt_template | self.model | parser
            result = chain.invoke({"input": full_prompt})
        
        if self.response_format == "code":
            result = self._sanitize_output(result)

        if self.use_cache and not image_paths:
            commit = self.compute_commit(query, pydantic_object)
            self.cache.append(commit, result)
        
        if self.response_format == "json":
            result = result.model_dump()
            tmp = {}
            for key, type in self.json_keys:
                if type == 'bool':
                    tmp[key] = bool(result[key])
                elif type == 'str':
                    tmp[key] = str(result[key])
                elif type == 'int':
                    tmp[key] = int(result[key])
                elif type == 'float':
                    tmp[key] = float(result[key])
                else:
                    raise ValueError(f"Unsupported type: {type}")
            result = tmp
        if self.response_format == "list":
            result = result.response
        return result
    
    def _sanitize_output(self, text: str):
        _, after = text.split("```python")
        return after.split("```")[0]
    
    def clear_cache(self):
        if self.use_cache:
            self.cache.clear()
        
    def compute_commit(self, query, pydantic_object):
        commit = query + "\n"+ self.system_desc + f"\nresponse_format={self.response_format}\nmodel_name={self.model_name}\nseed={self.seed}\ntemperature={self.temperature}"
        if pydantic_object is not None:
            commit += f"\npydantic_object={pydantic_object.__name__}"
        return commit