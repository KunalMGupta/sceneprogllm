# **SceneProgLLM**

**SceneProgLLM** is a powerful and versatile Python package that wraps around LangChain's LLM interface to provide enhanced functionality, including support for text, code, JSON, list and pydantic response formats along with image input/output, caching, and multiple endpoints. This project is built to support SceneProg projects. 

---

## **Features**
1. **Flexible Response Formats**: 
   - Supports text, code, list, JSON, pydantic and image outputs.
2. **Image Input and Output**: 
   - Accepts image inputs and enables image generation through Stable Diffusion (SD) or OpenAI's image generation API.
3. **Caching**: 
   - Integrated caching system to store and retrieve previous query responses for faster execution.
---

## **Installation**
To install the package and its dependencies, use the following command:
```bash
pip install sceneprogllm
```

For proper usage, create a ```.env``` file in the package root with following fields:
```text
TEXT2IMGSD=<endpoint for text to image generation>
OPENAI_API_KEY=<Your OpenAI key>
OLLAMA_HOST=Ollama host IP address
OLLAMA_PORT=Ollama host Port
```

## **Getting Started**
Importing the Package
```python
from sceneprogllm import LLM
```

## **Usage Examples**
1. **Generating Text Responses**
```python
llm = LLM(name="text_bot", response_format="text")
response = llm("What is the capital of France?")
print(response)

>> The capital of France is Paris.
```
2. **Generating JSON Responses**
```python
llm = LLM(
    name="json_bot",
    response_format="json",
    json_keys=["capital:str", "currency:str"]
)
query = "What is capital and currency of India?"
response = llm(query)
print(response)

>> {'capital': 'New Delhi', 'currency': 'Indian Rupee'}
```

3. **Generating List Responses**
```python
llm = LLM(
   name="list_bot",
   response_format="list",
)
query = "List G7 countries"
response = llm(query)
print(response)

>> ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'United Kingdom', 'United States']
```

4. **Generating Pydantic Responses**
```python
from pydantic import BaseModel, Field
class mypydantic(BaseModel):
    country: str = Field(description="Name of the country")
    capital: str = Field(description="Capital city of the country")

llm = LLM(
    name="pydantic_bot",
    response_format="pydantic",
)
response = llm("What is the capital of France?", pydantic_object=mypydantic)
print(response)

>> country='France' capital='Paris'
```

5. **Generating Python Code**
```python
llm = LLM(name="code_bot", response_format="code")
query = "Write a Python function to calculate factorial of a number."
response = llm(query)
print(response)

>>
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
```
6. **Generating images from text**
```python
llm = LLM(name="image_bot", response_format="image")
response = llm("Generate an image of a futuristic cityscape.")
response.save("futuristic_city.jpg")

>> 
```
![Futuristic City](assets/futuristic_city.jpg)

7. **Query using Images**
```python
llm = LLM(name="image_bot", response_format="json", json_keys=["count:int"])
image_paths = ["assets/lions.png"]
response = llm("How many lions are there in the image?", image_paths=image_paths)
print(response)

>> {'count': 6}
```
![lions](assets/lions.png)

8. **Clear LLM cache**
```python
from sceneprogllm import clear_llm_cache
clear_llm_cache()
```

9. **Set seed and temperature**
```python
llm = LLM(
         name="seed_bot",
         seed=0,
         temperature=1.0
         )
```

10. **Control behavious via system description**
```python
llm = LLM(
   name="system_bot",
   system_desc="You are a funny AI assistant",
)
response = llm("What is the capital of France")
print(response)

>> 

Ah, the capital of France! That's Paris, the city of romance, lights, and baguettes longer than your arm! Just imagine the Eiffel Tower wearing a beret and saying, "Bonjour!"
```
### **Using Ollama**

Additional models like Deekseek-R1 and Llama3.2-vision are available via Ollama with the `LLM` class, make sure Ollama is installed. Then, when initializing the `LLM` object, specify the Ollama Model via `model_name`. For example, `"llama3.2-vision"`. 

See [Ollama model site](https://ollama.com/search) for the available options. Note that different model will support different modes (text, image, etc.).


Here is an example:

```python
from sceneprogllm import LLM

# Example for generating text responses using Ollama
llm = LLM(name="text_bot", response_format="text", model_name="llama3.2-vision:90b")
```

## Queries

Please send your questions to k5gupta@ucsd.edu

