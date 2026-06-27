# **SceneProgLLM**

**SceneProgLLM** is a powerful and versatile Python package that wraps around LangChain's LLM interface to provide enhanced functionality, including support for text, code, JSON, list, pydantic, image, speech and embedding response formats along with image inputs. This project is built to support SceneProg projects and supports both **OpenAI** (GPT) and **Anthropic** (Claude) backends — the right backend and API key are selected automatically based on the `model_name` you pass.

---

## **Features**
1. **Flexible Response Formats**: 
   - Supports text, code, list, JSON, pydantic, image, speech and embedding outputs.
2. **Image Input and Output**: 
   - Accepts image inputs and enables image generation through OpenAI's image generation API.
3. **System Template**:
    - Allows users to set a system description template containing placeholders which can be later filled with values. 
---

## **Installation**
To install the package and its dependencies, use the following command:
```bash
pip install sceneprogllm
```

For proper usage, create a ```.env``` file in the package root with the keys for the backends you intend to use:
```text
OPENAI_API_KEY=<Your OpenAI key>
ANTHROPIC_API_KEY=<Your Anthropic key>
```

The backend is chosen from `model_name`: any model starting with `claude` (e.g. `claude-opus-4-8`, `claude-sonnet-4-6`, `claude-haiku-4-5`) uses Anthropic with `ANTHROPIC_API_KEY`; everything else (e.g. `gpt-5-nano`) uses OpenAI with `OPENAI_API_KEY`. You can still pass a key explicitly via `api_key=...`.

> Note: `image`, `speech` and `embedding` response formats are generated through OpenAI APIs regardless of `model_name`, since Anthropic does not provide equivalent endpoints. These always require `OPENAI_API_KEY`.

## **Getting Started**
Importing the Package
```python
from sceneprogllm import LLM
```

## **Usage Examples**
1. **Generating Text Responses**
```python
llm = LLM(response_format="text")
response = llm("What is the capital of France?")
print(response)

>> The capital of France is Paris.
```

2. **Generating JSON Responses**
```python
llm = LLM(
    response_format="json",
    response_params=["capital":"str", "currency":"str"]
)
query = "What is capital and currency of India?"
response = llm(query)
print(response)

>> {'capital': 'New Delhi', 'currency': 'Indian Rupee'}
```

3. **Generating List Responses**
```python
llm = LLM(
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
    response_format="pydantic",
)
response = llm("What is the capital of France?", pydantic_object=mypydantic)
print(response)

>> country='France' capital='Paris'
```

5. **Generating Python Code**
```python
llm = LLM(response_format="code")
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
llm = LLM(response_format="image", response_params={"size":"1024x1536", "quality":"auto"})
response = llm("Generate an image of a futuristic cityscape.")
response.save("futuristic_city.png")

>> 
```

7. **Generating speech from text**
```python
llm = LLM(response_format="speech", response_params={"output_path":"speech.wav", "voice":"coral"})
speech_path = llm("This is how I speak!")

>> 
```

8. **Generating embeddings from text**
```python
llm = LLM(response_format="embedding")
response = llm(["Hello World", "I like you"])

>> 
```

9. **Query using Images**
```python
llm = LLM(response_format="json", response_params=["count":"int"])
image_paths = ["assets/lions.png"]
response = llm("How many lions are there in the image?", image_paths=image_paths)
print(response)

>> {'count': 6}
```

10. **Generating images from text and image**
```python
llm = LLM(response_format="image", response_params={"size":"1024x1536", "quality":"auto"})
response = llm("Make the picture realistic", image_paths=["futuristic_city.png"])
response.save("real_futuristic_city.png")

>> 
```

11. **Set seed and temperature**
```python
llm = LLM(
         seed=0,
         temperature=1.0
         )
```

12. **Control behavious via system description**
```python
llm = LLM(
   system_desc="You are a funny AI assistant",
)
response = llm("What is the capital of France")
print(response)

>> 

Ah, the capital of France! That's Paris, the city of romance, lights, and baguettes longer than your arm! Just imagine the Eiffel Tower wearing a beret and saying, "Bonjour!"
```

13. **Using Template**
```python
from sceneprogllm import LLM
llm = LLM(
    system_desc="You are a helpful assistant. {description}",
)

response = llm("What is the capital of France?", system_desc_keys={"description": "You are a funny AI assistant"})
print(response)
```

14. **Using a Claude model**
```python
# Just change model_name — the Anthropic backend and ANTHROPIC_API_KEY are picked automatically.
llm = LLM(model_name="claude-sonnet-4-6", response_format="text")
response = llm("What is the capital of France?")
print(response)

# Structured outputs work the same across backends:
llm = LLM(model_name="claude-opus-4-8", response_format="json", response_params={"capital": "str"})
print(llm("What is the capital of India?"))
```


## Queries

Please send your questions to k5gupta@ucsd.edu

