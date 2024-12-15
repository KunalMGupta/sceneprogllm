# **SceneProgLLM**

**SceneProgLLM** is a powerful and versatile Python package that wraps around LangChain's LLM interface to provide enhanced functionality, including support for text, code, and JSON response formats, image input/output, caching, and multiple endpoints. This project is built to support SceneProg projects. 

---

## **Features**
1. **Flexible Response Formats**: 
   - Supports text, code, JSON, and image outputs.
2. **Image Input and Output**: 
   - Accepts image inputs and enables image generation through Stable Diffusion (SD) or OpenAI's image generation API.
3. **Caching**: 
   - Integrated caching system to store and retrieve previous query responses for faster execution.
4. **Strict JSON Validation**: 
   - Ensures correct JSON structure, particularly when specific keys are required.
---

## **Installation**
To install the package and its dependencies, use the following command:
```bash
pip install sceneprogllm
```

For proper usage, export the respective variables
```bash
export OPENAI_API_KEY= YOUR_OPENAI_API_KEY
export AWS_ACCESS_KEY= YOUR_AWS_ACCESS_KEY
export AWS_SECRET_KEY= YOUR_AWS_SECRET_KEY
export AWS_REGION = AWS_REGION
export AWS_S3_BUCKET= AWS_S3_BUCKET
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
response = llm.run("What is the capital of France?")
print(response)
```
2. **Generating JSON Responses**
```python
llm = LLM(
    name="json_bot",
    response_format="json",
    json_keys=["capital", "currency"]
)
query = "What is capital and currency of India?"
response = llm.run(query)
print(response)
```
3. **Generating Python Code**
```python
llm = LLM(name="code_bot", response_format="code")
query = "Write a Python function to calculate factorial of a number."
response = llm.run(query)
print(response)
```
4. **Generating images from text**
```python
llm = LLM(name="image_bot", response_format="image")
response = llm.run("Generate an image of a futuristic cityscape.")
response.save("futuristic_city.jpg")
```
5. **Query using Images**
```python
llm = LLM(name="image_bot", response_format="text", num_images=1, image_generator="SD")
image_paths = ["path/to/input_image.jpg"]
response = llm.run("What is the color of the object in the image?", image_paths=image_paths)
```
6. **Clear LLM cache**
```python
from sceneprogllm import clear_llm_cache
clear_llm_cache()
```


### **Using Ollama**

To use Ollama with the `LLM` class, you need to set the `use_ollama` parameter to `True` when initializing the `LLM` object. 

Furthermore, you can specify the Ollama Model via `ollama_model_name`, the default is `"llama3.2-vision"`. See [Ollama model site](https://ollama.com/search) for the available options. Note that different model will support different modes (text, image, etc.).

Here is an example:

```python
from sceneprogllm import LLM

# Example for generating text responses using Ollama
llm = LLM(name="text_bot", response_format="text", use_ollama=True)
```