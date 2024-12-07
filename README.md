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
```
pip install sceneprogllm
```

## **Getting Started**
Importing the Package
```
from sceneprogllm import LLM
```

## **Usage Examples**
1. **Generating Text Responses**
```
llm = LLM(name="text_bot", response_format="text")
response = llm.run("What is the capital of France?")
print(response)
```
2. **Generating JSON Responses**
```
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
```
llm = LLM(name="code_bot", response_format="code")
query = "Write a Python function to calculate factorial of a number."
response = llm.run(query)
print(response)
```
4. **Generating images from text**
```
llm = LLM(name="image_bot", response_format="image")
response = llm.run("Generate an image of a futuristic cityscape.")
response.save("futuristic_city.jpg")
```
5. **Query using Images**
```
llm = LLM(name="image_bot", response_format="text", num_images=1, image_generator="SD")
image_paths = ["path/to/input_image.jpg"]
response = llm.run("What is the color of the object in the image?", image_paths=image_paths)
```
6. **Clear LLM cache**
```
from sceneprogllm import clear_llm_cache
clear_llm_cache()
```# sceneprogllm
