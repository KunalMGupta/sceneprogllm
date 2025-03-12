from sceneprogllm import LLM

llm = LLM(name="image_bot", response_format="json", model_name='llama3.2-vision', use_cache=True, json_keys=["response"])
# image_paths = ["dog.jpg"]
image_paths=None
response = llm("What's today's date")
print(response)