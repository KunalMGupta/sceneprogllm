from sceneprogllm.llm import LLM

llm = LLM(
    name="json_bot",
    response_format="json",
    json_keys=["capital", "currency"],
    llm_type="gemini"
)
query = "What is capital and currency of India?"
response = llm.run(query)
print(response)
