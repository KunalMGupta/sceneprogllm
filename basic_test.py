import unittest
import ast
from sceneprogllm.llm import LLM

class TestLLM(unittest.TestCase):

    def test_text_response(self):
        self.llm = LLM(name="test", model_name="llama3.2-vision", response_format="text", use_cache=False)
        response = self.llm("Hello")
        self.assertIsInstance(response, str)

    def test_json_response(self):
        self.llm = LLM(name="test", model_name="llama3.2-vision", response_format="json", use_cache=False)
        response = self.llm("Hello")
        # If response is a string, convert it to a dictionary.
        if isinstance(response, str):
            try:
                response = ast.literal_eval(response)
            except Exception as e:
                self.fail(f"Response string could not be parsed as dict: {e}")
        self.assertIsInstance(response, dict)


    def test_list_response(self):
        self.llm = LLM(name="test", model_name="llama3.2-vision", response_format="list", use_cache=False)
        response = self.llm("List items")
        self.assertIsInstance(response, list)

if __name__ == "__main__":
    unittest.main()