import unittest
from sceneprogllm.llm import LLM

class TestLLM(unittest.TestCase):

    def setUp(self):
        self.llm = LLM(name="test")

    def test_text_response(self):
        response = self.llm("Hello")
        self.assertIsInstance(response, str)

    def test_json_response(self):
        self.llm.response_format = "json"
        response = self.llm("Hello")
        self.assertIsInstance(response, dict)

    def test_list_response(self):
        self.llm.response_format = "list"
        response = self.llm("List items")
        self.assertIsInstance(response, list)

    def test_cache(self):
        self.llm.use_cache = True
        query = "Hello"
        response1 = self.llm(query)
        response2 = self.llm(query)
        self.assertEqual(response1, response2)

if __name__ == "__main__":
    unittest.main()