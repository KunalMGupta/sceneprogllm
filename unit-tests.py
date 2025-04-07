import unittest
import ast
import os
from pydantic import BaseModel, Field
from sceneprogllm import LLM, clear_llm_cache

ASSET_IMAGE = "assets/lions.png"

class TestLLMBasics(unittest.TestCase):
    def setUp(self):
        self.model_name = "llama3.2-vision:90b"

    def test_text_response(self):
        llm = LLM(name="text_test", model_name=self.model_name, response_format="text", use_cache=False)
        response = llm("Hello")
        self.assertIsInstance(response, str)
        self.assertTrue(response)

    def test_json_response(self):
        llm = LLM(name="json_test", model_name=self.model_name, response_format="json", json_keys=["country:str"], use_cache=False)
        response = llm("What country is Tokyo in?")
        if isinstance(response, str):
            response = ast.literal_eval(response)
        self.assertIsInstance(response, dict)

    def test_list_response(self):
        llm = LLM(name="list_test", model_name=self.model_name, response_format="list", use_cache=False)
        response = llm("List programming languages")
        self.assertIsInstance(response, list)

    def test_pydantic_response(self):
        class CountryCapital(BaseModel):
            country: str = Field(description="Country name")
            capital: str = Field(description="Capital city")

        llm = LLM(name="pydantic_test", model_name=self.model_name, response_format="pydantic", use_cache=False)
        response = llm("Capital of Spain?", pydantic_object=CountryCapital)
        self.assertIsInstance(response, CountryCapital)

    def test_code_response(self):
        llm = LLM(name="code_test", model_name=self.model_name, response_format="code", use_cache=False)
        response = llm("Python function to add two numbers")
        self.assertIsInstance(response, str)
        self.assertTrue("def" in response or "lambda" in response)

    def test_image_generation(self):
        llm = LLM(name="image_gen_test", response_format="image", use_cache=False)
        response = llm("Generate an image of a robot in a forest")
        save_path = "test_output.jpg"
        response.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        os.remove(save_path)

    def test_image_query(self):
        if not os.path.exists(ASSET_IMAGE):
            self.skipTest(f"Test image not found: {ASSET_IMAGE}")

        llm = LLM(name="image_query_test", model_name=self.model_name, response_format="json", json_keys=["count:int"], use_cache=False)
        response = llm("How many lions?", image_paths=[ASSET_IMAGE])
        if isinstance(response, str):
            response = ast.literal_eval(response)
        self.assertIsInstance(response, dict)

    def test_clear_cache(self):
        try:
            clear_llm_cache()
        except Exception as e:
            self.fail(f"clear_llm_cache raised an error: {e}")

    def test_seed_and_temperature(self):
        llm = LLM(name="seed_temp_test", model_name=self.model_name, response_format="text", seed=42, temperature=0.7, use_cache=False)
        response = llm("Hello!")
        self.assertIsInstance(response, str)

    def test_system_description(self):
        llm = LLM(name="funny_bot", model_name=self.model_name, response_format="text", system_desc="You are a funny assistant", use_cache=False)
        response = llm("What is the capital of Japan?")
        self.assertIsInstance(response, str)


class TestLLMGPT4OConfig(unittest.TestCase):
    def setUp(self):
        self.model_name = "gpt-4o"

    def test_all_configs_with_gpt4o(self):
        llm = LLM(name="gpt4o_test", model_name=self.model_name, response_format="text", use_cache=False, seed=123, temperature=0.8, system_desc="Be brief.")
        response = llm("Say something.")
        self.assertIsInstance(response, str)


class TestLLMImageInputFormats(unittest.TestCase):
    def setUp(self):
        self.model_name = "gpt-4o"
        if not os.path.exists(ASSET_IMAGE):
            self.skipTest(f"Test image not found: {ASSET_IMAGE}")

    def test_image_input_text_output(self):
        llm = LLM(name="img_text", model_name=self.model_name, response_format="text", use_cache=False)
        response = llm("Describe this image.", image_paths=[ASSET_IMAGE])
        self.assertIsInstance(response, str)

    def test_image_input_list_output(self):
        llm = LLM(name="img_list", model_name=self.model_name, response_format="list", use_cache=False)
        response = llm("List all animals you see.", image_paths=[ASSET_IMAGE])
        self.assertIsInstance(response, list)

    def test_image_input_json_output(self):
        llm = LLM(name="img_json", model_name=self.model_name, response_format="json", json_keys=["count:int"], use_cache=False)
        response = llm("How many animals?", image_paths=[ASSET_IMAGE])
        if isinstance(response, str):
            response = ast.literal_eval(response)
        self.assertIsInstance(response, dict)

    def test_image_input_pydantic_output(self):
        class AnimalCount(BaseModel):
            count: int = Field(description="Number of animals")

        llm = LLM(name="img_pydantic", model_name=self.model_name, response_format="pydantic", use_cache=False)
        response = llm("How many animals are shown?", image_paths=[ASSET_IMAGE], pydantic_object=AnimalCount)
        self.assertIsInstance(response, AnimalCount)

    def test_image_input_code_output(self):
        llm = LLM(name="img_code", model_name=self.model_name, response_format="code", use_cache=False)
        response = llm("Write Python code to count objects in this image.", image_paths=[ASSET_IMAGE])
        self.assertIsInstance(response, str)
        self.assertTrue("def" in response or "import" in response)


if __name__ == "__main__":
    unittest.main()