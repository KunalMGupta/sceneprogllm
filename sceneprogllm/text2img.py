import os
import bentoml
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

def text2imgSD(text):
    with bentoml.SyncHTTPClient(os.getenv('TEXT2IMGSD')) as client:
        result = client.txt2img(
            prompt=text,
            num_inference_steps=1,
            guidance_scale=0.0
        )
    result = Image.open(result)
    return result

def text2imgOpenAI(text):
    from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
    from io import BytesIO
    import requests
    
    image_url = DallEAPIWrapper(api_key=os.getenv('OPENAI_API_KEY')).run(text)
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img