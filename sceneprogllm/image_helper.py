import base64
from io import BytesIO
from PIL import Image

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

def convert_to_base64(image_path):
    """
    Convert images to Base64 encoded strings

    :param image_path: Path to the image file
    :return: Base64 encoded string
    """
    with Image.open(image_path) as pil_image:
        buffered = BytesIO()
        format = pil_image.format  # Get the image format (e.g., JPEG, PNG)
        pil_image.save(buffered, format=format)  # Save the image in its original format
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str, format.lower()

class ImageHelper:
    def __init__(self, system_desc):
        self.system_desc = system_desc

    def prepare_image_prompt_template(self, image_paths):
        messages = [
            ChatPromptTemplate.from_messages([
                ("system", self.system_desc),
                ("human", "{input}")
            ]),
        ]
        for i in range(len(image_paths)):
            img_str, format = convert_to_base64(image_paths[i])
            messages.append(HumanMessagePromptTemplate.from_template(
                [{"image_url": {"url": f"data:image/{format};base64,{{image_data{i+1}}}"}}]
            ))

        return messages
    def invoke_image_prompt_template(self, chain, prompt, image_paths):
        to_invoke = {"input": prompt}

        for i in range(len(image_paths)):
            img_str, _ = convert_to_base64(image_paths[i])
            to_invoke[f"image_data{i+1}"] = img_str

        return chain.invoke(to_invoke)