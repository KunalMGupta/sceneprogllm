import base64

from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

class ImageHelper:
    def __init__(self, system_desc, num_images, image_detail='low'):
        self.num_images = num_images
        self.system_desc = system_desc
        self.image_detail = image_detail
    
    def upload_images_to_s3(self, image_paths):
        import os
        import boto3
        from PIL import Image, ImageOps
        from concurrent.futures import ThreadPoolExecutor
        from botocore.exceptions import NoCredentialsError

        def compress_image(input_path, max_size_kb=50, quality=85, max_resolution=(512,512)):
            """
            Compress an image intelligently by adjusting resolution, quality, and format.

            Args:
                input_path (str): Path to the input image file.
                output_path (str): Path to save the compressed image file.
                max_size_kb (int): Maximum desired file size in kilobytes (default: 500KB).
                quality (int): Initial quality setting for compression (default: 85).
                max_resolution (tuple): Maximum resolution for the image (default: 1920x1080).

            Returns:
                str: Path to the compressed image.
            """
            # Open the image
            img = Image.open(input_path)

            # Convert to RGB mode if not already (to avoid issues with PNG/alpha)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Step 1: Resize the image if it's larger than max_resolution
            img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
            if img.size[0] > max_resolution[0] or img.size[1] > max_resolution[1]:
                img.thumbnail(max_resolution, Image.Resampling.LANCZOS)

            # Step 2: Save with initial quality and calculate size
            temp_path = input_path + "_temp.jpg"
            img.save(temp_path, format="JPEG", quality=quality, optimize=True)
            file_size_kb = os.path.getsize(temp_path) // 1024

            # Step 3: Adjust quality iteratively until file size is within limit
            while file_size_kb > max_size_kb and quality > 10:
                quality -= 5
                img.save(temp_path, format="JPEG", quality=quality, optimize=True)
                file_size_kb = os.path.getsize(temp_path) // 1024

            # Step 4: Save the final image
            os.rename(temp_path, input_path)
            # print(f"Compressed image saved at {input_path}, size: {file_size_kb}KB, quality: {quality}")

            return input_path

        def upload_image_to_s3(image_path, folder='transfer'):
            """
            Uploads an image to AWS S3 and returns a public URL.

            Args:
                image_path (str): Path to the image file on your local system.
                bucket_name (str): Name of the S3 bucket.
                folder (str): Folder in the S3 bucket where the file will be stored.
                object_name (str, optional): Name of the object in S3. Defaults to the image file name.
                aws_access_key (str): AWS Access Key ID.
                aws_secret_key (str): AWS Secret Access Key.
                region (str, optional): AWS region of the bucket. Default is 'us-east-1'.

            Returns:
                str: Public URL of the uploaded image, or None if upload fails.
            """
            bucket_name = os.getenv('AWS_S3_BUCKET')
            aws_access_key=os.getenv('AWS_ACCESS_KEY')
            aws_secret_key=os.getenv('AWS_SECRET_KEY')
            region=os.getenv('AWS_REGION')

            compress_image(image_path)

            object_name = image_path.split('/')[-1]  # Use file name if object name is not provided

            # Prepend the folder to the object name
            object_name = f"{folder}/{object_name}"

            # Initialize S3 client
            s3 = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
            try:
                # Upload the file to S3
                s3.upload_file(
                    Filename=image_path,
                    Bucket=bucket_name,
                    Key=object_name
                )

                # Generate the public URL
                url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_name}"
                return url

            except NoCredentialsError:
                print("Credentials not available.")
                return None
            except Exception as e:
                print(f"Failed to upload image: {e}")
                return None
            
        
        def upload_images_to_s3_multithreaded(image_paths, upload_function, max_threads=10):
            """
            Uploads multiple images to S3 using multi-threading.

            Args:
                image_paths (list): List of file paths to images.
                upload_function (function): The function to upload a single image to S3.
                max_threads (int): Maximum number of threads to use (default: 10).

            Returns:
                list: A list of image URLs.
            """
            image_urls = []

            def upload_wrapper(image_path):
                return upload_function(image_path)

            # Use ThreadPoolExecutor for multi-threading
            with ThreadPoolExecutor(max_threads) as executor:
                results = executor.map(upload_wrapper, image_paths)
            
            # Collect results
            image_urls.extend(results)

            return image_urls

        image_urls = upload_images_to_s3_multithreaded(image_paths, upload_image_to_s3)
        return image_urls
    
    def invoke_image_prompt_template(self, chain, prompt, image_paths):
        if self.num_images == 1:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'detail_parameter': self.image_detail})
        elif self.num_images == 2:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'detail_parameter': self.image_detail})
        elif self.num_images == 3:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'image_path3': image_paths[2], 'detail_parameter': self.image_detail})
        elif self.num_images == 4:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'image_path3': image_paths[2], 'image_path4': image_paths[3], 'detail_parameter': self.image_detail})
        elif self.num_images == 5:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'image_path3': image_paths[2], 'image_path4': image_paths[3], 'image_path5': image_paths[4], 'detail_parameter': self.image_detail})
        elif self.num_images == 6:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'image_path3': image_paths[2], 'image_path4': image_paths[3], 'image_path5': image_paths[4], 'image_path6': image_paths[5], 'detail_parameter': self.image_detail})
        elif self.num_images == 7:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'image_path3': image_paths[2], 'image_path4': image_paths[3], 'image_path5': image_paths[4], 'image_path6': image_paths[5], 'image_path7': image_paths[6], 'detail_parameter': self.image_detail})
        elif self.num_images == 8:
            return chain.invoke({'input':prompt, 'image_path1': image_paths[0], 'image_path2': image_paths[1], 'image_path3': image_paths[2], 'image_path4': image_paths[3], 'image_path5': image_paths[4], 'image_path6': image_paths[5], 'image_path7': image_paths[6], 'image_path8': image_paths[7], 'detail_parameter': self.image_detail})
        else:
            raise ValueError("Number of images should be between 1 and 8.")
    
    def native_invoke_image_prompt_template(self, chain, prompt, image_paths):
        image_urls = []
        to_invoke = {"input": prompt, "detail_parameter": self.image_detail}

        for i in range(self.num_images):
            with open(image_paths[i], "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
            to_invoke[f"image_data{i+1}"] = image_data
        
        return chain.invoke(to_invoke)    

    def native_prepare_image_prompt_template(self):
        messages = [
            ChatPromptTemplate.from_messages([
                ("system", self.system_desc),
                ("human", "{input}")
            ]),
        ]
        for i in range(self.num_images):
            messages.append(HumanMessagePromptTemplate.from_template(
                [{"image_url": {"url": f"data:image/jpeg;base64,{{image_data{i+1}}}"}}]
            ))

        return messages
        
    def prepare_image_prompt_template(self):
            if self.num_images == 1:
                return self.image1()
            elif self.num_images == 2:
                return self.image2()
            elif self.num_images == 3:
                return self.image3()
            elif self.num_images == 4:
                return self.image4()
            elif self.num_images == 5:
                return self.image5()
            elif self.num_images == 6:
                return self.image6()
            elif self.num_images == 7:
                return self.image7()
            elif self.num_images == 8:
                return self.image8()
            else:
                raise ValueError("Number of images should be between 1 and 8.")

    def image1(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image2(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image3(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path3}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image4(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path3}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path4}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image5(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path3}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path4}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path5}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image6(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path3}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path4}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path5}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path6}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image7(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path3}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path4}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path5}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path6}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path7}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages

    def image8(self):
        messages=[
                ChatPromptTemplate.from_messages([
                    ("system", self.system_desc),
                    ("human", "{input}")
                ]),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path1}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path2}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path3}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path4}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path5}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path6}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path7}', 'detail': '{detail_parameter}'}}]
                ),
                HumanMessagePromptTemplate.from_template(
                    [{'image_url': {'url': '{image_path8}', 'detail': '{detail_parameter}'}}]
                )
            ]
        return messages