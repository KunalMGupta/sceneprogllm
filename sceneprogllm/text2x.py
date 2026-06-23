import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI, api_key
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from .image_helper import convert_to_base64

def text2img(
    text,
    image_paths=None,
    mask_path=None,
    size="auto",
    quality="auto",
    background="transparent",
    api_key=None,
):
    assert size in ["1024x1024", "1536x1024", "1024x1536", "auto"], \
        "Invalid size. Must be one of '1024x1024', '1536x1024', '1024x1536', 'auto'"
    assert quality in ["low", "medium", "high", "auto"], \
        "Invalid quality. Must be one of 'low', 'medium', 'high', 'auto'"
    assert background in ["transparent", "opaque", "auto"], \
        "Invalid background. Must be one of 'transparent', 'opaque', 'auto'"

    if mask_path and not image_paths:
        raise ValueError("mask_path requires at least one base image in image_paths (the mask applies to the first image).")

    client = OpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))

    # ---- Build input (text + optional reference images) ----
    if image_paths:
        content = [{"type": "input_text", "text": text}]

        for path in image_paths:
            img_b64, fmt = convert_to_base64(path)
            content.append({
                "type": "input_image",
                "image_url": f"data:image/{fmt};base64,{img_b64}",
            })

        input_payload = [{"role": "user", "content": content}]
    else:
        input_payload = text  # text-only

    # ---- Build tool options (add mask if provided) ----
    # Transparent backgrounds are only supported by gpt-image-1; the Responses
    # API's newer default image model rejects them. Use the default (newer) model
    # in general, and only pin gpt-image-1 when transparency is actually needed.
    tool = {
        "type": "image_generation",
        "quality": quality,
        "size": size,
    }

    if background == "transparent":
        tool["model"] = "gpt-image-1"
        tool["background"] = background
    elif background != "auto":
        tool["background"] = background

    if mask_path:
        # Masks are typically expected to be PNG with alpha. Convert if needed.
        mask_b64, mask_fmt = convert_to_base64(mask_path)

        if mask_fmt != "png":
            # Convert to PNG in-memory to preserve alpha channel behavior.
            with Image.open(mask_path) as m:
                m = m.convert("RGBA")
                buf = BytesIO()
                m.save(buf, format="PNG")
                mask_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                mask_fmt = "png"

        tool["input_image_mask"] = {
            "image_url": f"data:image/{mask_fmt};base64,{mask_b64}"
        }

    # ---- Call API ----
    response = client.responses.create(
        model="gpt-5",
        input=input_payload,
        tools=[tool],
    )

    # ---- Extract image ----
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    if not image_data:
        raise RuntimeError(f"No image returned. Output: {response.output}")

    image_bytes = base64.b64decode(image_data[0])
    return Image.open(BytesIO(image_bytes))

def text2speech(
    text: str,
    output_path: str | Path,
    *,
    model: str = "gpt-4o-mini-tts",
    voice: str = "coral",
    instructions: str | None = None,
    response_format: str = "mp3",
    speed: float = 1.0,
    stream_format: str = "audio",
    api_key=None,
):
    """
    Convert text to speech and stream audio to a file.

    :param text: Text to synthesize (max 4096 chars)
    :param output_path: Path to write the audio file
    :param model: tts-1, tts-1-hd, or gpt-4o-mini-tts
    :param voice: alloy, ash, ballad, coral, echo, fable, onyx,
                  nova, sage, shimmer, or verse
    :param instructions: Optional style/tone instructions
                         (only supported for gpt-4o-mini-tts)
    :param response_format: mp3, opus, aac, flac, wav, or pcm
    :param speed: Playback speed (0.25–4.0)
    :param stream_format: audio or sse (sse unsupported for tts-1/hd)
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))

    request = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": response_format,
        "speed": speed,
        "stream_format": stream_format,
    }

    # Instructions only supported on gpt-4o-mini-tts
    if instructions and model == "gpt-4o-mini-tts":
        request["instructions"] = instructions

    with client.audio.speech.with_streaming_response.create(**request) as response:
        response.stream_to_file(output_path)

    return output_path

def text2embeddings(
    texts: list[str],
    *,
    model: str = "text-embedding-3-large",
    api_key=None,
) -> list[list[float]]:
    """
    Create embeddings for a list of texts in a single request.

    :param texts: List of input strings
    :param model: Embedding model
    :return: List of embedding vectors
    """
    if type(texts) is str:
        texts = [texts]
    cleaned = [t.replace("\n", " ") for t in texts]
    client = OpenAI(api_key=api_key if api_key is not None else os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(
        model=model,
        input=cleaned,
    )

    return np.array([item.embedding for item in response.data])