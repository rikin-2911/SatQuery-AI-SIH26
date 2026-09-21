## Grounding Agent/Node
## Make bounding box on the images and give the info about the image + VQA


## image.tif ---> normalize rgb --> 
import os
import base64
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN_2")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")


MODEL = "Qwen/Qwen3.5-27B"

# NODE
def grounding_node(tiff_image, query):

    if not tiff_image:
        raise Exception("There is no Input Image for Geospatial Detection or Grounding Analysis")

    # Convert the .tiff to .jpeg image file
    def tiff_to_jpeg(tiff_image):

        output_path = Path(tiff_image).with_suffix(".jpeg")

        with rasterio.open(tiff_image) as src:
            tiff = src.read(1).astype(np.float32)

        # Robust contrast stretching
        p2, p98 = np.nanpercentile(tiff, (2, 98))

        # Avoid division by zero
        if p98 <= p2:
            tiff_vis = np.zeros_like(tiff, dtype=np.float32)
        else:
            # normalization
            tiff_vis = np.clip((tiff - p2) / (p98 - p2), 0, 1)

        # 8-bit image for display
        jpeg_uint8 = (tiff_vis * 255).astype(np.uint8)

        # save to image format
        Image.fromarray(jpeg_uint8).save(output_path, format="JPEG", quality=85)

        return str(output_path)
        

    # resize the image pixels to 1080 x 1080
    def resize__image(jpeg_path):

        # jepg image path
        jpeg_path = Path(jpeg_path)

        # Output filename
        output_path = jpeg_path.with_name(jpeg_path.stem + "_1080.jpeg")

        img = Image.open(jpeg_path)

        # Resize while preserving the complete scene
        img = img.resize((1080, 1080), Image.Resampling.LANCZOS)

        # Qwen vision input: convert grayscale SAR visualization to 3-channel
        img = img.convert("RGB")

        img.save(output_path, format="JPEG", quality=85, optimize=True)

        return str(output_path)

        
    # jpeg image to data url
    def image_to_data_url(resized_jpeg_path):

        with open(resized_jpeg_path, "rb") as f:
            image_bytes = f.read()

        encoded = base64.b64encode(image_bytes).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"


    # tiff to resize jpeg
    jpeg_path = tiff_to_jpeg(tiff_image)

    resized_jpeg_path = resize__image(jpeg_path)

    image_url = image_to_data_url(resized_jpeg_path)

    system_prompt = """
    You are an expert remote-sensing visual grounding agent.

    Your task is to identify and localize requested geographic features in
    satellite/aerial imagery using visual evidence only.

    Rules:

    1. Identify only features clearly supported by the image. Never
    hallucinate objects or regions.

    2. Return normalized bounding boxes:
    [x_min, y_min, x_max, y_max]

    Coordinate system:
    (0,0) = top-left
    (1,1) = bottom-right

    3. Bounding boxes must satisfy:
    0 <= x_min <= x_max <= 1
    0 <= y_min <= y_max <= 1

    4. Make boxes as tight as the visible evidence allows. If boundaries
    are uncertain, use an approximate box and lower confidence.

    5. If the requested feature is not reliably visible:
    found = false
    bounding_box = null
    confidence = 0.0

    6. Confidence must be between 0 and 1 and represent confidence in both
    feature identification and localization.

    7. Use remote-sensing visual cues such as texture, spatial patterns,
    land cover, terrain, vegetation, water, and built-up areas. Do not
    invent sensor information, coordinates, dates, CRS, or geographic
    names.

    8. If multiple distinct instances are requested, return separate
    detections.

    9. Return ONLY valid JSON. No Markdown, explanations, or reasoning.

    Output format:

    {
    "detections": [
        {
        "feature": "feature name",
        "found": true,
        "bounding_box": [0.0, 0.0, 0.0, 0.0],
        "confidence": 0.0,
        "description": "Brief evidence-based description."
        }
    ]
    }
    """

    # user input query from langgarpg state
    user_query = query

    # client for Qwen Model
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B",

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ],

        max_tokens=2048,
        temperature=0.7,
        top_p=0.8,

        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
        },
    )

    message = response.choices[0].message
    return message.content




# Testing
# Input image
#tiff_img = "/home/rikin/satquery-ai/satquery_sar_test/sample_VV.tif"

# user query
#query = """
#    Identify the approximate locations of the following features in the SAR image:

#    1. The main water body
#    2. The major mountainous region
#    3. The densest forested region
#    4. Any visible built-up or settlement area

#    """

#print(grounding_node(tiff_img, query))
