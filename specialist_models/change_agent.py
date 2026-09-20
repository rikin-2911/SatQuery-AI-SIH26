## Change or Bi-Temporal NODE 
## In this file, the change_node function inputs two images (Before and After) + User Query

import os
import base64
import json
from pathlib import Path
from PIL import Image
import rasterio
import numpy as np


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN_2")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

# Model name
MODEL = "Qwen/Qwen3.5-27B"

## CHANGE NODE
def change_node(before_tiff_image, after_tiff_image, query):

    # check both images are uploaded or not...
    if not before_tiff_image:
        raise Exception("Earlier or Image 1 is missing for Analysis !")

    if not after_tiff_image:
        raise Exception("After or Image 2 is missing for Analysis !")
    
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


    # Image 1 -- tiff to resize jpeg
    image_1_jpeg_path = tiff_to_jpeg(before_tiff_image)
    resized_jpeg_path = resize__image(image_1_jpeg_path)
    image_1_url = image_to_data_url(resized_jpeg_path)

    # Image 1 -- tiff to resize jpeg
    image_2_jpeg_path = tiff_to_jpeg(after_tiff_image)
    resized_jpeg_path = resize__image(image_2_jpeg_path)
    image_2_url = image_to_data_url(resized_jpeg_path)


    system_prompt = """
    You are an expert remote-sensing and satellite image analyst.

    You are given TWO satellite images representing the same geographic area at different points in time.

    Image 1 = EARLIER observation.
    Image 2 = LATER observation.

    Your task is to identify meaningful visual changes between the two
    observations.

    Compare the images spatially and temporally.

    Look for changes such as:

    - newly constructed buildings
    - demolished buildings
    - expansion of built-up areas
    - vegetation loss
    - vegetation growth
    - agricultural changes
    - changes in water bodies
    - newly developed roads or infrastructure
    - changes in exposed/bare land
    - other clearly visible land-cover changes

    Do NOT report a change merely because of small differences in:

    - illumination
    - image intensity
    - noise
    - shadows
    - sensor artifacts
    - seasonal appearance

    Only report changes that have reasonable visual evidence in BOTH images.

    For each detected change provide:

    - change_type
    - description
    - approximate location
    - normalized bounding box
    - confidence (Importantly)

    Bounding box format: [x_min, y_min, x_max, y_max]

    Coordinate convention:
    (0,0) = top-left
    (1,1) = bottom-right

    Confidence must be between 0 and 1.

    Do not invent geographic coordinates, place names, dates,
    sensor parameters, or information that cannot be inferred
    from the images.

    If no meaningful change is visible, return an empty changes list.

    For every detected change, return:
    
        - change_type
        - description
        - approximate_location
        - bounding_box
        - confidence
    
        Return ONLY valid JSON using exactly this structure:
    
        {
        "changes": [
            {
            "change_type": "...",
            "description": "...",
            "approximate_location": "...",
            "bounding_box": [0.0, 0.0, 0.0, 0.0],
            "confidence": 0.0
            }
        ]
        }
    """

    # User input query -> take from the langgraph state.
    user_query = query

    # OpenAI client for HuggingFace
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

    # Getting the response from the model
    response = client.chat.completions.create(
        model=MODEL,
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
                        "type": "text",
                        "text": "EARLIER OBSERVATION:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_1_url,
                        },
                    },

                    {
                        "type": "text",
                        "text": "LATER OBSERVATION:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_2_url,
                        },
                    },
                ],
            },
        ],

        max_tokens=2048,
        temperature=0.2,
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

  

# Earlier Observation
before_image_1 = "/home/rikin/satquery-ai/satquery_sar_test/sample_VH.tif"

# Later Observation
after_image_2 = "/home/rikin/satquery-ai/satquery_sar_test/sample_VV.tif"

# User Query
query = "Compare the EARLIER image and the LATER image. Identify the significant changes between them."

## Testing the agent
print(change_node(before_image_1, after_image_2, query))
