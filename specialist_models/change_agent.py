## Change or Bi-Temporal NODE 
## In this file, the change_node function inputs two images (Before and After) + User Query

import os
import base64
import json


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN_2")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

# Model name
MODEL = "Qwen/Qwen3.5-27B"

## CHANGE NODE
def change_node(before_image, after_image, query):

    # check both images are uploaded or not...
    if not before_image:
        raise Exception("Earlier or Image 1 is missing for Analysis !")

    if not after_image:
        raise Exception("After or Image 2 is missing for Analysis !")
    

    # Image to data url
    def image_to_data_url(path: str) -> str:
        with open(path, "rb") as f:
            image_bytes = f.read()

        encoded = base64.b64encode(image_bytes).decode("utf-8")

        return f"data:image/jpeg;base64,{encoded}"


    image_1_url = image_to_data_url(before_image)
    image_2_url = image_to_data_url(after_image)


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
    #print(message)
    return message.content
    #int("QWEN3.5-27B BI-TEMPORAL CHANGE ANALYSIS")
    #print("=" * 70)
    #print(message.content)
    #print("=" * 70)


"""
# Earlier Observation
before_image_1 = "/home/rikin/satquery-ai/bi-temporal-testing/T1.png"

# Later Observation
after_image_2 = "/home/rikin/satquery-ai/bi-temporal-testing/T2.png"

# User Query
query = "Compare the EARLIER image and the LATER image. Identify the significant changes between them."

## Testing the agent
print(change_node(before_image_1, after_image_2, query))
"""