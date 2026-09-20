## SAR Agent
import os
import base64
from pathlib import Path
from typing import List, Dict

import numpy as np
import rasterio
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

## SARAgent analyses the given SAR image in geotiff, tiff format 
class SARAgent:

    """
    SAR analysis agent using AlignEarth-SAR-ViT-B-16.

    Input:
        Sentinel-1 SAR GeoTIFF

    Output:
        Zero-shot semantic predictions based on
        image-text similarity.
    """

    MODEL_NAME = "BiliSakura/AlignEarth-SAR-ViT-B-16"

    candidate_labels = [
            "urban area",
            "water body",
            "forest",
            "agricultural land",
            "bare soil",
            "road",
            "buildings",
        ]

    def __init__(self, device: str | None = None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SARAgent] Using Device: {self.device}")
        print(f"[SARAgent] Loading: {self.MODEL_NAME}")

        # CLIP Processor
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)

        # CLIP Model
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(self.device)

        self.model.eval()

        print("[SARAgent] Loaded Successfully.")

    def read_geotiff(self, image_path: str):
        """
        Read SAR GeoTIFF using Rasterio.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"SAR image not found: {image_path}")

        # read the tiff/geotiff file using the rasterio
        with rasterio.open(image_path) as src:

            data = src.read()

            # storing the metadata of the tiff file

            metadata = {
                "crs": str(src.crs),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes,
                "resolution": src.res,
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "transform": src.transform,
            }

        return data, metadata

    def sar_to_image(self, sar_data: np.ndarray) -> Image.Image:

        """
        Convert single-band SAR backscatter into
        an RGB visualization for the CLIP model.

        NOTE:
        This is only a model-input representation.
        Original SAR values remain untouched.
        """

        # First band
        sar = sar_data[0].astype(np.float32)

        # Remove invalid values
        sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)

        # Robust contrast stretching
        p2, p98 = np.percentile(sar, [2, 98])

        if p98 <= p2:
            normalized = np.zeros_like(sar)
        else:
            normalized = (sar - p2) / (p98 - p2)

        normalized = np.clip(normalized, 0, 1)

        # Convert grayscale to RGB
        rgb = np.stack([normalized] * 3, axis=-1)

        rgb = (rgb * 255).astype(np.uint8)

        return Image.fromarray(rgb)


    # inference from model
    @torch.inference_mode()
    def predict(self, image_path: str, candidate_labels=candidate_labels) -> Dict:

        # read geotiff
        sar_data, metadata = self.read_geotiff(image_path)

        # convert SAR to Model image
        image = self.sar_to_image(sar_data)

        # text prompt list comprehension
        texts = [f"a SAR satellite image of {label}" for label in candidate_labels]

        # Creating the inputs for the model
        image_inputs = self.processor.image_processor(
            images=image,
            return_tensors="pt"
        )

        text_inputs = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        print(
            "[SARAgent] pixel_values:",
            image_inputs["pixel_values"].shape
        )

        print(
            "[SARAgent] input_ids:",
            text_inputs["input_ids"].shape
        )

        # Moving tensors to Device -- GPU/CPU
        image_inputs = {
            key: value.to(self.device)
            for key, value in image_inputs.items()
            if torch.is_tensor(value)
        }

        text_inputs = {
            key: value.to(self.device)
            for key, value in text_inputs.items()
            if torch.is_tensor(value)
        }
        
        # Outpurs
        outputs = self.model(
            pixel_values=image_inputs["pixel_values"],
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )

        # logits
        logits = outputs.logits_per_image

        # Probabilities
        probabilities = torch.softmax(logits, dim=1)[0]

        # Sort probs -> High -> Low
        sorted_indices = torch.argsort(probabilities, descending=True)

        predictions = []

        for idx in sorted_indices:

            index = idx.item()

            predictions.append(
                {
                    "label": candidate_labels[index],
                    "confidence": float(probabilities[index].item())
                }
            )

        top_prediction = predictions[0],
        predictions = predictions
                
        return metadata, top_prediction, predictions
            #"agent": "SARAgent",
            #"task": "sar_scene_analysis",
            #"model": self.MODEL_NAME,

            #"input": {
            #    "image": str(image_path),
            #    "modality": "SAR",
            #    "bands": ["VV", "VH"]
            #},

            #"execution_summary": {
            #    "model": self.MODEL_NAME,
            #    "task": "SAR semantic scene analysis",
            #    "input_modality": "SAR",
            #    "input_bands": ["VV", "VH"]
            #}
        #}


HF_TOKEN = os.getenv("HUGGINGFACE_HUB_ACCESS_TOKEN_2")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set.")

MODEL = "Qwen/Qwen3.5-27B"

# NODE
def sar_node(tiff_image, query, metadata, top_prediction, predictions):

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
        You are an expert SAR (Synthetic Aperture Radar) and remote-sensing
    analysis agent.

    Your task is to answer the user's query using THREE sources of evidence:

    1. The provided SAR image
    2. Predictions from a SAR-specific vision model
    3. Metadata from the original SAR GeoTIFF

    Your response must be grounded in the available evidence. Do not
    invent information that cannot be supported by the image, predictions,
    metadata, or user query.

    ==================================================
    EVIDENCE PRIORITY
    ==================================================

    Use the evidence in this order:

    1. VISUAL EVIDENCE FROM THE SAR IMAGE
    2. SAR MODEL PREDICTIONS AS SUPPORTING EVIDENCE
    3. GEO-TIFF METADATA FOR SPATIAL/TECHNICAL CONTEXT

    The SAR model predictions are NOT ground truth.

    They are candidate semantic predictions produced by a specialist
    SAR vision model and should be treated as supporting evidence.

    If the visual evidence conflicts with the SAR model prediction,
    acknowledge the uncertainty rather than blindly following the model.

    ==================================================
    SAR IMAGE INTERPRETATION
    ==================================================

    Analyze the image as SAR imagery, not as an ordinary RGB photograph.

    Consider SAR-relevant visual characteristics such as:

    - radar backscatter
    - bright and dark scattering regions
    - surface roughness
    - vegetation response
    - water surfaces
    - built-up structures
    - bare soil
    - agricultural areas
    - terrain geometry
    - ridges
    - valleys
    - slopes
    - radar shadow
    - possible layover
    - spatial texture
    - linear structures
    - scattering patterns

    Important:

    Do NOT assume:

    - bright = urban
    - dark = water
    - bright = forest
    - dark = shadow

    These interpretations depend on terrain, incidence geometry, surface
    properties, acquisition conditions, and other factors that may not be
    available.

    Only make a physical SAR interpretation when supported by visible
    evidence.

    ==================================================
    SAR MODEL PREDICTIONS
    ==================================================

    The SAR specialist model provides predictions in the form:

    [
    {
        "label": "...",
        "confidence": 0.0
    }
    ]

    These values are model scores over the candidate classes supplied to
    the SAR model.

    Treat them as semantic evidence, NOT calibrated probabilities and NOT
    ground truth.

    Use them to:

    - support visual observations
    - identify potentially relevant land-cover classes
    - resolve reasonable ambiguities
    - provide additional evidence for the final answer

    Do NOT claim that a class is definitely present solely because it has
    the highest model score.

    For example, if the model predicts:

    forest: 0.30
    bare soil: 0.25
    agricultural land: 0.17

    this means forest received the highest score among the supplied
    candidate classes. It does NOT mean that the image contains 30%
    forest or that forest is present with 30% probability.

    ==================================================
    GEO-TIFF METADATA
    ==================================================

    The metadata may contain:

    - CRS
    - image width
    - image height
    - number of bands
    - data type
    - spatial resolution
    - image bounds
    - affine transform

    Use this metadata only for information it actually provides.

    For example:

    - Use width and height to understand image dimensions.
    - Use resolution to understand nominal pixel spacing.
    - Use CRS to identify the coordinate reference system.
    - Use bounds to describe the geographic extent if appropriate.

    Do NOT infer:

    - acquisition date
    - satellite name
    - sensor
    - polarization
    - orbit
    - incidence angle
    - geographic location beyond the supplied metadata

    unless explicitly provided.

    ==================================================
    SPATIAL REASONING
    ==================================================

    If the user asks WHERE a feature is located, provide an approximate
    normalized bounding box when the feature can be visually localized.

    Use:

    [x_min, y_min, x_max, y_max]

    Coordinate convention:

    (0,0) = top-left
    (1,1) = bottom-right

    The box must satisfy:

    0 <= x_min <= x_max <= 1
    0 <= y_min <= y_max <= 1

    Do not fabricate bounding boxes.

    If a feature cannot be reliably localized, return:

    "bounding_box": null

    and explain the uncertainty briefly.

    If geographic coordinates are explicitly requested and sufficient
    GeoTIFF metadata is available, distinguish between:

    - image/pixel localization
    - geographic localization

    Do not invent geographic coordinates.

    ==================================================
    QUERY FOLLOWING
    ==================================================

    Answer the user's actual query.

    Do not automatically describe the entire SAR scene when the user asks
    about a specific feature.

    For example:

    If the user asks:
    "Is there a water body?"

    Focus on water-related evidence.

    If the user asks:
    "What is the dominant land-cover type?"

    Use the visual evidence together with the SAR model predictions.

    If the user asks:
    "Where is the forest?"

    Provide spatial localization if supported.

    If the user asks:
    "Describe the SAR characteristics."

    Discuss relevant backscatter, texture, terrain, and scattering
    characteristics visible in the image.

    ==================================================
    UNCERTAINTY
    ==================================================

    Be explicit when evidence is weak, ambiguous, or conflicting.

    Use language such as:

    - "The image suggests..."
    - "The SAR model predicts..."
    - "Visual evidence is consistent with..."
    - "The feature is not clearly identifiable..."
    - "The evidence is ambiguous..."

    Do not present uncertain interpretations as established facts.

    ==================================================
    OUTPUT FORMAT
    ==================================================

    Return ONLY valid JSON.

    Do not return Markdown.
    Do not return code fences.
    Do not return chain-of-thought.
    Do not provide long reasoning.

    Use this structure:

    {
    "answer": "Concise answer to the user's query.",
    "observations": [
        {
        "feature": "feature name",
        "description": "Brief evidence-based observation.",
        "model_support": "Brief description of relevant SAR model evidence.",
        "confidence": 0.0
        }
    ],
    "spatial_evidence": [
        {
        "feature": "feature name",
        "bounding_box": [0.0, 0.0, 0.0, 0.0],
        "confidence": 0.0
        }
    ],
    "uncertainties": [
        "Brief uncertainty if applicable."
    ]
    }

    If no spatial localization is requested or supported, return:

    "spatial_evidence": []

    If there is no meaningful uncertainty, return:

    "uncertainties": []

    ==================================================
    FINAL PRINCIPLE
    ==================================================

    Use the SAR image for visual evidence.

    Use the SAR specialist model for supporting semantic evidence.

    Use GeoTIFF metadata for technical and spatial context.

    Combine these sources carefully to answer the user's query.

    Never treat model predictions as ground truth.

    Never invent unsupported geographic or SAR information.

    Prioritize accuracy, evidence, spatial correctness, and calibrated
    uncertainty over producing a confident-looking answer. 
    """

    # user input query from langgraph state
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

#agent = SARAgent()

#metadata, top_prediction, predictions = agent.predict(
#       image_path=tiff_img,
        #candidate_labels=candidate_labels
#    )

# user query
#query = """
#    Identify the approximate of the following features in the SAR image:

#    1. The main water body
#    2. The major mountainous region
#    3. The densest forested region
#    4. Any visible built-up or settlement area

#    """

#print(sar_node(tiff_img, query, metadata, top_prediction, predictions))