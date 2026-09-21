## SAR Agent
import os
import base64
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


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

        """
        print(
            "[SARAgent] pixel_values:",
            image_inputs["pixel_values"].shape
        )

        print(
            "[SARAgent] input_ids:",
            text_inputs["input_ids"].shape
        )
        """

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

        top_prediction = predictions[0]
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