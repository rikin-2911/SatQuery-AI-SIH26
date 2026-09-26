"""
SatQuery AI
Multispectral Remote-Sensing VLM Agent

Responsibilities:
    1. Load Qwen3.5-2B + trained LoRA adapter
    2. Load the trained 12 -> 3 spectral adapter
    3. Replace Qwen's RGB patch embedding with the multispectral version
    4. Convert 12-band Sentinel-2 patches into Qwen-compatible tensors
    5. Run remote-sensing VQA / captioning inference
    6. Return a structured result for LangGraph

Expected input:
    patch: torch.Tensor [12, H, W]
           Training-normalized Sentinel-2 bands

Output:
    {
        "answer": str,
        "agent": str,
        "model": str,
        "task": str,
        "confidence": float | None
    }
"""

import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from unsloth import FastVisionModel


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_NAME = "Qwen3.5-2B-SatQuery-6K"

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 1. Spectral Adapter
# ============================================================

class SpectralAdapter(nn.Module):
    """
    Converts 12 Sentinel-2 spectral bands into
    the 3-channel representation expected by
    Qwen's original vision patch embedding.

    Architecture:
        12 bands -> 3 channels
        1x1 Conv2D

    IMPORTANT:
        This must match the adapter used during training.
    """

    def __init__(self):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels=12,
            out_channels=3,
            kernel_size=1,
            bias=True,
            dtype=torch.float32,
        )

        # Same initialization used during training.
        nn.init.constant_(
            self.proj.weight,
            1.0 / 12.0
        )

        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:

        # Keep spectral adapter numerically stable in FP32.
        x = x.float()

        x = self.proj(x)

        return x


# ============================================================
# 2. Multispectral Patch Embedding
# ============================================================

class MultispectralPatchEmbed(nn.Module):
    """
    Adapter around Qwen's original patch embedding.

    Qwen expects RGB input.

    Our trained pipeline:
        12-band image
            ↓
        SpectralAdapter
            ↓
        3-channel representation
            ↓
        Original Qwen Conv3D patch embedding
    """

    def __init__(
        self,
        original_patch_embed: nn.Module,
        spectral_adapter: SpectralAdapter,
    ):
        super().__init__()

        self.original = original_patch_embed

        self.spectral_adapter = spectral_adapter

        self.patch_size = 16
        self.temporal_patch_size = 2

        self.embed_dim = (
            original_patch_embed.proj.out_channels
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:

        """
        Expected input:

            [N, 6144]

        where:

            6144 = 2 * 12 * 16 * 16

        We reconstruct:

            [N, 2, 12, 16, 16]
        """

        N = hidden_states.shape[0]

        # ----------------------------------------------------
        # [N, 6144]
        # ->
        # [N, 2, 12, 16, 16]
        # ----------------------------------------------------

        hidden_states = hidden_states.reshape(
            N,
            2,
            12,
            16,
            16,
        )

        # ----------------------------------------------------
        # Merge temporal dimension for spectral adapter
        #
        # [N, 2, 12, 16, 16]
        # ->
        # [N*2, 12, 16, 16]
        # ----------------------------------------------------

        hidden_states = hidden_states.reshape(
            N * 2,
            12,
            16,
            16,
        )

        # ----------------------------------------------------
        # 12 spectral bands -> 3 channels
        # ----------------------------------------------------

        hidden_states = self.spectral_adapter(
            hidden_states
        )

        # ----------------------------------------------------
        # Restore temporal dimension
        #
        # [N*2, 3, 16, 16]
        # ->
        # [N, 2, 3, 16, 16]
        # ----------------------------------------------------

        hidden_states = hidden_states.reshape(
            N,
            2,
            3,
            16,
            16,
        )

        # ----------------------------------------------------
        # Qwen Conv3D expects:
        #
        # [N, C, T, H, W]
        # ----------------------------------------------------

        hidden_states = hidden_states.permute(
            0,
            2,
            1,
            3,
            4,
        )

        # ----------------------------------------------------
        # Match original Qwen projection dtype
        # ----------------------------------------------------

        hidden_states = hidden_states.to(
            dtype=self.original.proj.weight.dtype
        )

        # ----------------------------------------------------
        # Original Qwen patch embedding
        # ----------------------------------------------------

        output = self.original.proj(
            hidden_states
        )

        # ----------------------------------------------------
        # Flatten exactly like training
        # ----------------------------------------------------

        output = output.flatten(1)

        return output


# ============================================================
# 3. Multispectral Qwen Agent
# ============================================================

class MultispectralVLM:
    """
    Main SatQuery multispectral VLM specialist.

    Example:

        agent = MultispectralVLM(
            model_path="models/qwen35_multispectral_6000",
            spectral_adapter_path="models/qwen35_multispectral_6000/spectral_adapter.pt"
        )

        result = agent.run(
            patch,
            "What type of land cover is present?"
        )
    """

    def __init__(
        self,
        model_path: str,
        spectral_adapter_path: str,
        device: Optional[str] = None,
    ):

        self.device = (
            device
            if device is not None
            else DEFAULT_DEVICE
        )

        self.model_path = model_path

        self.spectral_adapter_path = (
            spectral_adapter_path
        )

        self.model_name = DEFAULT_MODEL_NAME

        print("=" * 60)
        print("Loading SatQuery Multispectral VLM")
        print("=" * 60)

        print(f"Model path: {model_path}")
        print(
            f"Spectral adapter: "
            f"{spectral_adapter_path}"
        )
        print(f"Device: {self.device}")

        # ----------------------------------------------------
        # Validate files
        # ----------------------------------------------------

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model path does not exist:\n{model_path}"
            )

        if not os.path.exists(
            spectral_adapter_path
        ):
            raise FileNotFoundError(
                "Spectral adapter not found:\n"
                f"{spectral_adapter_path}"
            )

        # ----------------------------------------------------
        # Load Qwen + LoRA
        # ----------------------------------------------------

        print("\n[1/4] Loading Qwen3.5-2B + LoRA...")

        self.model, self.tokenizer = (
            FastVisionModel.from_pretrained(
                model_path,
                load_in_4bit=True,
                dtype=torch.float16,
            )
        )

        print("  Qwen + LoRA loaded")

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        self.processor = self.tokenizer

        # ----------------------------------------------------
        # Locate visual encoder
        # ----------------------------------------------------

        print(
            "\n[2/4] Locating Qwen visual encoder..."
        )

        visual = self._get_visual_module()

        print("  Visual encoder located")

        # ----------------------------------------------------
        # Create spectral adapter
        # ----------------------------------------------------

        print(
            "\n[3/4] Loading spectral adapter..."
        )

        spectral_adapter = SpectralAdapter()

        state_dict = torch.load(
            spectral_adapter_path,
            map_location="cpu",
        )

        spectral_adapter.load_state_dict(
            state_dict
        )

        print("  Spectral adapter weights loaded")

        # ----------------------------------------------------
        # Replace patch embedding
        # ----------------------------------------------------

        visual.patch_embed = (
            MultispectralPatchEmbed(
                original_patch_embed=(
                    visual.patch_embed
                ),
                spectral_adapter=(
                    spectral_adapter
                ),
            )
        )

        print(
            "  Qwen patch embedding replaced "
            "with multispectral adapter"
        )

        # ----------------------------------------------------
        # Move model
        # ----------------------------------------------------

        print(
            "\n[4/4] Moving model to device..."
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print("Model ready")

        print("=" * 60)
        print("SatQuery Multispectral VLM READY")
        print("=" * 60)

    # ========================================================
    # Locate Visual Encoder
    # ========================================================

    def _get_visual_module(self):

        """
        PEFT-wrapped Qwen hierarchy:

            model
             └── base_model
                  └── model
                       └── model
                            └── visual

        We first try the hierarchy observed
        during your training.
        """

        # Current PEFT hierarchy
        try:
            visual = (
                self.model
                .base_model
                .model
                .model
                .visual
            )

            return visual

        except AttributeError:
            pass

        # Fallback hierarchy
        try:
            visual = (
                self.model
                .model
                .visual
            )

            return visual

        except AttributeError:
            pass

        raise RuntimeError(
            "Could not locate Qwen visual encoder.\n"
            "Please inspect the loaded model hierarchy."
        )

    # ========================================================
    # Qwen Multispectral Pixel Preparation
    # ========================================================

    @staticmethod
    def prepare_multispectral_pixels(
        patch: torch.Tensor
    ) -> torch.Tensor:

        """
        Convert a 12-band patch into the exact
        tensor representation used during training.

        Input:

            [12, H, W]

        Output:

            [256, 6144]
        """

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if not isinstance(
            patch,
            torch.Tensor
        ):
            patch = torch.tensor(
                patch,
                dtype=torch.float32,
            )

        if patch.ndim != 3:

            raise ValueError(
                "Expected patch with shape "
                "[12, H, W]. "
                f"Received {tuple(patch.shape)}"
            )

        if patch.shape[0] != 12:

            raise ValueError(
                "Multispectral Qwen expects "
                f"12 bands, got {patch.shape[0]}"
            )

        # ----------------------------------------------------
        # FP32 preprocessing
        # ----------------------------------------------------

        patch = patch.float()

        # ----------------------------------------------------
        # Resize to 256 × 256
        # ----------------------------------------------------

        patch = patch.unsqueeze(0)

        patch = F.interpolate(
            patch,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )

        patch = patch.squeeze(0)

        # ----------------------------------------------------
        # Split into 16 × 16 patches
        # ----------------------------------------------------

        patch = patch.reshape(
            12,
            16,
            16,
            16,
            16,
        )

        # ----------------------------------------------------
        # Rearrange
        # ----------------------------------------------------

        patch = patch.permute(
            1,
            2,
            0,
            3,
            4,
        )

        # ----------------------------------------------------
        # [256, 12, 16, 16]
        # ----------------------------------------------------

        patch = patch.reshape(
            256,
            12,
            16,
            16,
        )

        # ----------------------------------------------------
        # Qwen temporal dimension
        #
        # Training duplicated the frame.
        # ----------------------------------------------------

        patch = torch.stack(
            [patch, patch],
            dim=1,
        )

        # ----------------------------------------------------
        # Final:
        #
        # [256, 2, 12, 16, 16]
        #
        # ->
        #
        # [256, 6144]
        # ----------------------------------------------------

        patch = patch.reshape(
            256,
            2 * 12 * 16 * 16,
        )

        return patch

    # ========================================================
    # Build Qwen Prompt
    # ========================================================

    def _build_prompt(
        self,
        question: str,
    ):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            }
        ]

        text = (
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        return text

    # ========================================================
    # Run Inference
    # ========================================================

    @torch.inference_mode()
    def predict(
        self,
        patch: torch.Tensor,
        question: str,
        max_new_tokens: int = 128,
    ) -> str:

        """
        Run Qwen inference.

        Parameters
        ----------
        patch:
            Tensor [12, H, W]

        question:
            Natural-language remote-sensing query.

        max_new_tokens:
            Maximum generated tokens.

        Returns
        -------
        str
            Generated answer.
        """

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # ----------------------------------------------------
        # Prepare multispectral pixels
        # ----------------------------------------------------

        pixel_values = (
            self.prepare_multispectral_pixels(
                patch
            )
        )

        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

        pixel_values = pixel_values.to(
            self.device,
            dtype=torch.float16,
        )

        # ----------------------------------------------------
        # Dummy RGB image
        #
        # Qwen's processor is used only to construct:
        #     input_ids
        #     attention_mask
        #     image tokens
        #     image_grid_thw
        #
        # We replace its RGB pixel_values afterwards.
        # ----------------------------------------------------

        dummy_image = Image.new(
            "RGB",
            (128, 128),
            (0, 0, 0),
        )

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        text = self._build_prompt(
            question
        )

        # ----------------------------------------------------
        # Qwen processor
        # ----------------------------------------------------

        inputs = self.processor(
            text=text,
            images=dummy_image,
            return_tensors="pt",
        )

        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        for key, value in list(
            inputs.items()
        ):

            if torch.is_tensor(value):

                inputs[key] = value.to(
                    self.device
                )

        # ----------------------------------------------------
        # Replace RGB pixels with
        # trained multispectral tensor
        # ----------------------------------------------------

        inputs["pixel_values"] = (
            pixel_values
        )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # ----------------------------------------------------
        # Remove prompt tokens
        # ----------------------------------------------------

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_ids = output_ids[
            :,
            input_length:,
        ]

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        answer = (
            self.processor.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )
        )

        return answer.strip()

    # ========================================================
    # Agent Interface
    # ========================================================

    def run(
        self,
        image: torch.Tensor,
        query: str,
        max_new_tokens: int = 128,
    ) -> dict:

        """
        Main interface used by LangGraph.

        Returns a structured SatQuery result.
        """

        answer = self.predict(
            patch=image,
            question=query,
            max_new_tokens=max_new_tokens,
        )

        return {
            "answer": answer,

            "agent": (
                "multispectral_vlm"
            ),

            "model": self.model_name,

            "task": (
                "remote_sensing_vqa"
            ),

            "confidence": None,
        }