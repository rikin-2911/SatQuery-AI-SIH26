from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


def create_evidence_preview(
    tiff_path: str,
    output_dir: str,
    max_size: int = 1080,
) -> str:

    tiff_path = Path(tiff_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{tiff_path.stem}_1080.jpeg"

    with rasterio.open(tiff_path) as src:
        band = src.read(1).astype(np.float32)

    valid_pixels = band[np.isfinite(band)]

    if valid_pixels.size == 0:
        preview = np.zeros(band.shape, dtype=np.uint8)

    else:
        p2, p98 = np.percentile(valid_pixels, (2, 98))

        if p98 <= p2:
            normalized = np.zeros_like(band)
        else:
            normalized = np.clip(
                (band - p2) / (p98 - p2),
                0,
                1,
            )

        preview = (normalized * 255).astype(np.uint8)

    image = Image.fromarray(preview).convert("RGB")

    # Preserve aspect ratio
    width, height = image.size

    scale = min(
        max_size / width,
        max_size / height,
        1.0,
    )

    if scale < 1.0:
        new_size = (
            int(width * scale),
            int(height * scale),
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS,
        )

    image.save(
        output_path,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    return str(output_path)