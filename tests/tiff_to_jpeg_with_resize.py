import rasterio
import numpy as np
from PIL import Image


TIFF_PATH = "satquery_sar_test/sample_VV.tif"

def tiff_to_jpeg(tiff_image):
    with rasterio.open(TIFF_PATH) as src:
        sar = src.read(1)

    # Robust contrast stretching
    p2, p98 = np.nanpercentile(sar, (2, 98))

    sar_vis = np.clip((sar - p2) / (p98 - p2), 0, 1)

    # 8-bit image for display
    sar_uint8 = (sar_vis * 255).astype(np.uint8)

    return Image.fromarray(sar_uint8).save("jpeg_image.jpeg")
    #jpeg_image = Image.fromarray(sar_uint8).save("jpeg_image.jpeg")

#tiff_to_jpeg(TIFF_PATH)


## Image compression to 1080 x 1080 pixels.
from PIL import Image
import os

src = "/home/rikin/satquery-ai/jpeg_image.jpeg"
dst = "/home/rikin/satquery-ai/jpeg_image_1080.jpeg"

img = Image.open(src).convert("L")

# Resize while preserving the complete scene
img = img.resize((1080, 1080), Image.Resampling.LANCZOS)

# Qwen vision input: convert grayscale SAR visualization to 3-channel
img = img.convert("RGB")

img.save(
    dst,
    format="JPEG",
    quality=85,
    optimize=True
)

print("Saved:", dst)
print("Size:", os.path.getsize(dst) / 1024 / 1024, "MB")