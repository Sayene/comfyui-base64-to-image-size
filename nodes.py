import sys
import os

import cv2
from PIL import Image, ImageOps
import folder_paths
import torch
import numpy as np
from io import BytesIO
import base64


class LoadImageFromBase64Size:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    OUTPUT_TOOLTIPS = ("The loaded image tensor.", "The alpha mask tensor.", "The width of the image.", "The height of the image.")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def convert_color(self, image):
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    def load_image(self, data):
        import hashlib
        import logging

        # Log the hash of the received base64 data for comparison
        data_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
        logging.info(f"Received base64 data hash: {data_hash}")

        # Clean the base64 string: remove whitespace and add padding if needed
        cleaned_data = data.strip().replace('\n', '').replace('\r', '')
        missing_padding = len(cleaned_data) % 4
        if missing_padding:
            cleaned_data += '=' * (4 - missing_padding)

        try:
            decoded_bytes = base64.b64decode(cleaned_data)
        except Exception as e:
            logging.error(f"Base64 decoding failed: {e}")
            raise

        nparr = np.frombuffer(base64.b64decode(data), np.uint8)

        result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        height, width = result.shape[:2]
        channels = cv2.split(result)
        if len(channels) > 3:
            mask = channels[3].astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones(channels[0].shape, dtype=torch.float32, device="cpu")

        result = self.convert_color(result)
        result = result.astype(np.float32) / 255.0
        image = torch.from_numpy(result)[None,]
        return image, mask.unsqueeze(0), width, height


NODE_CLASS_MAPPINGS = {
    "LoadImageFromBase64Size": LoadImageFromBase64Size,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromBase64Size": "Load Image From Base64",
}
