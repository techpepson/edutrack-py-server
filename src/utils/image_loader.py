import httpx
import cv2
import numpy as np


from typing import List


async def load_images_from_urls(urls: List[str]) -> List[np.ndarray]:
    images = []
    async with httpx.AsyncClient() as client:
        for url in urls:
            response = await client.get(url)
            response.raise_for_status()
            image_data = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image from the provided URL: {url}")
            images.append(image)
    return images
