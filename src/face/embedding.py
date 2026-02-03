from .models import face_app
from ..utils.image_loader import load_images_from_urls
import cv2
import numpy as np


async def detect_and_embed(image_urls: list[str]):
    images = await load_images_from_urls(image_urls)
    all_embeddings = []
    for img in images:
        # normalize color space
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(img)
        if not faces:
            continue
        for face in faces:
            w = face.bbox[2] - face.bbox[0]
            h = face.bbox[3] - face.bbox[1]
            # quality gates
            if face.det_score < 0.6:
                continue
            if min(w, h) < 80:
                continue
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)  # enforce normalization
            all_embeddings.append(emb)
    if not all_embeddings:
        return None
    avg_embedding = np.mean(all_embeddings, axis=0)
    return {"embedding": avg_embedding.tolist(), "count": len(all_embeddings)}
