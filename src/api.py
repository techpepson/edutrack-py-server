import logging

from fastapi import APIRouter, HTTPException

from .face.embedding import detect_and_embed
from .qdrant.client import delete_user_embeddings, save_embedding, search_embedding

router = APIRouter()
logger = logging.getLogger(__name__)


from typing import List
from pydantic import BaseModel


class EnrollRequest(BaseModel):
    user_id: str
    image_urls: List[str]


@router.post("/enroll")
async def enroll(request: EnrollRequest):
    result = await detect_and_embed(request.image_urls)
    if not result:
        raise HTTPException(404, "No valid face detected")
    save_embedding(request.user_id, result["embedding"])
    return {
        "status": "enrolled",
        "embeddings_saved": result["count"],
    }


@router.post("/recognize")
async def recognize(image_url: str):
    result = await detect_and_embed([image_url])
    if not result:
        return {"match": False}
    search_result = search_embedding(result["embedding"])
    if search_result.points:
        best = search_result.points[0]
        return {
            "match": True,
            "user_id": best.payload["user_id"],
            "score": best.score,
        }
    return {"match": False}


@router.delete("/user/{user_id}")
async def delete_user(user_id: str):
    delete_user_embeddings(user_id)
    return {"status": "deleted", "user_id": user_id}
