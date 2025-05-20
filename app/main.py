import os
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from typing import List
import logging

from app.core.config import IMAGE_DIR, OPENAI_API_KEY
from app.services.embedding_service import get_embedding
from app.services.milvus_service import MilvusService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Vector Search API")

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables.")

try:
    milvus_service = MilvusService()
except Exception as e:
    logger.error(f"Failed to initialize MilvusService: {e}")
    milvus_service = None 


if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    logger.info(f"Created image directory: {IMAGE_DIR}")

app.mount(f"/{IMAGE_DIR}", StaticFiles(directory=IMAGE_DIR), name="images")


def construct_image_url(filename: str, request_url_base: str) -> str:
    """Constructs the full URL for an image file."""
    if not str(request_url_base).endswith('/'):
        base = str(request_url_base) + '/'
    else:
        base = str(request_url_base)
    
    return f"{base}{IMAGE_DIR}/{filename}"


# --- API Endpoints ---

@app.post("/index-images/", summary="Index images from the configured directory")
async def index_images():
    if not milvus_service:
        raise HTTPException(status_code=503, detail="Milvus service is not available.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    logger.info(f"Starting image indexing process from directory: {IMAGE_DIR}")
    
    image_files = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    if not image_files:
        return {"message": "No images found in the directory to index."}

    indexed_count = 0
    failed_count = 0
    
    filenames_to_index = []
    embeddings_to_index = []
    
    for filename in image_files:
        try:
            text_to_embed = os.path.splitext(filename)[0]
            embedding = get_embedding(text_to_embed)
            filenames_to_index.append(filename)
            embeddings_to_index.append(embedding)
            logger.info(f"Generated embedding for: {filename}")
        except ValueError as ve:
            logger.error(f"ValueError for {filename}: {ve}")
            failed_count += 1
        except Exception as e:
            logger.error(f"Failed to generate embedding for {filename}: {e}")
            failed_count += 1
            
    if embeddings_to_index:
        try:
            milvus_service.insert_vectors(embeddings_to_index, filenames_to_index)
            indexed_count = len(filenames_to_index)
            logger.info(f"Successfully indexed {indexed_count} new images.")
        except Exception as e:
            logger.error(f"Failed to insert embeddings into Milvus: {e}")
            failed_count += len(filenames_to_index)
            indexed_count = 0


    return {
        "message": f"Image indexing complete. Processed {len(image_files)} files.",
        "indexed_count": indexed_count,
        "failed_count": failed_count,
        "total_in_milvus_after_op": milvus_service.count_entities()
    }

@app.get("/search/", summary="Search for images by text query")
async def search_images(
    request: Request,
    q: str = Query(..., min_length=1, description="Text query to search for images."),
    n: int = Query(5, gt=0, le=100, description="Number of top matches to return.")
):
    if not milvus_service:
        raise HTTPException(status_code=503, detail="Milvus service is not available.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    logger.info(f"Received search query: '{q}', n: {n}")
    try:
        query_embedding = get_embedding(q)
    except ValueError as ve:
        logger.error(f"Invalid query for embedding: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid query for embedding: {ve}")
    except Exception as e:
        logger.error(f"Failed to get embedding for query '{q}': {e}")
        raise HTTPException(status_code=500, detail="Failed to process query embedding.")

    try:
        search_results = milvus_service.search_vectors(query_embedding, n)
    except Exception as e:
        logger.error(f"Failed to search Milvus for query '{q}': {e}")
        raise HTTPException(status_code=500, detail="Search operation failed.")
    
    request_url_base = f"{request.url.scheme}://{request.url.netloc}"

    image_urls = []
    for res in search_results:
        file_path = res.get('file_path')
        if file_path:
            image_urls.append(construct_image_url(file_path, request_url_base))
        else:
            logger.warning(f"Search result with ID {res.get('id')} missing file_path.")
            
    logger.info(f"Returning {len(image_urls)} search results for query '{q}'.")
    return {"data": image_urls, "code": 200, "msg": "Success"}


@app.get("/-debug-milvus-count", summary="Get current count of items in Milvus collection (for debugging)")
async def milvus_count():
    if not milvus_service:
        raise HTTPException(status_code=503, detail="Milvus service is not available.")
    try:
        count = milvus_service.count_entities()
        return {"collection_name": milvus_service.collection_name, "count": count}
    except Exception as e:
        logger.error(f"Failed to get Milvus count: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Milvus count: {e}")


@app.on_event("startup")
async def startup_event():
    logger.info("Application startup: Initializing resources...")
    global milvus_service
    if not milvus_service:
        try:
            milvus_service = MilvusService()
            logger.info("MilvusService re-initialized successfully on startup.")
        except Exception as e:
            logger.error(f"Critical: Failed to re-initialize MilvusService on startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutdown: Releasing resources...")
    logger.info("Milvus connection managed by MilvusClient. No explicit disconnect needed for local DB.")

# To run the app (if this file is executed directly, for development):
# uvicorn app.main:app --reload
# Remember to set your OPENAI_API_KEY in your environment or .env file.
# And have Milvus Lite (or a server) running/configurable. 