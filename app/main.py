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


@app.post("/index-images/", summary="Index images from the configured directory")
async def index_images(
    category: str = Query(None, description="Optional: Specific category (subdirectory in IMAGE_DIR) to index. If not provided, all images in IMAGE_DIR and its subdirectories will be indexed.")
):
    if not milvus_service:
        raise HTTPException(status_code=503, detail="Milvus service is not available.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    logger.info(f"Starting image indexing process from directory: {IMAGE_DIR}")
    
    milvus_category_prefix = None
    base_indexing_path = IMAGE_DIR
    if category:
        clean_category = category.strip(os.path.sep)
        base_indexing_path = os.path.join(IMAGE_DIR, clean_category)
        if not os.path.isdir(base_indexing_path):
            raise HTTPException(status_code=404, detail=f"Category directory '{clean_category}' not found.")
        logger.info(f"Indexing images for category: {clean_category} in path: {base_indexing_path}")
        milvus_category_prefix = f"{clean_category}{os.path.sep}" 
    else:
        logger.info(f"Indexing all images in path: {IMAGE_DIR} and its subdirectories.")

    try:
        indexed_file_paths_in_milvus = milvus_service.get_all_file_paths(category_prefix=milvus_category_prefix)
        logger.info(f"Found {len(indexed_file_paths_in_milvus)} files in Milvus for prefix '{milvus_category_prefix}'.")
    except Exception as e:
        logger.error(f"Failed to retrieve existing file paths from Milvus: {e}")
        raise HTTPException(status_code=500, detail="Failed to communicate with Milvus to get existing files.")

    local_image_relative_paths = set()
    image_files_to_process_map = {} 

    for root, _, files in os.walk(base_indexing_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, IMAGE_DIR) 

                if category:
                    if not relative_path.startswith(milvus_category_prefix):
                        expected_prefix_for_relative_path = milvus_category_prefix
                        if not relative_path.startswith(expected_prefix_for_relative_path):
                            continue
                
                local_image_relative_paths.add(relative_path)
                image_files_to_process_map[relative_path] = full_path
    
    logger.info(f"Found {len(local_image_relative_paths)} image files on disk for the specified scope.")

    files_to_add_relative_paths = local_image_relative_paths - indexed_file_paths_in_milvus
    files_to_delete_relative_paths = indexed_file_paths_in_milvus - local_image_relative_paths
    
    logger.info(f"Files to add: {len(files_to_add_relative_paths)}")
    logger.info(f"Files to delete: {len(files_to_delete_relative_paths)}")
    if files_to_delete_relative_paths:
        logger.info(f"DEBUG: Attempting to delete the following files: {list(files_to_delete_relative_paths)}")

    added_count = 0
    deleted_count = 0
    failed_add_count = 0
    
    if files_to_add_relative_paths:
        embeddings_to_index = []
        filenames_to_index = []
        
        for relative_path in files_to_add_relative_paths:
            full_path = image_files_to_process_map[relative_path]
            try:
                text_to_embed = os.path.splitext(relative_path)[0] 
                embedding = get_embedding(text_to_embed)
                embeddings_to_index.append(embedding)
                filenames_to_index.append(relative_path)
                logger.info(f"Generated embedding for new file: {relative_path}")
            except ValueError as ve:
                logger.error(f"ValueError for new file {relative_path}: {ve}")
                failed_add_count += 1
            except Exception as e:
                logger.error(f"Failed to generate embedding for new file {relative_path}: {e}")
                failed_add_count += 1
                
        if embeddings_to_index:
            try:
                milvus_service.insert_vectors(embeddings_to_index, filenames_to_index)
                added_count = len(filenames_to_index)
                logger.info(f"Successfully added {added_count} new image embeddings to Milvus.")
            except Exception as e:
                logger.error(f"Failed to insert new embeddings into Milvus: {e}")
                failed_add_count += len(filenames_to_index)
                added_count = 0
    
    if files_to_delete_relative_paths:
        try:
            deleted_count = milvus_service.delete_vectors_by_file_paths(list(files_to_delete_relative_paths))
            logger.info(f"Successfully deleted {deleted_count} image embeddings from Milvus.")
        except Exception as e:
            logger.error(f"Failed to delete vectors from Milvus: {e}")

    total_in_milvus_after_op = milvus_service.count_entities()

    return {
        "message": f"Image indexing complete for category '{category if category else 'all'}'.",
        "files_on_disk_scanned": len(local_image_relative_paths),
        "files_in_milvus_before_op_for_category": len(indexed_file_paths_in_milvus),
        "new_files_added_to_milvus": added_count,
        "files_deleted_from_milvus": deleted_count,
        "failed_to_add_count": failed_add_count,
        "total_in_milvus_after_op": total_in_milvus_after_op
    }

@app.get("/search/", summary="Search for images by text query")
async def search_images(
    request: Request,
    q: str = Query(..., min_length=1, description="Text query to search for images."),
    n: int = Query(5, gt=0, le=100, description="Number of top matches to return."),
    category: str = Query(None, description="Optional: Specific category (subdirectory in IMAGE_DIR) to search within.")
):
    if not milvus_service:
        raise HTTPException(status_code=503, detail="Milvus service is not available.")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    logger.info(f"Received search query: '{q}', n: {n}, category: '{category}'")
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
    
    if search_results:
        if category:
            clean_category = category.strip(os.path.sep)
            category_prefix_to_match = f"{clean_category}{os.path.sep}" if clean_category else None

            if category_prefix_to_match:
                logger.info(f"Filtering search results for category prefix: '{category_prefix_to_match}'")
                for res in search_results:
                    file_path = res.get('file_path')
                    if file_path:
                        if file_path.startswith(category_prefix_to_match):
                            image_urls.append(construct_image_url(file_path, request_url_base))
                    else:
                        logger.warning(f"Search result with ID {res.get('id')} missing file_path.")
            else: 
                logger.info("Empty or invalid category provided, processing as if no category specified.")
                for res in search_results: 
                    file_path = res.get('file_path')
                    if file_path:
                        image_urls.append(construct_image_url(file_path, request_url_base))
                    else:
                        logger.warning(f"Search result with ID {res.get('id')} missing file_path.")
        else:
            logger.info("No category specified, processing all search results.")
            for res in search_results:
                file_path = res.get('file_path')
                if file_path:
                    image_urls.append(construct_image_url(file_path, request_url_base))
                else:
                    logger.warning(f"Search result with ID {res.get('id')} missing file_path.")

    if not image_urls and search_results:
        log_message_prefix = ""
        if category:
            if any(res.get('file_path') for res in search_results):
                 log_message_prefix = f"No results found for category '{category}' after filtering. "

        if not log_message_prefix and not any(res.get('file_path') for res in search_results):
             logger.warning("Milvus returned results, but none have a valid 'file_path'. Cannot provide a fallback image.")
        else:
            logger.info(f"{log_message_prefix}Attempting fallback to return the top available result.")
            for res in search_results: 
                file_path = res.get('file_path')
                if file_path:
                    image_urls.append(construct_image_url(file_path, request_url_base))
                    logger.info(f"Fallback: Selected top available result '{file_path}'.")
                    break
    
    filtered_results_count = len(image_urls)
    logger.info(f"Returning {filtered_results_count} search results for query '{q}' (category: '{category if category else 'all'}').")
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

# uvicorn app.main:app --reload