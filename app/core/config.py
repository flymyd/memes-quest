import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "ritrieve_zh_v1")

IMAGE_DIR = os.getenv("IMAGE_DIR", "images")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "image_vectors")

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1792))

MILVUS_URI = os.getenv("MILVUS_URI", "./memes_quest.db")