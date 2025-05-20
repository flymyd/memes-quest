from openai import OpenAI
from app.core.config import OPENAI_API_KEY, OPENAI_API_BASE_URL, OPENAI_EMBEDDING_MODEL

client_params = {"api_key": OPENAI_API_KEY}
if OPENAI_API_BASE_URL:
    client_params["base_url"] = OPENAI_API_BASE_URL

client = OpenAI(**client_params)

def get_embedding(text: str) -> list[float]:
    """Generates embedding for the given text using the configured OpenAI model."""
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string.")
    try:
        response = client.embeddings.create(input=[text], model=OPENAI_EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding with model {OPENAI_EMBEDDING_MODEL}: {e}")
        raise 