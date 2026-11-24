from functools import lru_cache

from .config import settings


@lru_cache
def encoder():
    from sentence_transformers import SentenceTransformer
    config = settings()
    model = SentenceTransformer(config.embedding_model, device=config.embedding_device, trust_remote_code=False)
    dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
    if dim != 384:
        raise ValueError("The embedding model must produce 384 dimensions for this schema.")
    return model


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
