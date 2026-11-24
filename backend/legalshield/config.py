from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    supabase_url: str = ""
    supabase_anon_key: SecretStr = SecretStr("")
    supabase_service_role_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    laven_enhanced: bool = False
    laven_verifier_path: str = ""
    laven_calibration_path: str = ""
    laven_regeneration_attempts: int = Field(1, ge=0, le=2)
    laven_sample_count: int = Field(1, ge=1, le=10)
    laven_answer_timeout: float = Field(55, gt=0, le=180)
    laven_max_entropy: float = Field(1., ge=0, le=4)
    laven_research_index: str = ""
    laven_reconstruction_path: str = ""
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


@lru_cache
def settings() -> Settings:
    return Settings()
