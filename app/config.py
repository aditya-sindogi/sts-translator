"""Runtime configuration, read from environment variables (prefix ``STS_``) or ``.env``."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STS_", env_file=ROOT / ".env", extra="ignore")

    # --- Server ---
    # Keep 127.0.0.1 on EC2 and reach it through an SSH tunnel (browsers only allow the
    # microphone on https:// or localhost).
    host: str = "127.0.0.1"
    port: int = 7860
    # STUN lets the server advertise its public IP for WebRTC audio (needed on EC2, where
    # the instance only sees its private 172.31.x.x address). Given to server and browser.
    ice_servers: list[str] = ["stun:stun.l.google.com:19302"]

    # --- LLM (any OpenAI-compatible server: Ollama locally, vLLM in production) ---
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "gemma4:12b"
    # Refuse to start when the LLM is unreachable (set true on EC2 so a missing Ollama
    # fails loudly instead of the app limping along).
    llm_required: bool = False
    llm_timeout_secs: float = 120.0
    llm_max_retries: int = 2
    # Gemma 4 "thinks" by default on Ollama, which costs ~10x tokens and latency here.
    llm_extra_body: dict = {"reasoning_effort": "none"}

    # --- STT: which engine transcribes each spoken language ---
    # Language (ISO 639-1) -> engine name from app/stt/providers.py. Only these languages
    # are accepted as spoken commands. e.g. {"en": "parakeet", "hi": "indic_zipformer"}
    stt: dict[str, str] = {"en": "parakeet"}
    # Pipecat built-in STT used if the main one fails ("moonshine", CPU) or "" for none.
    stt_fallback: str = "moonshine"
    # Engine settings
    parakeet_backend: str = "auto"  # "mlx" (Apple Silicon), "nemo" (NVIDIA GPU), or "auto"
    parakeet_mlx_model: str = "mlx-community/parakeet-tdt-0.6b-v2"
    parakeet_nemo_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    indic_asr_url: str = "ws://52.6.34.53:6000"  # team's Zipformer server (12 Indic languages)

    # --- Spoken language ID ---
    lid_model: str = "speechbrain/lang-id-voxlingua107-ecapa"
    lid_min_speech_secs: float = 1.0
    # A turn is rejected only when LID is at least this confident it is a language with no STT engine.
    lid_reject_confidence: float = 0.70
    # Document and translation-target languages.
    supported_languages: list[str] = ["en"]

    # --- TTS: which provider and voice speaks each language ---
    # Language of the text -> "provider:voice", providers from app/tts/providers.py.
    # e.g. {"en": "kokoro:af_heart", "hi": "kokoro:hf_alpha", "ur": "indic_parler:urdu_divya"}
    tts: dict[str, str] = {"en": "kokoro:af_heart"}
    indic_tts_url: str = "http://52.6.34.53:6001"  # team's Indic Parler-TTS server

    # --- Documents ---
    max_pdf_mb: int = 20
    max_pages: int = 200
    min_chars_per_page: int = 25
    max_doc_chars: int = 600_000
    doc_ttl_secs: int = 24 * 3600

    # --- Translation ---
    chunk_target_chars: int = 6000
    translate_concurrency: int = 2
    same_lang_min_chrf: float = 0.80
    summary_max_words: int = 60

    @property
    def models_dir(self) -> Path:
        return ROOT / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
