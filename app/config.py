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
    # Refuse to start when the LLM is unreachable (set true on EC2 so a missing vLLM
    # fails loudly instead of the app limping along).
    llm_required: bool = False
    llm_timeout_secs: float = 120.0
    llm_max_retries: int = 2
    # Gemma 4 "thinks" by default on Ollama, which costs ~10x tokens and latency here.
    # Ollama: {"reasoning_effort": "none"}.
    # vLLM:   {"chat_template_kwargs": {"enable_thinking": false}}  (off by default, made explicit)
    llm_extra_body: dict = {"reasoning_effort": "none"}

    # --- STT ---
    # "auto" picks "mlx" on Apple Silicon and "nemo" elsewhere.
    stt_backend: str = "auto"
    parakeet_mlx_model: str = "mlx-community/parakeet-tdt-0.6b-v2"
    parakeet_nemo_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    stt_moonshine_fallback: bool = True
    # Team's k2-sherpa Zipformer ASR server (12 Indic languages). Empty = disabled.
    # e.g. ws://52.6.34.53:6000. Languages listed here are routed to it; English stays on
    # Parakeet (measured more accurate on our English commands).
    indic_asr_url: str = ""
    indic_asr_languages: list[str] = ["hi"]

    # --- Spoken language ID ---
    lid_model: str = "speechbrain/lang-id-voxlingua107-ecapa"
    lid_min_speech_secs: float = 1.0
    # A turn is rejected only when LID is at least this confident it is NOT a supported language.
    lid_reject_confidence: float = 0.70
    # Spoken languages accepted as commands (LID gate). Add "hi" once indic_asr_url is set.
    speech_languages: list[str] = ["en"]
    # Document and translation-target languages.
    supported_languages: list[str] = ["en"]

    # --- TTS ---
    kokoro_voice: str = "af_heart"
    # Team's Indic Parler-TTS server. Empty = disabled. e.g. http://52.6.34.53:6001
    indic_tts_url: str = ""
    # Language of the spoken text -> Parler voice key. Used for languages Kokoro lacks (ur)
    # or does poorly; set tts_english_provider="indic_parler" to also use it for English.
    indic_tts_voices: dict[str, str] = {"hi": "hindi_rohit", "ur": "urdu_divya", "en": "english_mary"}
    tts_english_provider: str = "kokoro"  # "kokoro" | "indic_parler"

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
