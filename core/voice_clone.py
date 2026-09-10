import logging

from config.config import config as service_config

logger = logging.getLogger(__name__)


def get_voice_list():
    # Qwen3-TTS CustomVoice ships a fixed set of predefined voices; voice
    # cloning is not available in local mode.
    voices = service_config.speech.tts_voices
    return [(name, name) for name in voices]


def upload_audio_file(file_path):
    logger.warning(
        "Voice cloning is not supported in local (Qwen3-TTS CustomVoice) mode."
    )
    return None


def clone(file_id, voice_name):
    logger.warning(
        "Voice cloning is not supported in local (Qwen3-TTS CustomVoice) mode."
    )
    return {"success": False, "message": "Voice cloning unavailable in local mode."}
