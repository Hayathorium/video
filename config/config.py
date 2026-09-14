import os
from dataclasses import dataclass

from omegaconf import OmegaConf

PATH_TO_YOUR_MODEL = "/workspace/video/model.pt"  # Local checkpoint (downloaded from zai-org/RealVideo)


@dataclass
class AudioConfig:
    sample_rate: int = 16000


@dataclass
class VideoConfig:
    fps: int = 16

    frame_width: int = 480
    frame_height: int = 640

    speaking_prompt: str = "A character is talking."
    silence_prompt: str = "A character is looking at the camera."


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8003
    diffusion_socket_port: int = 9090
    app_socket_port: int = 9094
    app_ready_socket_port: int = 9092
    diffusion_ready_socket_port: int = 9093


@dataclass
class LocalSpeechConfig:
    # Groq (OpenAI-compatible) — cloud LLM, keeps the local GPU free for the DiT/VAE.
    llm_server_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_api_key: str = ""

    # vLLM-Omni `vllm serve ... --omni` (OpenAI /v1/audio/speech) — local Qwen3-TTS
    tts_server_url: str = "http://127.0.0.1:8091/v1"
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    tts_sample_rate: int = 24000
    tts_voices: tuple = (
        "vivian",
        "ryan",
        "aiden",
        "dylan",
        "eric",
        "ono_anna",
        "serena",
        "sohee",
        "uncle_fu",
    )


@dataclass
class LipSyncConfig:
    fps: int = 16

    s2v_segment_latent_length = 80

    self_forcing_config_path: str = (
        "self_forcing/configs/sample_14B_s2v_sparse_nfb2_2steps.yaml"
    )
    # self_forcing_config_path: str = 'self_forcing/configs/sample_14B_s2v_sparse_nfb2_2steps.yaml'

    checkpoint_path: str = PATH_TO_YOUR_MODEL

    audio_padding_div = 16
    audio_padding_rem = 0
    audio_min_length = 16
    audio_segment_length = 80
    s2v_video_refresh_interval = 20
    compile = True
    # Adds a torch.cuda.synchronize() + CUDA-event pair around every DiT
    # block purely to log its timing; that forces a sync barrier each block
    # and blocks any cross-block GPU stream overlap. Leave off outside of
    # active latency debugging.
    profile = False
    fp8_quantize = False
    no_refresh_inference = True

    dit_config = OmegaConf.load(self_forcing_config_path)
    default_config = OmegaConf.load("self_forcing/configs/default_config.yaml")
    dit_config = OmegaConf.merge(default_config, dit_config)


class Config:
    def __init__(self):
        self.audio = AudioConfig()
        self.video = VideoConfig()
        self.server = ServerConfig()
        self.lip_sync = LipSyncConfig()
        self.speech = LocalSpeechConfig()

        self._load_from_env()

    def _load_from_env(self):
        self.log_level = os.getenv("LOG_LEVEL", "DEBUG")

        # LLM (Groq cloud API) + TTS (local Qwen3-TTS) overrides
        self.speech.llm_server_url = os.getenv(
            "LLM_SERVER_URL", self.speech.llm_server_url
        )
        self.speech.llm_model = os.getenv("LLM_MODEL", self.speech.llm_model)
        self.speech.llm_api_key = os.getenv("GROQ_API_KEY", self.speech.llm_api_key)
        self.speech.tts_server_url = os.getenv(
            "TTS_SERVER_URL", self.speech.tts_server_url
        )
        self.speech.tts_model = os.getenv("TTS_MODEL", self.speech.tts_model)
        self.speech.tts_sample_rate = int(
            os.getenv("TTS_SAMPLE_RATE", self.speech.tts_sample_rate)
        )
        self.self_focing_config_path = os.getenv("CONFIG_PATH", "")
        self.audio_samples_per_video_block = round(
            self.audio.sample_rate
            / self.video.fps
            * self.lip_sync.dit_config.num_frame_per_block
            * 4
        )  # in audio samples, (4 for vae)
        self.lip_sync.audio_min_length = (
            4 * self.lip_sync.dit_config.num_frame_per_block
        )  # in frames, (4 for vae)


config = Config()
