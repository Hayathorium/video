import asyncio
import base64
import logging
import re
import time

import aiohttp
import orjson

from config.config import config as service_config

logger = logging.getLogger(__name__)

NO_EMOJI_INSTRUCTION = (
    "Do not use emoji, emoticons, or other pictographic symbols in your "
    "responses; reply in plain text only."
)

# Qwen3-TTS is given `language: "Auto"` and picks the spoken language per
# input string; an emoji slipping through (LLMs add them despite the system
# prompt) can confuse that detection and make it switch to an unrelated
# language mid-sentence. Belt-and-suspenders: strip them before TTS regardless
# of whether the LLM followed instructions.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001f000-\U0001ffff"  # emoticons, symbols & pictographs, transport, supplemental, extended-A
    "\U00002600-\U000027bf"  # misc symbols & dingbats
    "\U00002300-\U000023ff"  # misc technical (e.g. watch/hourglass symbols used as emoji)
    "\U0000fe0f"  # variation selector-16 (forces emoji presentation)
    "\U0000200d"  # zero-width joiner (compound emoji)
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return re.sub(r" {2,}", " ", _EMOJI_PATTERN.sub("", text))


class TTSPipeline:
    def __init__(
        self,
        vae_idle_event: asyncio.Event,
        model_name_llm=None,
        model_name_tts=None,
    ):
        self.vae_idle_event = vae_idle_event

        # Local speech config (llama.cpp LLM + vLLM-Omni Qwen3-TTS).
        self.llm_server_url = service_config.speech.llm_server_url.rstrip("/")
        self.tts_server_url = service_config.speech.tts_server_url.rstrip("/")
        self.model_name_llm = model_name_llm or service_config.speech.llm_model
        self.model_name_tts = model_name_tts or service_config.speech.tts_model
        self.tts_sample_rate = service_config.speech.tts_sample_rate
        self.tts_voices = service_config.speech.tts_voices
        self.llm_api_key = service_config.speech.llm_api_key
        if not self.llm_api_key:
            logger.warning(
                "GROQ_API_KEY is not set; requests to the Groq LLM endpoint will be rejected."
            )

        self.stc_split_pattern = r"([。？！?!\n]”?|[.?!]\s?)"
        self.substc_split_pattern = "(，|, )"
        self.stc_min_length = 10
        self.stc_max_length = 50
        self.chat_history = []
        self.async_tasks_started = False
        # Localhost services must bypass any HTTP(S)_PROXY env var.
        self.proxy = None
        self.llm_task = None
        self.tts_task = None

    def reset_status(self):
        self.chat_history = []

    def start_async_tasks(
        self, text_input_queue: asyncio.Queue, output_queue: asyncio.Queue
    ):
        if not self.async_tasks_started:
            sentence_queue = asyncio.Queue(32)
            self.llm_task = asyncio.create_task(
                self.llm_worker_async(text_input_queue, sentence_queue)
            )
            self.tts_task = asyncio.create_task(
                self.tts_worker_async(sentence_queue, output_queue)
            )
            self.async_tasks_started = True
            logger.info("LLM & TTS tasks created")

    async def llm_worker_async(self, text_input_queue: asyncio.Queue, sentence_queue):
        # Groq exposes an OpenAI-compatible /v1/chat/completions.
        llm_url = f"{self.llm_server_url}/chat/completions"
        llm_headers = {"Authorization": f"Bearer {self.llm_api_key}"}
        body_template = {
            "model": self.model_name_llm,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
        }

        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    text_item = await text_input_queue.get()
                    profile = text_item.get("profile", None)
                    text_input = text_item["text"]
                    voice_id = text_item.get("voice_id", None)
                    logger.info(f"LLM processing: {text_item}")

                    system_prompt = (
                        f"{profile}\n\n{NO_EMOJI_INSTRUCTION}"
                        if profile
                        else NO_EMOJI_INSTRUCTION
                    )
                    body = {
                        "messages": [{"role": "system", "content": system_prompt}]
                        + self.chat_history
                        + [{"role": "user", "content": text_input}]
                    }
                    body.update(body_template)

                    buffer = b""
                    chunk_resp = b""
                    text_buffer = ""
                    finished = False
                    text_response = ""
                    current_sentence = ""

                    start = time.time()

                    logger.info(f"Creating LLM stream response for input: {text_input}")
                    async with session.post(
                        llm_url, json=body, headers=llm_headers, proxy=self.proxy
                    ) as response:
                        response.raise_for_status()
                        logger.info(
                            "LLM stream response for input %s created, %.3fms elapsed"
                            % (text_input, 1000 * (time.time() - start))
                        )

                        while True:
                            await asyncio.sleep(0)
                            await self.vae_idle_event.wait()
                            chunk_resp = await response.content.readline()

                            buffer += chunk_resp
                            if not buffer:
                                break

                            pos = buffer.find(b"\n")
                            if pos == -1 and not chunk_resp:
                                pos = len(buffer)

                            while pos > -1:
                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()

                                bline = buffer[: pos + 1]
                                buffer = buffer[pos + 1 :]
                                pos = buffer.find(b"\n")

                                if not bline:
                                    break

                                bline = bline.strip()
                                if not bline or not bline.startswith(b"data:"):
                                    continue

                                if bline.startswith(b"data: [DONE]"):
                                    break

                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()
                                chunk = orjson.loads(
                                    bline[5:].strip()
                                )  # remove 'data:'

                                finished = chunk["choices"][0].get("finish_reason", "")
                                if finished == "stop" or finished == "stop_sequence":
                                    break

                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    text_chunk = content
                                    text_buffer += text_chunk

                                    while True:
                                        await asyncio.sleep(0)
                                        await self.vae_idle_event.wait()
                                        m = re.search(
                                            self.stc_split_pattern, text_buffer
                                        )
                                        if m is not None:
                                            if m.end() > self.stc_max_length:
                                                pass

                                            current_sentence = strip_emoji(
                                                text_buffer[: m.end()]
                                            ).strip()
                                            text_buffer = text_buffer[m.end() :]
                                            if current_sentence:
                                                await sentence_queue.put(
                                                    {
                                                        "sentence": current_sentence,
                                                        "voice_id": voice_id,
                                                    }
                                                )
                                                logger.info(
                                                    "LLM current_sentence: %s time used: %.3fms"
                                                    % (
                                                        current_sentence,
                                                        ((time.time() - start) * 1000),
                                                    )
                                                )
                                                text_response += current_sentence

                                        else:
                                            break

                    text_buffer = strip_emoji(text_buffer).strip()
                    if text_buffer:
                        await sentence_queue.put(
                            {"sentence": text_buffer, "voice_id": voice_id}
                        )
                        text_response += text_buffer

                    await sentence_queue.put(None)
                    self.chat_history.append(
                        {"role": "assistant", "content": text_response}
                    )

                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                except Exception as e:
                    logger.exception(f"Exception in LLM worker: {e}")

    async def tts_worker_async(
        self, sentence_queue: asyncio.Queue, output_queue: asyncio.Queue
    ):
        # vLLM-Omni `vllm serve ... --omni` exposes an OpenAI-compatible
        # /v1/audio/speech. With `stream: true` + `response_format: "pcm"` it
        # streams raw 16-bit signed little-endian mono PCM (no SSE envelope).
        tts_url = f"{self.tts_server_url}/audio/speech"
        timeout = aiohttp.ClientTimeout(total=None)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    sentence_item = await sentence_queue.get()
                    if sentence_item is None:  # llm response finish
                        await output_queue.put(None)
                        continue

                    sentence = sentence_item.get("sentence", None)
                    voice_id = sentence_item.get("voice_id", None)
                    if not sentence:
                        continue

                    # Map the frontend voice id straight through to a
                    # Qwen3-TTS CustomVoice name (e.g. "vivian").
                    voice = voice_id if voice_id in self.tts_voices else self.tts_voices[0]
                    logger.info(f"TTS processing: {sentence_item} (voice={voice})")

                    body = {
                        "model": self.model_name_tts,
                        "input": sentence,
                        "voice": voice,
                        "language": "Auto",
                        "response_format": "pcm",
                        "stream": True,
                    }

                    chunk_id = 0
                    start = time.time()
                    logger.info(f"Creating TTS stream for sentence: {sentence}")
                    async with session.post(
                        tts_url, json=body, proxy=self.proxy
                    ) as response:
                        response.raise_for_status()
                        logger.info(
                            "TTS stream response for %s created, %.3fms elapsed"
                            % (sentence, 1000 * (time.time() - start))
                        )

                        while True:
                            await asyncio.sleep(0)
                            await self.vae_idle_event.wait()
                            pcm_chunk = await response.content.read(4096)
                            if not pcm_chunk:
                                break

                            logger.info(
                                "Processing audio chunk %d (bytes=%d)"
                                % (chunk_id, len(pcm_chunk))
                            )
                            await output_queue.put(
                                {
                                    "audio_base64": base64.b64encode(
                                        pcm_chunk
                                    ).decode("ascii"),
                                    "sample_rate": self.tts_sample_rate,
                                    "chunk_id": chunk_id,
                                    "time": time.time(),
                                }
                            )
                            chunk_id += 1

                except Exception as e:
                    logger.exception(f"Exception in TTS worker: {e}")
