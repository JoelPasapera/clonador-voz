"""TTS.api simulado con las MISMAS firmas que coqui-tts 0.28.0 (test_fidelidad.py lo comprueba contra el código real)."""

import os

import numpy as np
import torch

NOMBRE_ESPERADO = "tts_models/multilingual/multi-dataset/xtts_v2"
registro = {"cargas": 0, "inferencias": [], "speaker": [], "gpt": []}


class _Args:
    input_sample_rate = 22050
    output_sample_rate = 24000


class _Config:
    languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi"]


class _Tokenizer:
    char_limits = {
        "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203, "pl": 224, "zh": 82, "ar": 166,
        "cs": 186, "ru": 182, "nl": 251, "tr": 226, "ja": 71, "hu": 224, "ko": 95, "hi": 150,
    }


class Xtts:
    def __init__(self) -> None:
        self.args, self.config, self.tokenizer = _Args(), _Config(), _Tokenizer()
        self._dispositivo = "cpu"

    @property
    def device(self):
        return self._dispositivo

    def to(self, dispositivo):
        self._dispositivo = str(dispositivo)
        return self

    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        assert sr == 22050, sr
        assert audio.datos.ndim == 2 and audio.shape[0] == 1, audio.shape
        assert audio.dispositivo == self._dispositivo, "el audio debe estar en el dispositivo del modelo"
        assert 0 < chunk_length <= length
        registro["gpt"].append({"muestras": audio.shape[1], "length": length, "chunk_length": chunk_length})
        return torch.Tensor(np.zeros((1, 32, 1024), np.float32), self._dispositivo)

    def get_speaker_embedding(self, audio, sr):
        assert sr == 22050, sr
        assert audio.datos.dtype == np.float32 and np.abs(audio.datos).max() <= 1.0
        assert audio.dispositivo == self._dispositivo
        registro["speaker"].append(audio.shape[1])
        return torch.Tensor(np.ones((1, 512, 1), np.float32), self._dispositivo)

    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature: float = 0.75,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 50,
        top_p: float = 0.85,
        do_sample: bool = True,
        num_beams: int = 1,
        speed: float = 1.0,
        enable_text_splitting: bool = False,
        **hf_generate_kwargs,
    ):
        assert not hf_generate_kwargs, f"argumentos inesperados para generate(): {hf_generate_kwargs}"
        assert not enable_text_splitting
        assert 0 < len(text) <= self.tokenizer.char_limits[language.split("-")[0]], (language, len(text))
        # Igual que los LogitsProcessor de transformers: «penalty has to be a strictly positive float»
        for nombre, valor in (("temperature", temperature), ("top_p", top_p), ("repetition_penalty", repetition_penalty),
                              ("length_penalty", length_penalty), ("speed", speed)):
            assert isinstance(valor, float) and valor > 0, f"{nombre} debe ser float positivo, llegó {valor!r}"
        assert isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0
        if "💥" in text:  # simula la aserción real de XTTS: «can only generate text with a maximum of 400 tokens»
            raise AssertionError(" ❗ XTTS can only generate text with a maximum of 400 tokens.")
        assert gpt_cond_latent.shape == (1, 32, 1024) and speaker_embedding.shape == (1, 512, 1)
        registro["inferencias"].append(
            {"texto": text, "idioma": language, "temperature": temperature, "top_p": top_p, "top_k": top_k,
             "repetition_penalty": repetition_penalty, "speed": speed}
        )
        sr = self.args.output_sample_rate
        rng = torch.generador()
        segundos_voz = max(0.3, 0.055 * len(text) / speed)
        t = np.arange(int(segundos_voz * sr)) / sr
        f0 = 120.0 + 15.0 * rng.standard_normal()
        envolvente = 0.3 * (1.0 - np.cos(2.0 * np.pi * 4.0 * t))  # «sílabas» a 4 Hz
        voz = envolvente * (np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t))
        silencio_inicial = rng.normal(0.0, 1e-4, int(0.3 * sr))
        silencio_final = rng.normal(0.0, 1e-4, int(0.6 * sr))
        wav = np.concatenate([silencio_inicial, voz, silencio_final]).astype(np.float32)
        return {"wav": wav, "gpt_latents": np.zeros((1, 8, 1024), np.float32), "speaker_embedding": speaker_embedding}


class _Synthesizer:
    def __init__(self) -> None:
        self.tts_model = Xtts()


class TTS:
    def __init__(
        self,
        model_name: str = "",
        *,
        model_path=None,
        config_path=None,
        vocoder_name=None,
        vocoder_path=None,
        vocoder_config_path=None,
        encoder_path=None,
        encoder_config_path=None,
        speakers_file_path=None,
        progress_bar: bool = True,
        gpu: bool = False,
    ) -> None:
        assert model_name == NOMBRE_ESPERADO, model_name
        assert os.environ.get("COQUI_TOS_AGREED") == "1", "la licencia debe aceptarse antes de descargar el modelo"
        registro["cargas"] += 1
        self.synthesizer = _Synthesizer()

    def to(self, dispositivo):
        self.synthesizer.tts_model.to(dispositivo)
        return self
