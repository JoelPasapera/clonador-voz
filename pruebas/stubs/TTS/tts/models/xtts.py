"""TTS.tts.models.xtts simulado: Xtts con las MISMAS firmas que coqui-tts 0.28.0 (test_fidelidad.py lo comprueba)."""

from pathlib import Path

import numpy as np
import torch

import simulacion

registro = {"inferencias": [], "speaker": [], "gpt": [], "puntos_de_control": []}
VOCES_INCLUIDAS = {"Ana Florence": 210.0, "Claribel Dervla": 250.0, "Andrew Chipper": 110.0, "Damien Black": 95.0}


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


class SpeakerManager:
    """Como TTS.tts.layers.xtts.xtts_manager.SpeakerManager: un dict nombre → {latente, embedding}."""

    def __init__(self, speakers) -> None:
        self.speakers = speakers


def voces_de_estudio() -> dict:
    return {
        nombre: {
            "gpt_cond_latent": torch.Tensor(np.zeros((1, 32, 1024), np.float32)),
            "speaker_embedding": torch.Tensor(simulacion.huella_de_frecuencia(f0).reshape(1, 512, 1)),
        }
        for nombre, f0 in VOCES_INCLUIDAS.items()
    }


class Xtts:
    def __init__(self, config=None) -> None:
        self.args, self.config, self.tokenizer = _Args(), config if config is not None else _Config(), _Tokenizer()
        self._dispositivo = "cpu"
        self.speaker_manager = None

    @property
    def device(self):
        return self._dispositivo

    def to(self, dispositivo):
        self._dispositivo = str(dispositivo)
        return self

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=True,
        use_deepspeed=False,
        speaker_file_path=None,
    ):
        carpeta = Path(checkpoint_dir)
        estado = torch.load(carpeta / "model.pth", map_location="cpu")["model"]  # weights_only=True, como en torch ≥ 2.6
        assert estado and not any(clave.startswith(("xtts.", "dvae.", "torch_mel_spectrogram")) for clave in estado)
        assert (carpeta / "vocab.json").is_file()
        if (carpeta / "speakers_xtts.pth").is_file():
            self.speaker_manager = SpeakerManager(torch.load(carpeta / "speakers_xtts.pth"))
        registro["puntos_de_control"].append(str(carpeta))

    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        assert sr == 22050, sr
        assert audio.datos.ndim == 2 and audio.shape[0] == 1, audio.shape
        assert audio.dispositivo == self._dispositivo, "el audio debe estar en el dispositivo del modelo"
        assert 0 < chunk_length <= length
        registro["gpt"].append({"muestras": audio.shape[1], "length": length, "chunk_length": chunk_length})
        return torch.Tensor(np.zeros((1, 32, 1024), np.float32), self._dispositivo)

    def get_speaker_embedding(self, audio, sr):
        assert audio.datos.dtype == np.float32 and np.abs(audio.datos).max() <= 1.0 + 1e-6
        assert audio.datos.ndim == 2 and audio.shape[0] == 1, audio.shape
        assert audio.dispositivo == self._dispositivo
        registro["speaker"].append(audio.shape[1] if sr == 22050 else ("huella", sr, audio.shape[1]))
        huella = simulacion.huella_de_frecuencia(simulacion.frecuencia_dominante(audio.datos, sr))
        return torch.Tensor(huella.reshape(1, 512, 1), self._dispositivo)

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
        voz = simulacion.frecuencia_de_huella(speaker_embedding.datos)
        registro["inferencias"].append(
            {"texto": text, "idioma": language, "temperature": temperature, "top_p": top_p, "top_k": top_k,
             "repetition_penalty": repetition_penalty, "speed": speed, "voz_hz": voz}
        )
        rng = torch.generador()
        f0 = voz + 8.0 * rng.standard_normal()
        wav = simulacion.voz_sintetica(text, self.args.output_sample_rate, f0, speed, rng)
        return {"wav": wav, "gpt_latents": np.zeros((1, 8, 1024), np.float32), "speaker_embedding": speaker_embedding}
