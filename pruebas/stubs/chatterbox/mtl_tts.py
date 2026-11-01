"""chatterbox.mtl_tts simulado con las MISMAS firmas que chatterbox-tts 0.1.7."""

import wave
from dataclasses import dataclass

import numpy as np
import torch

import simulacion
from chatterbox.models.t3.modules.cond_enc import T3Cond

SUPPORTED_LANGUAGES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fi": "Finnish",
    "fr": "French", "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
    "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}
S3GEN_SR = 24000
registro = {"cargas": 0, "generaciones": [], "referencias": []}


@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for clave, valor in self.gen.items():
            if torch.is_tensor(valor):
                self.gen[clave] = valor.to(device)
        return self


class VoiceEncoder:
    def embeds_from_wavs(self, wavs, sample_rate, as_spk=False, batch_size=32, trim_top_db=20, **kwargs):
        assert sample_rate == 16000, "el cuaderno debe pasar audio a 16 kHz (la frecuencia del codificador)"
        huellas = np.stack([simulacion.huella_de_frecuencia(simulacion.frecuencia_dominante(w, sample_rate)) for w in wavs])
        return simulacion.huella_de_frecuencia(simulacion.frecuencia_dominante(np.concatenate(wavs), sample_rate)) if as_spk else huellas


class ChatterboxMultilingualTTS:
    def __init__(self, device) -> None:
        self.sr = S3GEN_SR
        self.device = device
        self.conds = None
        self.ve = VoiceEncoder()

    @classmethod
    def from_pretrained(cls, device):
        assert device == "cuda"
        registro["cargas"] += 1
        return cls(device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        with wave.open(str(wav_fpath), "rb") as archivo:
            assert archivo.getnchannels() == 1
            frecuencia = archivo.getframerate()
            muestras = np.frombuffer(archivo.readframes(archivo.getnframes()), "<i2").astype(np.float32) / 32768
        registro["referencias"].append({"frecuencia": frecuencia, "segundos": muestras.size / frecuencia})
        huella = simulacion.huella_de_frecuencia(simulacion.frecuencia_dominante(muestras, frecuencia))
        t3 = T3Cond(
            speaker_emb=torch.Tensor(huella.reshape(1, -1)),
            cond_prompt_speech_tokens=torch.Tensor(np.zeros((1, 150), np.int64)),
            emotion_adv=torch.Tensor(np.full((1, 1, 1), exaggeration, np.float32)),
        ).to(device=self.device)
        gen = {"embedding": torch.Tensor(huella.reshape(1, -1), self.device), "prompt_token_len": torch.Tensor(np.array([150]), self.device)}
        self.conds = Conditionals(t3, gen)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    ):
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language_id '{language_id}'.")
        assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"
        assert self.conds.t3.speaker_emb.dispositivo == self.device, "condicionales en otro dispositivo"
        for nombre, valor in (("exaggeration", exaggeration), ("cfg_weight", cfg_weight), ("temperature", temperature)):
            assert isinstance(valor, float), f"{nombre} debe ser float, llegó {valor!r}"
        voz = simulacion.frecuencia_de_huella(self.conds.t3.speaker_emb.datos)
        registro["generaciones"].append({"texto": text, "idioma": language_id, "exageracion": exaggeration, "cfg": cfg_weight, "voz_hz": voz})
        rng = torch.generador()
        wav = simulacion.voz_sintetica(text, self.sr, voz + 8.0 * rng.standard_normal(), 1.0, rng)
        return torch.Tensor(wav).unsqueeze(0)
