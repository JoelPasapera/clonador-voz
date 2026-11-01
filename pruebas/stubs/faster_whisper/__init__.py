"""faster_whisper simulado.

Con un arreglo (verificación de fragmentos) «entiende» el último texto sintetizado, salvo «zzmurmullo» en su primer
intento. Con un archivo (datos de entrenamiento) inventa palabras cada 0,4 s y cierra frase cada 8 palabras.
"""

import wave
from dataclasses import dataclass

import numpy as np

import simulacion

cargados = []
transcripciones = []


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list


class WhisperModel:
    def __init__(self, model_size_or_path, device="auto", device_index=0, compute_type="default", **_):
        assert device == "cuda" and compute_type == "float16"
        cargados.append(model_size_or_path)
        self.nombre = model_size_or_path

    def transcribe(self, audio, language=None, task="transcribe", beam_size=5, word_timestamps=False, vad_filter=False,
                   condition_on_previous_text=True, without_timestamps=False, **_):
        transcripciones.append({"modelo": self.nombre, "idioma": language, "palabras": word_timestamps})
        if isinstance(audio, np.ndarray):
            assert audio.dtype == np.float32 and audio.ndim == 1
            texto = simulacion.estado["ultimo_texto"]
            if "zzmurmullo" in texto and simulacion.estado["intentos"].get(texto, 0) == 1:
                texto = "nada que ver con lo pedido"
            return iter([Segment(0.0, audio.size / 16000, f" {texto}", None)]), None
        with wave.open(str(audio), "rb") as archivo:
            duracion = archivo.getnframes() / archivo.getframerate()
        palabras = []
        inicio, numero = 0.5, 0
        while inicio + 0.3 < duracion - 0.3:
            numero += 1
            fin_de_frase = numero % 8 == 0
            palabras.append(Word(inicio, inicio + 0.3, f" palabra{numero}" + ("." if fin_de_frase else ""), 0.9))
            inicio += 0.8 if fin_de_frase else 0.4
        return iter([Segment(0.0, duracion, "".join(p.word for p in palabras), palabras)]), None
