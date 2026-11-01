"""XttsConfig simulado: lee el config.json del modelo afinado (los archivos simulados no siempre son JSON)."""

import json

from TTS.tts.models.xtts import _Config


class XttsConfig(_Config):
    def load_json(self, ruta: str) -> None:
        with open(ruta, encoding="utf-8") as archivo:
            contenido = archivo.read()
        self.cargado_de = ruta
        try:
            datos = json.loads(contenido)
        except json.JSONDecodeError:
            return
        if isinstance(datos, dict) and "languages" in datos:
            self.languages = datos["languages"]
