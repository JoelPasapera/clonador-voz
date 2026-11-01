"""TTS.api simulado con las MISMAS firmas que coqui-tts 0.28.0 (test_fidelidad.py lo comprueba contra el código real).

Al «descargar» XTTS-v2 lo hace como coqui-tts 0.28.0: snapshot_download del repositorio de Hugging Face (sin
dvae.pth) a la caché de Hugging Face; el entrenamiento luego reutiliza esa caché.
"""

import os

import huggingface_hub
import torch

from TTS.tts.models.xtts import SpeakerManager, Xtts, _Config, _Tokenizer, registro as _registro_xtts, voces_de_estudio
from TTS.utils.manage import ModelManager

NOMBRE_ESPERADO = "tts_models/multilingual/multi-dataset/xtts_v2"
registro = _registro_xtts
registro["cargas"] = 0
huggingface_hub.contenidos[("tts-hub/XTTS-v2", "speakers_xtts.pth")] = lambda ruta: torch.save(voces_de_estudio(), ruta)


class _Synthesizer:
    def __init__(self) -> None:
        self.tts_model = Xtts(_Config())
        self.tts_model.speaker_manager = SpeakerManager(voces_de_estudio())


def simular_descarga() -> None:
    entrada = ModelManager().models_dict["tts_models"]["multilingual"]["multi-dataset"]["xtts_v2"]
    huggingface_hub.snapshot_download(entrada["repo_id"], ignore_patterns=entrada.get("ignore"))


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
        simular_descarga()
        self.synthesizer = _Synthesizer()

    def to(self, dispositivo):
        self.synthesizer.tts_model.to(dispositivo)
        return self
