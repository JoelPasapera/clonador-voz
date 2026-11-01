"""TTS.utils.manage simulado: misma entrada de XTTS-v2 que el .models.json de coqui-tts 0.28.0 (test_fidelidad lo
comprueba) y misma carpeta de la licencia que get_user_data_dir("tts")."""

import os
from pathlib import Path


def get_user_data_dir(appname: str) -> Path:
    if os.environ.get("TTS_HOME"):
        base = Path(os.environ["TTS_HOME"])
    elif os.environ.get("XDG_DATA_HOME"):
        base = Path(os.environ["XDG_DATA_HOME"])
    else:
        base = Path.home() / ".local" / "share"
    return base.joinpath(appname)


class ModelManager:
    def __init__(self, models_file=None, output_prefix=None, progress_bar: bool = False) -> None:
        self.progress_bar = progress_bar
        self.output_prefix = get_user_data_dir("tts") if output_prefix is None else Path(output_prefix) / "tts"
        self.models_dict = {
            "tts_models": {
                "multilingual": {
                    "multi-dataset": {
                        "xtts_v2": {"repo_id": "tts-hub/XTTS-v2", "ignore": ["samples/*", "dvae.pth"], "license": "CPML"},
                    }
                }
            }
        }
