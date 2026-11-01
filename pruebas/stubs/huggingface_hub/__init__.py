"""huggingface_hub simulado con caché: lo ya descargado (p. ej. por TTS() en la celda 3) no se vuelve a descargar.

`descargas` registra cada archivo que de verdad se «bajó»; `contenidos` permite que otros simuladores definan qué
contiene un archivo (p. ej. speakers_xtts.pth con voces de estudio que torch.load pueda leer).
"""

import fnmatch
import os
from pathlib import Path

descargas = []
contenidos = {}  # (repo_id, archivo) → función(ruta) que escribe el archivo
REPOSITORIOS = {
    "tts-hub/XTTS-v2": ["config.json", "dvae.pth", "hash.md5", "mel_stats.pth", "model.pth", "speakers_xtts.pth", "vocab.json"],
}


def _carpeta(repo_id: str) -> Path:
    raiz = Path(os.environ.get("HF_HOME", "/tmp/hf_simulado")) / "hub" / f"models--{repo_id.replace('/', '--')}"
    return raiz / "snapshots" / "simulada"


def hf_hub_download(repo_id, filename, **_):
    assert filename in REPOSITORIOS.get(repo_id, []), f"{repo_id} no tiene {filename}"
    ruta = _carpeta(repo_id) / filename
    if not ruta.is_file():
        ruta.parent.mkdir(parents=True, exist_ok=True)
        escribir = contenidos.get((repo_id, filename))
        if escribir is not None:
            escribir(ruta)
        else:
            ruta.write_text(f"{repo_id}/{filename}", encoding="utf-8")
        descargas.append((repo_id, filename))
    return str(ruta)


def snapshot_download(repo_id, allow_patterns=None, ignore_patterns=None, **_):
    for nombre in REPOSITORIOS[repo_id]:
        if ignore_patterns and any(fnmatch.fnmatch(nombre, patron) for patron in ignore_patterns):
            continue
        hf_hub_download(repo_id, nombre)
    return str(_carpeta(repo_id))
