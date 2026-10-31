"""Banco de pruebas: ejecuta las celdas REALES del .ipynb en un «Colab» simulado (Drive, GPU, torch y XTTS falsos)."""

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RAIZ = Path(__file__).resolve().parent.parent
STUBS = Path(__file__).resolve().parent / "stubs"
CUADERNO = RAIZ / "Clonador_de_voz_XTTS.ipynb"
MARCADOR = "# ─── Ejecución ───"
CONTENT = Path("/content")
MI_UNIDAD = CONTENT / "drive" / "MyDrive"
PATRON_PARAM = re.compile(r"^(?P<nombre>\w+)\s*=\s*(?P<valor>.+?)(?P<comentario>\s*# @param.*)$")

sys.path.insert(0, str(STUBS))


def celdas_de_codigo() -> dict[str, str]:
    cuaderno = json.loads(CUADERNO.read_text(encoding="utf-8"))
    return {c["metadata"]["id"]: "".join(c["source"]) for c in cuaderno["cells"] if c["cell_type"] == "code"}


def con_parametros(fuente: str, **valores) -> str:
    """Simula que el usuario cambió campos del formulario."""
    pendientes = dict(valores)
    lineas = []
    for linea in fuente.split("\n"):
        m = PATRON_PARAM.match(linea)
        if m and m["nombre"] in pendientes:
            linea = f"{m['nombre']} = {pendientes.pop(m['nombre'])!r}{m['comentario']}"
        lineas.append(linea)
    assert not pendientes, f"esos parámetros no existen en el formulario: {sorted(pendientes)}"
    return "\n".join(lineas)


class ShellFalso:
    def __init__(self) -> None:
        self.tipos: tuple = ()
        self.manejador = None

    def set_custom_exc(self, tipos, manejador) -> None:
        self.tipos, self.manejador = tipos, manejador


class Sesion:
    """Un kernel de Colab: un único espacio de nombres compartido por todas las celdas."""

    def __init__(self) -> None:
        self.shell = ShellFalso()
        self.ns: dict = {"__name__": "__main__", "get_ipython": lambda: self.shell}
        self.fuentes = celdas_de_codigo()

    def _correr(self, codigo: str, nombre: str) -> None:
        exec(compile(codigo, f"<celda {nombre}>", "exec"), self.ns)

    def ejecutar(self, celda: str, **parametros) -> None:
        self._correr(con_parametros(self.fuentes[celda], **parametros), celda)

    def definir(self, celda: str, **parametros) -> None:
        """Solo la parte de definiciones (antes del marcador de ejecución)."""
        definiciones, encontrado, _ = con_parametros(self.fuentes[celda], **parametros).partition(MARCADOR)
        assert encontrado, f"la celda {celda} no tiene marcador de ejecución"
        self._correr(definiciones, celda)

    def cargar_funciones(self) -> None:
        """Celdas 2a, 2b y 2c (sin parámetros)."""
        for celda in ("archivos", "audio", "texto"):
            self.ejecutar(celda)

    def solo_ejecucion(self, celda: str, **parametros) -> None:
        _, _, ejecucion = con_parametros(self.fuentes[celda], **parametros).partition(MARCADOR)
        self._correr(ejecucion, celda)

    def error(self):
        return self.ns["ErrorUsuario"]


def crear_bin_con_gpu(carpeta: Path) -> Path:
    """`nvidia-smi` simulado, creado al vuelo: el permiso de ejecución se pierde al descomprimir en Windows o al
    subir archivos por la web de GitHub, así que no puede venir en el repositorio."""
    carpeta.mkdir(parents=True, exist_ok=True)
    nvidia_smi = carpeta / "nvidia-smi"
    nvidia_smi.write_text('#!/bin/sh\necho "Tesla T4, 15360 MiB"\n', encoding="utf-8")
    nvidia_smi.chmod(0o755)
    return carpeta


class Colab:
    def __init__(self, sitio: Path, monkeypatch, bin_gpu: Path) -> None:
        self.sitio = sitio
        self.monkeypatch = monkeypatch
        self.bin_gpu = bin_gpu

    def instalar(self, nombre: str, version: str) -> None:
        carpeta = self.sitio / f"{nombre.replace('-', '_')}-{version}.dist-info"
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {nombre}\nVersion: {version}\n")
        importlib.invalidate_caches()

    def instalar_todo(self) -> None:
        for nombre, version in [
            ("torch", "2.10.0+cu126"), ("torchaudio", "2.10.0+cu126"), ("torchcodec", "0.10.0"), ("coqui-tts", "0.28.0"),
        ]:
            self.instalar(nombre, version)

    def sin_gpu(self) -> None:
        self.monkeypatch.setenv("PATH", os.environ["PATH"].replace(str(self.bin_gpu) + os.pathsep, ""))

    def sesion(self) -> Sesion:
        return Sesion()


@pytest.fixture
def colab(tmp_path, monkeypatch):
    # Se vacía /content sin borrar la carpeta: en GitHub Actions la crea root y el usuario solo es su dueño.
    CONTENT.mkdir(exist_ok=True)
    for hijo in CONTENT.iterdir():
        if hijo.is_dir() and not hijo.is_symlink():
            shutil.rmtree(hijo)
        else:
            hijo.unlink()
    bin_gpu = crear_bin_con_gpu(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(bin_gpu) + os.pathsep + os.environ["PATH"])
    monkeypatch.delenv("COQUI_TOS_AGREED", raising=False)
    sitio = tmp_path / "sitio"
    sitio.mkdir()
    monkeypatch.syspath_prepend(str(sitio))

    import torch
    from google.colab import drive, files
    from IPython import display
    from TTS import api

    torch.cuda.disponible = True
    api.registro.update(cargas=0, inferencias=[], speaker=[], gpt=[])
    drive.montajes.clear()
    drive.fallos_pendientes = 0
    drive.montado = False
    files.descargas.clear()
    files.para_subir.clear()
    display.mostrados.clear()
    es_montaje_real = os.path.ismount
    monkeypatch.setattr(
        os.path, "ismount", lambda ruta: (drive.montado and os.fspath(ruta) == "/content/drive") or es_montaje_real(ruta)
    )
    yield Colab(sitio, monkeypatch, bin_gpu)


@pytest.fixture
def lista(colab):
    """Sesión con las celdas 1, 2 y 3 ejecutadas (todo instalado, licencia aceptada)."""
    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    sesion.ejecutar("modelo")  # la casilla de uso no comercial viene marcada por defecto
    return sesion


@pytest.fixture
def con_voz(lista):
    """Sesión con la voz de referencia ya creada (celdas 1 a 4)."""
    crear_grabacion(MI_UNIDAD / "mi_voz.wav", voz_s=15.0)
    lista.ejecutar("referencia")  # valor por defecto del formulario: "mi_voz.wav"
    return lista


@pytest.fixture
def funciones(colab):
    """Sesión con las funciones base cargadas (sin pasar por la instalación)."""
    sesion = colab.sesion()
    sesion.definir("entorno")
    sesion.cargar_funciones()
    return sesion


# ── Audio de prueba ────────────────────────────────────────────────────────────


def senal_de_voz(segundos: float, frecuencia: int = 44100, f0: float = 140.0, amplitud: float = 0.5) -> np.ndarray:
    """Señal parecida a la voz: armónicos con «sílabas» de 4 Hz."""
    t = np.arange(int(segundos * frecuencia)) / frecuencia
    envolvente = 0.5 * (1 - np.cos(2 * np.pi * 4 * t))
    return (amplitud * envolvente * (np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 3 * f0 * t)) / 1.4).astype(np.float32)


def crear_grabacion(destino: Path, *, voz_s: float, silencio_ini: float = 1.0, silencio_fin: float = 1.5) -> Path:
    """Grabación de prueba (silencio + voz + silencio) en el formato que indique la extensión del destino."""
    frecuencia = 44100
    muestras = np.concatenate([
        np.zeros(int(silencio_ini * frecuencia), np.float32),
        senal_de_voz(voz_s, frecuencia),
        np.zeros(int(silencio_fin * frecuencia), np.float32),
    ])
    destino.parent.mkdir(parents=True, exist_ok=True)
    wav = destino.with_suffix(".tmp.wav")
    pcm = np.round(muestras * 32767).astype("<i2")
    import wave

    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2)  # estéreo, como muchos celulares
        w.setsampwidth(2)
        w.setframerate(frecuencia)
        w.writeframes(np.repeat(pcm, 2).tobytes())
    extension = destino.suffix.lower()
    if extension == ".wav":
        wav.rename(destino)
        return destino
    codecs = {".mp3": ["-c:a", "libmp3lame"], ".m4a": ["-c:a", "aac"], ".opus": ["-c:a", "libopus"], ".ogg": ["-c:a", "libopus"]}
    if extension in codecs:
        comando = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), *codecs[extension], str(destino)]
    else:  # video con pista de audio
        duracion = silencio_ini + voz_s + silencio_fin
        comando = [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:d={duracion}",
            "-i", str(wav), "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(destino),
        ]
    subprocess.run(comando, check=True)
    wav.unlink()
    return destino


def duracion_ffprobe(ruta: Path) -> float:
    salida = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(ruta)],
        capture_output=True, text=True, check=True,
    )
    return float(salida.stdout.strip())


def flujo_de_audio_ffprobe(ruta: Path) -> dict:
    salida = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name,sample_rate,bit_rate",
         "-of", "json", str(ruta)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(salida.stdout)["streams"][0]
