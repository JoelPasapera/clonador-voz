"""Instala las librerías con el MISMO código de la celda 1 (uv con respaldo en pip), como lo haría Colab.

Uso: python pruebas/real/preparar_entorno.py xtts|chatterbox
En CI se agrega el índice de PyTorch para CPU, así se instalan sus ruedas «+cpu» en lugar de las de CUDA.
"""

import sys

from cuaderno import Cuaderno

INDICE_CPU = "https://download.pytorch.org/whl/cpu"


def main(motor: str) -> None:
    cuaderno = Cuaderno()
    cuaderno.definir("entorno")
    ns = cuaderno.ns
    pendientes = ns["paquetes_pendientes"](motor)
    pendientes.append(f"faster-whisper=={ns['VERSION_FASTER_WHISPER']}")
    indices = sorted(set(ns["indices_pytorch"](pendientes)) | {INDICE_CPU})
    print("📦 Instalando con el código de la celda 1:", ", ".join(pendientes), flush=True)
    ns["instalar"](pendientes, indices)
    paquete = "chatterbox-tts" if motor == "chatterbox" else "coqui-tts"
    for nombre in ("torch", "torchaudio", "torchcodec", "torchvision", paquete, "transformers", "faster-whisper"):
        print(f"   {nombre} {ns['version_instalada'](nombre)}")


if __name__ == "__main__":
    main(sys.argv[1])
