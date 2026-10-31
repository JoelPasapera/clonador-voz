"""google.colab.files simulado: descargas registradas y subidas desde una lista preparada por la prueba."""

import os
import shutil

descargas = []
para_subir = []  # rutas que «elige» el usuario en el botón de subida


def download(filename):
    assert os.path.isfile(filename), filename
    descargas.append(filename)


def upload(target_dir=""):
    """Como el real: escribe cada archivo en target_dir y devuelve {ruta local: contenido}."""
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    subidos = {}
    for origen in para_subir:
        destino = os.path.join(target_dir, os.path.basename(origen))
        shutil.copyfile(origen, destino)
        with open(destino, "rb") as archivo:
            subidos[destino] = archivo.read()
    return subidos
