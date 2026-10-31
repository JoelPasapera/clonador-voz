"""IPython.display simulado: registra lo que el cuaderno muestra."""

import os

mostrados = []


class Audio:
    def __init__(self, data=None, filename=None, url=None, embed=None, rate=None, autoplay=False, normalize=True, *, element_id=None):
        if filename is not None:
            assert os.path.isfile(filename), f"Audio(filename=...) apunta a un archivo inexistente: {filename}"
        self.filename = filename


def display(*objetos, **_):
    mostrados.extend(objetos)
