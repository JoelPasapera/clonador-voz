"""torch simulado: solo lo que usa el cuaderno, respaldado por NumPy (no hay PyTorch en el banco de pruebas)."""

import contextlib

import numpy as np

__version__ = "2.10.0+cu126"
_rng = np.random.default_rng(0)


class Tensor:
    def __init__(self, datos, dispositivo: str = "cpu") -> None:
        self.datos = np.asarray(datos)
        self.dispositivo = dispositivo

    @property
    def shape(self):
        return self.datos.shape

    def size(self, dim=None):
        return self.datos.shape if dim is None else self.datos.shape[dim]

    def unsqueeze(self, eje):
        return Tensor(np.expand_dims(self.datos, eje), self.dispositivo)

    def to(self, dispositivo):
        return Tensor(self.datos, str(dispositivo))

    def mean(self, dim=0):
        return Tensor(self.datos.mean(axis=dim), self.dispositivo)

    def cpu(self):
        return Tensor(self.datos, "cpu")

    def numpy(self):
        return self.datos


def from_numpy(arreglo):
    assert isinstance(arreglo, np.ndarray) and arreglo.flags.c_contiguous, "from_numpy exige un arreglo contiguo"
    return Tensor(arreglo)


def cat(tensores, dim=0):
    assert len({t.dispositivo for t in tensores}) == 1, "torch.cat con tensores en dispositivos distintos"
    return Tensor(np.concatenate([t.datos for t in tensores], axis=dim), tensores[0].dispositivo)


def stack(tensores, dim=0):
    assert len({t.dispositivo for t in tensores}) == 1, "torch.stack con tensores en dispositivos distintos"
    return Tensor(np.stack([t.datos for t in tensores], axis=dim), tensores[0].dispositivo)


def inference_mode(modo: bool = True):
    return contextlib.nullcontext()


def manual_seed(semilla: int) -> None:
    global _rng
    _rng = np.random.default_rng(semilla)


def generador():
    """RNG que usa el XTTS simulado: así la semilla del cuaderno hace repetible la salida."""
    return _rng


class cuda:  # noqa: N801 (imita torch.cuda)
    disponible = True

    @staticmethod
    def is_available() -> bool:
        return cuda.disponible

    @staticmethod
    def manual_seed_all(semilla: int) -> None:
        pass

    @staticmethod
    def get_device_name(indice: int = 0) -> str:
        return "Tesla T4 (simulada)"
