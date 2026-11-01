"""torch simulado: solo lo que usa el cuaderno, respaldado por NumPy (no hay PyTorch en el banco de pruebas)."""

import contextlib
import pickle

import numpy as np

__version__ = "2.10.0+cu126"
_rng = np.random.default_rng(0)
float16 = np.float16


class Tensor:
    def __init__(self, datos, dispositivo: str = "cpu") -> None:
        self.datos = np.asarray(datos)
        self.dispositivo = dispositivo

    @property
    def shape(self):
        return self.datos.shape

    @property
    def dtype(self):
        return self.datos.dtype

    @property
    def device(self):
        return self.dispositivo

    def size(self, dim=None):
        return self.datos.shape if dim is None else self.datos.shape[dim]

    def unsqueeze(self, eje):
        return Tensor(np.expand_dims(self.datos, eje), self.dispositivo)

    def squeeze(self, eje=None):
        return Tensor(np.squeeze(self.datos) if eje is None else np.squeeze(self.datos, eje), self.dispositivo)

    def to(self, dispositivo=None, dtype=None, **_):
        datos = self.datos if dtype is None else self.datos.astype(dtype)
        return Tensor(datos, self.dispositivo if dispositivo is None else str(dispositivo))

    def mean(self, dim=0):
        return Tensor(self.datos.mean(axis=dim), self.dispositivo)

    def cpu(self):
        return Tensor(self.datos, "cpu")

    def detach(self):
        return self

    def float(self):
        return Tensor(self.datos.astype(np.float32), self.dispositivo)

    def half(self):
        return Tensor(self.datos.astype(np.float16), self.dispositivo)

    def is_floating_point(self) -> bool:
        return bool(np.issubdtype(self.datos.dtype, np.floating))

    def numpy(self):
        assert self.dispositivo == "cpu", "numpy() de un tensor en la GPU: falta .cpu()"
        return self.datos


def is_tensor(objeto) -> bool:
    return isinstance(objeto, Tensor)


def from_numpy(arreglo):
    assert isinstance(arreglo, np.ndarray) and arreglo.flags.c_contiguous, "from_numpy exige un arreglo contiguo"
    return Tensor(arreglo)


def cat(tensores, dim=0):
    assert len({t.dispositivo for t in tensores}) == 1, "torch.cat con tensores en dispositivos distintos"
    return Tensor(np.concatenate([t.datos for t in tensores], axis=dim), tensores[0].dispositivo)


def stack(tensores, dim=0):
    assert len({t.dispositivo for t in tensores}) == 1, "torch.stack con tensores en dispositivos distintos"
    return Tensor(np.stack([t.datos for t in tensores], axis=dim), tensores[0].dispositivo)


def zeros(*forma):
    return Tensor(np.zeros(forma, np.float32))


def ones(*forma):
    return Tensor(np.ones(forma, np.float32))


def inference_mode(modo: bool = True):
    return contextlib.nullcontext()


def manual_seed(semilla: int) -> None:
    global _rng
    _rng = np.random.default_rng(semilla)


def generador():
    """RNG que usan los modelos simulados: así la semilla del cuaderno hace repetible la salida."""
    return _rng


# ── torch.save / torch.load (con la restricción real de weights_only=True) ──

_PERMITIDOS = (str, int, float, bool, type(None), Tensor)


def _solo_pesos(objeto) -> bool:
    if isinstance(objeto, dict):
        return all(isinstance(k, (str, int)) and _solo_pesos(v) for k, v in objeto.items())
    if isinstance(objeto, (list, tuple)):
        return all(_solo_pesos(v) for v in objeto)
    return isinstance(objeto, _PERMITIDOS)


def save(objeto, ruta) -> None:
    with open(ruta, "wb") as archivo:
        pickle.dump(objeto, archivo)


def load(ruta, map_location=None, weights_only=True, **_):
    with open(ruta, "rb") as archivo:
        objeto = pickle.load(archivo)
    if weights_only and not _solo_pesos(objeto):
        raise pickle.UnpicklingError("Weights only load failed: el archivo contiene objetos que no son tensores")
    return objeto


class cuda:  # noqa: N801 (imita torch.cuda)
    disponible = True
    vaciados = 0

    @staticmethod
    def is_available() -> bool:
        return cuda.disponible

    @staticmethod
    def manual_seed_all(semilla: int) -> None:
        pass

    @staticmethod
    def empty_cache() -> None:
        cuda.vaciados += 1

    @staticmethod
    def get_device_name(indice: int = 0) -> str:
        return "Tesla T4 (simulada)"
