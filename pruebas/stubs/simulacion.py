"""Piezas comunes de los modelos simulados (XTTS, Chatterbox y Whisper).

La «voz» de una señal es su frecuencia fundamental: la huella del hablante es una campana centrada en ella, así las
pruebas pueden comprobar parecidos y diálogos. Marcas en el texto provocan fallos solo en el primer intento:
  zzbalbuceo  → audio 3 veces más largo (balbuceo)  zzsilencio → 2 s de silencio a mitad de frase
  zzmurmullo  → Whisper entiende otra cosa          zzsiempre  → balbucea en TODOS los intentos
"""

import numpy as np

DIMENSION = 512
ANCHO_HZ = 20.0
estado = {"ultimo_texto": "", "intentos": {}}


def reiniciar() -> None:
    estado["ultimo_texto"] = ""
    estado["intentos"] = {}


def huella_de_frecuencia(f0: float) -> np.ndarray:
    indices = np.arange(DIMENSION, dtype=np.float64)
    campana = np.exp(-(((indices - f0) / ANCHO_HZ) ** 2))
    return (campana / np.linalg.norm(campana)).astype(np.float32)


def frecuencia_dominante(muestras: np.ndarray, frecuencia: int) -> float:
    x = np.asarray(muestras, dtype=np.float64).reshape(-1)
    espectro = np.abs(np.fft.rfft(x))
    hz = np.fft.rfftfreq(x.size, 1.0 / frecuencia)
    banda = (hz >= 60) & (hz <= 400)
    return float(hz[banda][np.argmax(espectro[banda])])


def frecuencia_de_huella(huella: np.ndarray) -> float:
    return float(np.argmax(np.asarray(huella).reshape(-1)))


def intento(texto: str) -> int:
    estado["intentos"][texto] = estado["intentos"].get(texto, 0) + 1
    return estado["intentos"][texto]


def voz_sintetica(texto: str, frecuencia: int, f0: float, velocidad: float, rng) -> np.ndarray:
    """Silencio + «sílabas» a 4 Hz (55 ms por carácter) + silencio, con los fallos simulados del primer intento."""
    numero = intento(texto)
    estado["ultimo_texto"] = texto
    segundos_voz = max(0.3, 0.055 * len(texto) / velocidad)
    if ("zzbalbuceo" in texto and numero == 1) or "zzsiempre" in texto:
        segundos_voz *= 3.0
    t = np.arange(int(segundos_voz * frecuencia)) / frecuencia
    envolvente = 0.3 * (1.0 - np.cos(2.0 * np.pi * 4.0 * t))
    voz = envolvente * (np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t))
    if "zzsilencio" in texto and numero == 1:
        mitad = voz.size // 2
        voz = np.concatenate([voz[:mitad], np.zeros(int(2.0 * frecuencia)), voz[mitad:]])
    silencio_inicial = rng.normal(0.0, 1e-4, int(0.3 * frecuencia))
    silencio_final = rng.normal(0.0, 1e-4, int(0.6 * frecuencia))
    return np.concatenate([silencio_inicial, voz, silencio_final]).astype(np.float32)
