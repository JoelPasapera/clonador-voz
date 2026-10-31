"""Volumen (celda 2): medidor de sonoridad ITU-R BS.1770-4 y limitador, contra valores de referencia."""

import numpy as np
import pytest


def seno(frecuencia_hz: float, segundos: float, amplitud: float, frecuencia: int) -> np.ndarray:
    t = np.arange(int(segundos * frecuencia)) / frecuencia
    return amplitud * np.sin(2 * np.pi * frecuencia_hz * t)


def test_coeficientes_a_48_khz_son_los_de_la_norma(funciones):
    """Tabla 1 y 2 de ITU-R BS.1770-4 (filtro K a 48 kHz)."""
    (b1, a1), (b2, a2) = funciones.ns["coeficientes_filtro_k"](48000)
    assert np.allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-8)
    assert np.allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-8)
    assert np.allclose(b2, [1.0, -2.0, 1.0])
    assert np.allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-8)


def _biquad_directo(b, a, x):
    """Filtro IIR en forma directa I, muestra a muestra (referencia lenta)."""
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for n, muestra in enumerate(x):
        salida = b[0] * muestra + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, muestra, y1, salida
        y[n] = salida
    return y


@pytest.mark.parametrize("frecuencia", [24000, 22050, 48000])
def test_filtro_k_por_fft_equivale_al_iir(funciones, frecuencia):
    ns = funciones.ns
    x = np.random.default_rng(3).standard_normal(int(0.3 * frecuencia)) * 0.2
    directo = x
    for b, a in ns["coeficientes_filtro_k"](frecuencia):
        directo = _biquad_directo(b, a, directo)
    assert np.allclose(ns["ponderar_k"](x, frecuencia), directo, atol=1e-9)


@pytest.mark.parametrize(("frecuencia", "tolerancia"), [(48000, 0.02), (24000, 0.1)])
@pytest.mark.parametrize("amplitud_db", [0.0, -20.0])
def test_seno_de_1khz(funciones, frecuencia, tolerancia, amplitud_db):
    """Un seno de 1 kHz de pico A (dBFS) en un canal mide A − 3,01 LUFS. A 48 kHz (la frecuencia de la norma) debe
    ser exacto; a 24 kHz el filtro rediseñado se desvía unas centésimas, dentro de los ±0,1 LU de EBU Tech 3341."""
    x = seno(1000, 5.0, 10 ** (amplitud_db / 20), frecuencia)
    assert funciones.ns["sonoridad_integrada"](x, frecuencia) == pytest.approx(amplitud_db - 3.01, abs=tolerancia)


def test_compuertas(funciones):
    ns, frecuencia = funciones.ns, 24000
    medir = ns["sonoridad_integrada"]
    tono = seno(1000, 4.0, 0.1, frecuencia)
    silencio = np.zeros(4 * frecuencia)
    muy_bajo = seno(1000, 4.0, 0.001, frecuencia)  # 40 dB por debajo: pasa la compuerta absoluta, no la relativa
    solo_tono = medir(tono, frecuencia)
    # Con silencio después, cuentan los 37 bloques llenos y 3 a medias (75 %, 50 % y 25 % de tono); el silencio, no.
    esperado = solo_tono + 10 * np.log10((37 + 0.75 + 0.5 + 0.25) / 40)
    assert medir(np.concatenate([tono, silencio]), frecuencia) == pytest.approx(esperado, abs=0.02)
    # Sin la compuerta relativa, la parte muy baja hundiría la medida unos 3 dB; con ella, da lo mismo que el silencio.
    assert medir(np.concatenate([tono, muy_bajo]), frecuencia) == pytest.approx(esperado, abs=0.02)
    assert medir(silencio, frecuencia) == float("-inf")
    assert np.isfinite(medir(seno(1000, 0.2, 0.1, frecuencia), frecuencia))  # más corto que un bloque


def test_minimo_deslizante(funciones):
    minimo = funciones.ns["minimo_deslizante"]
    rng = np.random.default_rng(5)
    for _ in range(60):
        x = rng.standard_normal(rng.integers(1, 300))
        ventana = int(rng.integers(1, 40))
        esperado = np.array([x[i : i + ventana].min() for i in range(x.size)])
        assert np.array_equal(minimo(x, ventana), esperado)


def test_largo_fft_rapido(funciones):
    largo = funciones.ns["largo_fft_rapido"]

    def suave(m: int) -> bool:
        for primo in (2, 3, 5):
            while m % primo == 0:
                m //= primo
        return m == 1

    for n in range(1, 3000):
        esperado = next(m for m in range(n, 2 * n + 2) if suave(m))
        assert largo(n) == esperado


def test_limitador_respeta_el_techo_y_no_toca_lo_demas(funciones):
    ns, frecuencia = funciones.ns, 24000
    x = seno(220, 2.0, 0.3, frecuencia)
    x[24000:24010] = 1.4  # un golpe muy por encima del techo
    y = ns["limitar_picos"](x, frecuencia, -1.0)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-7
    lejos = np.r_[0:20000, 28000:x.size]
    assert np.allclose(y[lejos], x[lejos], atol=1e-6)  # lejos del golpe la señal queda intacta
    sin_picos = seno(220, 1.0, 0.5, frecuencia)
    assert np.array_equal(ns["limitar_picos"](sin_picos, frecuencia), sin_picos.astype(np.float32))


@pytest.mark.parametrize("objetivo", [-20.0, -16.0, -14.0])
def test_nivelar_volumen(funciones, objetivo):
    ns, frecuencia = funciones.ns, 24000
    rng = np.random.default_rng(11)
    t = np.arange(6 * frecuencia) / frecuencia
    envolvente = 0.5 * (1 - np.cos(2 * np.pi * 3 * t))  # «sílabas»
    voz = 0.05 * envolvente * (np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)) + 0.002 * rng.standard_normal(t.size)
    y = ns["nivelar_volumen"](voz.astype(np.float32), frecuencia, objetivo)
    assert y.dtype == np.float32 and np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    assert ns["sonoridad_integrada"](y, frecuencia) == pytest.approx(objetivo, abs=0.5)


def test_volumen_parejo_entre_audios_distintos(funciones):
    ns, frecuencia = funciones.ns, 24000
    bajo = seno(300, 3.0, 0.02, frecuencia).astype(np.float32)
    alto = seno(300, 3.0, 0.4, frecuencia).astype(np.float32)
    medidas = [ns["sonoridad_integrada"](ns["nivelar_volumen"](x, frecuencia, -16.0), frecuencia) for x in (bajo, alto)]
    assert abs(medidas[0] - medidas[1]) < 0.1


def test_silencio_no_se_amplifica(funciones):
    ns = funciones.ns
    assert not ns["nivelar_volumen"](np.zeros(24000, np.float32), 24000, -16.0).any()
