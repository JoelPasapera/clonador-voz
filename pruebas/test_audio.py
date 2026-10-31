"""Audio (celda 2): DSP con NumPy y preparación con FFmpeg real sobre varios formatos."""

import numpy as np
import pytest

from conftest import crear_grabacion, duracion_ffprobe


def test_rms_por_tramas_coincide_con_el_calculo_directo(funciones):
    rms = funciones.ns["rms_por_tramas"]
    x = np.random.default_rng(1).uniform(-1, 1, 10007).astype(np.float32)
    ventana, salto = 441, 220
    directo = [np.sqrt(np.mean(x[i : i + ventana].astype(np.float64) ** 2)) for i in range(0, x.size - ventana + 1, salto)]
    assert np.allclose(rms(x, ventana, salto), directo, atol=1e-7)
    assert rms(x[:100], ventana, salto).shape == (1,)
    assert rms(x[:0], ventana, salto).size == 0


def test_recortar_silencios(funciones):
    recortar = funciones.ns["recortar_silencios"]
    sr = 24000
    tono = (0.5 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    x = np.concatenate([np.zeros(sr // 2, np.float32), tono, np.zeros(int(0.8 * sr), np.float32)])
    y = recortar(x, sr)
    assert abs(y.size / sr - (1.0 + 2 * 0.06)) < 0.03
    assert recortar(np.zeros(sr, np.float32), sr).size == 0


def test_fundidos_normalizacion_y_concatenacion(funciones):
    ns = funciones.ns
    sr = 24000
    x = np.full(1000, 0.25, np.float32)
    y = ns["aplicar_fundidos"](x, sr)
    assert y[0] == 0.0 and y[-1] == 0.0 and y[500] == 0.25 and x[0] == 0.25  # no modifica la entrada
    assert np.isclose(np.abs(ns["normalizar_pico"](x)).max(), 10 ** (-1 / 20), atol=1e-6)
    unidos = ns["concatenar_con_pausas"]([x, x], [0.5, 0.0], sr)
    assert unidos.size == 2000 + 12000
    assert not unidos[1000:13000].any() and (unidos[13000:] == 0.25).all()


def test_wav_ida_y_vuelta(funciones, tmp_path):
    x = np.random.default_rng(2).uniform(-1, 1, 5000).astype(np.float32)
    funciones.ns["escribir_wav"](tmp_path / "a.wav", x, 24000)
    y, sr = funciones.ns["leer_wav"](tmp_path / "a.wav")
    assert sr == 24000 and y.dtype == np.float32 and np.abs(x - y).max() <= 1.5 / 32767


@pytest.mark.parametrize("extension", [".wav", ".mp3", ".m4a", ".opus", ".mp4"])
def test_preparar_referencia_de_cualquier_formato(funciones, tmp_path, extension):
    ns = funciones.ns
    origen = crear_grabacion(tmp_path / f"voz{extension}", voz_s=8.0, silencio_ini=1.0, silencio_fin=1.5)
    destino = tmp_path / "ref.wav"
    ns["preparar_referencia"](origen, destino, filtros=ns["filtros_referencia"](True, False), max_segundos=60)
    muestras, sr = ns["leer_wav"](destino)
    assert sr == 22050
    assert 7.4 < muestras.size / sr < 8.7, muestras.size / sr  # sin los silencios de los bordes
    assert np.abs(muestras).max() > 0.05

    ns["preparar_referencia"](origen, destino, filtros=ns["filtros_referencia"](False, False), max_segundos=60)
    assert 10.3 < ns["leer_wav"](destino)[0].size / 22050 < 10.7  # sin limpiar: duración completa

    ns["preparar_referencia"](origen, destino, filtros=ns["filtros_referencia"](True, True), max_segundos=5)
    assert 4.85 < ns["leer_wav"](destino)[0].size / 22050 <= 5.0  # tope de segundos (el remuestreo come unos ms)


def test_mp3_y_error_de_ffmpeg(funciones, tmp_path):
    ns = funciones.ns
    ns["escribir_wav"](tmp_path / "a.wav", np.zeros(24000, np.float32) + 0.1, 24000)
    ns["convertir_a_mp3"](tmp_path / "a.wav", tmp_path / "a.mp3")
    assert abs(duracion_ffprobe(tmp_path / "a.mp3") - 1.0) < 0.1
    (tmp_path / "no_es_audio.mp3").write_text("hola")
    with pytest.raises(ns["ErrorUsuario"], match="FFmpeg no pudo"):
        ns["preparar_referencia"](tmp_path / "no_es_audio.mp3", tmp_path / "b.wav", filtros="", max_segundos=10)
