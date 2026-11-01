"""Audio (celda 2): DSP con NumPy y preparación con FFmpeg real sobre varios formatos."""

import numpy as np
import pytest

from conftest import crear_grabacion, duracion_ffprobe, senal_de_voz


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


# ── v2: estéreo, ensamblado, remuestreo, silencios, FFmpeg en memoria y música ──


def test_wav_estereo(funciones, tmp_path):
    import wave

    x = np.stack([np.linspace(-0.5, 0.5, 1000), np.linspace(0.5, -0.5, 1000)], axis=1).astype(np.float32)
    funciones.ns["escribir_wav"](tmp_path / "e.wav", x, 48000)
    with wave.open(str(tmp_path / "e.wav"), "rb") as archivo:
        assert archivo.getnchannels() == 2 and archivo.getframerate() == 48000 and archivo.getnframes() == 1000
        leido = np.frombuffer(archivo.readframes(1000), "<i2").reshape(-1, 2) / 32767
    assert np.allclose(leido, x, atol=1e-4)


def test_ensamblar_devuelve_las_posiciones(funciones):
    audio, posiciones = funciones.ns["ensamblar"]([np.ones(10, np.float32), np.ones(5, np.float32)], [0.5, 0.0], 10)
    assert audio.size == 20 and posiciones == [(0, 10), (15, 20)] and not audio[10:15].any()


@pytest.mark.parametrize(("origen", "destino"), [(24000, 16000), (22050, 16000), (24000, 48000), (44100, 24000)])
def test_remuestrear_conserva_la_frecuencia_y_el_largo(funciones, origen, destino):
    t = np.arange(round(1.3 * origen)) / origen
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    y = funciones.ns["remuestrear"](x, origen, destino)
    assert y.dtype == np.float32 and y.size == round(x.size * destino / origen)
    esperado = 0.5 * np.sin(2 * np.pi * 440 * np.arange(y.size) / destino)
    centro = slice(y.size // 10, -y.size // 10)  # lejos de los bordes
    assert np.max(np.abs(y[centro] - esperado[centro])) < 1e-3


def test_remuestrear_quita_lo_que_no_cabe(funciones):
    t = np.arange(24000) / 24000
    x = np.sin(2 * np.pi * 10000 * t).astype(np.float32)  # 10 kHz no existe a 16 kHz (Nyquist: 8 kHz)
    y = funciones.ns["remuestrear"](x, 24000, 16000)
    assert np.abs(y[2000:-2000]).max() < 0.02  # (en los bordes queda el salto brusco del corte, que sí es de banda ancha)


def test_silencio_interno_maximo(funciones):
    ns, sr = funciones.ns, 24000
    voz = senal_de_voz(1.0, sr)
    x = np.concatenate([np.zeros(sr), voz, np.zeros(2 * sr), voz, np.zeros(3 * sr)])
    assert ns["silencio_interno_maximo"](x, sr) == pytest.approx(2.0, abs=0.05)  # los bordes no cuentan
    assert ns["silencio_interno_maximo"](voz, sr) < 0.1
    assert ns["silencio_interno_maximo"](np.zeros(sr, np.float32), sr) == 0.0


def test_decodificar_audio_en_estereo(funciones, tmp_path):
    origen = crear_grabacion(tmp_path / "musica.mp3", voz_s=2.0)
    musica = funciones.ns["decodificar_audio"](origen, 48000, canales=2)
    assert musica.shape[1] == 2 and musica.dtype == np.float32
    assert 4.3 < musica.shape[0] / 48000 < 4.7
    corta = funciones.ns["decodificar_audio"](origen, 16000, max_segundos=1.0)
    assert corta.ndim == 1 and corta.size == 16000
    with pytest.raises(funciones.ns["ErrorUsuario"], match="no pudo leer"):
        (tmp_path / "roto.mp3").write_bytes(b"no es audio")
        funciones.ns["decodificar_audio"](tmp_path / "roto.mp3", 16000)


def test_atempo_en_memoria(funciones):
    x = senal_de_voz(3.0, 24000)
    y = funciones.ns["procesar_con_ffmpeg"](x, 24000, "atempo=1.25")
    assert y.size / 24000 == pytest.approx(3.0 / 1.25, abs=0.02)


def test_sonoridad_y_limitador_en_estereo(funciones):
    ns, sr = funciones.ns, 48000
    t = np.arange(3 * sr) / sr
    mono = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    doble = np.stack([mono, mono], axis=1)
    assert ns["sonoridad_integrada"](doble, sr) == pytest.approx(ns["sonoridad_integrada"](mono, sr) + 3.01, abs=0.02)
    estereo = np.stack([mono * 5, mono], axis=1)  # el izquierdo pasa el techo
    limitado = ns["limitar_picos"](estereo, sr)
    assert np.abs(limitado).max() <= 10 ** (-1 / 20) + 1e-6
    proporcion = limitado[sr : 2 * sr, 0] / np.where(np.abs(limitado[sr : 2 * sr, 1]) > 1e-3, limitado[sr : 2 * sr, 1], np.nan)
    assert np.nanstd(proporcion) < 1e-3  # misma ganancia en los dos canales: la imagen no se mueve
    nivelado = ns["nivelar_volumen"](doble, sr, -16.0)
    assert ns["sonoridad_integrada"](nivelado, sr) == pytest.approx(-16.0, abs=0.3)


def test_mezcla_con_musica(funciones):
    ns, sr = funciones.ns, 48000
    voz = np.concatenate([senal_de_voz(2.0, sr), np.zeros(2 * sr, np.float32), senal_de_voz(2.0, sr)])
    t = np.arange(sr) / sr
    musica = np.stack([np.sin(2 * np.pi * 330 * t), np.sin(2 * np.pi * 330 * t)], axis=1).astype(np.float32) * 0.5
    mezcla = ns["mezclar_con_musica"](voz, musica, sr, nivel_db=-18.0)
    entrada, salida = round(1.5 * sr), round(2.5 * sr)
    assert mezcla.shape == (entrada + voz.size + salida, 2)
    assert np.abs(mezcla[0]).max() < 1e-3 and np.abs(mezcla[-1]).max() < 1e-3  # fundidos de entrada y salida
    # la música sola en la pausa larga (sube) es más fuerte que bajo la voz
    pausa = mezcla[entrada + round(3.4 * sr) : entrada + round(3.5 * sr), 0]
    bajo_la_voz = mezcla[entrada + round(1.0 * sr) : entrada + round(1.1 * sr), 0] - voz[round(1.0 * sr) : round(1.1 * sr)]
    assert 20 * np.log10(np.abs(pausa).max() / np.abs(bajo_la_voz).max()) > 6.0
    mono = ns["mezclar_con_musica"](voz, musica[:, 0], sr, nivel_db=-18.0)
    assert mono.shape == (mezcla.shape[0], 1)


def test_dilatar_activa_antes_y_despues(funciones):
    activas = np.zeros(20, dtype=bool)
    activas[10] = True
    assert np.flatnonzero(funciones.ns["dilatar"](activas, antes=2, despues=3)).tolist() == [8, 9, 10, 11, 12, 13]
