"""Salida (celda 6): subtítulos .srt sincronizados, música de fondo con atenuación y modo lote."""

import re
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import MI_UNIDAD, audios_en_drive

PATRON_TIEMPO = re.compile(r"(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})")


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def leer_srt(ruta: Path) -> list[tuple[float, float, str]]:
    subtitulos = []
    for bloque in ruta.read_text(encoding="utf-8").strip().split("\n\n"):
        numero, tiempos, *lineas = bloque.split("\n")
        assert numero.isdecimal() and int(numero) == len(subtitulos) + 1
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, PATRON_TIEMPO.fullmatch(tiempos).groups())
        assert 1 <= len(lineas) <= 2 and all(len(linea) <= 42 for linea in lineas)
        subtitulos.append((h1 * 3600 + m1 * 60 + s1 + ms1 / 1000, h2 * 3600 + m2 * 60 + s2 + ms2 / 1000, " ".join(lineas)))
    return subtitulos


def crear_musica(ruta: Path, segundos: float = 4.0, hz: float = 2000.0) -> Path:
    """Tono estéreo de prueba (como música), en WAV a 44,1 kHz."""
    frecuencia = 44100
    t = np.arange(int(segundos * frecuencia)) / frecuencia
    tono = 0.3 * np.sin(2 * np.pi * hz * t)
    pcm = np.round(np.stack([tono, tono], axis=1) * 32767).astype("<i2")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(ruta), "wb") as archivo:
        archivo.setnchannels(2)
        archivo.setsampwidth(2)
        archivo.setframerate(frecuencia)
        archivo.writeframes(pcm.tobytes())
    return ruta


def amplitud_del_tono(x: np.ndarray, frecuencia: int, desde: float, hz: float = 2000.0, ventana: float = 0.1) -> float:
    """Amplitud de una sinusoide de `hz` en una ventana (detección síncrona con ventana de Hann): la voz simulada
    (140–300 Hz) no se cuela en la medida."""
    inicio = round(desde * frecuencia)
    trozo = x[inicio : inicio + round(ventana * frecuencia)]
    t = np.arange(trozo.size) / frecuencia
    hann = np.hanning(trozo.size)
    seno, coseno = np.sum(trozo * hann * np.sin(2 * np.pi * hz * t)), np.sum(trozo * hann * np.cos(2 * np.pi * hz * t))
    return float(2 * np.hypot(seno, coseno) / hann.sum())


# ── Subtítulos ──────────────────────────────────────────────────────────────────


def test_subtitulos_sincronizados_con_el_audio(con_voz):
    from google.colab import files

    texto = "Primera frase del subtítulo. Segunda frase, un poco más larga que la primera, para probar el reparto."
    con_voz.generar(texto=texto, subtitulos=True, formato="wav", pausa_entre_frases=0.5, descargar_al_terminar=True)
    toma = con_voz.ns["ULTIMA_GENERACION"].tomas[0]
    assert toma.subtitulos == toma.archivo.with_suffix(".srt")
    subtitulos = leer_srt(toma.subtitulos)
    assert [s[2] for s in subtitulos][0] == "Primera frase del subtítulo."
    assert " ".join(s[2] for s in subtitulos) == texto
    assert subtitulos[0][0] == 0.0
    assert subtitulos[1][0] - subtitulos[0][1] == pytest.approx(0.5, abs=0.002)  # la pausa entre frases
    audio, sr = con_voz.ns["leer_wav"](toma.archivo)
    assert subtitulos[-1][1] == pytest.approx(audio.size / sr, abs=0.002)  # el último termina con el audio
    assert all(a < b for a, b, _ in subtitulos)
    assert audios_en_drive() == ["mi_voz_clonada.srt", "mi_voz_clonada.wav"]
    assert files.descargas == [str(toma.archivo), str(toma.subtitulos)]


def test_audio_y_subtitulos_comparten_nombre_libre(con_voz):
    carpeta = MI_UNIDAD / "Clonador de voz"
    (carpeta / "mi_voz_clonada.srt").write_text("de otro video", encoding="utf-8")  # solo existe el .srt
    con_voz.generar(texto="Una frase con subtítulos de prueba.", subtitulos=True, formato="mp3")
    assert audios_en_drive() == ["mi_voz_clonada.srt", "mi_voz_clonada_2.mp3", "mi_voz_clonada_2.srt"]
    assert (carpeta / "mi_voz_clonada.srt").read_text(encoding="utf-8") == "de otro video"


def test_subtitulos_con_varias_tomas_se_guardan_con_la_elegida(con_voz):
    con_voz.generar(texto="Frase para elegir entre varias tomas.", subtitulos=True, tomas=2, semilla=8, formato="wav")
    generacion = con_voz.ns["ULTIMA_GENERACION"]
    assert [t.subtitulos.name for t in generacion.tomas] == ["mi_voz_clonada_toma1.srt", "mi_voz_clonada_toma2.srt"]
    con_voz.ejecutar("elegir", toma_elegida="2")
    assert audios_en_drive() == ["mi_voz_clonada.srt", "mi_voz_clonada.wav"]
    copia = (MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.srt").read_bytes()
    assert copia == generacion.tomas[1].subtitulos.read_bytes()


# ── Música de fondo ─────────────────────────────────────────────────────────────


def test_musica_de_fondo_entra_antes_y_baja_mientras_se_habla(con_voz, capsys):
    crear_musica(MI_UNIDAD / "Música" / "fondo.wav")
    texto = "Bienvenidos al programa de hoy, con música de fondo. Esta segunda frase también lleva música detrás."
    con_voz.generar(texto=texto, musica_de_fondo="fondo", formato="wav", subtitulos=True, guardar_en_drive=False)
    toma = con_voz.ns["ULTIMA_GENERACION"].tomas[0]
    with wave.open(str(toma.archivo), "rb") as archivo:
        assert archivo.getnchannels() == 2 and archivo.getframerate() == 48000
        estereo = np.frombuffer(archivo.readframes(archivo.getnframes()), "<i2").reshape(-1, 2) / 32768
    assert np.array_equal(estereo[:, 0], estereo[:, 1])  # voz al centro y música en estéreo idéntico
    izquierda, sr = estereo[:, 0], 48000
    subtitulos = leer_srt(toma.subtitulos)
    assert subtitulos[0][0] == pytest.approx(1.5, abs=0.002)  # la voz empieza tras la entrada de la música
    fin_voz = subtitulos[-1][1]
    assert izquierda.size / sr == pytest.approx(fin_voz + 2.5, abs=0.01)  # y la música sigue 2,5 s
    sola = amplitud_del_tono(izquierda, sr, 1.15)  # música sola (ya entró del todo, aún no baja)
    hablando = amplitud_del_tono(izquierda, sr, subtitulos[0][0] + 1.0)
    assert sola > 0.0 and 20 * np.log10(hablando / sola) < -9.0  # baja ~12 dB mientras se habla
    assert amplitud_del_tono(izquierda, sr, izquierda.size / sr - 0.1) < 0.05 * sola  # termina desvanecida
    assert con_voz.ns["sonoridad_integrada"](estereo.astype(np.float32), sr) == pytest.approx(-16.0, abs=0.5)
    assert "🎵 Música de fondo: Mi unidad/Música/fondo.wav" in capsys.readouterr().out


def test_musica_corta_se_repite_en_bucle(con_voz):
    crear_musica(MI_UNIDAD / "corta.wav", segundos=1.0)
    con_voz.generar(texto="Un texto bastante más largo que la música corta, para que la música tenga que repetirse varias veces.", musica_de_fondo="corta.wav", formato="wav", guardar_en_drive=False)
    with wave.open(str(con_voz.ns["ULTIMA_GENERACION"].tomas[0].archivo), "rb") as archivo:
        assert archivo.getnframes() / archivo.getframerate() > 5.0


def test_musica_inexistente(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(musica_de_fondo="no_existe.mp3", formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "No encontré" in str(excinfo.value)


# ── Modo lote ───────────────────────────────────────────────────────────────────

LOTE = (
    "# Capítulo uno\nTexto del primer capítulo, bastante tranquilo.\n"
    "---\nTexto del segundo bloque, que no tiene título.\n"
    "---\n# Final\nÚltimo bloque del lote, con su propio nombre."
)


def test_lote_un_audio_por_bloque(con_voz, capsys):
    from TTS import api

    con_voz.generar(texto=LOTE, modo_lote=True, tomas=2, semilla=21, subtitulos=True)
    lote = con_voz.ns["ULTIMO_LOTE"]
    assert [g.nombre for g in lote] == ["Capítulo uno", "mi_voz_clonada_02", "Final"]
    assert all(g.lote and len(g.tomas) == 2 for g in lote)
    assert audios_en_drive() == [
        "Capítulo uno.mp3", "Capítulo uno.srt", "Final.mp3", "Final.srt", "mi_voz_clonada_02.mp3", "mi_voz_clonada_02.srt",
    ]
    for generacion in lote:  # en Drive queda la recomendada de cada bloque
        elegida = generacion.tomas[generacion.recomendada - 1]
        assert elegida.en_drive is not None and elegida.en_drive.read_bytes() == elegida.archivo.read_bytes()
        assert all(t.en_drive is None for t in generacion.tomas if t is not elegida)
    leidos = [ll["texto"] for ll in api.registro["inferencias"]]
    assert "Capítulo uno" not in " ".join(leidos) and "Final" not in " ".join(leidos)  # los títulos no se leen
    salida = capsys.readouterr().out
    assert "📚 Modo lote: 3 bloque(s)." in salida and "✅ Lote terminado: 3 audio(s)." in salida
    assert con_voz.ns["ULTIMA_GENERACION"] is lote[-1]

    con_voz.ejecutar("elegir", toma_elegida="1")  # en lote, la celda 7 no hace nada por su cuenta
    assert "ya guardó la mejor toma de cada bloque" in capsys.readouterr().out
    assert len(audios_en_drive()) == 6


def test_lote_un_bloque_con_problemas_no_detiene_a_los_demas(con_voz, capsys):
    texto = "Primer bloque que funciona bien.\n---\nuno💥 dos💥 tres💥\n---\nTercer bloque que también funciona."
    con_voz.generar(texto=texto, modo_lote=True, formato="wav")
    assert [g.nombre for g in con_voz.ns["ULTIMO_LOTE"]] == ["mi_voz_clonada_01", "mi_voz_clonada_03"]
    salida = capsys.readouterr().out
    assert "❌ Bloque 2 («mi_voz_clonada_02»): El modelo no produjo audio" in salida
    assert "con errores: mi_voz_clonada_02." in salida


def test_lote_sin_bloques(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(texto="---\n   \n---", modo_lote=True)
    assert es_error_usuario(excinfo) and "No hay bloques" in str(excinfo.value)


def test_lote_muestra_pocos_reproductores(con_voz, capsys):
    from IPython import display

    texto = "\n---\n".join(f"Bloque número {n} del lote de prueba." for n in range(1, 5))
    con_voz.ejecutar("guion", modo_lote=True)
    con_voz.ns["CUADRO"].value = texto
    antes = len(display.mostrados)
    con_voz.definir("generar", formato="wav", guardar_en_drive=False)
    con_voz.ns["MAXIMO_REPRODUCTORES_LOTE"] = 2  # la celda define 6; aquí se baja para la prueba
    con_voz.solo_ejecucion("generar")
    reproductores = [m for m in display.mostrados[antes:] if hasattr(m, "filename")]
    assert len(reproductores) == 2 and "primeros 2 bloques" in capsys.readouterr().out
