"""Celdas 5 a 7 · texto, generación y elección de tomas, con Drive, GPU y XTTS simulados."""

import re
from pathlib import Path

import pytest

from conftest import CONTENT, MI_UNIDAD, audios_en_drive, duracion_ffprobe, flujo_de_audio_ffprobe

SALIDAS = CONTENT / "clonador_voz" / "salidas"
EJEMPLO = (
    "Hola, esta es una prueba de mi voz clonada. Puedo leer cualquier texto que escriba aquí, "
    "con pausas naturales entre frases."
)


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def archivo_de(sesion, toma: int = 1) -> Path:
    """Archivo local de una toma de la última generación."""
    return sesion.ns["ULTIMA_GENERACION"].tomas[toma - 1].archivo


# ── Celda 5 · el cuadro de texto ────────────────────────────────────────────────


def test_el_cuadro_muestra_el_ejemplo_y_guarda_un_borrador(con_voz, capsys):
    from IPython import display

    con_voz.ejecutar("guion")
    cuadro = con_voz.ns["CUADRO"]
    assert cuadro.value == EJEMPLO and display.mostrados[-1].children[0] is cuadro
    contador = display.mostrados[-1].children[1]
    assert contador.value.startswith(f"{len(EJEMPLO)} caracteres")
    cuadro.value = "Primera línea.\nSegunda línea del texto."
    borrador = CONTENT / "clonador_voz" / "borrador.txt"
    assert borrador.read_text(encoding="utf-8") == cuadro.value and contador.value.startswith("39 caracteres")
    assert "ejecuta la celda 6" in capsys.readouterr().out

    con_voz.ejecutar("guion")  # volver a ejecutar la celda conserva lo escrito
    assert con_voz.ns["CUADRO"].value == "Primera línea.\nSegunda línea del texto."
    del con_voz.ns["CUADRO"]  # como tras reiniciar la sesión: queda el borrador
    con_voz.ejecutar("guion")
    assert con_voz.ns["CUADRO"].value == "Primera línea.\nSegunda línea del texto."


def test_la_celda_6_lee_lo_ultimo_que_se_escribio(con_voz):
    from TTS import api

    con_voz.ejecutar("guion")
    con_voz.ns["CUADRO"].value = "Este texto se escribió después de ejecutar la celda cinco."
    con_voz.ejecutar("generar", formato="wav", guardar_en_drive=False)  # sin volver a ejecutar la celda 5
    assert [ll["texto"] for ll in api.registro["inferencias"]] == ["Este texto se escribió después de ejecutar la celda cinco."]


def test_archivo_vacio_en_modo_archivo(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("guion", fuente_del_texto="Archivo .txt de Drive")
    assert es_error_usuario(excinfo) and "archivo_de_texto" in str(excinfo.value)
    assert con_voz.ns["PEDIDO_TEXTO"] is None


# ── Celda 6 · generación ────────────────────────────────────────────────────────


def test_genera_mp3_en_colab_y_en_drive(con_voz, capsys):
    from IPython import display
    from TTS import api

    con_voz.generar()
    local = SALIDAS / "mi_voz_clonada.mp3"
    en_drive = MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.mp3"
    assert archivo_de(con_voz) == local and local.is_file() and en_drive.is_file()
    assert not (local.parent / "_temporal.wav").exists()
    llamadas = api.registro["inferencias"]
    assert [ll["idioma"] for ll in llamadas] == ["es"] * len(llamadas)
    assert llamadas[0] | {"texto": None, "voz_hz": None} == {
        "texto": None, "idioma": "es", "temperature": 0.75, "top_p": 0.85, "top_k": 50, "repetition_penalty": 5.0,
        "speed": 1.0, "voz_hz": None,
    }
    assert " ".join(ll["texto"] for ll in llamadas) == EJEMPLO
    assert 2.0 < duracion_ffprobe(local) < 12.0
    mp3 = flujo_de_audio_ffprobe(local)
    assert mp3["codec_name"] == "mp3" and mp3["sample_rate"] == "44100" and mp3["bit_rate"] == "192000"
    assert display.mostrados[-1].filename == str(local)
    salida = capsys.readouterr().out
    assert "Guardado en Mi unidad/Clonador de voz/mi_voz_clonada.mp3" in salida and "semilla" in salida
    assert re.search(r"semilla \d+ · parecido 0\.\d{3}", salida)  # parecido con la grabación, medido en cada toma

    con_voz.generar()  # no sobrescribe
    assert (local.parent / "mi_voz_clonada_2.mp3").is_file()
    assert (en_drive.parent / "mi_voz_clonada_2.mp3").is_file()


def test_formato_ogg_opus(con_voz):
    con_voz.generar(formato="ogg", guardar_en_drive=False)
    flujo = flujo_de_audio_ffprobe(archivo_de(con_voz))
    assert archivo_de(con_voz).suffix == ".ogg" and flujo["codec_name"] == "opus" and flujo["sample_rate"] == "48000"


def test_semilla_hace_repetible_el_resultado(con_voz):
    def generar(semilla, nombre):
        con_voz.generar(semilla=semilla, formato="wav", nombre_salida=nombre, guardar_en_drive=False)
        return archivo_de(con_voz).read_bytes()

    assert generar(1234, "a") == generar(1234, "b")
    assert generar(1234, "c") != generar(99, "d")
    assert audios_en_drive() == []


def test_texto_largo_desde_drive_con_pausas(con_voz):
    from google.colab import files
    from TTS import api

    parrafo = "Esta es una oración de prueba bastante larga para el modelo, con comas y detalles. " * 4
    (MI_UNIDAD / "Guiones").mkdir()
    (MI_UNIDAD / "Guiones" / "guion.txt").write_text(f"{parrafo}\n{parrafo}[pausa 2] Cierre del guion de prueba.", encoding="utf-8")
    con_voz.generar(
        archivo_de_texto="guion", formato="wav", idioma="English", velocidad=1.25, pausa_entre_frases=0.4,
        temperatura=0.6, top_p=0.9, top_k=30, penalizacion_repeticion=7.5, semilla=5, descargar_al_terminar=True,
    )
    ns = con_voz.ns
    segmentos = ns["dividir_texto"](ns["PEDIDO_TEXTO"].leer(), 250 - 10, 0.4)
    llamadas = api.registro["inferencias"]
    assert [ll["texto"] for ll in llamadas] == [s.texto for s in segmentos]
    assert all(ll["idioma"] == "en" and ll["speed"] == 1.25 and ll["top_k"] == 30 for ll in llamadas)
    assert llamadas[0]["repetition_penalty"] == 7.5 and llamadas[0]["temperature"] == 0.6
    assert [s.pausa for s in segmentos].count(2.0) == 1 and 0.8 in [s.pausa for s in segmentos]
    audio, sr = ns["leer_wav"](archivo_de(con_voz))
    # duración ≈ fragmentos recortados (voz simulada + márgenes de 60 ms) + pausas
    voz = sum(max(0.3, 0.055 * len(s.texto) / 1.25) + 0.12 for s in segmentos)
    assert abs(audio.size / sr - (voz + sum(s.pausa for s in segmentos))) < 0.05 * len(segmentos)
    assert files.descargas == [str(archivo_de(con_voz))]


def test_texto_vacio(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(texto="   ")
    assert es_error_usuario(excinfo) and "No hay texto" in str(excinfo.value)


def test_deslizadores_en_valores_enteros(con_voz):
    """Colab escribe «7» y no «7.0» si el deslizador cae en un entero; transformers exige float."""
    from TTS import api

    con_voz.generar(
        penalizacion_repeticion=7, temperatura=1, velocidad=1, top_p=1, pausa_entre_frases=1, volumen_musica=-18,
        formato="wav", guardar_en_drive=False,
    )
    llamada = api.registro["inferencias"][0]
    assert llamada["repetition_penalty"] == 7.0 and isinstance(llamada["repetition_penalty"], float)
    assert isinstance(llamada["temperature"], float) and isinstance(llamada["speed"], float)


def test_fragmento_rechazado_se_reintenta_por_mitades(con_voz, capsys):
    from TTS import api

    texto = "Primera frase normal y tranquila. Segunda frase con 💥 que el modelo rechaza siempre. Tercera frase final sin problemas."
    con_voz.generar(texto=texto, formato="wav", guardar_en_drive=False, control_de_calidad=False)
    pedidos = [llamada["texto"] for llamada in api.registro["inferencias"]]  # solo las llamadas que funcionaron
    # la frase 2 falla → «Segunda frase con 💥 que el» | «modelo rechaza siempre.» → la 1.ª mitad se parte otra vez
    assert pedidos == [
        "Primera frase normal y tranquila.",
        "Segunda frase",
        "modelo rechaza siempre.",
        "Tercera frase final sin problemas.",
    ]
    salida = capsys.readouterr().out
    assert "Se omitió «con 💥 que el»" in salida and "✅ 3 fragmento(s)" in salida
    assert archivo_de(con_voz).is_file()


def test_si_ningun_fragmento_suena_avisa(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(texto="uno💥 dos💥 tres💥 cuatro💥", formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "no produjo audio" in str(excinfo.value)


def test_texto_sin_nada_pronunciable(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(texto="💥💥💥 … [pausa 2] ¡!", formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "No hay texto" in str(excinfo.value)


def test_carpeta_de_drive_con_barra_inicial(con_voz):
    con_voz.generar(carpeta_en_drive="/Audios clonados/2026", formato="wav")
    assert (MI_UNIDAD / "Audios clonados" / "2026" / "mi_voz_clonada.wav").is_file()
    assert not Path("/Audios clonados").exists()


def test_audio_largo_muestra_solo_una_vista_previa(con_voz, capsys):
    from IPython import display

    con_voz.ns["LIMITE_REPRODUCTOR_MB"] = 0.05
    con_voz.generar(formato="wav", guardar_en_drive=False)
    vista = Path(display.mostrados[-1].filename)
    assert vista == CONTENT / "clonador_voz" / "vistas_previas" / "mi_voz_clonada.mp3"
    assert vista.stat().st_size < archivo_de(con_voz).stat().st_size
    assert "primer minuto" in capsys.readouterr().out


def test_txt_guardado_en_windows_1252(con_voz):
    from TTS import api

    contenido = "“Comillas curvas” y puntos suspensivos… — guion largo incluido en el texto."
    (MI_UNIDAD / "windows.txt").write_bytes(contenido.encode("cp1252"))
    con_voz.generar(archivo_de_texto="windows.txt", formato="wav", guardar_en_drive=False)
    pedidos = [llamada["texto"] for llamada in api.registro["inferencias"]]
    assert pedidos == ["“Comillas curvas” y puntos suspensivos…", "— guion largo incluido en el texto."]


def test_idioma_que_el_modelo_no_habla(con_voz):
    con_voz.ns["MOTOR"].idiomas = frozenset({"en"})
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "no habla el idioma «es»" in str(excinfo.value)


# ── Drive al guardar ────────────────────────────────────────────────────────────


def test_drive_ya_montado_no_se_vuelve_a_montar(con_voz):
    from google.colab import drive

    con_voz.generar(formato="wav")
    assert len(drive.montajes) == 1 and (MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.wav").is_file()


def test_si_drive_falla_al_guardar_el_audio_no_se_pierde(con_voz, capsys):
    from google.colab import drive, files

    drive.montado, drive.fallos_pendientes = False, 99  # Drive se desconectó durante la sesión
    con_voz.generar(descargar_al_terminar=True)
    salida = capsys.readouterr().out
    local = archivo_de(con_voz)
    assert "No se pudo guardar en Drive (ValueError: mount failed)" in salida and "celda 7" in salida
    assert local.is_file() and files.descargas == [str(local)]

    drive.fallos_pendientes = 0  # Drive vuelve: la celda 7 reintenta el guardado
    con_voz.ejecutar("elegir", toma_elegida="1")
    assert (MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.mp3").read_bytes() == local.read_bytes()


def test_celda_7_descarga_aunque_drive_falle(con_voz):
    from google.colab import drive, files

    con_voz.generar(tomas=2, semilla=3)
    drive.montado, drive.fallos_pendientes = False, 99
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("elegir", toma_elegida="2", descargar_tambien=True)
    assert type(excinfo.value).__name__ == "DriveNoDisponible"
    assert files.descargas == [str(archivo_de(con_voz, 2))]


# ── Tomas y celda 7 ─────────────────────────────────────────────────────────────


def test_varias_tomas_y_guardar_la_elegida(con_voz, capsys):
    from google.colab import files
    from IPython import display

    con_voz.generar(tomas=3, semilla=100, descargar_al_terminar=True)
    generacion = con_voz.ns["ULTIMA_GENERACION"]
    assert [t.semilla for t in generacion.tomas] == [100, 101, 102]
    assert [t.archivo for t in generacion.tomas] == [SALIDAS / f"mi_voz_clonada_toma{n}.mp3" for n in (1, 2, 3)]
    assert all(t.archivo.is_file() and t.en_drive is None for t in generacion.tomas)
    assert audios_en_drive() == [] and files.descargas == []  # nada sale hasta elegir
    assert [Path(a.filename) for a in display.mostrados[-3:]] == [t.archivo for t in generacion.tomas]
    salida = capsys.readouterr().out
    assert "✅ Toma 2/3 · 2 fragmento(s)" in salida and "semilla 101" in salida and "celda 7" in salida
    assert f"la recomendada es la {generacion.recomendada}" in salida and "⭐ recomendada" in salida

    con_voz.ejecutar("elegir")  # «—»: solo orienta (así «Ejecutar todas» no guarda nada solo)
    assert "Escucha las 3 tomas" in capsys.readouterr().out
    assert audios_en_drive() == []

    con_voz.ejecutar("elegir", toma_elegida="2")
    destino = MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.mp3"
    assert destino.read_bytes() == generacion.tomas[1].archivo.read_bytes()
    assert "Toma 2 · semilla 101" in capsys.readouterr().out
    con_voz.ejecutar("elegir", toma_elegida="2")  # no la duplica
    assert "Ya estaba guardada" in capsys.readouterr().out
    assert audios_en_drive() == ["mi_voz_clonada.mp3"]

    con_voz.ejecutar("elegir", toma_elegida="3", descargar_tambien=True)
    assert (destino.parent / "mi_voz_clonada_2.mp3").read_bytes() == generacion.tomas[2].archivo.read_bytes()
    assert files.descargas == [str(generacion.tomas[2].archivo)]

    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("elegir", toma_elegida="4")
    assert es_error_usuario(excinfo) and "tiene 3 toma(s)" in str(excinfo.value)


def test_elegir_la_recomendada(con_voz, capsys):
    con_voz.generar(tomas=3, semilla=40, formato="wav")
    generacion = con_voz.ns["ULTIMA_GENERACION"]
    con_voz.ejecutar("elegir", toma_elegida="Recomendada")
    elegida = generacion.tomas[generacion.recomendada - 1]
    assert elegida.en_drive == MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.wav"
    assert f"Toma {generacion.recomendada} · semilla {elegida.semilla}" in capsys.readouterr().out


def test_cada_toma_se_puede_repetir_con_su_semilla(con_voz):
    con_voz.generar(tomas=3, semilla=500, formato="wav")
    segunda = con_voz.ns["ULTIMA_GENERACION"].tomas[1]
    assert segunda.semilla == 501
    con_voz.generar(tomas=1, semilla=501, formato="wav", guardar_en_drive=False)
    assert archivo_de(con_voz).read_bytes() == segunda.archivo.read_bytes()


def test_tomas_al_azar_tienen_semillas_consecutivas(con_voz):
    con_voz.generar(tomas=4, formato="wav")
    semillas = [t.semilla for t in con_voz.ns["ULTIMA_GENERACION"].tomas]
    base = semillas[0]
    assert semillas == [(base + i) % (2**31 - 1) for i in range(4)]
    assert len({archivo_de(con_voz, n).read_bytes() for n in range(1, 5)}) == 4


def test_celda_7_con_una_sola_toma(con_voz, capsys):
    con_voz.generar()  # guarda directamente en Drive
    con_voz.ejecutar("elegir")
    assert "ya está en Mi unidad/Clonador de voz/mi_voz_clonada.mp3" in capsys.readouterr().out
    con_voz.ejecutar("elegir", toma_elegida="1")
    assert "Ya estaba guardada" in capsys.readouterr().out
    assert audios_en_drive() == ["mi_voz_clonada.mp3"]

    con_voz.generar(guardar_en_drive=False, nombre_salida="sin_drive")
    con_voz.ejecutar("elegir")
    assert "elige «1» si quieres guardarla" in capsys.readouterr().out
    con_voz.ejecutar("elegir", toma_elegida="1")
    assert (MI_UNIDAD / "Clonador de voz" / "sin_drive.mp3").is_file()


# ── Pronunciación y cifras ──────────────────────────────────────────────────────


def test_pronunciacion_y_cifras_en_el_flujo(con_voz, capsys):
    from TTS import api

    (MI_UNIDAD / "Clonador de voz" / "pronunciacion.txt").write_text(
        "# reglas de prueba\nJr. = junior\nDNI = de ene i\n", encoding="utf-8"
    )
    con_voz.generar(
        texto="Trae tu DNI al local del Jr. Ica 123. La matrícula cuesta S/ 1,500.50 y el libro US$ 21, unos 1,200 en total.",
        pronunciacion="Jr. = jirón",  # corrige la regla del archivo
        archivo_de_pronunciacion="Clonador de voz/pronunciacion.txt",
        formato="wav",
        guardar_en_drive=False,
    )
    leido = " ".join(llamada["texto"] for llamada in api.registro["inferencias"])
    assert "Trae tu de ene i al local del jirón Ica 123." in leido
    assert "mil quinientos soles con cincuenta céntimos y el libro veintiún dólares, unos 1200 en total." in leido
    salida = capsys.readouterr().out
    assert "🔤 Pronunciación: 2 regla(s) · 2 reemplazo(s)" in salida
    assert "💲 Montos y cifras: 3 adaptado(s)" in salida


def test_cifras_solo_en_espanol_y_desactivables(con_voz):
    from TTS import api

    con_voz.generar(texto="The course costs S/ 150 and that is a fair price.", idioma="English", formato="wav", guardar_en_drive=False)
    con_voz.generar(texto="El curso cuesta S/ 150 y es un precio justo.", leer_montos_y_cifras=False, formato="wav", guardar_en_drive=False)
    pedidos = [llamada["texto"] for llamada in api.registro["inferencias"]]
    assert pedidos == ["The course costs S/ 150 and that is a fair price.", "El curso cuesta S/ 150 y es un precio justo."]


def test_reglas_mal_escritas_se_avisan(con_voz, capsys):
    con_voz.generar(texto="Esto es una prueba con reglas.", pronunciacion="DNI de ene i; = nada; prueba = test", formato="wav", guardar_en_drive=False)
    salida = capsys.readouterr().out
    assert salida.count("Regla ignorada") == 2 and "🔤 Pronunciación: 1 regla(s) · 1 reemplazo(s)" in salida


def test_cifras_que_alargan_la_frase_se_vuelven_a_partir(con_voz):
    """El límite se aplica a lo que se LEE: una frase corta con cifras puede pasarlo al expandirse."""
    from TTS import api

    montos = ", ".join(f"S/ {n},999.99" for n in range(101, 106))
    texto = f"Los precios del catálogo son {montos} y eso es todo."
    con_voz.generar(texto=texto, formato="wav", guardar_en_drive=False, subtitulos=True)
    pedidos = [llamada["texto"] for llamada in api.registro["inferencias"]]
    assert len(texto) < 229 and len(pedidos) > 1 and all(len(p) <= 229 for p in pedidos)
    srt = con_voz.ns["ULTIMA_GENERACION"].tomas[0].subtitulos.read_text(encoding="utf-8")
    assert "101,999.99" in srt and "noventa y nueve" not in srt  # el subtítulo muestra lo que se escribió
