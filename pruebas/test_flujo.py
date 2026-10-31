"""Flujo completo celda por celda, con Drive, GPU y XTTS simulados."""

import os
import shutil
import sys
import time
import types
import unicodedata
from pathlib import Path

import pytest

from conftest import CONTENT, MI_UNIDAD, crear_grabacion, duracion_ffprobe, flujo_de_audio_ffprobe


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def archivo_de(sesion, toma: int = 1) -> Path:
    """Archivo local de una toma de la última generación."""
    return sesion.ns["ULTIMA_GENERACION"].tomas[toma - 1].archivo


# ── Celda 1 · entorno ───────────────────────────────────────────────────────────


def test_sin_gpu_muestra_un_mensaje_claro(colab, capsys):
    colab.sin_gpu()
    assert shutil.which("nvidia-smi") is None
    sesion = colab.sesion()
    with pytest.raises(Exception) as excinfo:
        sesion.ejecutar("entorno")
    assert es_error_usuario(excinfo) and "T4 GPU" in str(excinfo.value)
    assert sesion.shell.tipos == (sesion.error(),)
    # IPython llama así al manejador (run_code → CustomTB) y descarta lo que devuelve
    assert sesion.shell.manejador(sesion.shell, type(excinfo.value), excinfo.value, None, tb_offset=None) is None
    assert capsys.readouterr().err == f"\x1b[1;31m❌ {excinfo.value}\x1b[0m\n"


def test_instala_solo_lo_que_falta(colab):
    colab.instalar("torch", "2.10.0+cu126")
    colab.instalar("torchaudio", "2.10.0+cu126")
    sesion = colab.sesion()
    sesion.definir("entorno")
    llamadas = []
    sesion.ns["pip_instalar"] = lambda paquetes, indices: llamadas.append((paquetes, indices))
    sesion.solo_ejecucion("entorno")
    assert llamadas == [(["coqui-tts==0.28.0", "torchcodec==0.10.*"], [])]


def test_torchaudio_faltante_se_instala_con_la_variante_cuda_de_torch(colab):
    colab.instalar("torch", "2.10.0+cu126")
    sesion = colab.sesion()
    sesion.definir("entorno")
    llamadas = []
    sesion.ns["pip_instalar"] = lambda paquetes, indices: llamadas.append((paquetes, indices))
    sesion.solo_ejecucion("entorno")
    assert llamadas == [
        (["coqui-tts==0.28.0", "torchaudio==2.10.0+cu126", "torchcodec==0.10.*"], ["https://download.pytorch.org/whl/cu126"])
    ]


def test_no_reinstala_si_todo_esta(colab, capsys):
    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.definir("entorno")
    sesion.ns["pip_instalar"] = lambda *_: pytest.fail("no debía instalar nada")
    sesion.solo_ejecucion("entorno")
    salida = capsys.readouterr().out
    assert "✅ Entorno listo" in salida and "Tesla T4" in salida and "coqui-tts 0.28.0" in salida


def test_tabla_de_torchcodec(colab):
    sesion = colab.sesion()
    sesion.definir("entorno")
    especificacion = sesion.ns["especificacion_torchcodec"]
    assert especificacion((2, 8)) is None
    assert especificacion((2, 9)) == "torchcodec==0.9.*"
    assert especificacion((2, 10)) == "torchcodec==0.10.*"
    assert especificacion((2, 11)) == especificacion((2, 14)) == "torchcodec>=0.12"
    assert sesion.ns["version_mayor_menor"]("2.11.0a0+git1234") == (2, 11)


def test_reinicia_si_cambia_una_libreria_ya_cargada(colab, monkeypatch):
    colab.instalar_todo()
    colab.instalar("transformers", "5.4.0")
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="4.57.6"))
    sesion = colab.sesion()
    sesion.definir("entorno")
    motivos = []
    sesion.ns["reiniciar_sesion"] = motivos.append
    sesion.solo_ejecucion("entorno")
    assert motivos == [["transformers 4.57.6 → 5.4.0"]]


# ── Celda 3 · modelo ────────────────────────────────────────────────────────────


def test_licencia_no_comercial_asumida_por_defecto_y_avisada(colab, capsys):
    from TTS import api

    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    sesion.ejecutar("modelo")  # sin tocar la casilla
    assert api.registro["cargas"] == 1 and os.environ["COQUI_TOS_AGREED"] == "1"
    assert "CPML aceptada para uso no comercial" in capsys.readouterr().out
    sesion.ejecutar("modelo")  # el aviso se repite en cada ejecución, aunque el modelo ya esté cargado
    salida = capsys.readouterr().out
    assert "CPML aceptada para uso no comercial" in salida and "ya estaba cargado" in salida
    assert api.registro["cargas"] == 1


def test_desmarcar_el_uso_no_comercial_detiene_la_carga(colab):
    from TTS import api

    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    with pytest.raises(Exception) as excinfo:
        sesion.ejecutar("modelo", uso_no_comercial=False)
    assert es_error_usuario(excinfo) and "no permite el uso comercial" in str(excinfo.value)
    assert api.registro["cargas"] == 0 and "COQUI_TOS_AGREED" not in os.environ
    assert sesion.ns["MOTOR"] is None


def test_desmarcar_despues_de_cargar_bloquea_las_celdas_siguientes(lista):
    crear_grabacion(MI_UNIDAD / "mi_voz.wav", voz_s=12.0)
    lista.ejecutar("referencia")
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("modelo", uso_no_comercial=False)
    assert es_error_usuario(excinfo)
    assert lista.ns["MOTOR"] is None and "COQUI_TOS_AGREED" not in os.environ
    with pytest.raises(RuntimeError, match="celdas 1 a 3"):
        lista.ejecutar("referencia")
    with pytest.raises(RuntimeError, match="celdas 1 a 4"):
        lista.ejecutar("generar")  # la voz ya existía, pero sin motor no se genera nada
    lista.ejecutar("modelo")  # volver a marcarla lo deja todo como antes
    lista.ejecutar("generar", formato="wav", guardar_en_drive=False)
    assert archivo_de(lista).is_file()


def test_sin_cuda_en_torch(colab):
    import torch

    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    torch.cuda.disponible = False
    with pytest.raises(Exception) as excinfo:
        sesion.ejecutar("modelo")
    assert es_error_usuario(excinfo) and "GPU" in str(excinfo.value)


def test_carga_una_sola_vez(lista):
    from TTS import api

    motor = lista.ns["MOTOR"]
    assert api.registro["cargas"] == 1 and os.environ["COQUI_TOS_AGREED"] == "1"
    assert motor.frecuencia == 24000 and motor.modelo.device == "cuda" and "es" in motor.idiomas
    assert motor.limite_caracteres("es") == 239 and motor.limite_caracteres("zh-cn") == 82
    lista.ejecutar("modelo")
    assert api.registro["cargas"] == 1 and lista.ns["MOTOR"] is not motor  # adaptador nuevo, mismo modelo


# ── Celda 4 · voz de referencia ─────────────────────────────────────────────────


def test_busca_por_nombre_sin_extension_ni_mayusculas(lista, capsys):
    from IPython import display
    from TTS import api

    crear_grabacion(MI_UNIDAD / "Grabaciones" / "2026" / "Mi Voz.m4a", voz_s=12.0)
    lista.ejecutar("referencia", audio_en_drive="mi voz")
    voz = lista.ns["VOZ"]
    assert 11.4 < voz.segundos < 12.2
    assert api.registro["speaker"] == [round(voz.segundos * 22050)]
    assert api.registro["gpt"] == [{"muestras": round(voz.segundos * 22050), "length": 30, "chunk_length": 4}]
    assert display.mostrados and display.mostrados[-1].filename.endswith("referencia_01.wav")
    assert "Mi unidad/Grabaciones/2026/Mi Voz.m4a" in capsys.readouterr().out


def test_ruta_exacta_y_tope_de_segundos(lista):
    from TTS import api

    crear_grabacion(MI_UNIDAD / "Audios" / "larga.wav", voz_s=20.0)
    lista.ejecutar("referencia", audio_en_drive="Mi unidad/Audios/larga.wav", segundos_por_archivo=10)
    voz = lista.ns["VOZ"]
    assert 9.85 < voz.segundos <= 10.0
    assert api.registro["speaker"] == [round(voz.segundos * 22050)]


def test_homonimos_usa_el_mas_reciente(lista):
    viejo = crear_grabacion(MI_UNIDAD / "A" / "voz.mp3", voz_s=6.0)
    nuevo = crear_grabacion(MI_UNIDAD / "B" / "voz.mp3", voz_s=9.0)
    hace_una_hora = time.time() - 3600
    os.utime(viejo, (hace_una_hora, hace_una_hora))
    lista.ejecutar("referencia", audio_en_drive="voz.mp3")
    assert 8.4 < lista.ns["VOZ"].segundos < 9.2


def test_carpeta_con_varios_formatos(lista):
    from TTS import api

    for nombre in ("uno.wav", "dos.opus", "tres.mp4"):
        crear_grabacion(MI_UNIDAD / "Muestras" / nombre, voz_s=5.0)
    (MI_UNIDAD / "Muestras" / "notas.txt").write_text("no es audio")
    lista.ejecutar("referencia", audio_en_drive="Muestras")
    voz = lista.ns["VOZ"]
    assert len(voz.archivos) == 3 and len(api.registro["speaker"]) == 3
    assert 14.0 < voz.segundos < 15.5


@pytest.mark.parametrize(
    "entrada",
    ["Grabación nueva.m4a", "grabación nueva", "Audios/grabación nueva.m4a", "audios/GRABACIÓN NUEVA", "/Audios/Grabación nueva.m4a"],
)
def test_nombres_subidos_desde_mac_en_forma_nfd(lista, entrada):
    nombre_nfd = unicodedata.normalize("NFD", "Grabación nueva.m4a")
    assert nombre_nfd != "Grabación nueva.m4a"
    crear_grabacion(MI_UNIDAD / "Audios" / nombre_nfd, voz_s=6.0)
    lista.ejecutar("referencia", audio_en_drive=entrada)
    assert lista.ns["VOZ"].segundos > 5.5


@pytest.mark.parametrize(
    ("entrada", "mensaje"),
    [("Audios/no_existe.wav", "No existe"), ("fantasma", "No encontré"), ("", "Falta el nombre")],
)
def test_errores_de_busqueda(lista, entrada, mensaje):
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("referencia", audio_en_drive=entrada)
    assert es_error_usuario(excinfo) and mensaje in str(excinfo.value)


def test_grabacion_demasiado_corta(lista):
    crear_grabacion(MI_UNIDAD / "corta.wav", voz_s=2.0)
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("referencia", audio_en_drive="corta.wav")
    assert es_error_usuario(excinfo) and "Solo hay" in str(excinfo.value)


# ── Celda 5 · generación ────────────────────────────────────────────────────────

def test_genera_mp3_en_colab_y_en_drive(con_voz, capsys):
    from IPython import display
    from TTS import api

    con_voz.ejecutar("generar")
    local = CONTENT / "clonador_voz" / "salidas" / "mi_voz_clonada.mp3"
    en_drive = MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.mp3"
    assert archivo_de(con_voz) == local and local.is_file() and en_drive.is_file()
    assert not (local.parent / "_temporal.wav").exists()
    llamadas = api.registro["inferencias"]
    assert [ll["idioma"] for ll in llamadas] == ["es"] * len(llamadas)
    assert llamadas[0] | {"texto": None} == {
        "texto": None, "idioma": "es", "temperature": 0.75, "top_p": 0.85, "top_k": 50, "repetition_penalty": 5.0, "speed": 1.0,
    }
    assert 2.0 < duracion_ffprobe(local) < 12.0
    mp3 = flujo_de_audio_ffprobe(local)
    assert mp3["codec_name"] == "mp3" and mp3["sample_rate"] == "44100" and mp3["bit_rate"] == "192000"
    assert display.mostrados[-1].filename == str(local)
    salida = capsys.readouterr().out
    assert "Guardado en Mi unidad/Clonador de voz/mi_voz_clonada.mp3" in salida and "semilla" in salida

    con_voz.ejecutar("generar")  # no sobrescribe
    assert (local.parent / "mi_voz_clonada_2.mp3").is_file()
    assert (en_drive.parent / "mi_voz_clonada_2.mp3").is_file()


def test_semilla_hace_repetible_el_resultado(con_voz):
    def generar(semilla, nombre):
        con_voz.ejecutar("generar", semilla=semilla, formato="wav", nombre_salida=nombre, guardar_en_drive=False)
        return archivo_de(con_voz).read_bytes()

    assert generar(1234, "a") == generar(1234, "b")
    assert generar(1234, "c") != generar(99, "d")
    assert not (MI_UNIDAD / "Clonador de voz").exists()


def test_texto_largo_desde_drive_con_pausas(con_voz):
    from TTS import api

    parrafo = "Esta es una oración de prueba bastante larga para el modelo, con comas y detalles. " * 4
    (MI_UNIDAD / "Guiones").mkdir()
    (MI_UNIDAD / "Guiones" / "guion.txt").write_text(f"{parrafo}\n{parrafo}[pausa 2] Cierre del guion de prueba.", encoding="utf-8")
    con_voz.ejecutar(
        "generar", archivo_de_texto="guion", formato="wav", idioma="English", velocidad=1.25, pausa_entre_frases=0.4,
        temperatura=0.6, top_p=0.9, top_k=30, penalizacion_repeticion=7.5, semilla=5, descargar_al_terminar=True,
    )
    ns = con_voz.ns
    segmentos = ns["dividir_texto"](ns["leer_texto"]("", "guion"), 250 - 10, 0.4)
    llamadas = api.registro["inferencias"]
    assert [ll["texto"] for ll in llamadas] == [s.texto for s in segmentos]
    assert all(ll["idioma"] == "en" and ll["speed"] == 1.25 and ll["top_k"] == 30 for ll in llamadas)
    assert llamadas[0]["repetition_penalty"] == 7.5 and llamadas[0]["temperature"] == 0.6
    assert [s.pausa for s in segmentos].count(2.0) == 1 and 0.8 in [s.pausa for s in segmentos]
    audio, sr = ns["leer_wav"](archivo_de(con_voz))
    # duración ≈ fragmentos recortados (voz simulada + márgenes de 60 ms) + pausas
    voz = sum(max(0.3, 0.055 * len(s.texto) / 1.25) + 0.12 for s in segmentos)
    assert abs(audio.size / sr - (voz + sum(s.pausa for s in segmentos))) < 0.05 * len(segmentos)

    from google.colab import files

    assert files.descargas == [str(archivo_de(con_voz))]


def test_texto_vacio(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("generar", texto="   ")
    assert es_error_usuario(excinfo) and "No hay texto" in str(excinfo.value)


def test_deslizadores_en_valores_enteros(con_voz):
    """Colab escribe «7» y no «7.0» si el deslizador cae en un entero; transformers exige float."""
    from TTS import api

    con_voz.ejecutar(
        "generar", penalizacion_repeticion=7, temperatura=1, velocidad=1, top_p=1, pausa_entre_frases=1,
        formato="wav", guardar_en_drive=False,
    )
    llamada = api.registro["inferencias"][0]
    assert llamada["repetition_penalty"] == 7.0 and isinstance(llamada["repetition_penalty"], float)
    assert isinstance(llamada["temperature"], float) and isinstance(llamada["speed"], float)


def test_fragmento_rechazado_se_reintenta_por_mitades(con_voz, capsys):
    from TTS import api

    texto = "Primera frase normal y tranquila. Segunda frase con 💥 que el modelo rechaza siempre. Tercera frase final sin problemas."
    con_voz.ejecutar("generar", texto=texto, formato="wav", guardar_en_drive=False)
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
        con_voz.ejecutar("generar", texto="uno💥 dos💥 tres💥 cuatro💥", formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "no produjo audio" in str(excinfo.value)


def test_texto_sin_nada_pronunciable(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("generar", texto="💥💥💥 … [pausa 2] ¡!", formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "No hay texto" in str(excinfo.value)


def test_carpeta_de_drive_con_barra_inicial(con_voz):
    con_voz.ejecutar("generar", carpeta_en_drive="/Audios clonados/2026", formato="wav")
    assert (MI_UNIDAD / "Audios clonados" / "2026" / "mi_voz_clonada.wav").is_file()
    assert not Path("/Audios clonados").exists()


def test_audio_largo_muestra_solo_una_vista_previa(con_voz, capsys):
    from IPython import display

    con_voz.ns["LIMITE_REPRODUCTOR_MB"] = 0.05
    con_voz.ejecutar("generar", formato="wav", guardar_en_drive=False)
    vista = Path(display.mostrados[-1].filename)
    assert vista == CONTENT / "clonador_voz" / "vistas_previas" / "mi_voz_clonada.mp3"
    assert vista.stat().st_size < archivo_de(con_voz).stat().st_size
    assert "primer minuto" in capsys.readouterr().out


def test_txt_guardado_en_windows_1252(con_voz):
    from TTS import api

    contenido = "“Comillas curvas” y puntos suspensivos… — guion largo incluido en el texto."
    (MI_UNIDAD / "windows.txt").write_bytes(contenido.encode("cp1252"))
    con_voz.ejecutar("generar", archivo_de_texto="windows.txt", formato="wav", guardar_en_drive=False)
    pedidos = [llamada["texto"] for llamada in api.registro["inferencias"]]
    assert pedidos == ["“Comillas curvas” y puntos suspensivos…", "— guion largo incluido en el texto."]


def test_orden_de_ejecucion_explicado(colab):
    sesion = colab.sesion()
    pasos = [
        ("archivos", "celda 1"), ("audio", "celdas 1 y 2a"), ("modelo", "celdas 1, 2a, 2b y 2c"),
        ("referencia", "celdas 1 a 3"), ("generar", "1 a 4"), ("elegir", "celda 5"),
    ]
    for celda, mensaje in pasos:
        with pytest.raises(RuntimeError, match=mensaje):
            sesion.ejecutar(celda)


# ── Conexión con Drive ──────────────────────────────────────────────────────────


def test_drive_falla_una_vez_y_se_reintenta(lista, capsys):
    from google.colab import drive

    crear_grabacion(MI_UNIDAD / "mi_voz.wav", voz_s=12.0)
    drive.fallos_pendientes = 1  # «ValueError: mount failed», como lo vio el usuario
    lista.ejecutar("referencia")
    assert drive.montajes == [("/content/drive", False), ("/content/drive", True)]
    assert "Reintentando" in capsys.readouterr().out and lista.ns["VOZ"].segundos > 11


def test_drive_no_disponible_explica_que_hacer(lista):
    from google.colab import drive

    drive.fallos_pendientes = 99
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("referencia")
    mensaje = str(excinfo.value)
    assert es_error_usuario(excinfo) is False and type(excinfo.value).__name__ == "DriveNoDisponible"
    assert isinstance(excinfo.value, lista.error())  # se muestra como error amigable, sin traza
    assert "MISMA cuenta" in mensaje and "TODOS los permisos" in mensaje and "mount failed" in mensaje
    assert "subir el audio desde tu computadora" in mensaje and len(drive.montajes) == 2


def test_drive_ya_montado_no_se_vuelve_a_montar(con_voz):
    from google.colab import drive

    con_voz.ejecutar("generar", formato="wav")
    assert len(drive.montajes) == 1 and (MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.wav").is_file()


def test_una_carpeta_mydrive_creada_a_mano_no_cuenta_como_montaje(lista):
    from google.colab import drive

    (MI_UNIDAD / "basura").mkdir(parents=True)  # existe /content/drive/MyDrive, pero Drive no está montado
    assert not lista.ns["drive_montado"]()
    lista.ns["montar_drive"]()
    assert drive.montajes == [("/content/drive", False)]


def test_subir_el_audio_desde_la_computadora(lista, tmp_path):
    from google.colab import drive, files

    files.para_subir.extend([crear_grabacion(tmp_path / "mi voz (celular).m4a", voz_s=10.0), tmp_path / "nota.txt"])
    (tmp_path / "nota.txt").write_text("no es audio")
    lista.ejecutar("referencia", origen_del_audio="Subir desde mi computadora")
    assert 9.4 < lista.ns["VOZ"].segundos < 10.3
    assert drive.montajes == []  # Drive ni se toca
    assert (CONTENT / "clonador_voz" / "subidas" / "mi voz (celular).m4a").is_file()


def test_subir_sin_elegir_audio(lista):
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("referencia", origen_del_audio="Subir desde mi computadora")
    assert es_error_usuario(excinfo) and "No subiste ningún audio" in str(excinfo.value)


def test_si_drive_falla_al_guardar_el_audio_no_se_pierde(con_voz, capsys):
    from google.colab import drive, files

    drive.montado, drive.fallos_pendientes = False, 99  # Drive se desconectó durante la sesión
    con_voz.ejecutar("generar", descargar_al_terminar=True)
    salida = capsys.readouterr().out
    local = archivo_de(con_voz)
    assert "No se pudo guardar en Drive (ValueError: mount failed)" in salida and "celda 6" in salida
    assert local.is_file() and files.descargas == [str(local)]

    drive.fallos_pendientes = 0  # Drive vuelve: la celda 6 reintenta el guardado
    con_voz.ejecutar("elegir", toma_elegida="1")
    assert (MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.mp3").read_bytes() == local.read_bytes()


def test_celda_6_descarga_aunque_drive_falle(con_voz):
    from google.colab import drive, files

    con_voz.ejecutar("generar", tomas=2, semilla=3)
    drive.montado, drive.fallos_pendientes = False, 99
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("elegir", toma_elegida="2", descargar_tambien=True)
    assert type(excinfo.value).__name__ == "DriveNoDisponible"
    assert files.descargas == [str(archivo_de(con_voz, 2))]


# ── Tomas y celda 6 ─────────────────────────────────────────────────────────────


def test_varias_tomas_y_guardar_la_elegida(con_voz, capsys):
    from google.colab import files
    from IPython import display

    con_voz.ejecutar("generar", tomas=3, semilla=100, descargar_al_terminar=True)
    generacion = con_voz.ns["ULTIMA_GENERACION"]
    salidas = CONTENT / "clonador_voz" / "salidas"
    assert [t.semilla for t in generacion.tomas] == [100, 101, 102]
    assert [t.archivo for t in generacion.tomas] == [salidas / f"mi_voz_clonada_toma{n}.mp3" for n in (1, 2, 3)]
    assert all(t.archivo.is_file() and t.en_drive is None for t in generacion.tomas)
    assert not (MI_UNIDAD / "Clonador de voz").exists() and files.descargas == []  # nada sale hasta elegir
    assert [Path(a.filename) for a in display.mostrados[-3:]] == [t.archivo for t in generacion.tomas]
    salida = capsys.readouterr().out
    assert "✅ Toma 2/3 · 2 fragmento(s)" in salida and "semilla 101" in salida and "celda 6" in salida

    con_voz.ejecutar("elegir")  # «—»: solo orienta (así «Ejecutar todas» no guarda nada solo)
    assert "Escucha las 3 tomas" in capsys.readouterr().out
    assert not (MI_UNIDAD / "Clonador de voz").exists()

    con_voz.ejecutar("elegir", toma_elegida="2")
    destino = MI_UNIDAD / "Clonador de voz" / "mi_voz_clonada.mp3"
    assert destino.read_bytes() == generacion.tomas[1].archivo.read_bytes()
    assert "Toma 2 · semilla 101" in capsys.readouterr().out
    con_voz.ejecutar("elegir", toma_elegida="2")  # no la duplica
    assert "Ya estaba guardada" in capsys.readouterr().out
    assert sorted(p.name for p in destino.parent.iterdir()) == ["mi_voz_clonada.mp3"]

    con_voz.ejecutar("elegir", toma_elegida="3", descargar_tambien=True)
    assert (destino.parent / "mi_voz_clonada_2.mp3").read_bytes() == generacion.tomas[2].archivo.read_bytes()
    assert files.descargas == [str(generacion.tomas[2].archivo)]

    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("elegir", toma_elegida="4")
    assert es_error_usuario(excinfo) and "tiene 3 toma(s)" in str(excinfo.value)


def test_cada_toma_se_puede_repetir_con_su_semilla(con_voz):
    con_voz.ejecutar("generar", tomas=3, semilla=500, formato="wav")
    segunda = con_voz.ns["ULTIMA_GENERACION"].tomas[1]
    assert segunda.semilla == 501
    con_voz.ejecutar("generar", tomas=1, semilla=501, formato="wav", guardar_en_drive=False)
    assert archivo_de(con_voz).read_bytes() == segunda.archivo.read_bytes()


def test_tomas_al_azar_tienen_semillas_consecutivas(con_voz):
    con_voz.ejecutar("generar", tomas=4, formato="wav")
    semillas = [t.semilla for t in con_voz.ns["ULTIMA_GENERACION"].tomas]
    base = semillas[0]
    assert semillas == [(base + i) % (2**31 - 1) for i in range(4)]
    assert len({archivo_de(con_voz, n).read_bytes() for n in range(1, 5)}) == 4


def test_celda_6_con_una_sola_toma(con_voz, capsys):
    con_voz.ejecutar("generar")  # guarda directamente en Drive
    con_voz.ejecutar("elegir")
    assert "ya está en Mi unidad/Clonador de voz/mi_voz_clonada.mp3" in capsys.readouterr().out
    con_voz.ejecutar("elegir", toma_elegida="1")
    assert "Ya estaba guardada" in capsys.readouterr().out
    assert sorted(p.name for p in (MI_UNIDAD / "Clonador de voz").iterdir()) == ["mi_voz_clonada.mp3"]

    con_voz.ejecutar("generar", guardar_en_drive=False, nombre_salida="sin_drive")
    con_voz.ejecutar("elegir")
    assert "elige «1» si quieres guardarla" in capsys.readouterr().out
    con_voz.ejecutar("elegir", toma_elegida="1")
    assert (MI_UNIDAD / "Clonador de voz" / "sin_drive.mp3").is_file()


# ── Pronunciación y cifras ──────────────────────────────────────────────────────


def test_pronunciacion_y_cifras_en_el_flujo(con_voz, capsys):
    from TTS import api

    (MI_UNIDAD / "Clonador de voz").mkdir()
    (MI_UNIDAD / "Clonador de voz" / "pronunciacion.txt").write_text(
        "# reglas de prueba\nJr. = junior\nDNI = de ene i\n", encoding="utf-8"
    )
    con_voz.ejecutar(
        "generar",
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

    con_voz.ejecutar("generar", texto="The course costs S/ 150 and that is a fair price.", idioma="English", formato="wav", guardar_en_drive=False)
    con_voz.ejecutar("generar", texto="El curso cuesta S/ 150 y es un precio justo.", leer_montos_y_cifras=False, formato="wav", guardar_en_drive=False)
    pedidos = [llamada["texto"] for llamada in api.registro["inferencias"]]
    assert pedidos == ["The course costs S/ 150 and that is a fair price.", "El curso cuesta S/ 150 y es un precio justo."]


def test_reglas_mal_escritas_se_avisan(con_voz, capsys):
    con_voz.ejecutar("generar", pronunciacion="DNI de ene i; = nada; prueba = test", formato="wav", guardar_en_drive=False)
    salida = capsys.readouterr().out
    assert salida.count("Regla ignorada") == 2 and "🔤 Pronunciación: 1 regla(s) · 1 reemplazo(s)" in salida
