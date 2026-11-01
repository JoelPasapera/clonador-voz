"""Celdas 3 y 4 · modelo (licencia, carga única, modelo afinado) y voz (grabación, Drive, perfiles guardados)."""

import os
import time
import unicodedata

import numpy as np
import pytest

from conftest import CONTENT, MI_UNIDAD, crear_grabacion

VOCES = MI_UNIDAD / "Clonador de voz" / "voces"


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


# ── Celda 3 · modelo ────────────────────────────────────────────────────────────


def test_licencia_no_comercial_asumida_por_defecto_y_avisada(colab, capsys):
    from TTS import api

    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    sesion.ejecutar("modelo")  # sin tocar la casilla
    assert api.registro["cargas"] == 1 and os.environ["COQUI_TOS_AGREED"] == "1"
    salida = capsys.readouterr().out
    assert "CPML aceptada para uso no comercial" in salida and "4 voces incluidas" in salida
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
    assert "Chatterbox" in str(excinfo.value)
    assert api.registro["cargas"] == 0 and "COQUI_TOS_AGREED" not in os.environ
    assert sesion.ns["MOTOR"] is None


def test_desmarcar_despues_de_cargar_bloquea_las_celdas_siguientes(lista):
    crear_grabacion(MI_UNIDAD / "mi_voz.wav", voz_s=12.0)
    lista.ejecutar("referencia")
    lista.ejecutar("guion")
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("modelo", uso_no_comercial=False)
    assert es_error_usuario(excinfo)
    assert lista.ns["MOTOR"] is None and "COQUI_TOS_AGREED" not in os.environ
    with pytest.raises(RuntimeError, match="celdas 1 a 3"):
        lista.ejecutar("referencia")
    with pytest.raises(RuntimeError, match="celdas 1 a 4"):
        lista.ejecutar("generar")  # la voz ya existía, pero sin motor no se genera nada
    lista.ejecutar("modelo")  # volver a marcarla lo deja todo como antes (el modelo sigue en memoria)
    lista.generar(formato="wav", guardar_en_drive=False)
    assert lista.ns["ULTIMA_GENERACION"].tomas[0].archivo.is_file()


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
    assert {"zh-cn", "ja", "ko"}.isdisjoint(motor.idiomas)  # requieren paquetes extra
    assert motor.limite_caracteres("es") == 239 and motor.limite_caracteres("zh-cn") == 82
    assert motor.voces_incluidas() == ["Ana Florence", "Andrew Chipper", "Claribel Dervla", "Damien Black"]
    lista.ejecutar("modelo")
    assert api.registro["cargas"] == 1 and lista.ns["MOTOR"] is not motor  # adaptador nuevo, mismo modelo


def test_modelo_afinado_inexistente_lo_explica(lista):
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("modelo", modelo_afinado="no_existe")
    assert es_error_usuario(excinfo) and "Entrenar XTTS con tu voz" in str(excinfo.value)


def test_orden_de_ejecucion_explicado(colab):
    sesion = colab.sesion()
    pasos = [
        ("archivos", "celda 1"), ("audio", "celdas 1 y 2a"), ("modelo", "celdas 1, 2a, 2b y 2c"),
        ("referencia", "celdas 1 a 3"), ("guion", "celdas 1, 2a, 2b y 2c"), ("generar", "1 a 4"), ("elegir", "celda 6"),
        ("datos", "celdas 1, 2a, 2b y 2c"), ("entrenamiento", "celda 3"), ("guardar_modelo", "celda 4"),
    ]
    for celda, mensaje in pasos:
        with pytest.raises(RuntimeError, match=mensaje):
            sesion.ejecutar(celda)


def test_generar_sin_la_celda_5(con_voz):
    with pytest.raises(RuntimeError, match="celda 5"):
        con_voz.ejecutar("generar")


# ── Celda 4 · voz desde una grabación ───────────────────────────────────────────


def test_busca_por_nombre_sin_extension_ni_mayusculas(lista, capsys):
    from IPython import display
    from TTS import api

    crear_grabacion(MI_UNIDAD / "Grabaciones" / "2026" / "Mi Voz.m4a", voz_s=12.0)
    lista.ejecutar("referencia", audio_en_drive="mi voz")
    voz = lista.ns["VOZ"]
    assert 11.4 < voz.segundos < 12.2 and voz.nombre == "mi_voz" and voz.motor == "xtts"
    assert api.registro["speaker"] == [round(voz.segundos * 22050)]
    assert api.registro["gpt"] == [{"muestras": round(voz.segundos * 22050), "length": 30, "chunk_length": 4}]
    assert display.mostrados and display.mostrados[-1].filename.endswith("referencia_01.wav")
    assert abs(float(np.argmax(voz.huella)) - 140) <= 2  # la «voz» de la grabación (140 Hz)
    salida = capsys.readouterr().out
    assert "Mi unidad/Grabaciones/2026/Mi Voz.m4a" in salida and "Voces incluidas para diálogos" in salida


def test_ruta_exacta_y_tope_de_segundos(lista):
    from TTS import api

    crear_grabacion(MI_UNIDAD / "Audios" / "larga.wav", voz_s=20.0)
    lista.ejecutar("referencia", audio_en_drive="Mi unidad/Audios/larga.wav", segundos_por_archivo=10)
    voz = lista.ns["VOZ"]
    assert 9.85 < voz.segundos <= 10.0
    assert api.registro["speaker"] == [round(voz.segundos * 22050)]


def test_homonimos_usa_el_mas_reciente(lista):
    viejo = crear_grabacion(MI_UNIDAD / "A" / "voz.mp3", voz_s=6.0)
    crear_grabacion(MI_UNIDAD / "B" / "voz.mp3", voz_s=9.0)
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
    assert lista.ns["VOZ"] is None and lista.ns["VOCES"] is None  # la celda 6 no usará una voz vieja


# ── Celda 4 · perfiles guardados ────────────────────────────────────────────────


def test_la_voz_se_guarda_como_perfil_y_se_carga_sin_procesar_audio(con_voz, capsys):
    from TTS import api

    assert (VOCES / "mi_voz.xtts.npz").is_file()
    capsys.readouterr()
    original = con_voz.ns["VOZ"]
    api.registro["speaker"].clear()
    api.registro["gpt"].clear()
    (MI_UNIDAD / "mi_voz.wav").unlink()  # ya no hace falta la grabación
    con_voz.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")
    cargada = con_voz.ns["VOZ"]
    assert api.registro["speaker"] == [] and api.registro["gpt"] == []  # ni FFmpeg ni análisis
    assert cargada.nombre == "mi_voz" and cargada.motor == "xtts" and cargada.segundos == pytest.approx(original.segundos)
    assert np.array_equal(cargada.huella, original.huella)
    for cargado, creado in zip(cargada.datos, original.datos, strict=True):
        assert np.array_equal(cargado.datos, creado.datos) and cargado.dispositivo == "cuda"
    assert "Voz «mi_voz» cargada de Mi unidad/Clonador de voz/voces/mi_voz.xtts.npz" in capsys.readouterr().out


def test_perfil_guardado_genera_igual_que_la_voz_original(con_voz):
    con_voz.generar(semilla=7, formato="wav", guardar_en_drive=False, nombre_salida="original")
    antes = con_voz.ns["ULTIMA_GENERACION"].tomas[0].archivo.read_bytes()
    con_voz.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")
    con_voz.generar(semilla=7, formato="wav", guardar_en_drive=False, nombre_salida="desde_perfil")
    assert con_voz.ns["ULTIMA_GENERACION"].tomas[0].archivo.read_bytes() == antes


def test_volver_a_crear_un_perfil_lo_reemplaza(con_voz, capsys):
    crear_grabacion(MI_UNIDAD / "otra.wav", voz_s=8.0)
    con_voz.ejecutar("referencia", audio_en_drive="otra.wav")
    assert "(reemplazó al anterior)" in capsys.readouterr().out
    assert sorted(p.name for p in VOCES.iterdir()) == ["mi_voz.xtts.npz"]


def test_sin_guardar_perfil(lista):
    crear_grabacion(MI_UNIDAD / "mi_voz.wav", voz_s=8.0)
    lista.ejecutar("referencia", guardar_perfil=False)
    assert not VOCES.exists()


def test_voz_incluida_como_voz_principal(lista, capsys):
    lista.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida", nombre_del_perfil="ana florence")
    voz = lista.ns["VOZ"]
    assert voz.nombre == "Ana Florence" and int(np.argmax(voz.huella)) == 210
    assert "Voz incluida «Ana Florence» lista" in capsys.readouterr().out


def test_perfil_inexistente_lista_lo_que_hay(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida", nombre_del_perfil="Carlos")
    mensaje = str(excinfo.value)
    assert es_error_usuario(excinfo) and "No encontré la voz «Carlos»" in mensaje
    assert "Perfiles de XTTS-v2 en Mi unidad/Clonador de voz/voces: mi_voz." in mensaje and "Ana Florence" in mensaje


def test_perfil_de_otro_modelo_se_explica(con_voz):
    (VOCES / "Carlos.chatterbox.pt").write_bytes(b"perfil de chatterbox")
    with pytest.raises(Exception) as excinfo:
        con_voz.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida", nombre_del_perfil="carlos")
    assert "Existe «Carlos.chatterbox.pt», pero es de otro modelo" in str(excinfo.value)


def test_si_drive_falla_al_guardar_el_perfil_la_voz_igual_queda_lista(lista, tmp_path, capsys):
    from google.colab import drive, files

    files.para_subir.append(crear_grabacion(tmp_path / "voz.m4a", voz_s=9.0))
    drive.fallos_pendientes = 99
    lista.ejecutar("referencia", origen_de_la_voz="Subir desde mi computadora")
    assert lista.ns["VOZ"].segundos > 8 and lista.ns["VOCES"] is not None
    assert "No se pudo guardar el perfil en Drive (ValueError: mount failed)" in capsys.readouterr().out


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
    lista.ejecutar("referencia", origen_de_la_voz="Subir desde mi computadora", guardar_perfil=False)
    assert 9.4 < lista.ns["VOZ"].segundos < 10.3
    assert drive.montajes == []  # Drive ni se toca
    assert (CONTENT / "clonador_voz" / "subidas" / "mi voz (celular).m4a").is_file()


def test_subir_sin_elegir_audio(lista):
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("referencia", origen_de_la_voz="Subir desde mi computadora")
    assert es_error_usuario(excinfo) and "No subiste ningún audio" in str(excinfo.value)
