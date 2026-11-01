"""Chatterbox Multilingual (licencia MIT) con la misma interfaz que XTTS-v2: celdas 1, 3, 4 y 6."""

import os

import numpy as np
import pytest

from conftest import MI_UNIDAD

VOCES = MI_UNIDAD / "Clonador de voz" / "voces"


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def generaciones():
    from chatterbox import mtl_tts

    return mtl_tts.registro["generaciones"]


def test_carga_sin_licencia_no_comercial_y_avisa_de_la_marca_de_agua(colab, capsys):
    from chatterbox import mtl_tts
    from TTS import api

    colab.instalar_todo("chatterbox")
    sesion = colab.sesion()
    sesion.ejecutar("entorno", modelo="Chatterbox Multilingual (uso comercial)")
    sesion.cargar_funciones()
    sesion.ejecutar("modelo", uso_no_comercial=False, modelo_afinado="mi_modelo")  # ninguno de los dos aplica
    motor = sesion.ns["MOTOR"]
    assert motor.nombre == "Chatterbox" and motor.identificador == "chatterbox" and motor.frecuencia == 24000
    assert mtl_tts.registro["cargas"] == 1 and api.registro["cargas"] == 0 and "COQUI_TOS_AGREED" not in os.environ
    assert {"es", "en", "sv", "zh"} <= motor.idiomas and "cs" not in motor.idiomas and motor.voces_incluidas() == []
    salida = capsys.readouterr().out
    assert "Chatterbox es MIT (uso comercial permitido)" in salida and "marca de agua" in salida
    assert "«modelo_afinado» solo aplica a XTTS-v2" in salida
    sesion.ejecutar("modelo")
    assert mtl_tts.registro["cargas"] == 1  # una sola carga por sesión


def test_voz_y_perfil_de_chatterbox(con_chatterbox, capsys):
    from chatterbox import mtl_tts

    voz = con_chatterbox.ns["VOZ"]
    assert voz.motor == "chatterbox" and abs(int(np.argmax(voz.huella)) - 140) <= 2
    assert mtl_tts.registro["referencias"][-1]["frecuencia"] == 24000  # sin remuestrear dos veces
    assert (VOCES / "mi_voz.chatterbox.pt").is_file()
    con_chatterbox.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")  # torch.load weights_only
    cargada = con_chatterbox.ns["VOZ"]
    assert np.array_equal(cargada.huella, voz.huella) and cargada.datos.t3.speaker_emb.dispositivo == "cuda"
    assert "Voz «mi_voz» cargada" in capsys.readouterr().out


def test_genera_con_sus_parametros(con_chatterbox):
    con_chatterbox.generar(
        texto="Hola, esta es mi voz con Chatterbox y con una sola frase.", exageracion=1, ritmo=0.3, semilla=2,
        formato="wav", guardar_en_drive=False,
    )
    pedidas = generaciones()
    assert [g["texto"] for g in pedidas] == ["Hola, esta es mi voz con Chatterbox y con una sola frase."]
    assert pedidas[0]["exageracion"] == 1.0 and pedidas[0]["cfg"] == 0.3 and pedidas[0]["idioma"] == "es"
    assert round(pedidas[0]["voz_hz"]) == 140
    assert con_chatterbox.ns["ULTIMA_GENERACION"].tomas[0].parecido > 0.7


def test_velocidad_con_atempo_de_ffmpeg(con_chatterbox):
    texto = "Una frase bastante larga para medir bien la velocidad del habla con Chatterbox."
    duraciones = []
    for velocidad in (1.0, 1.25):
        con_chatterbox.generar(texto=texto, velocidad=velocidad, semilla=5, formato="wav", guardar_en_drive=False)
        audio, sr = con_chatterbox.ns["leer_wav"](con_chatterbox.ns["ULTIMA_GENERACION"].tomas[0].archivo)
        duraciones.append(audio.size / sr)
    assert duraciones[1] / duraciones[0] == pytest.approx(1 / 1.25, abs=0.03)


def test_perfil_guardado_genera_igual(con_chatterbox):
    con_chatterbox.generar(texto="Frase para comparar el perfil guardado.", semilla=9, formato="wav", guardar_en_drive=False, nombre_salida="a")
    antes = con_chatterbox.ns["ULTIMA_GENERACION"].tomas[0].archivo.read_bytes()
    con_chatterbox.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")
    con_chatterbox.generar(texto="Frase para comparar el perfil guardado.", semilla=9, formato="wav", guardar_en_drive=False, nombre_salida="b")
    assert con_chatterbox.ns["ULTIMA_GENERACION"].tomas[0].archivo.read_bytes() == antes


def test_idioma_que_chatterbox_no_habla(con_chatterbox):
    with pytest.raises(Exception) as excinfo:
        con_chatterbox.generar(idioma="Čeština", formato="wav", guardar_en_drive=False)
    assert es_error_usuario(excinfo) and "Chatterbox no habla el idioma «cs»" in str(excinfo.value)


def test_dialogo_con_perfiles_y_sin_voces_incluidas(con_chatterbox):
    from conftest import crear_grabacion

    crear_grabacion(MI_UNIDAD / "lucia.wav", voz_s=10.0, f0=230.0)
    con_chatterbox.ejecutar("referencia", audio_en_drive="lucia.wav", nombre_del_perfil="Lucía")
    con_chatterbox.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")
    con_chatterbox.generar(texto="Te presento a mi amiga Lucía. [voz: lucía]¡Hola! Un gusto saludarte.[/voz]", formato="wav", guardar_en_drive=False)
    assert [round(g["voz_hz"]) for g in generaciones()] == [140, 230]
    with pytest.raises(Exception) as excinfo:
        con_chatterbox.generar(texto="[voz: Ana Florence]Hola.[/voz] Y nada más por hoy.", formato="wav")
    assert "No encontré la voz «Ana Florence»" in str(excinfo.value) and "Voces incluidas" not in str(excinfo.value)


def test_un_perfil_de_xtts_no_sirve_en_chatterbox(con_chatterbox):
    (VOCES / "Pedro.xtts.npz").write_bytes(b"perfil de xtts")
    with pytest.raises(Exception) as excinfo:
        con_chatterbox.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida", nombre_del_perfil="Pedro")
    assert "Existe «Pedro.xtts.npz», pero es de otro modelo" in str(excinfo.value)


def test_el_entrenamiento_es_solo_para_xtts(con_chatterbox):
    with pytest.raises(Exception) as excinfo:
        con_chatterbox.ejecutar("datos")
    assert es_error_usuario(excinfo) and "solo para XTTS-v2" in str(excinfo.value)
