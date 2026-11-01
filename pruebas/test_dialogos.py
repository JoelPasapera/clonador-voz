"""Diálogos (celdas 4 y 6): [voz: …] con voces incluidas del modelo y con perfiles guardados en Drive."""

import pytest

from conftest import MI_UNIDAD, crear_grabacion


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def voces_usadas():
    from TTS import api

    return [(llamada["texto"], round(llamada["voz_hz"])) for llamada in api.registro["inferencias"]]


def test_dialogo_con_una_voz_incluida(con_voz, capsys):
    texto = (
        "Hoy tenemos una invitada muy especial en el programa.\n"
        "[voz: Ana Florence]Hola a todos, gracias por invitarme.[/voz]\n"
        "Bienvenida, empecemos con la primera pregunta."
    )
    con_voz.generar(texto=texto, formato="wav", guardar_en_drive=False, subtitulos=True)
    assert voces_usadas() == [
        ("Hoy tenemos una invitada muy especial en el programa.", 140),
        ("Hola a todos, gracias por invitarme.", 210),  # la huella de Ana Florence
        ("Bienvenida, empecemos con la primera pregunta.", 140),
    ]
    assert "🎭 Voces: la tuya, Ana Florence." in capsys.readouterr().out
    srt = con_voz.ns["ULTIMA_GENERACION"].tomas[0].subtitulos.read_text(encoding="utf-8")
    assert "Hola a todos, gracias por invitarme." in srt


def test_nombres_sin_mayusculas_y_voz_principal_por_su_nombre(con_voz):
    con_voz.generar(
        texto="[voz: ana florence]Primera frase de Ana en el diálogo.[/voz] [voz: mi_voz]Y esta es mía, con mi voz.[/voz]",
        formato="wav", guardar_en_drive=False,
    )
    assert [voz for _, voz in voces_usadas()] == [210, 140]


def test_dialogo_con_un_perfil_guardado(con_voz):
    crear_grabacion(MI_UNIDAD / "carlos.wav", voz_s=10.0, f0=100.0)
    con_voz.ejecutar("referencia", audio_en_drive="carlos.wav", nombre_del_perfil="Carlos")  # guarda «Carlos»
    con_voz.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")  # vuelve «mi_voz»
    con_voz.generar(
        texto="Te presento a mi amigo Carlos, que viene de lejos. [voz: carlos]¡Mucho gusto! Encantado de estar aquí.[/voz]",
        formato="wav", guardar_en_drive=False,
    )
    assert [voz for _, voz in voces_usadas()] == [140, 100]


def test_voz_desconocida_falla_antes_de_sintetizar(con_voz):
    with pytest.raises(Exception) as excinfo:
        con_voz.generar(texto="Hola, ¿quién habla? [voz: Fantasma]Nadie, solo el viento.[/voz]", formato="wav")
    assert es_error_usuario(excinfo) and "No encontré la voz «Fantasma»" in str(excinfo.value)
    assert voces_usadas() == []  # ni un fragmento se sintetizó en vano


def test_el_parecido_se_mide_solo_con_tu_voz(con_voz):
    """En un diálogo, la toma se compara con tu grabación usando solo tus fragmentos."""
    largo = "Esta es una intervención bastante larga de la invitada, que habla durante muchos segundos seguidos sin parar."
    con_voz.generar(texto=f"Una frase mía. [voz: Ana Florence]{largo}[/voz]", semilla=4, formato="wav", guardar_en_drive=False)
    assert con_voz.ns["ULTIMA_GENERACION"].tomas[0].parecido > 0.7  # con los fragmentos de Ana sería ~0
