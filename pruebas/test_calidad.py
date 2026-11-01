"""Control de calidad (celda 6): ritmo, silencios, Whisper, reintentos repetibles y orden de las tomas."""

import numpy as np
import pytest


def pedidos():
    from TTS import api

    return [llamada["texto"] for llamada in api.registro["inferencias"]]


def test_un_balbuceo_se_detecta_y_se_regenera(con_voz, capsys):
    texto = "Esta frase normal no tiene ningún problema. Esta otra trae un zzbalbuceo al final del todo."
    con_voz.generar(texto=texto, formato="wav", guardar_en_drive=False)
    segunda = "Esta otra trae un zzbalbuceo al final del todo."
    assert pedidos() == ["Esta frase normal no tiene ningún problema.", segunda, segunda]
    salida = capsys.readouterr().out
    assert "🔁 Control de calidad: 1 fragmento(s) se regeneraron." in salida
    toma = con_voz.ns["ULTIMA_GENERACION"].tomas[0]
    assert toma.avisos == 0
    audio, sr = con_voz.ns["leer_wav"](toma.archivo)
    esperado = sum(0.055 * len(t) + 0.12 for t in pedidos()[:2]) + 0.3  # sin los 2× de balbuceo
    assert abs(audio.size / sr - esperado) < 0.15


def test_un_silencio_a_mitad_de_frase_se_regenera(con_voz):
    con_voz.generar(texto="Aquí viene una frase con zzsilencio en el medio.", formato="wav", guardar_en_drive=False)
    assert len(pedidos()) == 2
    audio, sr = con_voz.ns["leer_wav"](con_voz.ns["ULTIMA_GENERACION"].tomas[0].archivo)
    assert con_voz.ns["silencio_interno_maximo"](audio, sr) < 0.5


def test_sin_control_de_calidad_no_hay_reintentos(con_voz, capsys):
    con_voz.generar(texto="Esta frase trae un zzbalbuceo larguísimo.", formato="wav", guardar_en_drive=False, control_de_calidad=False)
    assert len(pedidos()) == 1 and "Control de calidad" not in capsys.readouterr().out


def test_si_ningun_intento_pasa_se_usa_el_mejor_y_se_avisa(con_voz, capsys):
    con_voz.generar(texto="Esta frase siempre falla: zzsiempre balbucea.", formato="wav", guardar_en_drive=False)
    assert len(pedidos()) == 3  # 1 + 2 reintentos
    salida = capsys.readouterr().out
    assert "dura demasiado para su texto" in salida and "se usó el mejor de 3 intentos" in salida
    assert con_voz.ns["ULTIMA_GENERACION"].tomas[0].avisos == 1


def test_whisper_verifica_lo_que_se_oye(con_voz, colab, capsys):
    import faster_whisper

    colab.instalar("faster-whisper", "1.2.1")  # así no se instala de verdad
    con_voz.ns["instalar"] = lambda *_: pytest.fail("faster-whisper ya estaba instalado")
    texto = "Primera frase que Whisper entiende bien. Segunda frase con zzmurmullo que se oye mal."
    con_voz.generar(texto=texto, formato="wav", guardar_en_drive=False, verificar_con_whisper=True)
    segunda = "Segunda frase con zzmurmullo que se oye mal."
    assert pedidos() == ["Primera frase que Whisper entiende bien.", segunda, segunda]
    assert faster_whisper.cargados == ["large-v3-turbo"]  # se carga una sola vez
    assert [t["idioma"] for t in faster_whisper.transcripciones] == ["es", "es", "es"]
    assert "🔁 Control de calidad: 1 fragmento(s) se regeneraron." in capsys.readouterr().out
    con_voz.generar(texto="Otra frase más para verificar.", formato="wav", guardar_en_drive=False, verificar_con_whisper=True)
    assert faster_whisper.cargados == ["large-v3-turbo"]


def test_whisper_se_instala_la_primera_vez(con_voz):
    instalados = []
    con_voz.ns["instalar"] = lambda paquetes, indices=None: instalados.append(paquetes)
    con_voz.generar(texto="Una frase para verificar con Whisper.", formato="wav", guardar_en_drive=False, verificar_con_whisper=True)
    assert instalados == [["faster-whisper==1.2.1"]]


def test_los_reintentos_son_repetibles_con_la_semilla(con_voz):
    """La semilla de cada reintento se deriva de la de la toma: repetir la semilla repite todo."""
    con_voz.generar(texto="Una frase cualquiera para generar.", formato="wav", guardar_en_drive=False)
    derivada = con_voz.ns["semilla_derivada"]
    assert derivada(5, 0, 1) == derivada(5, 0, 1)
    assert len({derivada(5, i, j) for i in range(20) for j in (1, 2)}) == 40
    assert derivada(2**31 - 2, 3, 2) < 2**31 - 1


def test_ritmo_habitual_por_voz_e_idioma(con_voz):
    con_voz.generar(texto="Una frase cualquiera para generar.", formato="wav", guardar_en_drive=False)
    ns = con_voz.ns
    revisor = ns["Revisor"](ns["OpcionesCalidad"](activo=True, whisper=False), 1.0)
    rapido = ns["Segmento"]("x" * 60, 0.0, "es", "Ana")
    for _ in range(3):  # Ana habla a 30 caracteres por segundo
        revisor.aceptar(np.ones(2 * 24000, np.float32), rapido, 24000)
    lenta = np.ones(4 * 24000, np.float32) * 0.1  # 60 caracteres en 4 s: 15/s, la mitad de lo habitual en Ana
    assert "dura demasiado" in revisor.revisar(lenta, rapido, 24000).problemas[0]
    propia = ns["Segmento"]("x" * 60, 0.0, "es", "")  # para otra voz rige el ritmo nominal (14/s)
    assert revisor.revisar(lenta, propia, 24000).problemas == ()


def test_las_tomas_se_ordenan_por_avisos_y_parecido(con_voz, capsys):
    con_voz.generar(tomas=3, semilla=11, formato="wav")
    generacion = con_voz.ns["ULTIMA_GENERACION"]
    parecidos = [t.parecido for t in generacion.tomas]
    assert all(0.0 < p <= 1.0 for p in parecidos) and len(set(parecidos)) == 3
    assert generacion.recomendada == 1 + int(np.argmax(parecidos))
    salida = capsys.readouterr().out
    for toma in generacion.tomas:
        assert f"Toma {toma.numero}: parecido {toma.parecido:.3f} · 0 fragmento(s) con avisos" in salida

    Toma, ordenar = con_voz.ns["Toma"], con_voz.ns["ordenar_tomas"]
    tomas = [Toma(1, 1, None, parecido=0.99, avisos=2), Toma(2, 2, None, parecido=0.80), Toma(3, 3, None, parecido=0.95)]
    assert [t.numero for t in ordenar(tomas)] == [3, 2, 1]  # primero sin avisos; luego el parecido


def test_el_parecido_distingue_voces(con_voz):
    """La huella de la toma se compara con la de la grabación: una voz ajena se parece menos."""
    ns = con_voz.ns
    con_voz.generar(texto="Una frase cualquiera para comparar voces.", semilla=3, formato="wav", guardar_en_drive=False)
    motor, voz = ns["MOTOR"], ns["VOZ"]
    propia = ns["leer_wav"](ns["ULTIMA_GENERACION"].tomas[0].archivo)[0]
    t = np.arange(24000 * 3) / 24000
    ajena = (0.3 * np.sin(2 * np.pi * 230 * t)).astype(np.float32)
    assert ns["parecido"](motor.huella(propia, 24000), voz.huella) > 0.5 > ns["parecido"](motor.huella(ajena, 24000), voz.huella)
