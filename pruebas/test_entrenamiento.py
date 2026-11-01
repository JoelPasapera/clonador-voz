"""Cuaderno de entrenamiento · afinado de XTTS-v2: frases desde Whisper (celda 3), entrenamiento (4), guardado en
Drive (5) y uso del modelo afinado en la celda 3 del clonador."""

import csv
import os
import wave

import numpy as np
import pytest

from conftest import CONTENT, MI_UNIDAD, crear_grabacion

ENTRENAMIENTO = CONTENT / "clonador_voz" / "entrenamiento"
MODELOS = MI_UNIDAD / "Clonador de voz" / "modelos"


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def palabras(funciones, especificacion):
    """[(inicio, fin, texto)] → Palabra de la celda 8a."""
    Palabra = funciones.ns["Palabra"]
    return [Palabra(inicio, fin, texto) for inicio, fin, texto in especificacion]


@pytest.fixture
def frases(funciones):
    funciones.ns["MOTOR_ELEGIDO"] = "xtts"  # lo que deja la celda 1
    funciones.definir("datos")
    return funciones


# ── Cortar en frases ────────────────────────────────────────────────────────────


def test_corta_en_los_finales_de_frase_con_margenes(frases):
    lista = palabras(frases, [(0.5, 0.8, " Hola"), (0.9, 1.3, " mundo."), (2.0, 2.4, " Otra"), (2.5, 3.0, " frase!")])
    clips = frases.ns["cortar_en_frases"](lista, duracion=4.0)
    assert [(round(c.inicio, 3), round(c.fin, 3), c.texto) for c in clips] == [
        (0.3, 1.5, "Hola mundo."),  # 0,2 s de aire a cada lado…
        (1.8, 3.2, "Otra frase!"),  # …sin invadir la palabra vecina (el punto medio de la pausa queda en 1,65)
    ]


def test_el_aire_no_invade_la_palabra_vecina(frases):
    lista = palabras(frases, [(0.0, 0.9, " Uno dos tres."), (1.0, 1.9, " Cuatro cinco.")])  # pausa de solo 0,1 s
    clips = frases.ns["cortar_en_frases"](lista, duracion=2.0)
    assert [(round(c.inicio, 3), round(c.fin, 3)) for c in clips] == [(0.0, 0.95), (0.95, 1.95)]


def test_una_frase_larga_se_corta_en_su_mejor_pausa(frases):
    especificacion, inicio = [], 0.0
    for numero in range(30):  # 30 palabras seguidas (12 s) sin punto
        coma = "," if numero == 14 else ""
        especificacion.append((inicio, inicio + 0.3, f" palabra{numero}{coma}"))
        inicio += 0.4 if numero != 14 else 0.7  # tras la coma hay una pausa más larga
    clips = frases.ns["cortar_en_frases"](palabras(frases, especificacion), duracion=inicio + 1)
    assert len(clips) == 2 and clips[0].texto.endswith("palabra14,")
    assert all(c.fin - c.inicio <= 11.0 for c in clips)
    assert " ".join(c.texto for c in clips) == " ".join(t.strip() for _, _, t in especificacion)


def test_descarta_frases_muy_cortas_o_sin_letras(frases):
    lista = palabras(frases, [(0.0, 0.1, " ¡Ah!"), (1.0, 1.1, " ..."), (2.0, 3.0, " Esta sí vale.")])
    assert [c.texto for c in frases.ns["cortar_en_frases"](lista, duracion=3.5)] == ["Esta sí vale."]


def test_texto_seguro_para_el_csv(frases):
    assert frases.ns["texto_para_csv"](' Dijo "hola" | y\nse fue ') == "Dijo hola , y se fue"


# ── Flujo completo: 8a → 8b → 8c → celda 3 con el modelo afinado ────────────────


@pytest.fixture
def con_grabaciones(lista, colab):
    colab.instalar("faster-whisper", "1.2.1")
    crear_grabacion(MI_UNIDAD / "Audios" / "Mi voz" / "clase 1.m4a", voz_s=40.0)
    crear_grabacion(MI_UNIDAD / "Audios" / "Mi voz" / "Más clases" / "clase 2.wav", voz_s=25.0)
    return lista


def test_8a_prepara_frases_y_csv(con_grabaciones, capsys):
    import faster_whisper

    con_grabaciones.ejecutar("datos")
    datos = con_grabaciones.ns["DATOS_ENTRENAMIENTO"]
    assert faster_whisper.cargados == ["large-v3"] and con_grabaciones.ns["_WHISPER"] == {}  # se libera al terminar
    assert {t["palabras"] for t in faster_whisper.transcripciones} == {True}
    assert datos.idioma == "es" and datos.frases >= 15
    with open(datos.csv_entrenamiento, newline="", encoding="utf-8") as archivo:
        entrenamiento = list(csv.DictReader(archivo, delimiter="|"))  # como el formateador «coqui»
    with open(datos.csv_evaluacion, newline="", encoding="utf-8") as archivo:
        evaluacion = list(csv.DictReader(archivo, delimiter="|"))
    assert len(entrenamiento) + len(evaluacion) == datos.frases
    assert len(evaluacion) == max(1, round(datos.frases * 0.15))
    assert {f["speaker_name"] for f in entrenamiento} == {"mi_voz"}
    assert entrenamiento[0]["text"] == entrenamiento[0]["text"].lower() and "palabra" in entrenamiento[0]["text"]
    assert any("002_clase 2" in f["audio_file"] for f in entrenamiento + evaluacion)  # incluye subcarpetas
    for fila in entrenamiento + evaluacion:
        with wave.open(str(datos.carpeta / fila["audio_file"]), "rb") as archivo:
            assert archivo.getframerate() == 22050 and archivo.getnchannels() == 1
            assert 0.6 <= archivo.getnframes() / 22050 <= 11.0
    assert "Revisa algunas transcripciones" in capsys.readouterr().out


def test_8a_con_muy_poco_audio(lista, colab):
    colab.instalar("faster-whisper", "1.2.1")
    crear_grabacion(MI_UNIDAD / "poco.wav", voz_s=6.0)
    with pytest.raises(Exception) as excinfo:
        lista.ejecutar("datos", audios_de_entrenamiento="poco.wav")
    assert es_error_usuario(excinfo) and "hacen falta al menos 10" in str(excinfo.value)


def test_entrenar_guardar_y_usar_el_modelo_afinado(con_grabaciones, capsys):
    import huggingface_hub
    import torch
    from TTS import api
    from TTS.demos.xtts_ft_demo.utils import gpt_train

    sesion = con_grabaciones  # aquí corrió la celda 3 del clonador: XTTS-v2 ya está en la caché de Hugging Face
    ya_descargados = list(huggingface_hub.descargas)
    assert ("tts-hub/XTTS-v2", "model.pth") in ya_descargados and ("tts-hub/XTTS-v2", "dvae.pth") not in ya_descargados
    sesion.ejecutar("datos")
    sesion.ejecutar("entrenamiento")
    llamada = gpt_train.llamadas[-1]
    datos = sesion.ns["DATOS_ENTRENAMIENTO"]
    assert (llamada["idioma"], llamada["epocas"], llamada["lote"], llamada["acumulacion"]) == ("es", 6, 2, 1)
    assert llamada["entrenamiento"] + llamada["evaluacion"] == datos.frases
    assert llamada["descargados"] == []  # train_gpt no descarga nada por su cuenta
    assert huggingface_hub.descargas[len(ya_descargados) :] == [("tts-hub/XTTS-v2", "dvae.pth")]  # solo lo que faltaba
    originales = ENTRENAMIENTO / "salida" / "run" / "training" / "XTTS_v2.0_original_model_files"
    assert (originales / "model.pth").is_symlink() and not (originales / "vocab.json").is_symlink()
    assert "Guárdalo con la celda 5" in capsys.readouterr().out

    sesion.ejecutar("guardar_modelo")
    carpeta = MODELOS / "mi_voz_afinada"
    assert sorted(p.name for p in carpeta.iterdir()) == [
        "config.json", "info.json", "model.pth", "referencia.wav", "speakers_xtts.pth", "vocab.json",
    ]
    estado = torch.load(carpeta / "model.pth")  # weights_only=True: solo tensores
    assert sorted(estado) == ["model"]
    assert sorted(estado["model"]) == ["gpt.peso", "gpt.posiciones", "hifigan_decoder.peso"]
    assert estado["model"]["gpt.peso"].dtype == np.float16 and estado["model"]["gpt.posiciones"].dtype == np.int64
    assert "modelo_afinado = «mi_voz_afinada»" in capsys.readouterr().out
    sesion.ejecutar("guardar_modelo")  # no sobrescribe
    assert (MODELOS / "mi_voz_afinada_2" / "model.pth").is_file()

    sesion.ejecutar("modelo", modelo_afinado="mi_voz_afinada")
    motor = sesion.ns["MOTOR"]
    assert motor.identificador == "xtts:mi_voz_afinada" and api.registro["puntos_de_control"] == [str(carpeta)]
    assert motor.modelo.device == "cuda" and motor.voces_incluidas()  # con las voces de estudio
    sesion.ejecutar("referencia", audio_en_drive="Clonador de voz/modelos/mi_voz_afinada/referencia.wav")
    assert sesion.ns["VOZ"].motor == "xtts:mi_voz_afinada" and sesion.ns["VOZ"].segundos > 25  # varias frases unidas
    sesion.generar(texto="Ahora hablo con mi modelo afinado.", formato="wav", guardar_en_drive=False)
    assert sesion.ns["ULTIMA_GENERACION"].tomas[0].archivo.is_file()


def test_perfil_creado_con_otro_modelo_avisa(con_grabaciones, capsys):
    sesion = con_grabaciones
    crear_grabacion(MI_UNIDAD / "mi_voz.wav", voz_s=10.0)
    sesion.ejecutar("referencia")  # perfil «mi_voz» con el XTTS-v2 original
    sesion.ejecutar("datos")
    sesion.ejecutar("entrenamiento")
    sesion.ejecutar("guardar_modelo")
    sesion.ejecutar("modelo", modelo_afinado="mi_voz_afinada")
    capsys.readouterr()
    sesion.ejecutar("referencia", origen_de_la_voz="Perfil guardado o voz incluida")
    assert "se creó con «xtts» y ahora usas «xtts:mi_voz_afinada»" in capsys.readouterr().out


def test_8b_sin_memoria_explica_que_hacer(con_grabaciones):
    con_grabaciones.ejecutar("datos")
    with pytest.raises(Exception) as excinfo:
        con_grabaciones.ejecutar("entrenamiento", tamano_de_lote=4)
    assert es_error_usuario(excinfo) and "baje" not in str(excinfo.value) and "«tamano_de_lote» a 1" in str(excinfo.value)
    assert con_grabaciones.ns["MODELO_ENTRENADO"] is None


def test_el_cuaderno_de_entrenamiento_funciona_solo(colab):
    """En su propio cuaderno no hay celda 3: los archivos originales de XTTS-v2 se bajan del mismo repositorio que usa
    coqui-tts (tts-hub/XTTS-v2), sin que train_gpt descargue nada por su cuenta."""
    import huggingface_hub
    from TTS.demos.xtts_ft_demo.utils import gpt_train

    colab.instalar_todo()
    colab.instalar("faster-whisper", "1.2.1")
    crear_grabacion(MI_UNIDAD / "Audios" / "Mi voz" / "clase.wav", voz_s=40.0)
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    sesion.ejecutar("datos")
    assert os.environ["COQUI_TOS_AGREED"] == "1"  # la casilla de uso no comercial, marcada por defecto
    sesion.ejecutar("entrenamiento")
    assert gpt_train.llamadas[-1]["descargados"] == []
    assert sorted(nombre for _, nombre in huggingface_hub.descargas) == [
        "config.json", "dvae.pth", "mel_stats.pth", "model.pth", "speakers_xtts.pth", "vocab.json",
    ]
    sesion.ejecutar("guardar_modelo")
    assert (MODELOS / "mi_voz_afinada" / "speakers_xtts.pth").is_file()


def test_sin_licencia_no_comercial_no_se_entrena(colab):
    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.ejecutar("entorno")
    sesion.cargar_funciones()
    with pytest.raises(Exception) as excinfo:
        sesion.ejecutar("datos", uso_no_comercial=False)
    assert es_error_usuario(excinfo) and "no permite el uso comercial" in str(excinfo.value)
    assert "COQUI_TOS_AGREED" not in os.environ
