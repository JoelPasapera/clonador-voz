"""Prueba real semanal: las celdas del cuaderno con coqui-tts, XTTS-v2 y faster-whisper de verdad (en CPU).

Lazo cerrado, sin grabaciones en el repositorio: una voz incluida de XTTS-v2 lee unas frases → esa grabación pasa por
la preparación de la celda 4 (FFmpeg) → se clona → la celda 6 genera dos tomas con control de calidad y Whisper,
subtítulos y volumen parejo. Se comprueba que suene (duración y volumen), que se parezca más a la voz clonada que a
otra voz, que el perfil guardado reproduzca exactamente lo mismo y que Whisper entienda lo que dice.
"""

import os
import sys
import time
import types

import numpy as np

from cuaderno import Cuaderno, Revision

REFERENCIA = [
    "Hola, esta es una grabación de referencia para probar el clonador de voz.",
    "La leo con calma, con pausas naturales, como si le explicara algo a un amigo.",
    "Si todo funciona bien, la voz clonada debería sonar muy parecida a esta.",
]
FRASE = "Esta frase sirve para comparar el perfil guardado con el original."
TEXTO = "Buenos días a todos. Hoy la matrícula cuesta S/ 1,500.50 y empieza el lunes. ¡Nos vemos en clase!"
MODELO_WHISPER_CPU = "small"  # en la GPU de Colab se usa large-v3-turbo; aquí, uno pequeño en CPU


def main() -> int:
    revisar = Revision()
    os.environ["COQUI_TOS_AGREED"] = "1"  # lo que hace la celda 3 con la casilla de uso no comercial marcada
    cuaderno = Cuaderno()
    cuaderno.definir("entorno")
    cuaderno.ejecutar("archivos", "audio", "texto")
    ns = cuaderno.ns
    ns["mostrar_reproductor"] = lambda ruta: None  # fuera de un cuaderno no hay dónde mostrarlo
    cuaderno.definir("modelo")

    from faster_whisper import WhisperModel
    from TTS.api import TTS

    inicio = time.perf_counter()
    motor = ns["MotorXTTS"](TTS(ns["NOMBRE_MODELO"], progress_bar=False).synthesizer.tts_model, "xtts")
    ns["MOTOR"] = motor
    print(f"⏱️ XTTS-v2 cargado en {time.perf_counter() - inicio:.0f} s")
    incluidas = motor.voces_incluidas()
    revisar(len(incluidas) >= 2 and motor.frecuencia == 24000, f"{len(incluidas)} voces incluidas, salida a {motor.frecuencia} Hz")
    origen = "Ana Florence" if "Ana Florence" in incluidas else incluidas[0]
    ajena = "Damien Black" if "Damien Black" in incluidas else incluidas[-1]

    # 1) Una voz incluida lee la referencia; la celda 4 la prepara (FFmpeg) y la clona.
    ajustes = ns["AjustesSintesis"]("es")
    carpeta = ns["preparar_carpeta"](ns["CARPETA_TRABAJO"] / "prueba_real", vaciar=True)
    motor.fijar_semilla(1)
    voz_origen = motor.voz_incluida(origen)
    partes = [ns["recortar_silencios"](motor.sintetizar(frase, voz_origen, ajustes), motor.frecuencia) for frase in REFERENCIA]
    ns["escribir_wav"](carpeta / "referencia_24k.wav", ns["concatenar_con_pausas"](partes, [0.4] * len(partes), motor.frecuencia), motor.frecuencia)
    ns["preparar_referencia"](
        carpeta / "referencia_24k.wav", carpeta / "referencia.wav", filtros=ns["filtros_referencia"](True, False),
        max_segundos=60, frecuencia=motor.frecuencia_referencia,
    )
    voz = motor.crear_voz([carpeta / "referencia.wav"], 60, nombre="prueba")
    revisar(8.0 < voz.segundos < 30.0, f"referencia de {voz.segundos:.1f} s clonada")

    # 2) El perfil guardado (celda 4) reproduce exactamente la misma voz.
    motor.guardar_voz(voz, carpeta / "prueba.xtts.npz")
    cargada, meta = motor.cargar_voz(carpeta / "prueba.xtts.npz")
    motor.fijar_semilla(5)
    original = motor.sintetizar(FRASE, voz, ajustes)
    motor.fijar_semilla(5)
    desde_perfil = motor.sintetizar(FRASE, cargada, ajustes)
    revisar(meta["motor"] == "xtts" and np.array_equal(original, desde_perfil), "el perfil guardado genera el mismo audio")

    # 3) Generación completa de la celda 6: 2 tomas, control de calidad con Whisper y subtítulos.
    cuaderno.definir("referencia", "guion")
    ns["VOZ"], ns["VOCES"] = voz, ns["RegistroDeVoces"](motor, voz)
    ns["PEDIDO_TEXTO"] = ns["PedidoTexto"]("Cuadro de texto", "", "es", False, True, "", "", types.SimpleNamespace(value=TEXTO))
    ns["_WHISPER"] = {ns["MODELO_WHISPER"]: WhisperModel(MODELO_WHISPER_CPU, device="cpu", compute_type="int8")}
    cuaderno.definir("generar")
    salida = ns["OpcionesSalida"](
        nombre="prueba_real", formato="mp3", sonoridad=-16.0, subtitulos=True, musica="", volumen_musica=-18.0,
        guardar_en_drive=False, carpeta_en_drive="", descargar=False,
    )
    inicio = time.perf_counter()
    generacion = ns["generar_audio"](
        ns["PEDIDO_TEXTO"], ajustes, ns["OpcionesCalidad"](activo=True, whisper=True),
        pausa_frase=0.3, semilla=7, tomas=2, salida=salida,
    )[0]
    print(f"⏱️ Dos tomas generadas en {time.perf_counter() - inicio:.0f} s")
    opciones = ns["OpcionesTexto"](pausa_frase=0.3, cifras=True, pronunciador=None)
    leido = " ".join(s.texto for s in ns["preparar_segmentos"](TEXTO, "es", opciones, motor.idiomas))
    revisar("mil quinientos soles con cincuenta céntimos" in leido, "los montos se leen en palabras")
    huella_ajena = motor.voz_incluida(ajena).huella
    for toma in generacion.tomas:
        audio = ns["decodificar_audio"](toma.archivo, motor.frecuencia)
        segundos = audio.size / motor.frecuencia
        esperado = len(leido) / 14.0
        revisar(0.5 * esperado < segundos < 2.0 * esperado, f"toma {toma.numero}: {segundos:.1f} s (≈ {esperado:.1f} s esperados)")
        lufs = ns["sonoridad_integrada"](audio, motor.frecuencia)
        revisar(abs(lufs + 16.0) < 1.5 and np.abs(audio).max() < 0.95, f"toma {toma.numero}: {lufs:.1f} LUFS, sin saturar")
        subtitulos = toma.subtitulos.read_text(encoding="utf-8") if toma.subtitulos else ""
        revisar("S/ 1,500.50" in subtitulos and "-->" in subtitulos, f"toma {toma.numero}: subtítulos con el texto escrito")
        huella = motor.huella(audio, motor.frecuencia)
        propia, otra = ns["parecido"](huella, voz.huella), ns["parecido"](huella, huella_ajena)
        revisar(propia > otra, f"toma {toma.numero}: parecido con la voz clonada {propia:.3f} > con {ajena} {otra:.3f}")
        transcripcion = ns["transcribir"](audio, motor.frecuencia, "es")
        error = ns["tasa_error_caracteres"](leido, transcripcion, "es")
        revisar(error < 0.35, f"toma {toma.numero}: Whisper entiende lo pedido (error {error:.0%}): «{transcripcion}»")
    revisar(generacion.recomendada in (1, 2), f"recomendada: toma {generacion.recomendada}")
    return revisar.resultado()


if __name__ == "__main__":
    sys.exit(main())
