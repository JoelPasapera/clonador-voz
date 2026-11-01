"""Prueba real semanal de Chatterbox Multilingual (chatterbox-tts de verdad, en CPU) con las celdas del cuaderno.

Lazo cerrado: la voz predeterminada de Chatterbox lee unas frases → la celda 4 prepara esa grabación (FFmpeg, a 24 kHz)
y la clona → el perfil guardado se vuelve a cargar (torch.load con weights_only=True) y debe generar lo mismo → la
celda 6 genera una toma con control de calidad, Whisper, subtítulos y velocidad 1.15 (atempo).
"""

import sys
import time
import types

import numpy as np

from cuaderno import Cuaderno, Revision

REFERENCIA = "Hola, esta es una grabación de referencia para probar el clonador de voz con Chatterbox. La leo con calma."
FRASE = "Esta frase compara el perfil guardado con el original."
TEXTO = "Buenos días a todos. Hoy la matrícula cuesta S/ 1,500.50 y empieza el lunes. ¡Nos vemos en clase!"


def main() -> int:
    revisar = Revision()
    cuaderno = Cuaderno()
    cuaderno.definir("entorno")
    cuaderno.ejecutar("archivos", "audio", "texto")
    ns = cuaderno.ns
    ns["mostrar_reproductor"] = lambda ruta: None
    cuaderno.definir("modelo")

    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from faster_whisper import WhisperModel

    inicio = time.perf_counter()
    modelo = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
    motor = ns["MotorChatterbox"](modelo)
    ns["MOTOR"] = motor
    print(f"⏱️ Chatterbox cargado en {time.perf_counter() - inicio:.0f} s")
    revisar(motor.frecuencia == 24000 and "es" in motor.idiomas, f"salida a {motor.frecuencia} Hz, {len(motor.idiomas)} idiomas")

    # 1) La voz predeterminada (conds.pt) lee la referencia; la celda 4 la prepara y la clona.
    carpeta = ns["preparar_carpeta"](ns["CARPETA_TRABAJO"] / "prueba_real", vaciar=True)
    motor.fijar_semilla(1)
    referencia = ns["a_numpy"](modelo.generate(REFERENCIA, language_id="es")).reshape(-1)
    ns["escribir_wav"](carpeta / "referencia_24k.wav", referencia, motor.frecuencia)
    ns["preparar_referencia"](
        carpeta / "referencia_24k.wav", carpeta / "referencia.wav", filtros=ns["filtros_referencia"](True, False),
        max_segundos=60, frecuencia=motor.frecuencia_referencia,
    )
    voz = motor.crear_voz([carpeta / "referencia.wav"], 60, nombre="prueba")
    revisar(3.0 < voz.segundos < 30.0, f"referencia de {voz.segundos:.1f} s clonada")

    # 2) El perfil guardado reproduce la misma voz.
    ajustes = ns["AjustesSintesis"]("es")
    motor.guardar_voz(voz, carpeta / "prueba.chatterbox.pt")
    cargada, meta = motor.cargar_voz(carpeta / "prueba.chatterbox.pt")
    motor.fijar_semilla(5)
    original = motor.sintetizar(FRASE, voz, ajustes)
    motor.fijar_semilla(5)
    desde_perfil = motor.sintetizar(FRASE, cargada, ajustes)
    revisar(meta["motor"] == "chatterbox" and np.allclose(original, desde_perfil, atol=1e-4), "el perfil guardado genera el mismo audio")

    # 3) Generación completa de la celda 6, más rápida (atempo).
    cuaderno.definir("referencia", "guion")
    ns["VOZ"], ns["VOCES"] = voz, ns["RegistroDeVoces"](motor, voz)
    ns["PEDIDO_TEXTO"] = ns["PedidoTexto"]("Cuadro de texto", "", "es", False, True, "", "", types.SimpleNamespace(value=TEXTO))
    ns["_WHISPER"] = {ns["MODELO_WHISPER"]: WhisperModel("small", device="cpu", compute_type="int8")}
    cuaderno.definir("generar")
    salida = ns["OpcionesSalida"](
        nombre="prueba_chatterbox", formato="wav", sonoridad=-16.0, subtitulos=True, musica="", volumen_musica=-18.0,
        guardar_en_drive=False, carpeta_en_drive="", descargar=False,
    )
    rapido = ns["AjustesSintesis"]("es", velocidad=1.15, temperatura=0.8)
    inicio = time.perf_counter()
    generacion = ns["generar_audio"](
        ns["PEDIDO_TEXTO"], rapido, ns["OpcionesCalidad"](activo=True, whisper=True),
        pausa_frase=0.3, semilla=7, tomas=1, salida=salida,
    )[0]
    print(f"⏱️ Toma generada en {time.perf_counter() - inicio:.0f} s")
    toma = generacion.tomas[0]
    audio, frecuencia = ns["leer_wav"](toma.archivo)
    opciones = ns["OpcionesTexto"](pausa_frase=0.3, cifras=True, pronunciador=None)
    leido = " ".join(s.texto for s in ns["preparar_segmentos"](TEXTO, "es", opciones, motor.idiomas))
    esperado = len(leido) / (14.0 * 1.15)
    revisar(0.4 * esperado < audio.size / frecuencia < 2.5 * esperado, f"{audio.size / frecuencia:.1f} s (≈ {esperado:.1f} s esperados)")
    lufs = ns["sonoridad_integrada"](audio, frecuencia)
    revisar(abs(lufs + 16.0) < 1.0, f"{lufs:.1f} LUFS")
    revisar("-->" in toma.subtitulos.read_text(encoding="utf-8"), "subtítulos creados")
    print(f"ℹ️ Parecido con la voz clonada: {toma.parecido:.3f} (solo informativo)")
    error = ns["tasa_error_caracteres"](leido, ns["transcribir"](audio, frecuencia, "es"), "es")
    revisar(error < 0.4, f"Whisper entiende lo pedido (error {error:.0%})")
    return revisar.resultado()


if __name__ == "__main__":
    sys.exit(main())
