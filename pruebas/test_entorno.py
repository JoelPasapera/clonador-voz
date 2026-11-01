"""Celda 1 · entorno: GPU, instalación (uv con respaldo en pip), elección del modelo y reinicio de la sesión."""

import shutil
import subprocess
import sys
import types

import pytest


def es_error_usuario(excinfo) -> bool:
    return type(excinfo.value).__name__ == "ErrorUsuario"


def preparada(colab, **instalados):
    for nombre, version in instalados.items():
        colab.instalar(nombre.replace("_", "-"), version)
    sesion = colab.sesion()
    sesion.definir("entorno")
    llamadas = []
    sesion.ns["instalar"] = lambda paquetes, indices=None: llamadas.append((paquetes, indices))
    return sesion, llamadas


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
    sesion, llamadas = preparada(colab, torch="2.10.0+cu126", torchaudio="2.10.0+cu126")
    sesion.solo_ejecucion("entorno")
    assert llamadas == [(["coqui-tts==0.28.0", "torchcodec==0.10.*"], [])]
    assert sesion.ns["MOTOR_ELEGIDO"] == "xtts"


def test_torchaudio_faltante_se_instala_con_la_variante_cuda_de_torch(colab):
    sesion, llamadas = preparada(colab, torch="2.10.0+cu126")
    sesion.solo_ejecucion("entorno")
    assert llamadas == [
        (["coqui-tts==0.28.0", "torchaudio==2.10.0+cu126", "torchcodec==0.10.*"], ["https://download.pytorch.org/whl/cu126"])
    ]


def test_no_reinstala_si_todo_esta(colab, capsys):
    colab.instalar_todo()
    sesion = colab.sesion()
    sesion.definir("entorno")
    sesion.ns["instalar"] = lambda *_: pytest.fail("no debía instalar nada")
    sesion.solo_ejecucion("entorno")
    salida = capsys.readouterr().out
    assert "✅ Entorno listo" in salida and "Tesla T4" in salida and "coqui-tts 0.28.0" in salida


def test_chatterbox_instala_su_paquete_con_torchvision_a_juego(colab, capsys):
    sesion, llamadas = preparada(colab, torch="2.10.0+cu126", torchaudio="2.10.0+cu126")
    sesion.ns["modelo"] = "Chatterbox Multilingual (uso comercial)"  # el campo del formulario
    sesion.solo_ejecucion("entorno")
    assert llamadas == [(["chatterbox-tts==0.1.7", "torchvision==0.21.0"], [])]
    assert sesion.ns["MOTOR_ELEGIDO"] == "chatterbox"


def test_volver_a_xtts_en_un_entorno_con_chatterbox_pide_uno_nuevo(colab):
    colab.instalar_todo("chatterbox")
    sesion = colab.sesion()
    with pytest.raises(Exception) as excinfo:
        sesion.ejecutar("entorno")
    assert es_error_usuario(excinfo) and "Desconectar y eliminar entorno" in str(excinfo.value)


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


# ── uv con respaldo en pip ──────────────────────────────────────────────────────


class SubprocesoFalso:
    """Reemplaza subprocess en la celda: registra cada comando y hace fallar a uv cuando se pide."""

    def __init__(self, falla_uv: bool) -> None:
        self.falla_uv = falla_uv
        self.comandos: list[list[str]] = []

    def run(self, comando, **_):
        self.comandos.append(list(comando))
        codigo = 1 if (self.falla_uv and comando[0].endswith("uv")) else 0
        return subprocess.CompletedProcess(comando, codigo, stdout="", stderr="error de uv" if codigo else "")


def test_instalar_prueba_uv_y_si_falla_usa_pip(colab):
    sesion = colab.sesion()
    sesion.definir("entorno")
    falso = SubprocesoFalso(falla_uv=True)
    sesion.ns["subprocess"] = falso
    sesion.ns["asegurar_uv"] = lambda: "/usr/local/bin/uv"
    sesion.ns["instalar"](["coqui-tts==0.28.0", "torchaudio==2.10.0+cu126"], ["https://download.pytorch.org/whl/cu126"])
    uv, pip = falso.comandos
    assert uv[:3] == ["/usr/local/bin/uv", "pip", "install"]
    assert uv[uv.index("--python") + 1] == sys.executable and "--system" in uv and "--break-system-packages" in uv
    assert uv[uv.index("--extra-index-url") + 1] == "https://download.pytorch.org/whl/cu126"
    assert uv[uv.index("--index-strategy") + 1] == "unsafe-best-match"
    assert pip[:4] == [sys.executable, "-m", "pip", "install"] and "--extra-index-url" in pip
    assert uv[-2:] == pip[-2:] == ["coqui-tts==0.28.0", "torchaudio==2.10.0+cu126"]


def test_instalar_con_uv_no_llama_a_pip(colab):
    sesion = colab.sesion()
    sesion.definir("entorno")
    falso = SubprocesoFalso(falla_uv=False)
    sesion.ns["subprocess"] = falso
    sesion.ns["asegurar_uv"] = lambda: "/usr/local/bin/uv"
    sesion.ns["instalar"](["faster-whisper==1.2.1"])
    assert len(falso.comandos) == 1 and "--index-strategy" not in falso.comandos[0]


def test_sin_uv_se_instala_con_pip(colab):
    sesion = colab.sesion()
    sesion.definir("entorno")
    falso = SubprocesoFalso(falla_uv=False)
    sesion.ns["subprocess"] = falso
    sesion.ns["asegurar_uv"] = lambda: None
    sesion.ns["instalar"](["faster-whisper==1.2.1"])
    assert [c[:4] for c in falso.comandos] == [[sys.executable, "-m", "pip", "install"]]


def test_si_nada_funciona_muestra_el_error(colab, capsys):
    sesion = colab.sesion()
    sesion.definir("entorno")

    class TodoFalla(SubprocesoFalso):
        def run(self, comando, **_):
            self.comandos.append(list(comando))
            return subprocess.CompletedProcess(comando, 1, stdout="", stderr="ERROR: No matching distribution")

    sesion.ns["subprocess"] = TodoFalla(falla_uv=True)
    sesion.ns["asegurar_uv"] = lambda: "/usr/local/bin/uv"
    with pytest.raises(RuntimeError, match="No se pudo instalar: paquete==1"):
        sesion.ns["instalar"](["paquete==1"])
    assert "No matching distribution" in capsys.readouterr().out


def test_instalar_si_falta_solo_cuando_falta(colab):
    sesion, llamadas = preparada(colab)
    sesion.ns["instalar_si_falta"]("faster-whisper", "faster-whisper==1.2.1")
    colab.instalar("faster-whisper", "1.2.1")
    sesion.ns["instalar_si_falta"]("faster-whisper", "faster-whisper==1.2.1")
    assert llamadas == [(["faster-whisper==1.2.1"], None)]
