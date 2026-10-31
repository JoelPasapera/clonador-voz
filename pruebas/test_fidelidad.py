"""Los simuladores imitan fielmente a coqui-tts: se comparan con su código fuente real, clonado en la versión que
fija la celda 1 (`COQUI_TTS_SRC` indica dónde; el flujo de GitHub Actions lo clona solo)."""

import ast
import inspect
import json
import os
import re
from pathlib import Path

import pytest

from conftest import celdas_de_codigo

REPO = Path(os.environ["COQUI_TTS_SRC"]) if os.environ.get("COQUI_TTS_SRC") else None
pytestmark = pytest.mark.skipif(
    REPO is None or not (REPO / "TTS").is_dir(), reason="define COQUI_TTS_SRC con el código fuente de coqui-tts"
)


def firma_real(archivo: str, clase: str, metodo: str) -> list[str]:
    arbol = ast.parse((REPO / archivo).read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.ClassDef) and nodo.name == clase:
            for funcion in nodo.body:
                if isinstance(funcion, ast.FunctionDef) and funcion.name == metodo:
                    a = funcion.args
                    nombres = [x.arg for x in a.posonlyargs + a.args] + [f"*{x.arg}" for x in a.kwonlyargs]
                    if a.vararg:
                        nombres.append(f"*args:{a.vararg.arg}")
                    if a.kwarg:
                        nombres.append(f"**{a.kwarg.arg}")
                    return nombres
    raise LookupError(f"{clase}.{metodo}")


def firma_simulada(funcion) -> list[str]:
    nombres = []
    for p in inspect.signature(funcion).parameters.values():
        if p.kind is p.KEYWORD_ONLY:
            nombres.append(f"*{p.name}")
        elif p.kind is p.VAR_POSITIONAL:
            nombres.append(f"*args:{p.name}")
        elif p.kind is p.VAR_KEYWORD:
            nombres.append(f"**{p.name}")
        else:
            nombres.append(p.name)
    return nombres


@pytest.mark.parametrize("metodo", ["inference", "get_gpt_cond_latents", "get_speaker_embedding"])
def test_firmas_de_xtts(metodo):
    from TTS.api import Xtts

    assert firma_simulada(getattr(Xtts, metodo)) == firma_real("TTS/tts/models/xtts.py", "Xtts", metodo)


def test_firma_de_tts_api():
    from TTS.api import TTS

    assert firma_simulada(TTS.__init__) == firma_real("TTS/api.py", "TTS", "__init__")


def test_datos_del_modelo_real():
    from TTS.api import NOMBRE_ESPERADO, _Tokenizer

    modelos = json.loads((REPO / "TTS/.models.json").read_text(encoding="utf-8"))
    _, idioma, dataset, nombre = NOMBRE_ESPERADO.split("/")
    entrada = modelos["tts_models"][idioma][dataset][nombre]
    assert entrada["license"] == "CPML" and entrada["tos_required"] is True

    xtts = (REPO / "TTS/tts/models/xtts.py").read_text(encoding="utf-8")
    assert '"wav": torch.cat(wavs, dim=0).numpy()' in xtts  # inference() devuelve un ndarray en "wav"
    assert "self.args.output_sample_rate" in xtts
    tokenizador = (REPO / "TTS/tts/layers/xtts/tokenizer.py").read_text(encoding="utf-8")
    limites = ast.literal_eval(re.search(r"self\.char_limits = (\{.*?\})", tokenizador, re.S).group(1))
    assert limites == _Tokenizer.char_limits

    gestor = (REPO / "TTS/utils/manage.py").read_text(encoding="utf-8")
    assert 'os.environ.get("COQUI_TOS_AGREED") == "1"' in gestor
    assert "def device(self)" in (REPO / "TTS/model.py").read_text(encoding="utf-8")
    assert "self.synthesizer = Synthesizer(" in (REPO / "TTS/api.py").read_text(encoding="utf-8")


def test_version_fijada_en_la_celda_1_es_la_revisada():
    version = re.search(r'^version = "(.+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert f'VERSION_COQUI_TTS = "{version}"' in celdas_de_codigo()["entorno"]
    iniciador = (REPO / "TTS/__init__.py").read_text(encoding="utf-8")
    assert 'is_torch_greater_or_equal("2.9") and not is_torchcodec_available()' in iniciador
