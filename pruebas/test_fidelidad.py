"""Los simuladores y las llamadas del cuaderno coinciden con las librerías reales: se comparan con su código fuente,
clonado en las versiones que fija la celda 1. `COQUI_TTS_SRC`, `CHATTERBOX_SRC` y `FASTER_WHISPER_SRC` indican dónde
está cada uno (el flujo de GitHub Actions los clona solo); sin ellos, sus pruebas se omiten."""

import ast
import inspect
import json
import os
import re
from pathlib import Path

import pytest

from conftest import celdas_de_codigo


def raiz(variable: str, marca: str) -> Path | None:
    ruta = Path(os.environ[variable]) if os.environ.get(variable) else None
    return ruta if ruta is not None and (ruta / marca).exists() else None


COQUI = raiz("COQUI_TTS_SRC", "TTS")
CHATTERBOX = raiz("CHATTERBOX_SRC", "src/chatterbox")
WHISPER = raiz("FASTER_WHISPER_SRC", "faster_whisper")
con_coqui = pytest.mark.skipif(COQUI is None, reason="define COQUI_TTS_SRC con el código fuente de coqui-tts")
con_chatterbox = pytest.mark.skipif(CHATTERBOX is None, reason="define CHATTERBOX_SRC con el código de chatterbox")
con_whisper = pytest.mark.skipif(WHISPER is None, reason="define FASTER_WHISPER_SRC con el código de faster-whisper")


# ── Utilidades: firmas reales (por AST, sin importar nada) y llamadas del cuaderno ──


def definicion(archivo: Path, nombre: str, clase: str | None = None) -> ast.FunctionDef:
    arbol = ast.parse(archivo.read_text(encoding="utf-8"))
    cuerpo = arbol.body
    if clase is not None:
        cuerpo = next(n for n in ast.walk(arbol) if isinstance(n, ast.ClassDef) and n.name == clase).body
    return next(n for n in cuerpo if isinstance(n, ast.FunctionDef) and n.name == nombre)


def firma_real(archivo: Path, nombre: str, clase: str | None = None) -> list[str]:
    funcion = definicion(archivo, nombre, clase)
    a = funcion.args
    nombres = [x.arg for x in a.posonlyargs + a.args] + [f"*{x.arg}" for x in a.kwonlyargs]
    es_de_clase = any(isinstance(d, ast.Name) and d.id in ("classmethod", "staticmethod") for d in funcion.decorator_list)
    if es_de_clase and nombres and nombres[0] == "cls":
        nombres = nombres[1:]
    if a.vararg:
        nombres.append(f"*args:{a.vararg.arg}")
    if a.kwarg:
        nombres.append(f"**{a.kwarg.arg}")
    return nombres


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


def campos_de_clase(archivo: Path, clase: str) -> list[str]:
    nodo = next(n for n in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))) if isinstance(n, ast.ClassDef) and n.name == clase)
    return [n.target.id for n in nodo.body if isinstance(n, ast.AnnAssign)]


def llamadas_del_cuaderno(nombre: str) -> list[ast.Call]:
    """Todas las llamadas a `nombre(…)` o `x.nombre(…)` en las celdas de código."""
    llamadas = []
    for fuente in celdas_de_codigo().values():
        for nodo in ast.walk(ast.parse(fuente)):
            if isinstance(nodo, ast.Call):
                funcion = nodo.func
                if (isinstance(funcion, ast.Attribute) and funcion.attr == nombre) or (isinstance(funcion, ast.Name) and funcion.id == nombre):
                    llamadas.append(nodo)
    assert llamadas, f"el cuaderno no llama a {nombre}"
    return llamadas


def comprobar_llamadas(nombre: str, archivo: Path, clase: str | None = None, metodo: str | None = None) -> None:
    """Cada argumento con nombre que pasa el cuaderno existe en la función real, y no sobran posicionales."""
    funcion = definicion(archivo, metodo or nombre, clase)
    a = funcion.args
    posicionales = [x.arg for x in a.posonlyargs + a.args]
    if posicionales and posicionales[0] in ("self", "cls"):
        posicionales = posicionales[1:]
    validos = set(posicionales) | {x.arg for x in a.kwonlyargs}
    for llamada in llamadas_del_cuaderno(nombre):
        nombres = {k.arg for k in llamada.keywords if k.arg is not None}
        assert nombres <= validos or a.kwarg is not None, f"{nombre}: {sorted(nombres - validos)} no existen"
        assert nombres <= validos, f"{nombre}: {sorted(nombres - validos)} irían a **{a.kwarg.arg}"
        assert len(llamada.args) <= len(posicionales) or a.vararg is not None, f"{nombre}: demasiados posicionales"


def constante(fuente: str, nombre: str) -> str:
    return re.search(rf'^{nombre} = "([^"]+)"', fuente, re.M).group(1)


# ── coqui-tts ───────────────────────────────────────────────────────────────────


@con_coqui
@pytest.mark.parametrize("metodo", ["inference", "get_gpt_cond_latents", "get_speaker_embedding", "load_checkpoint"])
def test_firmas_de_xtts(metodo):
    from TTS.tts.models.xtts import Xtts

    assert firma_simulada(getattr(Xtts, metodo)) == firma_real(COQUI / "TTS/tts/models/xtts.py", metodo, "Xtts")


@con_coqui
def test_firma_de_tts_api():
    from TTS.api import TTS

    assert firma_simulada(TTS.__init__) == firma_real(COQUI / "TTS/api.py", "__init__", "TTS")


@con_coqui
def test_firmas_del_afinado():
    from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
    from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners
    from TTS.utils.manage import ModelManager

    demo = COQUI / "TTS/demos/xtts_ft_demo/utils/gpt_train.py"
    assert firma_simulada(train_gpt) == firma_real(demo, "train_gpt")
    assert firma_simulada(multilingual_cleaners) == firma_real(COQUI / "TTS/tts/layers/xtts/tokenizer.py", "multilingual_cleaners")
    assert firma_simulada(ModelManager.__init__) == firma_real(COQUI / "TTS/utils/manage.py", "__init__", "ModelManager")


@con_coqui
def test_llamadas_del_cuaderno_a_coqui():
    xtts = COQUI / "TTS/tts/models/xtts.py"
    comprobar_llamadas("inference", xtts, "Xtts")
    comprobar_llamadas("get_gpt_cond_latents", xtts, "Xtts")
    comprobar_llamadas("load_checkpoint", xtts, "Xtts")
    comprobar_llamadas("train_gpt", COQUI / "TTS/demos/xtts_ft_demo/utils/gpt_train.py")


@con_coqui
def test_datos_del_modelo_real():
    from TTS.api import NOMBRE_ESPERADO, _Tokenizer

    modelos = json.loads((COQUI / "TTS/.models.json").read_text(encoding="utf-8"))
    _, idioma, dataset, nombre = NOMBRE_ESPERADO.split("/")
    entrada = modelos["tts_models"][idioma][dataset][nombre]
    assert entrada["license"] == "CPML" and entrada["tos_required"] is True
    assert entrada["repo_id"] == "tts-hub/XTTS-v2" and "dvae.pth" in entrada["ignore"]  # de ahí sale dvae.pth (8b)

    xtts = (COQUI / "TTS/tts/models/xtts.py").read_text(encoding="utf-8")
    assert '"wav": torch.cat(wavs, dim=0).numpy()' in xtts  # inference() devuelve un ndarray en "wav"
    assert "self.args.output_sample_rate" in xtts
    assert "self.speaker_manager.speakers[speaker].values()" in xtts  # voces incluidas: (latente, embedding)
    tokenizador = (COQUI / "TTS/tts/layers/xtts/tokenizer.py").read_text(encoding="utf-8")
    limites = ast.literal_eval(re.search(r"self\.char_limits = (\{.*?\})", tokenizador, re.S).group(1))
    assert limites == _Tokenizer.char_limits

    gestor = (COQUI / "TTS/utils/manage.py").read_text(encoding="utf-8")
    assert 'os.environ.get("COQUI_TOS_AGREED") == "1"' in gestor
    assert 'model_full_name = f"{model_type}--{lang}--{dataset}--{model}"' in gestor  # carpeta que busca la 8b
    assert "def device(self)" in (COQUI / "TTS/model.py").read_text(encoding="utf-8")
    assert "self.synthesizer = Synthesizer(" in (COQUI / "TTS/api.py").read_text(encoding="utf-8")
    voces = (COQUI / "TTS/tts/layers/xtts/xtts_manager.py").read_text(encoding="utf-8")
    assert "self.speakers = torch.load(speaker_file_path" in voces


@con_coqui
def test_xtts_se_descarga_en_la_cache_de_hugging_face():
    """El entrenamiento toma los archivos originales de la caché de Hugging Face porque coqui-tts 0.28.0 descarga los
    modelos con «repo_id» mediante snapshot_download (no en ~/.local/share/tts, donde solo deja la licencia)."""
    from TTS.utils.manage import ModelManager

    gestor = (COQUI / "TTS/utils/manage.py").read_text(encoding="utf-8")
    assert re.search(r'elif "repo_id" in model_item:\s+output_path = self\._download_hf_model\(model_item\)', gestor)
    assert "snapshot_download(" in gestor and 'ignore_patterns=model_item.get("ignore")' in gestor
    assert "self.models_dict = json.loads(" in gestor
    real = json.loads((COQUI / "TTS/.models.json").read_text(encoding="utf-8"))["tts_models"]["multilingual"]["multi-dataset"]["xtts_v2"]
    simulada = ModelManager().models_dict["tts_models"]["multilingual"]["multi-dataset"]["xtts_v2"]
    assert (simulada["repo_id"], simulada["ignore"]) == (real["repo_id"], real["ignore"])


@con_coqui
def test_compactado_igual_al_de_coqui():
    """La celda 8c deja exactamente lo que Xtts.get_compatible_checkpoint_state_dict conservaría."""
    xtts = (COQUI / "TTS/tts/models/xtts.py").read_text(encoding="utf-8")
    ignoradas = ast.literal_eval(re.search(r"ignore_keys = (\[.*?\])", xtts).group(1))
    claves = ast.literal_eval(re.search(r"CLAVES_DEL_ENTRENADOR = (\(.*?\))", celdas_de_codigo()["guardar_modelo"]).group(1))
    assert sorted(ignoradas) == sorted(clave.rstrip(".") for clave in claves)
    assert 'if key.startswith("xtts."):' in xtts and '["model"]' in xtts


@con_coqui
def test_archivos_que_espera_train_gpt():
    demo = (COQUI / "TTS/demos/xtts_ft_demo/utils/gpt_train.py").read_text(encoding="utf-8")
    assert 'os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")' in demo
    assert 'OUT_PATH = os.path.join(output_path, "run", "training")' in demo
    for archivo in ("dvae.pth", "mel_stats.pth", "vocab.json", "model.pth", "config.json"):
        assert f"/main/{archivo}" in demo
    assert "return XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref" in demo
    formateador = (COQUI / "TTS/tts/datasets/formatters.py").read_text(encoding="utf-8")
    coqui = formateador[formateador.index("def coqui(") : formateador.index("def tweb(")]
    assert 'csv.DictReader(f, delimiter="|")' in coqui and 'row["audio_file"]' in coqui and 'row["text"]' in coqui


@con_coqui
def test_version_fijada_en_la_celda_1_es_la_revisada():
    version = re.search(r'^version = "(.+)"', (COQUI / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
    assert constante(celdas_de_codigo()["entorno"], "VERSION_COQUI_TTS") == version
    iniciador = (COQUI / "TTS/__init__.py").read_text(encoding="utf-8")
    assert 'is_torch_greater_or_equal("2.9") and not is_torchcodec_available()' in iniciador


# ── Chatterbox ──────────────────────────────────────────────────────────────────


@con_chatterbox
@pytest.mark.parametrize("metodo", ["generate", "prepare_conditionals", "from_pretrained"])
def test_firmas_de_chatterbox(metodo):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    real = firma_real(CHATTERBOX / "src/chatterbox/mtl_tts.py", metodo, "ChatterboxMultilingualTTS")
    assert firma_simulada(getattr(ChatterboxMultilingualTTS, metodo)) == real


@con_chatterbox
def test_estructuras_de_chatterbox():
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, VoiceEncoder

    mtl = CHATTERBOX / "src/chatterbox/mtl_tts.py"
    assert campos_de_clase(mtl, "Conditionals") == ["t3", "gen"]
    assert campos_de_clase(CHATTERBOX / "src/chatterbox/models/t3/modules/cond_enc.py", "T3Cond") == [
        "speaker_emb", "clap_emb", "cond_prompt_speech_tokens", "cond_prompt_speech_emb", "emotion_adv",
    ]
    codificador = CHATTERBOX / "src/chatterbox/models/voice_encoder/voice_encoder.py"
    assert firma_simulada(VoiceEncoder.embeds_from_wavs) == firma_real(codificador, "embeds_from_wavs", "VoiceEncoder")
    fuente = mtl.read_text(encoding="utf-8")
    reales = ast.literal_eval(re.search(r"SUPPORTED_LANGUAGES = (\{.*?\})", fuente, re.S).group(1))
    assert reales == SUPPORTED_LANGUAGES
    assert "self.sr = S3GEN_SR" in fuente and "weights_only=True" in fuente  # el perfil guardado debe cargar así
    assert "self.conds = Conditionals(t3_cond, s3gen_ref_dict)" in fuente
    constantes = (CHATTERBOX / "src/chatterbox/models/s3gen/const.py").read_text(encoding="utf-8")
    assert "S3GEN_SR = 24000" in constantes
    assert re.search(r"MotorChatterbox.*?frecuencia_referencia = 24000", celdas_de_codigo()["modelo"], re.S)


@con_chatterbox
def test_llamadas_del_cuaderno_a_chatterbox():
    mtl = CHATTERBOX / "src/chatterbox/mtl_tts.py"
    comprobar_llamadas("generate", mtl, "ChatterboxMultilingualTTS")
    comprobar_llamadas("prepare_conditionals", mtl, "ChatterboxMultilingualTTS")
    comprobar_llamadas("from_pretrained", mtl, "ChatterboxMultilingualTTS")
    comprobar_llamadas("embeds_from_wavs", CHATTERBOX / "src/chatterbox/models/voice_encoder/voice_encoder.py", "VoiceEncoder")


@con_chatterbox
def test_version_de_chatterbox_y_su_torch():
    proyecto = (CHATTERBOX / "pyproject.toml").read_text(encoding="utf-8")
    entorno = celdas_de_codigo()["entorno"]
    assert constante(entorno, "VERSION_CHATTERBOX") == re.search(r'^version = "(.+)"', proyecto, re.M).group(1)
    assert '"torch==2.6.0; python_version < \'3.14\'"' in proyecto  # Colab usa 3.13: torch 2.6 …
    assert constante(entorno, "TORCHVISION_DE_CHATTERBOX") == "0.21.0"  # … y su torchvision es la 0.21


# ── faster-whisper ──────────────────────────────────────────────────────────────


@con_whisper
def test_llamadas_del_cuaderno_a_whisper():
    transcribe = WHISPER / "faster_whisper/transcribe.py"
    comprobar_llamadas("WhisperModel", transcribe, "WhisperModel", metodo="__init__")
    comprobar_llamadas("transcribe", transcribe, "WhisperModel")
    assert campos_de_clase(transcribe, "Word")[:3] == ["start", "end", "word"]
    assert "words" in campos_de_clase(transcribe, "Segment") and "text" in campos_de_clase(transcribe, "Segment")


@con_whisper
def test_modelos_y_version_de_whisper():
    utilidades = (WHISPER / "faster_whisper/utils.py").read_text(encoding="utf-8")
    modelos = ast.literal_eval(re.search(r"_MODELS = (\{.*?\})", utilidades, re.S).group(1))
    celdas = celdas_de_codigo()
    assert constante(celdas["audio"], "MODELO_WHISPER") in modelos
    assert constante(celdas["datos"], "MODELO_WHISPER_ENTRENAMIENTO") in modelos
    version = re.search(r'__version__ = "(.+)"', (WHISPER / "faster_whisper/version.py").read_text(encoding="utf-8")).group(1)
    assert constante(celdas["entorno"], "VERSION_FASTER_WHISPER") == version
