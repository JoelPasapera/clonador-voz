"""El .ipynb es válido para Colab: estructura, modo formulario y parámetros con tipos y rangos correctos."""

import ast
import json
import re

import pytest

from conftest import CUADERNO, CUADERNO_ENTRENAMIENTO, celdas_de_codigo

PARAM = re.compile(r"^(?P<nombre>\w+)\s*=\s*(?P<valor>.+?)\s*# @param\s*(?P<spec>.+)$")


ESTRUCTURAS = {
    CUADERNO: ["intro", "entorno", "archivos", "audio", "texto", "modelo", "referencia", "guion", "generar", "elegir", "ayuda"],
    CUADERNO_ENTRENAMIENTO: [
        "intro_entrenamiento", "entorno", "archivos", "audio", "texto", "datos", "entrenamiento", "guardar_modelo",
        "ayuda_entrenamiento",
    ],
}


@pytest.mark.parametrize("ruta", list(ESTRUCTURAS), ids=lambda ruta: ruta.stem)
def test_estructura_nbformat_y_gpu(ruta):
    cuaderno = json.loads(ruta.read_text(encoding="utf-8"))
    assert cuaderno["nbformat"] == 4
    assert cuaderno["metadata"]["accelerator"] == "GPU"
    assert cuaderno["metadata"]["colab"]["gpuType"] == "T4" and cuaderno["metadata"]["colab"]["private_outputs"] is True
    assert cuaderno["metadata"]["colab"]["name"] == ruta.name
    ids = [c["metadata"]["id"] for c in cuaderno["cells"]]
    assert ids == ESTRUCTURAS[ruta]  # el entrenamiento NO está en el clonador: «Ejecutar todas» nunca entrena
    tipos = [c["cell_type"] for c in cuaderno["cells"]]
    assert tipos == ["markdown"] + ["code"] * (len(ids) - 2) + ["markdown"]
    for c in cuaderno["cells"]:
        assert all(isinstance(linea, str) for linea in c["source"])
        assert all(linea.endswith("\n") for linea in c["source"][:-1])
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            assert c["metadata"]["cellView"] == "form"
            assert re.match(r'# @title \d[a-c]? · .+ \{ display-mode: "form" \}\n$', c["source"][0])


def test_licencia_marcada_por_defecto_y_explicada():
    fuente = celdas_de_codigo()["modelo"]
    assert re.search(r'^uso_no_comercial = True  # @param \{type:"boolean"\}$', fuente, re.M)
    markdown = " ".join(linea for linea in fuente.splitlines() if linea.startswith("# @markdown"))
    assert "uso asumido por defecto: no comercial" in markdown and "CPML" in markdown and "Desmárcala" in markdown
    assert "Chatterbox" in markdown  # la alternativa para uso comercial
    intro = "".join(json.loads(CUADERNO.read_text(encoding="utf-8"))["cells"][0]["source"])
    assert "asume ese uso por defecto" in intro and "licencia MIT" in intro and "marca de agua" in intro


def test_modelo_xtts_por_defecto():
    fuente = celdas_de_codigo()["entorno"]
    assert re.search(r'^modelo = "XTTS-v2 \(uso no comercial\)"  # @param \[', fuente, re.M)


def test_parametros_validos_y_usados():
    total = 0
    for celda, fuente in celdas_de_codigo().items():
        ast.parse(fuente)
        for linea in fuente.splitlines():
            m = PARAM.match(linea)
            if m is None:
                assert "@param" not in linea, f"@param mal formado en {celda}: {linea}"
                continue
            total += 1
            nombre, valor, spec = m["nombre"], ast.literal_eval(m["valor"]), m["spec"].strip()
            usos = len(re.findall(rf"\b{nombre}\b", fuente))
            assert usos >= 2, f"{celda}: el campo «{nombre}» no se usa"
            if spec.startswith("["):
                assert valor in ast.literal_eval(spec), f"{celda}.{nombre}: valor fuera de la lista"
                continue
            tipo = re.search(r'type:\s*"(\w+)"', spec).group(1)
            if tipo == "boolean":
                assert isinstance(valor, bool)
            elif tipo == "string":
                assert isinstance(valor, str)
            elif tipo == "integer":
                assert isinstance(valor, int) and not isinstance(valor, bool)
            elif tipo == "slider":
                limites = {k: float(v) for k, v in re.findall(r"(min|max|step):\s*(-?[\d.]+)", spec)}
                assert limites["min"] <= valor <= limites["max"], f"{celda}.{nombre} fuera de rango"
                pasos = (valor - limites["min"]) / limites["step"]
                assert abs(pasos - round(pasos)) < 1e-6, f"{celda}.{nombre} no cae en un paso del deslizador"
            else:
                raise AssertionError(f"tipo desconocido {tipo}")
    # 1 (entorno) + 2 (modelo) + 7 (referencia) + 7 (guion) + 21 (generar) + 2 (elegir) + entrenamiento: 4 + 3 + 1
    assert total == 48
