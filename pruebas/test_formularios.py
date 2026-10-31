"""El .ipynb es válido para Colab: estructura, modo formulario y parámetros con tipos y rangos correctos."""

import ast
import json
import re

from conftest import CUADERNO, celdas_de_codigo

PARAM = re.compile(r"^(?P<nombre>\w+)\s*=\s*(?P<valor>.+?)\s*# @param\s*(?P<spec>.+)$")


def test_estructura_nbformat_y_gpu():
    cuaderno = json.loads(CUADERNO.read_text(encoding="utf-8"))
    assert cuaderno["nbformat"] == 4
    assert cuaderno["metadata"]["accelerator"] == "GPU"
    assert cuaderno["metadata"]["colab"]["gpuType"] == "T4"
    ids = [c["metadata"]["id"] for c in cuaderno["cells"]]
    assert len(ids) == len(set(ids))
    tipos = [c["cell_type"] for c in cuaderno["cells"]]
    assert tipos == ["markdown"] + ["code"] * 8 + ["markdown"]
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
    assert "Uso asumido por defecto: no comercial" in markdown and "CPML" in markdown and "Desmárcala" in markdown
    intro = json.loads(CUADERNO.read_text(encoding="utf-8"))["cells"][0]
    assert "asume ese uso por defecto" in "".join(intro["source"])


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
    assert total == 28  # 1 (modelo) + 5 (referencia) + 20 (generar) + 2 (elegir)
