"""Imprime una constante de versión que fija el cuaderno (p. ej. VERSION_COQUI_TTS): la usan los flujos de GitHub."""

import json
import re
import sys
from pathlib import Path

CUADERNO = Path(__file__).resolve().parent.parent / "Clonador_de_voz_XTTS.ipynb"


def version(nombre: str) -> str:
    cuaderno = json.loads(CUADERNO.read_text(encoding="utf-8"))
    fuente = "".join("".join(celda["source"]) for celda in cuaderno["cells"] if celda["cell_type"] == "code")
    coincidencia = re.search(rf'^{re.escape(nombre)} = "([^"]+)"', fuente, re.M)
    if coincidencia is None:
        raise SystemExit(f"El cuaderno no define {nombre}")
    return coincidencia.group(1)


if __name__ == "__main__":
    print(version(sys.argv[1]))
