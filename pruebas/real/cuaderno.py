"""Ejecuta las celdas REALES del cuaderno fuera de Colab, con las librerías de verdad (pruebas reales semanales)."""

import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CUADERNO = RAIZ / "Clonador_de_voz_XTTS.ipynb"
MARCADOR = "# ─── Ejecución ───"


class Cuaderno:
    """Un espacio de nombres compartido por las celdas, como el kernel de Colab."""

    def __init__(self) -> None:
        cuaderno = json.loads(CUADERNO.read_text(encoding="utf-8"))
        self.fuentes = {c["metadata"]["id"]: "".join(c["source"]) for c in cuaderno["cells"] if c["cell_type"] == "code"}
        self.ns: dict = {"__name__": "__main__"}

    def _correr(self, codigo: str, nombre: str) -> None:
        exec(compile(codigo, f"<celda {nombre}>", "exec"), self.ns)  # noqa: S102 — código del propio cuaderno

    def ejecutar(self, *celdas: str) -> None:
        for celda in celdas:
            self._correr(self.fuentes[celda], celda)

    def definir(self, *celdas: str) -> None:
        """Solo las definiciones (lo que va antes del marcador de ejecución): sin formularios ni efectos."""
        for celda in celdas:
            definiciones, encontrado, _ = self.fuentes[celda].partition(MARCADOR)
            assert encontrado, f"la celda {celda} no tiene marcador de ejecución"
            self._correr(definiciones, celda)


class Revision:
    """Acumula comprobaciones: imprime cada una y al final dice si todas pasaron."""

    def __init__(self) -> None:
        self.fallas: list[str] = []

    def __call__(self, condicion: bool, mensaje: str) -> None:
        print(("✅ " if condicion else "❌ ") + mensaje, flush=True)
        if not condicion:
            self.fallas.append(mensaje)

    def resultado(self) -> int:
        if self.fallas:
            print(f"\n{len(self.fallas)} comprobación(es) fallaron:", *self.fallas, sep="\n  • ")
            return 1
        print("\nTodas las comprobaciones pasaron.")
        return 0
