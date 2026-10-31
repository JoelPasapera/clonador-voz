"""Montos, cifras y diccionario de pronunciación (celda 2)."""

import random
import re
import unicodedata

import pytest

# ── Números en palabras ─────────────────────────────────────────────────────────

CASOS_NUMEROS = {
    0: "cero",
    1: "uno",
    15: "quince",
    16: "dieciséis",
    21: "veintiuno",
    22: "veintidós",
    30: "treinta",
    31: "treinta y uno",
    100: "cien",
    101: "ciento uno",
    115: "ciento quince",
    200: "doscientos",
    999: "novecientos noventa y nueve",
    1000: "mil",
    1001: "mil uno",
    1500: "mil quinientos",
    2000: "dos mil",
    21000: "veintiún mil",
    31000: "treinta y un mil",
    100000: "cien mil",
    101000: "ciento un mil",
    999999: "novecientos noventa y nueve mil novecientos noventa y nueve",
    1_000_000: "un millón",
    1_000_001: "un millón uno",
    2_500_000: "dos millones quinientos mil",
    21_000_000: "veintiún millones",
    1_000_000_000: "mil millones",
    1_001_000_000: "mil un millones",
    10**12: "un billón",
    2 * 10**12 + 3: "dos billones tres",
}


@pytest.mark.parametrize(("numero", "esperado"), CASOS_NUMEROS.items())
def test_numero_en_palabras(funciones, numero, esperado):
    assert funciones.ns["numero_en_palabras"](numero) == esperado


@pytest.mark.parametrize(
    ("numero", "esperado"),
    [(1, "un"), (21, "veintiún"), (31, "treinta y un"), (101, "ciento un"), (1001, "mil un"), (2, "dos"), (1_000_000, "un millón")],
)
def test_apocope_ante_masculino(funciones, numero, esperado):
    assert funciones.ns["numero_en_palabras"](numero, apocope=True) == esperado


def test_numero_fuera_de_rango(funciones):
    with pytest.raises(ValueError):
        funciones.ns["numero_en_palabras"](-1)
    with pytest.raises(ValueError):
        funciones.ns["numero_en_palabras"](10**24)


# ── Montos ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cifra", "esperado"),
    [
        ("150", (150, 0)), ("1,500", (1500, 0)), ("1.500", (1500, 0)), ("1,500.50", (1500, 50)),
        ("1.500,50", (1500, 50)), ("12.5", (12, 50)), ("12,50", (12, 50)), ("0.05", (0, 5)),
        ("1,234,567.891", (1234567, 89)), ("12.345.678", (12345678, 0)), ("7", (7, 0)),
    ],
)
def test_parsear_monto(funciones, cifra, esperado):
    assert funciones.ns["parsear_monto"](cifra) == esperado


@pytest.mark.parametrize(
    ("entero", "centimos", "esperado"),
    [
        (1, 0, "un sol"),
        (0, 0, "cero soles"),
        (21, 0, "veintiún soles"),
        (1500, 50, "mil quinientos soles con cincuenta céntimos"),
        (0, 50, "cincuenta céntimos"),
        (2, 1, "dos soles con un céntimo"),
        (1_000_000, 0, "un millón de soles"),
        (3_000_000, 0, "tres millones de soles"),
        (1_200_000, 0, "un millón doscientos mil soles"),
    ],
)
def test_monto_en_soles(funciones, entero, centimos, esperado):
    ns = funciones.ns
    assert ns["monto_en_palabras"](entero, centimos, ns["SOLES"]) == esperado


def test_normalizar_cifras(funciones):
    normalizar = funciones.ns["normalizar_cifras"]
    texto, cambios = normalizar(
        "Pagué S/ 150, luego S/. 1,500.50 y S/12.5; el libro costó US$ 21 y la guía $1,000. "
        "Vinieron 1,200 personas. La versión 2.5 salió en 2026. Sin cifra: S/N y $."
    )
    assert texto == (
        "Pagué ciento cincuenta soles, luego mil quinientos soles con cincuenta céntimos y doce soles con cincuenta "
        "céntimos; el libro costó veintiún dólares y la guía mil dólares. "
        "Vinieron 1200 personas. La versión 2.5 salió en 2026. Sin cifra: S/N y $."
    )
    assert cambios == 6


# ── Reglas ──────────────────────────────────────────────────────────────────────


def test_leer_reglas(funciones):
    leer = funciones.ns["leer_reglas"]
    reglas, avisos = leer("# comentario\nDNI = de ene i; Jr. =  jirón \n\nsin igual\n= vacía\nDni = d n i")
    assert [(r.texto, r.lectura) for r in reglas] == [("Jr.", "jirón"), ("Dni", "d n i")]  # la última repetida gana
    assert len(avisos) == 2 and all("Regla ignorada" in aviso for aviso in avisos)


def test_pronunciador_palabras_completas_y_mayusculas(funciones):
    ns = funciones.ns
    reglas, _ = ns["leer_reglas"]("DNI = de ene i; Jr. = jirón; Machu = macho; Machu Picchu = machu pikchu")
    pronunciador = ns["Pronunciador"](reglas)
    texto, cambios = pronunciador.aplicar("Mi dni y mi DNI, pero no DNIs. Vivo en el Jr. Ica; visité Machu  Picchu y Machu.")
    assert texto == "Mi de ene i y mi de ene i, pero no DNIs. Vivo en el jirón Ica; visité machu pikchu y macho."
    assert cambios == 5


def test_pronunciador_unicode_nfc(funciones):
    ns = funciones.ns
    pronunciador = ns["Pronunciador"](ns["leer_reglas"]("Perú = pe ru")[0])
    texto, cambios = pronunciador.aplicar(unicodedata.normalize("NFD", "Viva el Perú."))
    assert (texto, cambios) == ("Viva el pe ru.", 1)


def test_pronunciador_sin_reglas(funciones):
    assert funciones.ns["Pronunciador"]([]).aplicar("Hola  mundo") == ("Hola mundo", 0)


def _reemplazo_ingenuo(lecturas: dict[str, str], texto: str) -> str:
    """Referencia directa: en cada inicio de palabra prueba las claves de la más larga a la más corta."""
    claves = sorted(lecturas, key=len, reverse=True)

    def es_palabra(caracter: str) -> bool:
        return re.match(r"\w", caracter) is not None

    salida, i = [], 0
    while i < len(texto):
        if i == 0 or not es_palabra(texto[i - 1]):
            for clave in claves:
                fin = i + len(clave)
                if texto[i:fin].lower() == clave and (fin == len(texto) or not es_palabra(texto[fin])):
                    salida.append(lecturas[clave])
                    i = fin
                    break
            else:
                salida.append(texto[i])
                i += 1
        else:
            salida.append(texto[i])
            i += 1
    return "".join(salida)


def test_trie_equivale_a_la_referencia_ingenua(funciones):
    """La regex compilada desde la trie debe dar exactamente lo mismo que el algoritmo directo."""
    ns = funciones.ns
    rng = random.Random(2026)
    alfabeto = "abAB./-"
    for _ in range(400):
        reglas = []
        for indice in range(rng.randint(1, 12)):
            palabras = ["".join(rng.choice(alfabeto) for _ in range(rng.randint(1, 3))) for _ in range(rng.randint(1, 2))]
            reglas.append(ns["Regla"](" ".join(palabras), f"<{indice}>"))
        texto = " ".join("".join(rng.choice(alfabeto) for _ in range(rng.randint(1, 4))) for _ in range(rng.randint(1, 15)))
        lecturas = {regla.texto.lower(): regla.lectura for regla in reglas}
        obtenido, _ = ns["Pronunciador"](reglas).aplicar(texto)
        assert obtenido == _reemplazo_ingenuo(lecturas, texto), (reglas, texto)
