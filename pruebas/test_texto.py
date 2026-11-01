"""División del texto (celda 2): frases, límites del modelo, marcas de pausa y párrafos."""

import random
import re


def textos(segmentos):
    return [s.texto for s in segmentos]


def pausas(segmentos):
    return [s.pausa for s in segmentos]


def test_frases_y_union_de_cortas(funciones):
    dividir = funciones.ns["dividir_texto"]
    segmentos = dividir("Hola, ¿cómo estás? Muy bien. Hoy es un gran día para aprender.", 229, 0.3)
    assert textos(segmentos) == ["Hola, ¿cómo estás? Muy bien.", "Hoy es un gran día para aprender."]
    assert pausas(segmentos) == [0.3, 0.0]


def test_abreviaturas_no_dejan_fragmentos_sueltos(funciones):
    dividir = funciones.ns["dividir_texto"]
    segmentos = dividir("El Dr. Pérez llegó a las 3.5 horas. Luego habló con la Sra. Ruiz sobre el proyecto.", 229, 0.3)
    assert textos(segmentos) == ["El Dr. Pérez llegó a las 3.5 horas.", "Luego habló con la Sra. Ruiz sobre el proyecto."]


def test_no_corta_en_abreviaturas_a_mitad_de_frase(funciones):
    dividir = funciones.ns["dividir_texto"]
    primera = (
        "Ayer en la tarde conversé largamente con el Dr. Pérez sobre la matrícula del curso, "
        "que cuesta S/. 150 al mes."
    )
    segunda = "Después pasamos por la Av. Grau con la Sra. Ruiz y su hijo menor."
    tercera = "El libro de J. K. Rowling, p. ej., se lee mucho en el colegio."
    assert textos(dividir(f"{primera} {segunda} {tercera}", 229, 0.3)) == [primera, segunda, tercera]


def test_finales_reales_si_cortan(funciones):
    dividir = funciones.ns["dividir_texto"]
    segmentos = dividir("Le dije que no. Mañana vamos al cine con todos ellos. Compré frutas, verduras, etc. Luego volví a casa temprano.", 229, 0.3)
    assert textos(segmentos) == [
        "Le dije que no. Mañana vamos al cine con todos ellos.",
        "Compré frutas, verduras, etc.",
        "Luego volví a casa temprano.",
    ]


def test_frase_larga_respeta_el_limite_y_corta_en_comas(funciones):
    dividir = funciones.ns["dividir_texto"]
    frase = ", ".join(f"elemento número {i} de la lista" for i in range(30)) + "."
    segmentos = dividir(frase, 120, 0.3)
    assert len(segmentos) > 1
    assert all(len(s.texto) <= 120 for s in segmentos)
    assert all(s.texto.endswith((",", ".")) for s in segmentos)
    assert " ".join(textos(segmentos)) == frase


def test_marcas_de_pausa_y_parrafos(funciones):
    dividir = funciones.ns["dividir_texto"]
    texto = "Primer párrafo con texto suficiente.\nSegundo párrafo, también largo. [pausa 2,5] Después de la pausa larga. [PAUSA] Fin del texto aquí."
    segmentos = dividir(texto, 229, 0.3)
    assert textos(segmentos) == [
        "Primer párrafo con texto suficiente.",
        "Segundo párrafo, también largo.",
        "Después de la pausa larga.",
        "Fin del texto aquí.",
    ]
    assert pausas(segmentos) == [0.6, 2.5, 1.0, 0.0]


def test_palabra_enorme_se_corta_en_seco(funciones):
    dividir = funciones.ns["dividir_texto"]
    segmentos = dividir("a" * 450, 100, 0.3)
    assert [len(s.texto) for s in segmentos] == [100, 100, 100, 100, 50]


def test_texto_vacio(funciones):
    assert funciones.ns["dividir_texto"]("   \n [pausa]  ", 229, 0.3) == []


def test_aleatorio_no_pierde_palabras_ni_excede_limites(funciones):
    dividir = funciones.ns["dividir_texto"]
    rng = random.Random(7)
    palabras = ["casa", "voz", "modelo", "aprender", "Perú", "rápido", "señal", "¿qué", "hoy!", "3.5", "inglés"]
    signos = ["", "", "", ",", ".", "?", ";", "…", ":"]
    for _ in range(300):
        limite = rng.randint(40, 239)
        partes = []
        for _ in range(rng.randint(1, 120)):
            partes.append(rng.choice(palabras) + rng.choice(signos))
            if rng.random() < 0.03:
                partes.append(rng.choice(["[pausa]", "[pausa 1.5]", "\n"]))
        texto = " ".join(partes)
        segmentos = dividir(texto, limite, 0.3)
        assert all(0 < len(s.texto) <= limite for s in segmentos)
        esperado = re.findall(r"\S+", re.sub(r"\[pausa[^\]]*\]", " ", texto, flags=re.IGNORECASE))
        assert re.findall(r"\S+", " ".join(textos(segmentos))) == esperado


# ── v2: lotes, subtítulos y comparación de textos ───────────────────────────────


def test_bloques_con_titulos_y_separadores(funciones):
    bloques = funciones.ns["dividir_en_bloques"](
        "# Capítulo 1\nHola mundo.\n---\n   \n  ----  \n#Dos\nAdiós.\n---\n# Solo título\n---\nSin título."
    )
    assert [(b.titulo, b.texto.strip()) for b in bloques] == [("Capítulo 1", "Hola mundo."), ("Dos", "Adiós."), ("", "Sin título.")]
    assert [b.texto for b in funciones.ns["dividir_en_bloques"]("Un solo bloque - con guiones -- sueltos.")] == [
        "Un solo bloque - con guiones -- sueltos."
    ]


def test_subtitulos_srt(funciones):
    srt = funciones.ns["subtitulos_srt"]([
        (0.0, 3.2, "Esta es una frase bastante larga que no cabe en una sola línea de subtítulo, así que se parte en dos o más."),
        (3661.5, 3662.25, "Corta."),
    ])
    bloques = srt.strip().split("\n\n")
    assert bloques[0].split("\n") == [
        "1", "00:00:00,000 --> 00:00:02,294", "Esta es una frase bastante larga que", "no cabe en una sola línea de subtítulo,",
    ]  # «no» no queda colgando al final de la primera línea
    assert all(len(linea) <= 42 for bloque in bloques for linea in bloque.split("\n")[2:])
    assert all(len(bloque.split("\n")) <= 4 for bloque in bloques)  # número, tiempos y hasta 2 líneas
    assert bloques[1].split("\n")[1] == "00:00:02,294 --> 00:00:03,200"  # el tiempo se reparte por caracteres (76 + 30)
    assert bloques[2] == "3\n01:01:01,500 --> 01:01:02,250\nCorta."


def test_dos_lineas_parejas(funciones):
    texto = funciones.ns["_en_dos_lineas"]("uno dos tres cuatro cinco seis siete ocho nueve diez".split(), 42)
    primera, segunda = texto.split("\n")
    assert abs(len(primera) - len(segunda)) <= 6


def test_distancia_edicion_frente_a_la_tabla_clasica(funciones):
    import random

    distancia = funciones.ns["distancia_edicion"]

    def tabla(a, b):
        fila = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            nueva = [i]
            for j, cb in enumerate(b, 1):
                nueva.append(min(fila[j] + 1, nueva[j - 1] + 1, fila[j - 1] + (ca != cb)))
            fila = nueva
        return fila[-1]

    rng = random.Random(7)
    for _ in range(2000):
        a = "".join(rng.choice("abcñá ") for _ in range(rng.randrange(0, 70)))
        b = "".join(rng.choice("abcñá ") for _ in range(rng.randrange(0, 70)))
        assert distancia(a, b) == tabla(a, b) == distancia(b, a)
    assert distancia("x" * 300 + "hola", "hola" + "y" * 250) == tabla("x" * 300 + "hola", "hola" + "y" * 250)


def test_tasa_de_error_por_caracteres(funciones):
    tasa = funciones.ns["tasa_error_caracteres"]
    assert tasa("Cuesta mil quinientos soles.", "cuesta 1500 soles", "es") == 0.0  # cifras como palabras
    assert tasa("¡Hola, Señor Pérez!", "hola senor perez") == 0.0  # sin tildes ni signos
    assert tasa("hola mundo", "") == 1.0
    assert 0.0 < tasa("buenos días a todos", "buenos días a todo", "es") < 0.1


def test_lineas_de_subtitulo_sin_palabras_colgando(funciones):
    import random

    lineas_voraces = funciones.ns["_lineas_voraces"]
    assert lineas_voraces("y empieza el lunes, a las ocho de la mañana en el aula".split(), 33) == [
        ["y", "empieza", "el", "lunes,", "a", "las", "ocho"], ["de", "la", "mañana", "en", "el", "aula"],
    ]
    assert funciones.ns["_en_dos_lineas"]("Hoy la matrícula cuesta S/ 1,500.50 y empieza el lunes.".split(), 42) == (
        "Hoy la matrícula cuesta\nS/ 1,500.50 y empieza el lunes."  # «S/» no se separa de su monto
    )
    rng = random.Random(3)
    palabras = "a el de la S/ y en un casa programa matrícula extraordinariamente supercalifragilisticoespialidoso".split()
    for _ in range(2000):
        texto = [rng.choice(palabras) for _ in range(rng.randrange(1, 40))]
        lineas = lineas_voraces(texto, 42)
        assert sum(lineas, []) == texto  # no se pierde ni se reordena nada
        assert all(len(" ".join(linea)) <= 42 or len(linea) == 1 for linea in lineas)
