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
