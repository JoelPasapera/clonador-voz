"""Idiomas mezclados: tramos (celda 2) y su paso por la generación (celda 5)."""

DISPONIBLES = frozenset({"es", "en", "fr", "pt", "it", "de"})


def tramos(funciones, texto, principal="es"):
    resultado, avisos = funciones.ns["dividir_por_idioma"](texto, principal, DISPONIBLES)
    return [(t.texto, t.idioma) for t in resultado], avisos


def test_tramos_basicos(funciones):
    assert tramos(funciones, "Repite: [en]Where is the library?[/en] Muy bien.") == (
        [("Repite: ", "es"), ("Where is the library?", "en"), (" Muy bien.", "es")],
        [],
    )


def test_la_puntuacion_tras_el_cierre_pasa_al_tramo_anterior(funciones):
    assert tramos(funciones, "Di [en]hello[/en]. Muy bien.")[0] == [("Di ", "es"), ("hello.", "en"), (" Muy bien.", "es")]
    assert tramos(funciones, "«[en]Thanks[/en]», dijo.")[0] == [("«", "es"), ("Thanks»,", "en"), (" dijo.", "es")]


def test_etiquetas_sin_cierre_mayusculas_y_seguidas(funciones):
    assert tramos(funciones, "Hola [EN]Good morning everyone")[0] == [("Hola ", "es"), ("Good morning everyone", "en")]
    assert tramos(funciones, "[en]One two three.[/en] [fr]Un deux trois.[/fr]")[0] == [
        ("One two three. ", "en"),
        ("Un deux trois.", "fr"),
    ]


def test_idioma_no_disponible_se_avisa_y_se_lee_en_el_principal(funciones):
    resultado, avisos = tramos(funciones, "Hola [ja]konnichiwa[/ja] amigos")
    assert resultado == [("Hola konnichiwa amigos", "es")]
    assert len(avisos) == 1 and "«ja»" in avisos[0]


def test_las_marcas_de_pausa_no_son_etiquetas(funciones):
    assert tramos(funciones, "Uno [pausa 2] dos [pausa] tres")[0] == [("Uno [pausa 2] dos [pausa] tres", "es")]


def opciones(funciones, pausa=0.3, cifras=True, reglas=""):
    ns = funciones.ns
    pronunciador = ns["Pronunciador"](ns["leer_reglas"](reglas)[0]) if reglas else None
    return ns["OpcionesTexto"](pausa_frase=pausa, cifras=cifras, pronunciador=pronunciador)


def test_segmentos_con_idioma_y_pausas_en_las_fronteras(con_voz, capsys):
    con_voz.definir("generar")  # solo las funciones de la celda 5, sin generar
    preparar = con_voz.ns["preparar_segmentos"]
    texto = (
        "Escucha y repite con atención: [en]Where is the library, please?[/en] Ahora en francés: "
        "[fr]Où est la bibliothèque ?[/fr] [pausa 2] Muy bien, sigamos.\n[en]See you tomorrow, everybody![/en]"
    )
    segmentos = preparar(texto, "es", opciones(con_voz, reglas="library = laibreri; sigamos = sigamos ya"))
    assert [(s.texto, s.idioma, s.pausa) for s in segmentos] == [
        ("Escucha y repite con atención:", "es", 0.3),
        ("Where is the library, please?", "en", 0.3),  # las reglas solo tocan el idioma principal
        ("Ahora en francés:", "es", 0.3),
        ("Où est la bibliothèque ?", "fr", 0.3 + 2.0),  # pausa de frase + [pausa 2]
        ("Muy bien, sigamos ya.", "es", 0.6),  # salto de línea: pausa de párrafo
        ("See you tomorrow, everybody!", "en", 0.0),
    ]
    salida = capsys.readouterr().out
    assert "🌐 Idiomas: es (3 fragmento(s)) · en (2 fragmento(s)) · fr (1 fragmento(s))" in salida
    assert "🔤 Pronunciación: 2 regla(s) · 1 reemplazo(s)" in salida


def test_cifras_en_tramos_en_espanol_aunque_el_principal_sea_otro(con_voz):
    con_voz.definir("generar")
    preparar = con_voz.ns["preparar_segmentos"]
    segmentos = preparar("The course costs $20 here. [es]En Lima cuesta S/ 1,500.50 al mes.[/es]", "en", opciones(con_voz))
    assert [(s.texto, s.idioma) for s in segmentos] == [
        ("The course costs $20 here.", "en"),  # en inglés lo resuelve el propio modelo
        ("En Lima cuesta mil quinientos soles con cincuenta céntimos al mes.", "es"),
    ]


def test_idiomas_mezclados_en_el_flujo(con_voz):
    from TTS import api

    con_voz.ejecutar(
        "generar",
        texto="La palabra del día es: [en]library, a place full of books.[/en] Repite conmigo, por favor.",
        formato="wav",
        guardar_en_drive=False,
    )
    pedidos = [(llamada["texto"], llamada["idioma"]) for llamada in api.registro["inferencias"]]
    assert pedidos == [
        ("La palabra del día es:", "es"),
        ("library, a place full of books.", "en"),
        ("Repite conmigo, por favor.", "es"),
    ]
    assert con_voz.ns["ULTIMA_GENERACION"].tomas[0].archivo.is_file()
