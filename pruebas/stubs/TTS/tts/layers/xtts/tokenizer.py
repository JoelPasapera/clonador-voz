"""multilingual_cleaners simulado: minúsculas y sin comillas (la real además expande números y abreviaturas)."""


def multilingual_cleaners(text, lang):
    return " ".join(text.replace('"', "").lower().split())
