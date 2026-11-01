"""ipywidgets simulado: Textarea con observe() y lo mínimo para mostrarlo."""


class Layout:
    def __init__(self, **atributos) -> None:
        self.__dict__.update(atributos)


class _Widget:
    def __init__(self, value="", **atributos) -> None:
        self._observadores = []
        self._value = value
        self.__dict__.update(atributos)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, nuevo) -> None:
        anterior, self._value = self._value, nuevo
        for funcion, nombres in self._observadores:
            if nombres in (None, "value"):
                funcion({"name": "value", "old": anterior, "new": nuevo, "owner": self})

    def observe(self, funcion, names=None) -> None:
        self._observadores.append((funcion, names))


class Textarea(_Widget):
    pass


class Label(_Widget):
    pass


class VBox:
    def __init__(self, children=()) -> None:
        self.children = tuple(children)
