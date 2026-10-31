# Clonador de voz · XTTS-v2 en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoelPasapera/clonador-voz/blob/main/Clonador_de_voz_XTTS.ipynb)

Cuaderno de Google Colab (GPU) que lee cualquier texto **con tu propia voz** a partir de una grabación tuya en Google Drive, o subida desde tu computadora, usando el modelo XTTS-v2. Cada celda muestra solo su formulario; el código está oculto pero se puede editar.

**Qué incluye:** varias tomas por texto para elegir la mejor, diccionario de pronunciación, lectura correcta de montos en soles y dólares, frases en otro idioma con la misma voz (`[en]…[/en]`) y volumen parejo en LUFS con limitador de picos.

## Cómo usarlo

1. Pulsa el botón **Abrir en Colab**. No hace falta descargar nada ni autorizar GitHub.
2. Sigue las instrucciones del propio cuaderno: ejecuta las celdas en orden o todas a la vez con **Ctrl + F9**.

## Tu copia personal

Este repositorio es público y funciona como plantilla. Lo que escribes en los formularios (el nombre de tu audio, tus textos, tus reglas de pronunciación) forma parte del cuaderno, así que **si lo guardas en GitHub se vuelve público**. Para tu uso diario:

1. Ábrelo con el botón y elige *Archivo → Guardar una copia en Drive*.
2. Trabaja siempre con esa copia (queda en tu Drive, en la carpeta «Colab Notebooks»): ahí se conservan tus valores y nadie más los ve.
3. Cuando este repositorio tenga una versión nueva, vuelve a guardar una copia en Drive.

Usa *Archivo → Guardar una copia en GitHub* solo para mejoras generales del cuaderno. Además, el cuaderno no guarda las salidas de las celdas (*Editar → Configuración del cuaderno → Omitir el resultado de las celdas de código al guardar*), para que los reproductores con tu voz nunca terminen en un archivo. Confirma que esa casilla esté marcada la primera vez.

## Qué contiene

| Ruta | Contenido |
|---|---|
| `Clonador_de_voz_XTTS.ipynb` | El cuaderno: entorno, funciones (2a–2c), modelo, voz de referencia, generación y elección de tomas. |
| `pruebas/` | Pruebas con pytest que ejecutan las celdas reales en un Colab simulado (Drive, GPU y XTTS falsos; FFmpeg real). |
| `.github/workflows/pruebas.yml` | Corre las pruebas en cada push, incluidos los guardados desde Colab. |

## Pruebas en tu computadora

```bash
python -m pip install numpy pytest          # y FFmpeg instalado en el sistema
sudo mkdir -p /content && sudo chown "$USER" /content   # las celdas usan las rutas de Colab
python -m pytest pruebas -q
```

Para comparar los simuladores con la librería real, clona `idiap/coqui-ai-TTS` en la versión que fija la celda 1 y define `COQUI_TTS_SRC` con esa ruta; sin ella, esas pruebas se omiten.

## Licencias y uso responsable

- **Código:** licencia [MIT](LICENSE).
- **Modelo:** XTTS-v2 no está en este repositorio; Colab lo descarga al ejecutar la celda 3. Se distribuye con la [Coqui Public Model License (CPML)](https://tts-hub.github.io/cpml), que solo permite **uso no comercial** del modelo y de los audios que genera.
- Clona únicamente tu voz o la de alguien que te haya dado su permiso.
