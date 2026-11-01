# Clonador de voz · XTTS-v2 y Chatterbox en Google Colab

| Cuaderno | Para qué | |
|---|---|---|
| **Clonador de voz** | Leer cualquier texto con tu voz | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoelPasapera/clonador-voz/blob/main/Clonador_de_voz_XTTS.ipynb) |
| **Entrenar XTTS-v2 con tu voz** | Opcional: afinar el modelo con tus grabaciones para un parecido mayor | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoelPasapera/clonador-voz/blob/main/Entrenar_XTTS_con_tu_voz.ipynb) |

Cuadernos de Google Colab (GPU) que leen cualquier texto **con tu propia voz** a partir de una grabación tuya en Google Drive, o subida desde tu computadora. Cada celda muestra solo su formulario; el código está oculto pero se puede editar.

**Dos modelos, a elegir en la celda 1:**

| Modelo | Licencia | Para qué |
|---|---|---|
| **XTTS-v2** (coqui-tts) | CPML: solo **uso no comercial** | El mejor parecido; voces de estudio para diálogos; se puede **afinar con tu voz** (segundo cuaderno). |
| **Chatterbox Multilingual** (Resemble AI) | MIT: **uso comercial** permitido | 23 idiomas y control de expresividad. Sus audios llevan una marca de agua inaudible (Perth). |

**Qué incluye**

- **Texto:** cuadro de varias líneas (o un `.txt` de Drive), pausas `[pausa 2]`, otros idiomas con tu voz `[en]…[/en]`, diccionario de pronunciación y lectura correcta de montos en soles y dólares.
- **Voces:** tu voz se guarda como **perfil** en Drive y se carga en segundos; **diálogos** con `[voz: Nombre]…[/voz]` usando voces incluidas o tus perfiles.
- **Calidad:** varias **tomas** ordenadas por calidad y parecido con tu voz; **control de calidad** automático de cada fragmento (ritmo y silencios, y opcionalmente **Whisper**) que regenera lo que falla.
- **Salida:** MP3, WAV u OGG con volumen parejo en LUFS y limitador; **subtítulos `.srt`** sincronizados; **música de fondo** que baja sola cuando hablas; **modo lote** (un audio por bloque separado con `---`).
- **Arranque rápido:** instala con [uv](https://github.com/astral-sh/uv) (y pip como respaldo) y no reinstala nada en la misma sesión.

## Cómo usarlo

1. Pulsa el botón **Abrir en Colab** del cuaderno que quieras. No hace falta descargar nada ni autorizar GitHub.
2. Sigue las instrucciones del propio cuaderno. En el clonador puedes ejecutar todas las celdas a la vez con **Ctrl + F9**; el de entrenamiento conviene ejecutarlo celda por celda.

## Tu copia personal

Este repositorio es público y funciona como plantilla. Lo que escribes en los formularios (el nombre de tu audio, tus reglas de pronunciación) forma parte del cuaderno, así que **si lo guardas en GitHub se vuelve público**. Para tu uso diario:

1. Ábrelo con el botón y elige *Archivo → Guardar una copia en Drive*.
2. Trabaja siempre con esa copia (queda en tu Drive, en la carpeta «Colab Notebooks»): ahí se conservan tus valores y nadie más los ve.
3. Cuando este repositorio tenga una versión nueva, vuelve a guardar una copia en Drive.

Haz lo mismo con el cuaderno de entrenamiento si lo usas. Usa *Archivo → Guardar una copia en GitHub* solo para mejoras generales de los cuadernos. El texto del cuadro de la celda 5 no se guarda en el cuaderno, y el cuaderno tampoco guarda las salidas de las celdas (*Editar → Configuración del cuaderno → Omitir el resultado de las celdas de código al guardar*), para que los reproductores con tu voz nunca terminen en un archivo. Confirma que esa casilla esté marcada la primera vez. Tus perfiles de voz y modelos afinados quedan solo en tu Drive (`Mi unidad/Clonador de voz`).

## Qué contiene

| Ruta | Contenido |
|---|---|
| `Clonador_de_voz_XTTS.ipynb` | El clonador: entorno (1), funciones (2a–2c), modelo (3), voz (4), texto (5), generación (6) y elección de tomas (7). |
| `Entrenar_XTTS_con_tu_voz.ipynb` | El afinado: entorno y funciones (los mismos del clonador), preparación de tus grabaciones con Whisper (3), entrenamiento (4) y guardado en Drive (5). |
| `pruebas/` | Pruebas con pytest que ejecutan las celdas reales de ambos cuadernos en un Colab simulado (Drive, GPU, XTTS, Chatterbox y Whisper falsos; FFmpeg real), más pruebas de fidelidad contra el código fuente real de cada librería. |
| `pruebas/real/` | Prueba real: las celdas con las librerías y los modelos de verdad, en CPU. |
| `.github/workflows/pruebas.yml` | Corre las pruebas en cada push, incluidos los guardados desde Colab. |
| `.github/workflows/prueba-real.yml` | Corre la prueba real cada lunes (y cuando quieras, desde la pestaña *Actions*): avisa si una versión nueva de una librería rompió el clonador. |

## Pruebas en tu computadora

```bash
python -m pip install numpy pytest          # y FFmpeg instalado en el sistema
sudo mkdir -p /content && sudo chown "$USER" /content   # las celdas usan las rutas de Colab
python -m pytest pruebas -q
```

Para comparar los simuladores con las librerías reales, clona `idiap/coqui-ai-TTS`, `resemble-ai/chatterbox` y `SYSTRAN/faster-whisper` en las versiones que fija la celda 1 (los comandos exactos están en `pruebas.yml`) y define `COQUI_TTS_SRC`, `CHATTERBOX_SRC` y `FASTER_WHISPER_SRC` con sus rutas; sin ellas, esas pruebas se omiten.

## Licencias y uso responsable

- **Código:** licencia [MIT](LICENSE).
- **Modelos:** no están en este repositorio; Colab los descarga al ejecutar la celda 3.
  - **XTTS-v2** se distribuye con la [Coqui Public Model License (CPML)](https://tts-hub.github.io/cpml), que solo permite **uso no comercial** del modelo, de los audios que genera y de los modelos que afines a partir de él.
  - **Chatterbox** tiene [licencia MIT](https://github.com/resemble-ai/chatterbox/blob/master/LICENSE), que permite el uso comercial. Sus audios llevan una marca de agua inaudible que los identifica como generados por IA.
- Clona únicamente tu voz o la de alguien que te haya dado su permiso.
