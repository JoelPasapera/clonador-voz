# Clonador de voz · XTTS-v2 en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RicardoMendoza2027/clonador-voz/blob/main/Clonador_de_voz_XTTS.ipynb)

Cuaderno de Google Colab (GPU) que lee cualquier texto **con tu propia voz** a partir de una grabación tuya en Google Drive, o subida desde tu computadora, usando el modelo XTTS-v2. Cada celda muestra solo su formulario; el código está oculto pero se puede editar.

## Abrirlo desde este repositorio privado

1. Pulsa el botón **Abrir en Colab**.
2. La primera vez, Colab te pide autorizar GitHub. Acepta y dale acceso a tus repositorios privados. Si prefieres abrirlo desde Colab: *Archivo → Abrir cuaderno → GitHub*, marca **Incluir repositorios privados** y elige este repositorio.
3. Usa el cuaderno normalmente. Lo que cambies se queda en tu sesión hasta que lo guardes.

## Guardar cambios

- *Archivo → Guardar una copia en GitHub* hace un commit en este repositorio (rama `main`).
- Los valores de los formularios (nombre de tu audio, carpeta, ajustes) se guardan dentro del cuaderno. Para no volver a escribirlos cada vez, guárdalos una vez con *Guardar una copia en GitHub* y quedarán como valores por defecto.
- El cuaderno está configurado para **no guardar las salidas de las celdas** (*Editar → Configuración del cuaderno → Omitir el resultado de las celdas de código al guardar*). Así el repositorio queda limpio y los reproductores con tu voz no terminan en GitHub. La primera vez, confirma que esa casilla esté marcada.

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

## Licencia del modelo y uso responsable

El modelo XTTS-v2 no está en este repositorio: Colab lo descarga al ejecutar la celda 3. Se distribuye con la [Coqui Public Model License (CPML)](https://tts-hub.github.io/cpml), que solo permite **uso no comercial** del modelo y de los audios que genera. Clona únicamente tu voz o la de alguien que te haya dado su permiso.
