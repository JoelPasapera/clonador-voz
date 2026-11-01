"""train_gpt simulado: misma firma y mismos valores de retorno que la demo de afinado de coqui-tts 0.28.0.

Lee los CSV como el formateador «coqui» (csv.DictReader con «|»), «descarga» lo que falte en la carpeta de archivos
originales y deja un best_model.pth con la estructura del entrenador (claves «xtts.», partes de entrenamiento,
optimizador y un objeto de configuración que obliga a weights_only=False).
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch

llamadas = []
ORIGINALES = ("dvae.pth", "mel_stats.pth", "vocab.json", "model.pth", "config.json")


class ConfiguracionDelEntrenador:
    epocas = 0


def _leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo, delimiter="|"))


def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995):
    salida = os.path.join(output_path, "run", "training")
    originales = Path(salida, "XTTS_v2.0_original_model_files")
    originales.mkdir(parents=True, exist_ok=True)
    descargados = [nombre for nombre in ORIGINALES if not (originales / nombre).is_file()]
    for nombre in descargados:
        (originales / nombre).write_text("descargado de coqui/XTTS-v2", encoding="utf-8")
    entrenamiento, evaluacion = _leer(train_csv), _leer(eval_csv)
    raiz = os.path.dirname(train_csv)
    for fila in entrenamiento + evaluacion:
        assert set(fila) == {"audio_file", "text", "speaker_name"}, fila
        assert os.path.exists(os.path.join(raiz, fila["audio_file"])), fila["audio_file"]
    llamadas.append({
        "idioma": language, "epocas": num_epochs, "lote": batch_size, "acumulacion": grad_acumm,
        "entrenamiento": len(entrenamiento), "evaluacion": len(evaluacion), "descargados": descargados,
        "max_audio_length": max_audio_length,
    })
    if batch_size >= 4:  # como en una T4: el entrenador imprime el error y termina con sys.exit(1)
        print("torch.OutOfMemoryError: CUDA out of memory.")
        raise SystemExit(1)
    carpeta = Path(salida, "GPT_XTTS_FT-October-31-2026_10+00AM-0000000")
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    modelo = {
        "xtts.gpt.peso": torch.Tensor(rng.standard_normal((4, 4)).astype(np.float32)),
        "xtts.gpt.posiciones": torch.Tensor(np.arange(6, dtype=np.int64)),
        "xtts.hifigan_decoder.peso": torch.Tensor(rng.standard_normal(8).astype(np.float32)),
        "dvae.codebook": torch.Tensor(np.ones(3, np.float32)),
        "torch_mel_spectrogram_dvae.mel": torch.Tensor(np.ones(2, np.float32)),
        "torch_mel_spectrogram_style_encoder.mel": torch.Tensor(np.ones(2, np.float32)),
    }
    torch.save({"model": modelo, "optimizer": [torch.Tensor(np.zeros(4))], "config": ConfiguracionDelEntrenador()}, carpeta / "best_model.pth")
    mas_larga = max(entrenamiento, key=lambda fila: len(fila["text"].split(" ")))
    return (
        str(originales / "config.json"),
        str(originales / "model.pth"),
        str(originales / "vocab.json"),
        str(carpeta),
        os.path.join(raiz, mas_larga["audio_file"]),
    )
