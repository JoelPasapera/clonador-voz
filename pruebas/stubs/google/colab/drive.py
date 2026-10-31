"""google.colab.drive simulado: «monta» creando /content/drive/MyDrive; puede fallar como el real («mount failed»)."""

import os

montajes = []  # (punto de montaje, force_remount) de cada llamada
fallos_pendientes = 0  # cuántas llamadas seguidas fallarán, como cuando no se conceden todos los permisos
montado = False  # conftest hace que os.path.ismount("/content/drive") devuelva esto


def mount(mountpoint, force_remount=False, timeout_ms=120000, readonly=False):
    global fallos_pendientes, montado
    montajes.append((mountpoint, force_remount))
    if fallos_pendientes > 0:
        fallos_pendientes -= 1
        raise ValueError("mount failed")
    os.makedirs(os.path.join(mountpoint, "MyDrive"), exist_ok=True)
    montado = True
