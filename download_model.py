#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descargador de alta velocidad con reanudación y reporte en tiempo real
para Tencent Hunyuan3D-2 (4.93 GB safetensors) directo a Disco D:
"""

import os
import sys
import time
import json
from pathlib import Path
import requests

from studio_env import MODELS_ROOT, HUNYUAN_LOCAL_DIR

MODEL_DIR = HUNYUAN_LOCAL_DIR / "hunyuan3d-dit-v2-0-turbo"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FILE = MODEL_DIR / "model.fp16.safetensors"
CONFIG_FILE = MODEL_DIR / "config.yaml"
STATUS_FILE = MODELS_ROOT / "download_status.json"
LOCK_FILE = MODELS_ROOT / "download_status.lock"

URL_SAFETENSORS = "https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors"
URL_CONFIG = "https://huggingface.co/tencent/Hunyuan3D-2/resolve/main/hunyuan3d-dit-v2-0-turbo/config.yaml"

TOTAL_SIZE_EXPECTED = 4930777530 # 4.93 GB

def update_status(downloaded: int, total: int, speed_mb: float, eta_sec: float, msg: str, error: str = ""):
    percent = round((downloaded / total) * 100, 1) if total > 0 else 0
    data = {
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "downloaded_mb": round(downloaded / (1024 * 1024), 1),
        "total_mb": round(total / (1024 * 1024), 1),
        "percent": percent,
        "speed_mb": round(speed_mb, 2),
        "eta_seconds": int(eta_sec),
        "message": msg,
        "is_complete": downloaded >= total and total > 0 and not error,
        "error": error,
        "pid": os.getpid()
    }
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def download_config():
    if not CONFIG_FILE.exists() or CONFIG_FILE.stat().st_size == 0:
        print("[1/2] Descargando config.yaml...")
        r = requests.get(URL_CONFIG, timeout=15)
        r.raise_for_status()
        with open(CONFIG_FILE, "wb") as f:
            f.write(r.content)
        print(" -> config.yaml listo.")


def download_safetensors():
    # Comprobar si ya está completo
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size == TOTAL_SIZE_EXPECTED:
        print(f" -> model.fp16.safetensors ya está completamente descargado ({TOTAL_SIZE_EXPECTED / (1024**3):.2f} GB).")
        update_status(TOTAL_SIZE_EXPECTED, TOTAL_SIZE_EXPECTED, 0, 0, "Pesos listos en disco D:")
        return

    part_file = MODEL_DIR / "model.fp16.safetensors.part"
    existing_bytes = part_file.stat().st_size if part_file.exists() else 0

    headers = {}
    mode = "wb"
    if existing_bytes > 0:
        if existing_bytes < TOTAL_SIZE_EXPECTED:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
            print(f"[2/2] Reanudando descarga desde {existing_bytes / (1024**2):.1f} MB...")
        else:
            existing_bytes = 0

    t_start = time.time()
    last_log_time = t_start
    downloaded_this_session = 0

    print(f"[2/2] Descargando model.fp16.safetensors ({TOTAL_SIZE_EXPECTED / (1024**3):.2f} GB) a disco D:...")

    with requests.get(URL_SAFETENSORS, headers=headers, stream=True, timeout=20) as r:
        if r.status_code not in (200, 206):
            r.raise_for_status()

        total_bytes = TOTAL_SIZE_EXPECTED
        current_downloaded = existing_bytes

        with open(part_file, mode) as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024): # 4MB chunks
                if chunk:
                    f.write(chunk)
                    chunk_len = len(chunk)
                    current_downloaded += chunk_len
                    downloaded_this_session += chunk_len

                    now = time.time()
                    if now - last_log_time >= 2.0:
                        elapsed = now - t_start
                        speed_mb = (downloaded_this_session / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                        remaining_bytes = max(0, total_bytes - current_downloaded)
                        eta = remaining_bytes / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0

                        pct = (current_downloaded / total_bytes) * 100
                        msg = f"Descargando {current_downloaded / (1024**2):.0f} / {total_bytes / (1024**2):.0f} MB ({pct:.1f}%) a {speed_mb:.1f} MB/s"
                        print(f" -> {msg} | ETA: {int(eta // 60)}m {int(eta % 60):02d}s")
                        update_status(current_downloaded, total_bytes, speed_mb, eta, msg)
                        last_log_time = now

    # Validar tamaño final y renombrar
    if part_file.stat().st_size == TOTAL_SIZE_EXPECTED:
        if TARGET_FILE.exists():
            TARGET_FILE.unlink()
        part_file.rename(TARGET_FILE)
        print("\n" + "="*70)
        print(" 🎉 ¡DESCARGA DE PESOS COMPLETADA EXITOSAMENTE!")
        print(f" -> Ubicación: {TARGET_FILE}")
        print(f" -> Tamaño: {TARGET_FILE.stat().st_size / (1024**3):.2f} GB")
        print("="*70 + "\n")
        update_status(TOTAL_SIZE_EXPECTED, TOTAL_SIZE_EXPECTED, 0, 0, "✅ Pesos completos en disco D:")
    else:
        raise RuntimeError(f"Tamaño inconsistente: {part_file.stat().st_size} vs {TOTAL_SIZE_EXPECTED} bytes")


def acquire_lock() -> bool:
    """Impide dos descargas simultáneas escribiendo sobre el mismo .part (lo corromperían)."""
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Bloqueo huérfano de una descarga que murió: se recupera si ese proceso ya no existe
        try:
            import psutil
            old_pid = int(LOCK_FILE.read_text().strip() or 0)
            if old_pid and psutil.pid_exists(old_pid):
                return False
        except Exception:
            return False
        LOCK_FILE.unlink(missing_ok=True)
        return acquire_lock()


def main() -> int:
    """Descarga config.yaml y model.fp16.safetensors. Devuelve 0 si los pesos quedan completos."""
    if not acquire_lock():
        print("[Aviso] Ya hay otra descarga de Hunyuan3D-2 en curso. Saliendo.")
        return 0
    try:
        download_config()
        download_safetensors()
        return 0
    except Exception as e:
        print(f"[ERROR] La descarga de pesos falló: {e}")
        part = MODEL_DIR / "model.fp16.safetensors.part"
        done = part.stat().st_size if part.exists() else 0
        update_status(done, TOTAL_SIZE_EXPECTED, 0, 0, "Descarga fallida", error=str(e))
        return 1
    finally:
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
