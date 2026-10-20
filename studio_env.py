#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Configuración de Entorno y Rutas de Almacenamiento (fuente única)
==============================================================================
Carga el .env del proyecto y define TODAS las rutas de modelos y cachés a partir
de una sola raíz (HY3DGEN_MODELS), la que eligió setup.bat. Debe importarse antes
que torch / huggingface / rembg para que usen esas rutas.

Prioridad de cada variable: entorno del proceso > .env > valor derivado de la raíz.
Raíz por defecto si no hay .env: D:\\AI_Models si existe D:, si no <proyecto>\\AI_Models.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"


def load_env_file(path: Path = ENV_FILE) -> None:
    """Carga KEY=VALUE del .env sin sobrescribir variables ya definidas en el entorno."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _default_models_root() -> Path:
    return Path("D:/AI_Models") if Path("D:/").exists() else PROJECT_DIR / "AI_Models"


load_env_file()

MODELS_ROOT = Path(os.environ.get("HY3DGEN_MODELS") or _default_models_root())
OUTPUT_DIR = MODELS_ROOT / "output"
TEMP_DIR = MODELS_ROOT / "temp"
REPOS_DIR = MODELS_ROOT / "repos"
HUNYUAN_LOCAL_DIR = MODELS_ROOT / "tencent" / "Hunyuan3D-2"
SF3D_LOCAL_DIR = MODELS_ROOT / "stabilityai" / "stable-fast-3d"

_DERIVED = {
    "HY3DGEN_MODELS": MODELS_ROOT,
    "HF_HOME": MODELS_ROOT / "huggingface",
    "HUGGINGFACE_HUB_CACHE": MODELS_ROOT / "huggingface" / "hub",
    "TORCH_HOME": MODELS_ROOT / "torch",
    "U2NET_HOME": MODELS_ROOT / "u2net",
    "PIP_CACHE_DIR": MODELS_ROOT / "pip_cache",
    "TEMP": TEMP_DIR,
    "TMP": TEMP_DIR,
    "TMPDIR": TEMP_DIR,
}

# TEMP/TMP siempre existen en Windows (apuntan a C:): se fuerzan a la raíz elegida
# para que los temporales de modelos y descargas no llenen el disco C:
_ALWAYS_SET = {"TEMP", "TMP", "TMPDIR"}

for _key, _path in _DERIVED.items():
    if _key in _ALWAYS_SET:
        os.environ[_key] = str(_path)
    else:
        os.environ.setdefault(_key, str(_path))

for _d in (MODELS_ROOT / "huggingface", MODELS_ROOT / "torch", MODELS_ROOT / "u2net", TEMP_DIR, OUTPUT_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def write_models_root(root: str) -> None:
    """
    Guarda en el .env la raíz de modelos elegida en setup.bat y todas las rutas derivadas.
    Conserva el resto de claves y comentarios; si no hay .env, parte de .env.example.
    """
    root_path = Path(root)
    values = {
        "HY3DGEN_MODELS": root_path,
        "HF_HOME": root_path / "huggingface",
        "HUGGINGFACE_HUB_CACHE": root_path / "huggingface" / "hub",
        "TORCH_HOME": root_path / "torch",
        "U2NET_HOME": root_path / "u2net",
        "PIP_CACHE_DIR": root_path / "pip_cache",
        "TEMP": root_path / "temp",
        "TMP": root_path / "temp",
    }
    source = ENV_FILE if ENV_FILE.exists() else PROJECT_DIR / ".env.example"
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except Exception:
        lines = []

    pending = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    out.extend(f"{k}={v}" for k, v in pending.items())
    ENV_FILE.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--set-root":
        write_models_root(sys.argv[2])
        print(f"[OK] Carpeta de modelos guardada en {ENV_FILE}: {sys.argv[2]}")
    else:
        print(f"Carpeta de modelos actual: {MODELS_ROOT}")
        print("Uso: python studio_env.py --set-root <ruta>")
