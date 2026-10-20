#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Selector de GPU al Arrancar (equipos con 2 o más tarjetas gráficas)
==============================================================================
Debe ejecutarse ANTES de `import torch`: CUDA fija las GPUs visibles en el
momento en que se inicializa, por lo que CUDA_VISIBLE_DEVICES tiene que estar
definido de antemano. La GPU elegida pasa a ser siempre 'cuda:0' dentro del
proceso, así que el resto del código no necesita cambios.

Prioridad de selección:
    1. Argumento --gpu <índice|cpu|ask>
    2. Variable de entorno STUDIO_GPU (mismo formato), o STUDIO_GPU=... en .env
    3. Menú interactivo en consola (si hay más de un adaptador)
    4. Última GPU elegida (guardada en .gpu_choice)
    5. Primera GPU NVIDIA disponible

Nota: CUDA solo reconoce tarjetas NVIDIA. Las GPUs Intel/AMD se muestran como
información, pero la IA no puede ejecutarse en ellas (solo NVIDIA o CPU).
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Optional

CHOICE_FILE = Path(__file__).resolve().parent / ".gpu_choice"
PROMPT_TIMEOUT_SEC = 15


def list_nvidia_gpus() -> list[dict]:
    """Enumera las GPUs NVIDIA con nvidia-smi (no requiere inicializar CUDA)."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15
        )
    except Exception:
        return []
    if res.returncode != 0:
        return []
    gpus = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            vram_gb = int(parts[3]) / 1024
        except ValueError:
            vram_gb = 0.0
        gpus.append({
            "index": int(parts[0]),
            "uuid": parts[1],
            "name": parts[2],
            "vram_gb": vram_gb,
            "bus_id": parts[4],
        })
    return gpus


def list_other_adapters() -> list[str]:
    """Adaptadores de video no-NVIDIA según Windows (Intel/AMD, solo informativo)."""
    if os.name != "nt":
        return []
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=15
        )
    except Exception:
        return []
    names = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    ignored = ("nvidia", "microsoft basic", "remote display", "virtual", "parsec", "meta")
    return [n for n in names if not any(k in n.lower() for k in ignored)]


def _env_file_value(key: str) -> Optional[str]:
    """Lee una clave del .env del proyecto (aquí aún no se ha cargado: torch no debe importarse antes)."""
    try:
        for line in (Path(__file__).resolve().parent / ".env").read_text(encoding="utf-8").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() == key and not line.lstrip().startswith("#"):
                return v.strip() or None
    except Exception:
        pass
    return None


def _load_saved_choice() -> Optional[str]:
    try:
        return json.loads(CHOICE_FILE.read_text(encoding="utf-8")).get("choice")
    except Exception:
        return None


def _save_choice(choice: str) -> None:
    try:
        CHOICE_FILE.write_text(json.dumps({"choice": choice}), encoding="utf-8")
    except Exception:
        pass


def _input_with_timeout(prompt: str, timeout: float, default: str) -> str:
    """input() con cuenta atrás: si no se pulsa nada en `timeout` segundos devuelve `default`."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    if os.name == "nt":
        import msvcrt
        buf = ""
        deadline = time.time() + timeout
        while True:
            # La cuenta atrás solo corre mientras no se haya empezado a escribir
            if not buf and time.time() >= deadline:
                print()
                return default
            if not msvcrt.kbhit():
                time.sleep(0.05)
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # Teclas especiales (flechas, F1...): descartar
                msvcrt.getwch()
                continue
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                print()
                return buf.strip() or default
            if ch == "\b":
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            buf += ch
            sys.stdout.write(ch)
            sys.stdout.flush()
    else:
        import select
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if not ready:
            print()
            return default
        return sys.stdin.readline().strip() or default


def _resolve(choice: Optional[str], gpus: list[dict]) -> Optional[str]:
    """Convierte una elección ('cpu', índice nvidia-smi o UUID) en 'cpu' o el UUID de la GPU."""
    if not choice:
        return None
    choice = choice.strip().lower()
    if choice in ("cpu", "c"):
        return "cpu"
    for g in gpus:
        if choice == str(g["index"]) or choice == g["uuid"].lower():
            return g["uuid"]
    return None


def _prompt_menu(gpus: list[dict], others: list[str], default: str) -> str:
    print("\n" + "=" * 76)
    print("   [SELECCIÓN DE TARJETA GRÁFICA PARA LA IA]")
    print("=" * 76)
    for g in gpus:
        mark = "  <- predeterminada" if g["uuid"] == default else ""
        print(f"   [{g['index']}] {g['name']} ({g['vram_gb']:.0f} GB VRAM){mark}")
    cpu_mark = "  <- predeterminada" if default == "cpu" else ""
    print(f"   [C] CPU (sin aceleración, muy lento){cpu_mark}")
    for name in others:
        print(f"    -  {name}: no compatible con CUDA (solo pantalla / navegador)")
    print("-" * 76)

    default_key = "C" if default == "cpu" else next(str(g["index"]) for g in gpus if g["uuid"] == default)
    while True:
        answer = _input_with_timeout(
            f"   Elige GPU [Enter = {default_key}, automático en {PROMPT_TIMEOUT_SEC}s]: ",
            PROMPT_TIMEOUT_SEC, default_key
        )
        resolved = _resolve(answer, gpus)
        if resolved:
            return resolved
        print(f"   Opción no válida: '{answer}'.")


def select_gpu(argv: Optional[list[str]] = None) -> dict:
    """
    Elige la GPU y fija CUDA_VISIBLE_DEVICES para este proceso (y sus subprocesos).
    Es idempotente: si ya se eligió en este proceso, devuelve la elección previa.
    Retorna {"choice": "cpu" | uuid, "name": str}.
    """
    if os.environ.get("STUDIO_GPU_SELECTED"):
        return {"choice": os.environ.get("STUDIO_GPU_CHOICE", ""), "name": os.environ.get("STUDIO_GPU_NAME", "")}

    if "torch" in sys.modules:
        print(" -> [GPU Select] Aviso: torch ya estaba importado; la selección podría no aplicarse.")

    argv = sys.argv[1:] if argv is None else argv
    cli_choice = None
    for i, arg in enumerate(argv):
        if arg == "--gpu" and i + 1 < len(argv):
            cli_choice = argv[i + 1]
        elif arg.startswith("--gpu="):
            cli_choice = arg.split("=", 1)[1]
    requested = cli_choice or os.environ.get("STUDIO_GPU") or _env_file_value("STUDIO_GPU")
    force_ask = (requested or "").strip().lower() == "ask"

    gpus = list_nvidia_gpus()
    choice = None if force_ask else _resolve(requested, gpus)
    if requested and not force_ask and choice is None:
        print(f" -> [GPU Select] '{requested}' no corresponde a ninguna GPU NVIDIA; se usará el menú o la predeterminada.")

    if choice is None:
        if not gpus:
            choice = "cpu"
        else:
            default = _resolve(_load_saved_choice(), gpus) or gpus[0]["uuid"]
            others = list_other_adapters()
            interactive = sys.stdin is not None and sys.stdin.isatty()
            if interactive and (force_ask or len(gpus) > 1 or others):
                choice = _prompt_menu(gpus, others, default)
            else:
                choice = default
            _save_choice("cpu" if choice == "cpu" else next(str(g["index"]) for g in gpus if g["uuid"] == choice))

    # PCI_BUS_ID hace que el orden de CUDA coincida con nvidia-smi (por defecto CUDA ordena "el más rápido primero")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if choice == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" en Windows borra la variable; -1 oculta todas
        name = "CPU"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = choice
        name = next(g["name"] for g in gpus if g["uuid"] == choice)

    os.environ["STUDIO_GPU_SELECTED"] = "1"
    os.environ["STUDIO_GPU_CHOICE"] = choice
    os.environ["STUDIO_GPU_NAME"] = name
    print(f" -> [GPU Select] Dispositivo asignado para IA: {name}" + ("" if choice == "cpu" else " (cuda:0 dentro del proceso)"))
    return {"choice": choice, "name": name}


if __name__ == "__main__":
    # Permite probar el selector por separado: python gpu_select.py [--gpu ask]
    select_gpu()
    import torch
    print(f" -> torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f" -> cuda:0 = {torch.cuda.get_device_name(0)}")
