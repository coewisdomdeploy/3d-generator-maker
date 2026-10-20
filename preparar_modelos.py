#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Comprobación y Descarga de Pesos de los Modelos (sin usar la GPU)
==============================================================================
Uso:
    python preparar_modelos.py status              Tabla con el estado de cada componente
    python preparar_modelos.py check <item>        Código de salida 0 si <item> está listo
    python preparar_modelos.py download <item>     Descarga <item> (reanudable)
    python preparar_modelos.py memoria             Diagnóstico de RAM/VRAM y qué cabrá al generar

Items: dit (geometría Hunyuan3D-2), paint (textura Hunyuan3D-Paint),
       sf3d (Stable Fast 3D), rasterizer (extensión CUDA custom_rasterizer; solo check)

No importa torch ni inicializa CUDA: es seguro ejecutarlo en cualquier momento.
"""

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

from studio_env import MODELS_ROOT, HUNYUAN_LOCAL_DIR, SF3D_LOCAL_DIR

import os

HUNYUAN_REPO = "tencent/Hunyuan3D-2"
SF3D_REPO = "stabilityai/stable-fast-3d"

# Archivos clave de Hunyuan3D-Paint (delight + multivista). El UNet multivista se carga
# desde el .bin (hy3dgen/texgen/hunyuanpaint/unet/modules.py): su .safetensors duplicado
# (~3.5 GB) no se descarga.
PAINT_KEY_FILES = [
    "hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors",
    "hunyuan3d-delight-v2-0/text_encoder/model.safetensors",
    "hunyuan3d-delight-v2-0/vae/diffusion_pytorch_model.safetensors",
    "hunyuan3d-paint-v2-0/unet/diffusion_pytorch_model.bin",
    "hunyuan3d-paint-v2-0/text_encoder/pytorch_model.bin",
]
PAINT_PATTERNS = ["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0/*"]
PAINT_IGNORE = ["hunyuan3d-paint-v2-0/unet/*.safetensors"]
PAINT_DOWNLOAD_GB = 9.5
SF3D_DOWNLOAD_GB = 4.0
DIT_DOWNLOAD_GB = 5.0

ITEMS = {
    "dit": "Geometría Hunyuan3D-2 Turbo (DiT, ~4.9 GB)",
    "paint": "Textura Hunyuan3D-Paint (delight + multivista, ~9 GB)",
    "sf3d": "Stable Fast 3D (~3.8 GB, requiere licencia en Hugging Face)",
    "rasterizer": "Extensión CUDA custom_rasterizer (necesaria para la textura Hunyuan)",
}


def _hf_snapshots(repo: str) -> list[Path]:
    hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    snaps = hub / f"models--{repo.replace('/', '--')}" / "snapshots"
    return [p for p in snaps.iterdir() if p.is_dir()] if snaps.exists() else []


def _file_ok(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.resolve().stat().st_size >= min_bytes
    except OSError:
        return False


def dit_ready() -> bool:
    rel = Path("hunyuan3d-dit-v2-0-turbo") / "model.fp16.safetensors"
    roots = [HUNYUAN_LOCAL_DIR, *_hf_snapshots(HUNYUAN_REPO)]
    return any(_file_ok(r / rel, 4 * 1024 ** 3) for r in roots)


def paint_ready() -> bool:
    roots = [HUNYUAN_LOCAL_DIR, *_hf_snapshots(HUNYUAN_REPO)]
    return any(all(_file_ok(r / f) for f in PAINT_KEY_FILES) for r in roots)


def sf3d_ready() -> bool:
    roots = [SF3D_LOCAL_DIR, *_hf_snapshots(SF3D_REPO)]
    return any(_file_ok(r / "model.safetensors", 1024 ** 3) and _file_ok(r / "config.yaml") for r in roots)


def rasterizer_ready() -> bool:
    # find_spec no importa el módulo (no inicializa CUDA)
    return importlib.util.find_spec("custom_rasterizer") is not None


CHECKS = {"dit": dit_ready, "paint": paint_ready, "sf3d": sf3d_ready, "rasterizer": rasterizer_ready}


def _free_gb() -> float:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(MODELS_ROOT).free / (1024 ** 3)


def _ensure_space(needed_gb: float) -> bool:
    free = _free_gb()
    if free < needed_gb + 2:
        print(f"[ERROR] Espacio insuficiente en {MODELS_ROOT}: {free:.1f} GB libres, se necesitan ~{needed_gb + 2:.0f} GB.")
        return False
    return True


def download_dit() -> int:
    if not _ensure_space(DIT_DOWNLOAD_GB):
        return 1
    import download_model
    return download_model.main()


def download_paint() -> int:
    if not _ensure_space(PAINT_DOWNLOAD_GB):
        return 1
    from huggingface_hub import snapshot_download
    print(f"[INFO] Descargando Hunyuan3D-Paint (~{PAINT_DOWNLOAD_GB:.0f} GB) a {os.environ['HUGGINGFACE_HUB_CACHE']} (reanudable)...")
    try:
        snapshot_download(HUNYUAN_REPO, allow_patterns=PAINT_PATTERNS, ignore_patterns=PAINT_IGNORE)
    except Exception as e:
        print(f"[ERROR] No se pudo descargar Hunyuan3D-Paint: {e}")
        return 1
    return 0 if paint_ready() else 1


def download_sf3d() -> int:
    if not _ensure_space(SF3D_DOWNLOAD_GB):
        return 1
    from huggingface_hub import snapshot_download
    print(f"[INFO] Descargando Stable Fast 3D a {SF3D_LOCAL_DIR}...")
    try:
        snapshot_download(SF3D_REPO, local_dir=str(SF3D_LOCAL_DIR), allow_patterns=["config.yaml", "model.safetensors"])
    except Exception as e:
        msg = str(e)
        if any(k in msg for k in ("401", "403", "gated", "Unauthorized", "restricted")):
            print("[ERROR] SF3D es un modelo con licencia: acepta los términos en")
            print("        https://huggingface.co/stabilityai/stable-fast-3d y configura tu token (huggingface-cli login).")
        else:
            print(f"[ERROR] No se pudo descargar SF3D: {e}")
        return 1
    return 0 if sf3d_ready() else 1


DOWNLOADS = {"dit": download_dit, "paint": download_paint, "sf3d": download_sf3d}

# Debe coincidir con MODEL_MEMORY_GB / RAM_SAFETY_MARGIN_GB de img2glb.py (aquí no se importa
# img2glb para no cargar torch ni inicializar CUDA)
SHAPE_VRAM_FULL_GB = 6.0
SHAPE_VRAM_MIN_GB = 3.2
NEEDS_RAM_GB = {"forma (carga directa a GPU)": 1.0, "SF3D": 4.5, "textura Hunyuan3D-Paint": 5.5}


def _nvidia_vram() -> list[tuple[str, float, float]]:
    """(nombre, total GB, libre GB) por GPU según nvidia-smi (no inicializa CUDA)."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            gpus.append((parts[0], int(parts[1]) / 1024, int(parts[2]) / 1024))
    return gpus


def print_memory() -> None:
    """Diagnóstico de RAM/VRAM y qué cabrá al generar."""
    import psutil
    margin = float(os.environ.get("RAM_SAFETY_MARGIN_GB", "2"))
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    gb = 1024 ** 3
    print("  RAM")
    print(f"    Total: {vm.total / gb:.1f} GB | En uso: {vm.percent:.0f}% | Disponible: {vm.available / gb:.1f} GB")
    print(f"    Archivo de paginación: {sw.total / gb:.1f} GB (usado {sw.percent:.0f}%)")
    if sw.total < 8 * gb:
        print("    [AVISO] Paginación pequeña o desactivada: actívala como 'Tamaño administrado por el sistema'.")

    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            procs.append((p.info["memory_info"].rss, p.info["name"] or "?", p.pid))
        except Exception:
            pass
    procs.sort(reverse=True)
    print("  Procesos que más RAM usan:")
    for rss, name, pid in procs[:8]:
        print(f"    {rss / gb:5.2f} GB  {name} (PID {pid})")
    pythons = [(rss, pid) for rss, name, pid in procs if name.lower().startswith("python") and pid not in (os.getpid(), os.getppid())]
    if pythons:
        print(f"    [AVISO] Hay {len(pythons)} proceso(s) python abiertos: si no tienes la app abierta, ciérralos")
        print("            (Administrador de tareas > Detalles > python.exe > Finalizar tarea).")

    gpus = _nvidia_vram()
    print("  VRAM (nvidia-smi)")
    if not gpus:
        print("    No se detectó GPU NVIDIA.")
    for name, total, free in gpus:
        print(f"    {name}: {free:.1f} GB libres de {total:.1f} GB")

    print("  Qué cabe ahora mismo")
    avail = vm.available / gb
    for label, need in NEEDS_RAM_GB.items():
        mark = "OK " if avail >= need + margin else "NO "
        print(f"    [{mark}] {label}: necesita ~{need + margin:.1f} GB de RAM libre (incluye {margin:.0f} GB de margen)")
    if gpus:
        free = max(g[2] for g in gpus)
        mode = "full (todo en GPU)" if free >= SHAPE_VRAM_FULL_GB else (
            "stream (por partes)" if free >= SHAPE_VRAM_MIN_GB else "insuficiente")
        print(f"    Modo de carga de la forma con {free:.1f} GB de VRAM libre: {mode}")


def print_status() -> None:
    print(f"  Carpeta de modelos: {MODELS_ROOT}  ({_free_gb():.0f} GB libres)")
    for key, label in ITEMS.items():
        mark = "[OK]   " if CHECKS[key]() else "[FALTA]"
        print(f"  {mark} {label}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba y descarga los pesos de los modelos 3D")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("memoria")
    p_check = sub.add_parser("check")
    p_check.add_argument("item", choices=list(CHECKS))
    p_dl = sub.add_parser("download")
    p_dl.add_argument("item", choices=list(DOWNLOADS))
    args = parser.parse_args()

    if args.cmd == "status":
        print_status()
        return 0
    if args.cmd == "memoria":
        print_memory()
        return 0
    if args.cmd == "check":
        return 0 if CHECKS[args.item]() else 1
    if CHECKS[args.item]():
        print(f"[OK] {ITEMS[args.item]} ya está descargado.")
        return 0
    return DOWNLOADS[args.item]()


if __name__ == "__main__":
    sys.exit(main())
