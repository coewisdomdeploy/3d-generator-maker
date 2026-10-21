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
    python preparar_modelos.py optimize paint      Crea la caché fp16 de Hunyuan3D-Paint (una vez)

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
# Estimador de profundidad del refuerzo guiado de "Profundidad mejorada" (debe coincidir con mesh_detail.py)
DEPTH_REPO = os.environ.get("DEPTH_ESTIMATOR_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")

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
    "dit_fast": "Geometría Hunyuan3D-2 Fast (OPCIONAL, ~4.9 GB: casilla 'Profundidad mejorada')",
    "dit_full": "Geometría Hunyuan3D-2 completo (OPCIONAL, ~4.9 GB: casilla 'Máxima calidad')",
    "depth": "Depth Anything V2 Small (OPCIONAL, ~100 MB: refuerzo guiado de 'Profundidad mejorada')",
    "paint": "Textura Hunyuan3D-Paint (delight + multivista, ~9 GB)",
    "sf3d": "Stable Fast 3D (~3.8 GB, requiere licencia en Hugging Face)",
    "rasterizer": "Extensión CUDA custom_rasterizer (OPCIONAL: sin ella se usa el rasterizador PyTorch)",
    "meshproc": "mesh_processor nativo C++ (OPCIONAL: sin él se usa la versión Numba, igual de rápida)",
    "paint_opt": "Caché optimizada fp16 de Hunyuan3D-Paint (carga más rápida; crear con: optimize paint)",
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


def _dit_variant_ready(subfolder: str) -> bool:
    # La app carga las variantes desde HUNYUAN_LOCAL_DIR (smart_load_model)
    rel = Path(subfolder) / "model.fp16.safetensors"
    return _file_ok(HUNYUAN_LOCAL_DIR / rel, 4 * 1024 ** 3) and _file_ok(HUNYUAN_LOCAL_DIR / rel.parent / "config.yaml")


def dit_fast_ready() -> bool:
    return _dit_variant_ready("hunyuan3d-dit-v2-0-fast")


def dit_full_ready() -> bool:
    return _dit_variant_ready("hunyuan3d-dit-v2-0")


def depth_ready() -> bool:
    return any(_file_ok(s / "model.safetensors", 10 * 1024 ** 2) for s in _hf_snapshots(DEPTH_REPO))


def paint_ready() -> bool:
    roots = [HUNYUAN_LOCAL_DIR, *_hf_snapshots(HUNYUAN_REPO)]
    return any(all(_file_ok(r / f) for f in PAINT_KEY_FILES) for r in roots)


def sf3d_ready() -> bool:
    roots = [SF3D_LOCAL_DIR, *_hf_snapshots(SF3D_REPO)]
    return any(_file_ok(r / "model.safetensors", 1024 ** 3) and _file_ok(r / "config.yaml") for r in roots)


def rasterizer_ready() -> bool:
    # find_spec no importa el módulo (no inicializa CUDA)
    return importlib.util.find_spec("custom_rasterizer") is not None


def meshproc_ready() -> bool:
    """
    mesh_processor compilado (.pyd) junto a hy3dgen, y además da el MISMO resultado que la
    versión Python en una malla diminuta (un .pyd incompatible nunca se da por bueno).
    Se carga el archivo directamente: no importa torch ni hy3dgen.texgen.
    """
    spec = importlib.util.find_spec("hy3dgen")
    if spec is None or not spec.submodule_search_locations:
        return False
    folder = Path(list(spec.submodule_search_locations)[0]) / "texgen" / "differentiable_renderer"
    natives = sorted(folder.glob("mesh_processor*.pyd")) + sorted(folder.glob("mesh_processor*.so"))
    if not natives:
        return False
    try:
        import numpy as np
        from importlib.machinery import ExtensionFileLoader
        loader = ExtensionFileLoader("mesh_processor", str(natives[0]))
        native = importlib.util.module_from_spec(importlib.util.spec_from_loader("mesh_processor", loader))
        loader.exec_module(native)
        py_spec = importlib.util.spec_from_file_location("_mesh_processor_py", folder / "mesh_processor.py")
        python_impl = importlib.util.module_from_spec(py_spec)
        py_spec.loader.exec_module(python_impl)

        tex = np.zeros((8, 8, 3), np.float32)
        mask = np.zeros((8, 8), np.uint8)
        tex[5, 2] = (1.0, 0.5, 0.25)
        mask[5, 2] = 255
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
        uv = np.array([[2 / 7, 1 - 5 / 7], [0.9, 0.9], [0.5, 0.1], [0.1, 0.6]], np.float32)
        idx = np.array([[0, 1, 2], [1, 3, 2]], np.int32)
        a_tex, a_mask = native.meshVerticeInpaint(tex.copy(), mask.copy(), pos, uv, idx, idx)
        b_tex, b_mask = python_impl.meshVerticeInpaint(tex.copy(), mask.copy(), pos, uv, idx, idx)
        return bool(np.allclose(a_tex, b_tex, atol=1e-4) and np.array_equal(a_mask, b_mask))
    except Exception:
        return False


CHECKS = {"dit": dit_ready, "dit_fast": dit_fast_ready, "dit_full": dit_full_ready, "depth": depth_ready,
          "paint": paint_ready, "sf3d": sf3d_ready, "rasterizer": rasterizer_ready, "meshproc": meshproc_ready}


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


def _download_dit_variant(subfolder: str, label: str) -> int:
    if not _ensure_space(DIT_DOWNLOAD_GB):
        return 1
    from huggingface_hub import snapshot_download
    print(f"[INFO] Descargando {label} (~4.6 GB) a {HUNYUAN_LOCAL_DIR} (reanudable)...")
    try:
        snapshot_download(HUNYUAN_REPO, local_dir=str(HUNYUAN_LOCAL_DIR), allow_patterns=[
            f"{subfolder}/config.yaml", f"{subfolder}/model.fp16.safetensors"])
    except Exception as e:
        print(f"[ERROR] No se pudo descargar {label}: {e}")
        return 1
    return 0 if _dit_variant_ready(subfolder) else 1


def download_dit_fast() -> int:
    """Variante Fast (destilada de guía, ~30 pasos): más fiel al volumen que la Turbo de 5 pasos."""
    return _download_dit_variant("hunyuan3d-dit-v2-0-fast", "Hunyuan3D-2 Fast")


def download_dit_full() -> int:
    """Modelo completo (sin destilar, 50 pasos con guía): el más fiel a la imagen ("Máxima calidad")."""
    return _download_dit_variant("hunyuan3d-dit-v2-0", "Hunyuan3D-2 completo")


def download_depth() -> int:
    from huggingface_hub import snapshot_download
    print(f"[INFO] Descargando {DEPTH_REPO} (~100 MB) a {os.environ['HUGGINGFACE_HUB_CACHE']}...")
    try:
        snapshot_download(DEPTH_REPO)
    except Exception as e:
        print(f"[ERROR] No se pudo descargar Depth Anything V2: {e}")
        return 1
    return 0 if depth_ready() else 1


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


DOWNLOADS = {"dit": download_dit, "dit_fast": download_dit_fast, "dit_full": download_dit_full,
             "depth": download_depth, "paint": download_paint, "sf3d": download_sf3d}

# ------------------------------------------------------------------------------
# Caché optimizada de Hunyuan3D-Paint: pesos en fp16 + safetensors, convertidos UNA vez.
# El modelo se ejecuta en fp16, así que el resultado es idéntico; cada generación lee
# ~2.5 GB menos del disco y no convierte fp32 -> fp16 en la CPU.
# ------------------------------------------------------------------------------
PAINT_OPT_DIR = MODELS_ROOT / "optimized" / "Hunyuan3D-2"
PAINT_OPT_MARKER = PAINT_OPT_DIR / "optimized_ok.json"
PAINT_OPT_VERSION = 1
PAINT_SUBFOLDERS = ("hunyuan3d-delight-v2-0", "hunyuan3d-paint-v2-0")


def _paint_source_root() -> Path | None:
    for root in [HUNYUAN_LOCAL_DIR, *_hf_snapshots(HUNYUAN_REPO)]:
        if all(_file_ok(root / f) for f in PAINT_KEY_FILES):
            return root
    return None


def paint_opt_ready() -> bool:
    import json
    try:
        info = json.loads(PAINT_OPT_MARKER.read_text(encoding="utf-8"))
    except Exception:
        return False
    return info.get("version") == PAINT_OPT_VERSION and all((PAINT_OPT_DIR / s).is_dir() for s in PAINT_SUBFOLDERS)


def _stream_save_safetensors(state: dict, dst: Path) -> None:
    """
    Escribe un .safetensors estándar tensor a tensor. safetensors.save_file arma el archivo
    entero en memoria (hasta +3.5 GB de RAM); así el pico es solo el tensor en curso.
    Formato: u64 longitud de cabecera + cabecera JSON (alineada a 8) + datos en orden.
    """
    import json
    import struct
    import torch
    names = {torch.float16: "F16", torch.float32: "F32", torch.bfloat16: "BF16", torch.float64: "F64",
             torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
             torch.uint8: "U8", torch.bool: "BOOL"}
    header, offset = {}, 0
    for key, t in state.items():
        size = t.numel() * t.element_size()
        header[key] = {"dtype": names[t.dtype], "shape": list(t.shape), "data_offsets": [offset, offset + size]}
        offset += size
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(dst, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for t in state.values():
            raw = t.detach().contiguous().cpu()
            # bytes del tensor sin conversión de tipo (bf16/bool incluidos)
            fh.write(raw.view(torch.uint8).numpy().tobytes() if raw.dtype != torch.bool else raw.numpy().tobytes())


def optimize_paint() -> int:
    """Convierte los pesos de Hunyuan3D-Paint a fp16 safetensors en <modelos>/optimized (solo CPU)."""
    import json
    import shutil as _sh
    src_root = _paint_source_root()
    if src_root is None:
        print("[ERROR] Faltan los pesos de Hunyuan3D-Paint. Ejecuta primero: preparar_modelos.py download paint")
        return 1
    if not _ensure_space(7.0):
        return 1
    import torch
    from safetensors import safe_open


    def read_fp16(path: Path) -> dict:
        """Lee tensor a tensor (mmap) y convierte a fp16: la RAM solo aloja la versión fp16."""
        if path.suffix == ".bin":
            state = torch.load(str(path.resolve()), map_location="cpu", weights_only=True, mmap=True)
            return {k: (v.to(torch.float16) if v.is_floating_point() else v).contiguous() for k, v in state.items()}
        out = {}
        with safe_open(str(path.resolve()), framework="pt", device="cpu") as f:
            for k in f.keys():
                v = f.get_tensor(k)
                out[k] = (v.to(torch.float16) if v.is_floating_point() else v).contiguous()
        return out

    def save(state: dict, dst: Path) -> None:
        _stream_save_safetensors(state, dst)

    tmp_dir = PAINT_OPT_DIR.with_name(PAINT_OPT_DIR.name + ".tmp")
    _sh.rmtree(tmp_dir, ignore_errors=True)
    t0 = __import__("time").time()
    for sub in PAINT_SUBFOLDERS:
        for src_dir in sorted({p.parent for p in (src_root / sub).rglob("*")} | {src_root / sub}):
            if not src_dir.is_dir():
                continue
            rel = src_dir.relative_to(src_root)
            dst_dir = tmp_dir / rel
            dst_dir.mkdir(parents=True, exist_ok=True)
            files = [f for f in src_dir.iterdir() if f.is_file() or f.is_symlink()]
            weights = [f for f in files if f.suffix in (".safetensors", ".bin")]
            for f in files:
                if f not in weights:
                    _sh.copyfile(f.resolve(), dst_dir / f.name)
            if not weights:
                continue
            # El UNet multivista se carga desde su .bin (su .safetensors no es el que usa hy3dgen)
            is_mv_unet = rel.as_posix() == "hunyuan3d-paint-v2-0/unet"
            bins = [f for f in weights if f.suffix == ".bin"]
            sts = [f for f in weights if f.suffix == ".safetensors"]
            source = bins[0] if (is_mv_unet and bins) else (sts[0] if sts else bins[0])
            state = read_fp16(source)
            out_name = "model.safetensors" if rel.name == "text_encoder" else "diffusion_pytorch_model.safetensors"
            save(state, dst_dir / out_name)
            del state
            print(f"  [OK] {rel.as_posix()}/{out_name} <- {source.name}")
    _sh.rmtree(PAINT_OPT_DIR, ignore_errors=True)
    PAINT_OPT_DIR.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.rename(PAINT_OPT_DIR)
    PAINT_OPT_MARKER.write_text(json.dumps({"version": PAINT_OPT_VERSION, "source": str(src_root)}), encoding="utf-8")
    size = sum(f.stat().st_size for f in PAINT_OPT_DIR.rglob("*") if f.is_file()) / (1024 ** 3)
    print(f"[OK] Caché optimizada de Hunyuan3D-Paint lista en {PAINT_OPT_DIR} ({size:.1f} GB, {__import__('time').time() - t0:.0f}s)")
    return 0

# Debe coincidir con MODEL_MEMORY_GB / RAM_SAFETY_MARGIN_GB de img2glb.py (aquí no se importa
# img2glb para no cargar torch ni inicializar CUDA)
SHAPE_VRAM_FULL_GB = 6.0
SHAPE_VRAM_MIN_GB = 3.2
NEEDS_RAM_GB = {"forma (carga directa a GPU)": 1.0, "SF3D": 4.5, "textura Hunyuan3D-Paint (carga directa a GPU)": 3.0}
PAINT_MV_VRAM_GB = {"gpu": 7.0, "lean": 5.2, "bloques": 3.2}   # = PAINT_MV_VRAM_GB de img2glb.py


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
        tier = next((t for t, need in PAINT_MV_VRAM_GB.items() if free >= need), None)
        paint = {"gpu": "gpu (todo en GPU, el más rápido)", "lean": "lean (unet fijo, unet_dual y text encoder al usarse)",
                 "bloques": "bloques (UNet transmitido por bloques; unos segundos más)"}.get(tier, "no cabe: textura de contingencia")
        print(f"    Textura Hunyuan3D-Paint (multivista) con {free:.1f} GB de VRAM libre: {paint}")


def print_status() -> None:
    print(f"  Carpeta de modelos: {MODELS_ROOT}  ({_free_gb():.0f} GB libres)")
    for key, label in ITEMS.items():
        mark = "[OK]   " if CHECKS[key]() else "[FALTA]"
        print(f"  {mark} {label}")


CHECKS["paint_opt"] = paint_opt_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprueba y descarga los pesos de los modelos 3D")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("memoria")
    p_opt = sub.add_parser("optimize")
    p_opt.add_argument("item", choices=["paint"])
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
    if args.cmd == "optimize":
        if paint_opt_ready():
            print(f"[OK] La caché optimizada de Hunyuan3D-Paint ya existe en {PAINT_OPT_DIR}")
            return 0
        return optimize_paint()
    if args.cmd == "check":
        return 0 if CHECKS[args.item]() else 1
    if CHECKS[args.item]():
        print(f"[OK] {ITEMS[args.item]} ya está descargado.")
        return 0
    return DOWNLOADS[args.item]()


if __name__ == "__main__":
    sys.exit(main())
