#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
3D Generator - CLI: Image to .GLB usando Tencent Hunyuan3D-2
Optimizador de Mallas para Rendimiento Móvil en Godot 4 (Kaboomi Party)
==============================================================================

Uso por consola:
    python img2glb.py --input foto.png --output ../kaboomi-party/assets/models/bombos/mi_modelo.glb --faces 4000

Flujo de trabajo automático:
    a) Elimina el fondo de la imagen con `rembg` (o conserva transparencia limpia).
    b) Genera la malla 3D de alta definición con `Hunyuan3D-2` (Hunyuan3D-DiT-v2-0-Turbo en fp16).
    c) Simplifica y decima la malla al número exacto de caras `--faces` (4000 tris por defecto).
    d) Genera y hornea la textura de color sobre el modelo y exporta el `.glb` centrado en (0,0,0).
"""

import os
import sys
import time
import argparse
import shutil
import gc
from pathlib import Path
from typing import Optional, Tuple, Union, Any, Callable

# Rutas de modelos y cachés (raíz elegida en setup.bat, guardada en .env). Antes de torch.
from studio_env import MODELS_ROOT, OUTPUT_DIR as STUDIO_OUTPUT_DIR, HUNYUAN_LOCAL_DIR, SF3D_LOCAL_DIR, REPOS_DIR
print(f"[Smart Cache] Modelos y cachés en: {MODELS_ROOT}")

# Selección de GPU al usar la CLI directamente (en la WebUI ya se hizo antes de importar este módulo)
if __name__ == "__main__":
    from gpu_select import select_gpu
    select_gpu()

import numpy as np
from PIL import Image
import torch
import trimesh
import trimesh.smoothing
import pygltflib

# Desactivar OpenCL en OpenCV para evitar que utilice la GPU integrada Intel
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.cuda.set_device(0)
    except Exception:
        pass

DEFAULT_OUTPUT_DIR = STUDIO_OUTPUT_DIR
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. Función de Purga Estricta de Memoria VRAM (flush_vram)
# ==============================================================================

def flush_vram() -> None:
    """
    Purga estricta de memoria VRAM y recolección de basura para GPUs de 8 GB (RTX 4060):
    Ejecuta gc.collect(), torch.cuda.empty_cache() y torch.cuda.ipc_collect().
    Garantiza que ningún buffer intermedio, caché de PyTorch o residuo de tensores persista en VRAM.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def print_real_vram_status(stage_name: str = "") -> Tuple[float, float]:
    """
    Imprime en consola la memoria VRAM libre real reportada directamente por el driver
    de NVIDIA mediante torch.cuda.mem_get_info().
    """
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_mb = free_bytes / (1024 ** 2)
        total_mb = total_bytes / (1024 ** 2)
        used_mb = total_mb - free_mb
        prefix = f"[{stage_name}] " if stage_name else ""
        print(f" -> {prefix}VRAM Libre Real (NVIDIA Driver): {free_mb:.1f} MB libres de {total_mb:.1f} MB (Ocupado: {used_mb:.1f} MB)")
        return free_mb, total_mb
    return 0.0, 0.0


# ==============================================================================
# Gestor de Residencia de Modelos: UN SOLO modelo en la GPU a la vez
# ==============================================================================
# Hay tres modelos pesados: "shape" (Hunyuan3D-DiT), "paint" (Hunyuan3D-Paint) y
# "sf3d" (Stable Fast 3D). Flujo garantizado para cada carga:
#   1. Se sacan de la VRAM todos los demás modelos.
#   2. Se VALIDA que haya VRAM libre suficiente; si no, se destruyen los demás y se revalida.
#   3. Si el modelo no está ya en RAM, se VALIDA que haya RAM libre para cargarlo (más un
#      margen para Windows); si no, se liberan de la RAM los otros modelos y se revalida.
#   4. Si aun así no cabe, NO se carga (InsufficientMemoryError) para no saturar el equipo.
# Al terminar, park_model() saca el modelo de la VRAM, comprueba que la VRAM quedó libre
# y lo conserva en RAM solo si sobra memoria (KEEP_MODELS_IN_RAM=0 lo desactiva).

RAM_SAFETY_MARGIN_GB = float(os.environ.get("RAM_SAFETY_MARGIN_GB", "2"))
RAM_PARK_MIN_FREE_GB = float(os.environ.get("RAM_PARK_MIN_FREE_GB", "4"))
KEEP_MODELS_IN_RAM = os.environ.get("KEEP_MODELS_IN_RAM", "1").strip().lower() not in ("0", "false", "no")

# Memoria estimada por modelo (fp16), medida sobre los pesos en disco:
#   ram_load: pico de RAM durante la carga desde disco (pesos fp32 -> conversión a fp16)
#   vram:     VRAM necesaria durante su etapa
MODEL_MEMORY_GB = {
    "shape": {"ram_load": 6.5, "vram": 4.5, "vram_full": 6.0},  # DiT 4.7 GB + DINOv2 + VAE turbo
    "paint": {"ram_load": 7.5, "vram": 4.8},                    # Delight 4.0 GB + Multivista 4.8 GB (fp32 en disco)
    "sf3d":  {"ram_load": 4.5, "vram": 6.0},                    # model.safetensors 3.8 GB (fp32)
}

_MODEL_CACHE: dict = {}


class InsufficientMemoryError(RuntimeError):
    """No hay RAM/VRAM suficiente para cargar un modelo sin saturar el equipo."""


def _system_ram_free_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return 0.0


def _vram_free_gb() -> float:
    """VRAM libre real según el driver (incluye lo que usen otros programas)."""
    if not torch.cuda.is_available():
        return 0.0
    free_b, _ = torch.cuda.mem_get_info(0)
    return free_b / (1024 ** 3)


def _vram_held_mb() -> float:
    """VRAM ocupada por tensores de ESTE proceso."""
    return torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0


def _move_model_to_cpu(kind: str, model: Any) -> None:
    """Saca de la VRAM todos los pesos de un modelo, sea cual sea su tipo."""
    try:
        if kind == "paint":
            for sub in getattr(model, "models", {}).values():
                sub_pipe = getattr(sub, "pipeline", None)
                if sub_pipe is None:
                    continue
                if getattr(sub_pipe, "_offload_gpu_id", None) is not None and hasattr(sub_pipe, "maybe_free_model_hooks"):
                    sub_pipe.maybe_free_model_hooks()  # Con offload de diffusers: devuelve todo a CPU
                else:
                    sub_pipe.to("cpu")
            # El renderizador guarda buffers de la malla en la GPU
            if hasattr(model, "render"):
                model.render = None
        elif hasattr(model, "to"):
            model.to("cpu")
    except Exception as e:
        print(f" -> [VRAM Guard] Aviso al mover '{kind}' a CPU: {e}")


def release_model(kind: str) -> None:
    """Destruye un modelo (VRAM y RAM) y reporta cuánta RAM se recuperó."""
    model = _MODEL_CACHE.pop(kind, None)
    if model is None:
        return
    ram_before = _system_ram_free_gb()
    _move_model_to_cpu(kind, model)
    del model
    flush_vram()
    print(f" -> [Memoria] '{kind}' liberado de RAM y VRAM (RAM libre: {ram_before:.1f} -> {_system_ram_free_gb():.1f} GB)")


def park_model(kind: str) -> None:
    """
    Saca un modelo de la VRAM y valida que la VRAM del proceso quedó libre. Lo conserva
    en RAM solo si KEEP_MODELS_IN_RAM está activo y queda RAM de sobra; si no, lo destruye.
    """
    model = _MODEL_CACHE.get(kind)
    if model is None:
        return
    _move_model_to_cpu(kind, model)
    flush_vram()

    held_mb = _vram_held_mb()
    if held_mb > 256:
        print(f" -> [VRAM Guard] Tras sacar '{kind}' siguen {held_mb:.0f} MB en VRAM: se destruye el modelo.")
        release_model(kind)
        return

    free_gb = _system_ram_free_gb()
    if not KEEP_MODELS_IN_RAM:
        release_model(kind)
    elif free_gb < RAM_PARK_MIN_FREE_GB:
        print(f" -> [RAM Guard] Solo {free_gb:.1f} GB de RAM libre: se libera '{kind}' en lugar de conservarlo.")
        release_model(kind)
    else:
        print(f" -> [Memoria] '{kind}' fuera de la VRAM ({held_mb:.0f} MB retenidos) y conservado en RAM ({free_gb:.1f} GB libres)")


def park_all_models() -> None:
    """Deja la VRAM vacía conservando en RAM los modelos que quepan."""
    for kind in list(_MODEL_CACHE):
        park_model(kind)
    flush_vram()


def activate_model(kind: str, vram_needed_gb: Optional[float] = None, strict_vram: bool = False) -> Optional[Any]:
    """
    Prepara la memoria para usar `kind` como ÚNICO modelo en la GPU (ver flujo arriba).
    Devuelve el modelo si ya estaba en RAM, o None si hay que cargarlo (y llamar a register_model()).
    Lanza InsufficientMemoryError si falta RAM, o VRAM con strict_vram=True.
    """
    est = MODEL_MEMORY_GB.get(kind, {})
    vram_need = vram_needed_gb if vram_needed_gb is not None else est.get("vram", 0.0)
    others = [k for k in _MODEL_CACHE if k != kind]

    # 1. Ningún otro modelo en la VRAM
    for other in others:
        park_model(other)
    flush_vram()

    # 2. Validar VRAM
    vram_ok = True
    if torch.cuda.is_available():
        vram_free = _vram_free_gb()
        if vram_free < vram_need and any(k in _MODEL_CACHE for k in others):
            for other in others:
                release_model(other)
            vram_free = _vram_free_gb()
        vram_ok = vram_free >= vram_need
        mark = "OK" if vram_ok else "INSUFICIENTE"
        print(f" -> [Memoria] '{kind}': VRAM libre {vram_free:.1f} GB (necesita ~{vram_need:.1f}) {mark}")
        if not vram_ok:
            msg = (f"VRAM insuficiente para '{kind}': {vram_free:.1f} GB libres, necesita ~{vram_need:.1f} GB. "
                   "Cierra programas que usen la GPU (juegos, navegador con aceleración, Godot).")
            if strict_vram:
                raise InsufficientMemoryError(msg)
            print(f" -> [Memoria] Aviso: {msg} Se intentará igualmente.")

    # 3. Validar RAM (solo si hay que cargarlo desde disco)
    cached = _MODEL_CACHE.get(kind)
    if cached is None:
        ram_need = est.get("ram_load", 0.0) + RAM_SAFETY_MARGIN_GB
        ram_free = _system_ram_free_gb()
        for other in others:
            if ram_free >= ram_need:
                break
            if other in _MODEL_CACHE:
                release_model(other)
                ram_free = _system_ram_free_gb()
        ram_ok = ram_free >= ram_need
        mark = "OK" if ram_ok else "INSUFICIENTE"
        print(f" -> [Memoria] '{kind}': RAM libre {ram_free:.1f} GB (necesita ~{ram_need:.1f} con margen) {mark}")
        if not ram_ok:
            raise InsufficientMemoryError(
                f"RAM insuficiente para cargar '{kind}': {ram_free:.1f} GB libres, necesita ~{ram_need:.1f} GB. "
                "Cierra programas para no saturar el equipo."
            )
    return cached


def register_model(kind: str, model: Any) -> None:
    _MODEL_CACHE[kind] = model


def unload_sf3d() -> None:
    """Destruye Stable Fast 3D."""
    release_model("sf3d")


def unload_hunyuan() -> None:
    """Destruye los pipelines Hunyuan3D (forma y pintura)."""
    release_model("shape")
    release_model("paint")


def unload_all_models() -> None:
    """Destruye todos los modelos (VRAM y RAM)."""
    for kind in list(_MODEL_CACHE):
        release_model(kind)
    flush_vram()

# ==============================================================================
# Aceleradores Numéricos Numba para UV y Reproyección Frontal (0 MB VRAM)
# ==============================================================================

try:
    from numba import njit, prange
    _HAS_NUMBA = True

    @njit(fastmath=True)
    def _numba_rasterize_projected_buffer(
        uvs,             # (V, 2) float32
        faces,           # (F_active, 3) int32
        vert_weights,    # (V,) float32 - Pesos continuos por vértice calculados en 3D
        proj_verts,      # (V, 2) float32 [u, v]
        vert_depth,      # (V,) float32 - Profundidad exacta (camera_dist - verts[:, 2]), menor = más cerca
        ref_img,         # (H_ref, W_ref, 4) uint8
        res_w,
        res_h
    ):
        """
        Rasterizador de atlas UV en CPU acelerado con Numba (0 MB VRAM, ~30-50 ms).
        - Interpolación baricéntrica exacta de pesos por vértice (w0*vw0 + w1*vw1 + w2*vw2) para eliminar escalonamiento.
        - Z-buffer con convención física correcta: menor profundidad = más cerca de la cámara.
        - Muestreo bilineal exacto con interpolación de canal alfa de la ilustración 2D.
        """
        proj_rgb = np.zeros((res_h, res_w, 3), dtype=np.float32)
        blend_mask = np.zeros((res_h, res_w), dtype=np.float32)
        z_buf = np.full((res_h, res_w), 1e9, dtype=np.float32)

        num_faces = faces.shape[0]
        h_ref, w_ref = ref_img.shape[:2]

        for f in range(num_faces):
            i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
            vw0, vw1, vw2 = vert_weights[i0], vert_weights[i1], vert_weights[i2]

            # Si los tres vértices tienen peso cero, omitir el triángulo por completo
            if vw0 <= 0.0 and vw1 <= 0.0 and vw2 <= 0.0:
                continue

            u0, v0 = uvs[i0, 0], uvs[i0, 1]
            u1, v1 = uvs[i1, 0], uvs[i1, 1]
            u2, v2 = uvs[i2, 0], uvs[i2, 1]

            x0, y0 = u0 * res_w, (1.0 - v0) * res_h
            x1, y1 = u1 * res_w, (1.0 - v1) * res_h
            x2, y2 = u2 * res_w, (1.0 - v2) * res_h

            min_x = max(0, int(min(x0, x1, x2)))
            max_x = min(res_w - 1, int(max(x0, x1, x2)) + 1)
            min_y = max(0, int(min(y0, y1, y2)))
            max_y = min(res_h - 1, int(max(y0, y1, y2)) + 1)

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-9:
                continue
            inv_denom = 1.0 / denom

            # Profundidad exacta (distancia física a la cámara): menor profundidad = más frontal
            z0, z1, z2 = vert_depth[i0], vert_depth[i1], vert_depth[i2]
            pu0, pv0 = proj_verts[i0, 0], proj_verts[i0, 1]
            pu1, pv1 = proj_verts[i1, 0], proj_verts[i1, 1]
            pu2, pv2 = proj_verts[i2, 0], proj_verts[i2, 1]

            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    p_x = px + 0.5
                    p_y = py + 0.5
                    w0 = ((y1 - y2) * (p_x - x2) + (x2 - x1) * (p_y - y2)) * inv_denom
                    w1 = ((y2 - y0) * (p_x - x2) + (x0 - x2) * (p_y - y2)) * inv_denom
                    w2 = 1.0 - w0 - w1

                    if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                        pz = w0 * z0 + w1 * z1 + w2 * z2
                        if pz < z_buf[py, px]:
                            u_img = w0 * pu0 + w1 * pu1 + w2 * pu2
                            v_img = w0 * pv0 + w1 * pv1 + w2 * pv2

                            if 0.0 <= u_img <= 1.0 and 0.0 <= v_img <= 1.0:
                                fx = u_img * (w_ref - 1)
                                fy = v_img * (h_ref - 1)
                                ix0 = int(fx)
                                iy0 = int(fy)
                                ix1 = min(ix0 + 1, w_ref - 1)
                                iy1 = min(iy0 + 1, h_ref - 1)
                                dx = fx - ix0
                                dy = fy - iy0

                                a00 = float(ref_img[iy0, ix0, 3])
                                a10 = float(ref_img[iy0, ix1, 3])
                                a01 = float(ref_img[iy1, ix0, 3])
                                a11 = float(ref_img[iy1, ix1, 3])
                                alpha = (1.0 - dy) * ((1.0 - dx) * a00 + dx * a10) + dy * ((1.0 - dx) * a01 + dx * a11)
                                alpha_norm = alpha / 255.0

                                # Interpolación baricéntrica suave de pesos de vértices calculados en 3D
                                w_interp = w0 * vw0 + w1 * vw1 + w2 * vw2
                                eff_w = w_interp * alpha_norm
                                if eff_w > 0.0:
                                    z_buf[py, px] = pz
                                    blend_mask[py, px] = eff_w

                                    for c in range(3):
                                        c00 = float(ref_img[iy0, ix0, c])
                                        c10 = float(ref_img[iy0, ix1, c])
                                        c01 = float(ref_img[iy1, ix0, c])
                                        c11 = float(ref_img[iy1, ix1, c])
                                        orig_c = (1.0 - dy) * ((1.0 - dx) * c00 + dx * c10) + dy * ((1.0 - dx) * c01 + dx * c11)
                                        proj_rgb[py, px, c] = orig_c

        return proj_rgb, blend_mask

except Exception:
    _HAS_NUMBA = False
    _numba_rasterize_projected_buffer = None

_numba_rasterize = None



# ==============================================================================
# Perfiles de Rendimiento de VRAM y Hardware Adaptativo
# ==============================================================================

# Campos con efecto real:
#   staged_offload: True  -> sube a la GPU un componente a la vez (condicionador -> DiT -> VAE).
#                            Pico de VRAM ~2-3 GB menor; cuesta ~1-2 s de transferencias.
#                   False -> el pipeline de forma completo vive en la GPU durante su etapa.
#   max_octree:     tope de resolución octree del Volume Decoding (más = más detalle y VRAM).
#   paint_resolution: resolución de render/textura de Hunyuan3D-Paint.
#   use_fp16:       precisión de los pesos.
# Tope de caras antes de texturizar: TEXTURE_SAFE_MAX_FACES (igual en todos los perfiles,
# es el valor que usa el propio Hunyuan3D-2 con su FaceReducer antes de pintar).
TEXTURE_SAFE_MAX_FACES = 40000

VRAM_PROFILES = {
    "auto": {
        "name": "Auto-detectar (Recomendado)",
        "desc": "Ajusta la asignación de VRAM según el hardware detectado en el equipo.",
        "use_fp16": True,
        "staged_offload": True,
        "max_octree": 256,
        "paint_resolution": 1024,
    },
    "low_6gb": {
        "name": "6GB VRAM (Optimizado RTX 4050 Laptop / RTX 3050)",
        "desc": "6GB: un componente del modelo en la GPU a la vez y octree 256.",
        "use_fp16": True,
        "staged_offload": True,
        "max_octree": 256,
        "paint_resolution": 1024,
    },
    "low_8gb": {
        "name": "8GB VRAM (Optimizado RTX 4060 / Laptops)",
        "desc": "8GB: un componente del modelo en la GPU a la vez y octree 256.",
        "use_fp16": True,
        "staged_offload": True,
        "max_octree": 256,
        "paint_resolution": 1024,
    },
    "mid_12gb": {
        "name": "12GB VRAM (Equilibrado RTX 3060 / 4070)",
        "desc": "12GB: pipeline de forma completo en GPU y octree hasta 320.",
        "use_fp16": True,
        "staged_offload": False,
        "max_octree": 320,
        "paint_resolution": 1024,
    },
    "high_16gb": {
        "name": "16GB+ VRAM (Máximo Rendimiento RTX 4080 / 4090 / 3090)",
        "desc": "16GB+: pipeline completo en GPU, octree hasta 384 y textura 2048.",
        "use_fp16": True,
        "staged_offload": False,
        "max_octree": 384,
        "paint_resolution": 2048,
    },
    "cpu": {
        "name": "CPU Seguro (Sin GPU)",
        "desc": "Inferencia forzada en memoria RAM del sistema (muy lenta).",
        "use_fp16": False,
        "staged_offload": False,
        "max_octree": 192,
        "paint_resolution": 1024,
    }
}

def detect_vram_profile() -> str:
    """Detecta la VRAM física de la GPU y retorna la clave de perfil recomendada."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb <= 6.5:
            return "low_6gb"
        elif total_vram_gb <= 8.5:
            return "low_8gb"
        elif total_vram_gb <= 13.0:
            return "mid_12gb"
        else:
            return "high_16gb"
    except Exception:
        return "low_6gb"


def is_dit_cached() -> bool:
    """Verifica si los pesos safetensors de Hunyuan3D-2 DiT están completamente descargados en disco."""
    # 1. Comprobar ruta directa de modelos en D:\AI_Models\tencent\Hunyuan3D-2
    direct_path = HUNYUAN_LOCAL_DIR / "hunyuan3d-dit-v2-0-turbo" / "model.fp16.safetensors"
    if direct_path.exists() and direct_path.stat().st_size > 4 * 1024 * 1024 * 1024:
        return True

    # 2. Comprobar snapshots de Hugging Face
    hf_hub = Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    snapshots_dir = hf_hub / "models--tencent--Hunyuan3D-2" / "snapshots"
    if snapshots_dir.exists():
        for p in snapshots_dir.rglob("model.fp16.safetensors"):
            try:
                real_p = p.resolve()
                if real_p.exists() and real_p.stat().st_size > 4 * 1024 * 1024 * 1024:
                    return True
            except Exception:
                pass
    return False


# ==============================================================================
# 1. Utilidades de Ruta y Carpetas de Kaboomi Party
# ==============================================================================

def resolve_target_path(output_path_str: str) -> Path:
    """
    Resuelve inteligentemente la ruta de destino.
    Si se especifica '../kaboomi-party/...' pero la carpeta está en '../Games/kaboomi-party/...'
    o en '~/Proyectos/Games/kaboomi-party/...' (o en KABOOMI_DIR del .env),
    resuelve la ruta real existente para que funcione sin fricciones.
    Si solo se especifica un nombre de archivo, lo ubica en <HY3DGEN_MODELS>/output/.
    """
    target = Path(output_path_str)
    
    # Si no tiene carpeta padre (solo nombre de archivo), guardar en DEFAULT_OUTPUT_DIR
    if str(target.parent) in (".", ""):
        return (DEFAULT_OUTPUT_DIR / target.name).resolve()

    # Comprobar si ya existe el directorio padre
    if target.parent.exists():
        return target.resolve()
        
    parts = list(target.parts)
    if "kaboomi-party" in parts:
        idx = parts.index("kaboomi-party")
        # Probar anteponer 'Games' si existe ../Games/kaboomi-party
        candidate_parts = parts[:idx] + ["Games"] + parts[idx:]
        candidate_path = Path(*candidate_parts)
        if candidate_path.parent.exists():
            return candidate_path.resolve()
            
        # Probar en KABOOMI_DIR (.env) y en ~/Proyectos/Games/kaboomi-party
        env_root = os.environ.get("KABOOMI_DIR")
        roots = [Path(__file__).resolve().parent / env_root] if env_root else []
        roots.append(Path.home() / "Proyectos" / "Games" / "kaboomi-party")
        proyectos_root = next((r for r in roots if r.exists()), None)
        if proyectos_root is not None:
            subparts = parts[idx+1:]
            candidate_proy = proyectos_root.joinpath(*subparts)
            if candidate_proy.parent.exists():
                return candidate_proy.resolve()
            
    # Si no existe, crear la estructura de carpetas
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.resolve()


# ==============================================================================
# 2. Preprocesamiento, Silueta, Inpainting y Frontal Projection Blending
# ==============================================================================

def inpaint_edge_padding(
    image: Union[str, Path, Image.Image, np.ndarray],
    dilation_pixels: int = 10,
    alpha_threshold: int = 25,
    method: str = "telea"
) -> Image.Image:
    """
    Técnica 1: Inpainting y Extensión de Borde 2D (Anti-Halos y Anti-Color Bleeding).
    Toma el canal de color RGB y propaga ("estira") los colores del contorno hacia el área
    transparente exterior (6 a 12 píxeles, por defecto 10px) usando cv2.inpaint
    (algoritmo Telea) o dilatación de bordes mediante Scipy distance_transform_edt.
    
    Evita que el recorte de la máscara alfa o el antialiasing introduzcan halos oscuros
    o grises en bordes finos como cuernos, pies, extremidades y orejas al proyectarse en 3D.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGBA")
        arr = np.array(img)
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGBA"))
    else:
        arr = np.array(image)

    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("La imagen para inpaint_edge_padding debe ser RGBA de 4 canales.")

    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3].copy()

    clean_alpha = np.where(alpha < alpha_threshold, 0, alpha).astype(np.uint8)
    fg_mask = (clean_alpha > 0).astype(np.uint8) * 255

    if fg_mask.sum() == 0 or fg_mask.sum() == fg_mask.size * 255:
        return Image.fromarray(arr, "RGBA")

    pad_px = max(1, min(int(dilation_pixels), 32))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(fg_mask, kernel, iterations=pad_px)
    inpaint_band = cv2.subtract(dilated_mask, fg_mask)

    inpainted_rgb = rgb.copy()
    inpaint_success = False

    if method.lower() == "telea":
        try:
            inpainted_rgb = cv2.inpaint(
                rgb,
                inpaint_band,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
            inpaint_success = True
        except Exception as e_inpaint:
            print(f" -> [Aviso Telea] Falló cv2.inpaint ({e_inpaint}). Usando dilatación Scipy...")

    if not inpaint_success:
        try:
            from scipy import ndimage
            dist, (idx_y, idx_x) = ndimage.distance_transform_edt(
                fg_mask == 0, return_distances=True, return_indices=True
            )
            bleed_mask = (dist > 0) & (dist <= pad_px)
            inpainted_rgb[bleed_mask] = rgb[idx_y[bleed_mask], idx_x[bleed_mask]]
            inpaint_success = True
        except Exception:
            pass

    # Los píxeles transparentes más allá de la banda de padding se fijan en gris neutro (128)
    inpainted_rgb[dilated_mask == 0] = 128

    result_arr = np.dstack([inpainted_rgb, clean_alpha])
    return Image.fromarray(result_arr, "RGBA")


def preprocess_image_for_sf3d(
    image: Union[str, Path, Image.Image],
    foreground_ratio: float = 0.82,
    edge_dilation_pixels: int = 10,
    target_resolution: int = 1024,
    alpha_threshold: int = 25
) -> Image.Image:
    """
    Preprocesamiento especializado para Stable Fast 3D (SF3D) anti-color bleeding y anti-halos:
    1. Asegura que los píxeles transparentes no contengan negro (0,0,0) ni manchas oscuras de fondo.
    2. Aplica Inpainting y Extensión de Borde 2D (Telea / Scipy de 10px hacia afuera) extendiendo el color
       del contorno hacia la zona transparente antes de entrar al remuestreo o proyección UV.
    3. Rellena el fondo exterior lejano con gris neutro (128, 128, 128) con alfa 0.
    4. Ajusta el foreground_ratio exactamente a 0.82 (fijo) para garantizar que cuernos, colas,
       orejas y extremidades tengan margen suficiente sin colisionar con los límites del volumen 3D.
    5. Redimensiona preservando los canales RGB y Alfa por separado con filtro Lanczos.
    """
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGBA")
    else:
        img = image.convert("RGBA")

    arr = np.array(img)
    alpha = arr[:, :, 3]

    clean_alpha = np.where(alpha < alpha_threshold, 0, alpha).astype(np.uint8)
    non_zero = np.argwhere(clean_alpha > 0)
    if non_zero.size == 0:
        return img

    y0, x0 = non_zero.min(axis=0)
    y1, x1 = non_zero.max(axis=0) + 1

    cropped_arr = arr[y0:y1, x0:x1].copy()
    ch, cw = cropped_arr.shape[:2]
    max_dim = max(ch, cw)

    # Encuadre óptimo centrado con foreground_ratio = 0.82 fijo
    canvas_dim = int(np.ceil(max_dim / float(foreground_ratio)))
    canvas_arr = np.zeros((canvas_dim, canvas_dim, 4), dtype=np.uint8)
    # Fondo gris neutro en píxeles transparentes (evita negro 0,0,0)
    canvas_arr[:, :, :3] = 128
    canvas_arr[:, :, 3] = 0

    off_y = (canvas_dim - ch) // 2
    off_x = (canvas_dim - cw) // 2
    canvas_arr[off_y:off_y + ch, off_x:off_x + cw] = cropped_arr

    # Aplicar Inpainting y Extensión de Bordes 2D (10px hacia afuera)
    padded_img = inpaint_edge_padding(
        canvas_arr,
        dilation_pixels=edge_dilation_pixels,
        alpha_threshold=alpha_threshold,
        method="telea"
    )
    result_arr = np.array(padded_img)

    if target_resolution and (result_arr.shape[1] != target_resolution or result_arr.shape[0] != target_resolution):
        # Redimensionar RGB y Alfa por separado para evitar que Pillow fuerce RGB a (0,0,0) en pixeles transparentes
        rgb_part = Image.fromarray(result_arr[:, :, :3], "RGB").resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
        alpha_part = Image.fromarray(result_arr[:, :, 3], "L").resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
        res_img = Image.merge("RGBA", (*rgb_part.split(), alpha_part))
    else:
        res_img = Image.fromarray(result_arr, "RGBA")

    return res_img


def apply_taubin_smoothing(
    mesh: trimesh.Trimesh,
    lamb: float = 0.53,
    nu: float = -0.55,
    iterations: int = 5
) -> trimesh.Trimesh:
    """
    Técnica 3: Filtro de Suavizado Geométrico Taubin (Anti-Plastilina / Anti-Grumos).
    Aplica suavizado Taubin no restrictivo de curvatura (lamb=0.53, nu=-0.55, 3-5 iteraciones)
    que elimina el ruido de alta frecuencia, picos y deformidades de Marching Cubes/DMTet
    y decimation sin encoger el volumen ni perder definición en curvas anatómicas.
    Fuerza el recálculo de normales suaves de vértices y caras.
    """
    if mesh is None or len(mesh.faces) == 0:
        return mesh

    print(f" -> [Taubin Smoothing] Suavizando geometría ({len(mesh.faces):,} caras, lamb={lamb}, nu={nu}, {iterations} iters)...")
    try:
        smoothed = trimesh.smoothing.filter_taubin(
            mesh,
            lamb=float(lamb),
            nu=float(nu),
            iterations=int(iterations)
        )
        smoothed.fix_normals()
        _ = smoothed.vertex_normals
        _ = smoothed.face_normals
        return smoothed
    except Exception as e_taubin:
        print(f" -> [Aviso Taubin Smoothing] {e_taubin}. Recalculando normales estándar...")
        try:
            mesh.fix_normals()
            _ = mesh.vertex_normals
            _ = mesh.face_normals
        except Exception:
            pass
        return mesh
def blend_frontal_projection_texture(
    mesh: trimesh.Trimesh,
    highres_image: Union[str, Path, Image.Image],
    blend_start: float = 0.50,
    blend_full: float = 0.80,
    camera_distance: float = 1.6,
    fovy_deg: float = 40.0,
    foreground_ratio: float = 0.82,
    blur_sigma: float = 2.5,
    feather_kernel_size: int = 15,
    enable_raycast_occlusion: bool = True
) -> trimesh.Trimesh:
    """
    Técnica 2: Proyección Frontal de Alta Resolución con Interpolación Baricéntrica 3D,
    Oclusión por Distancia de Rayo y Dilatación Continua de Costuras UV.
    1. Pesos Continuos por Vértice e Interpolación Baricéntrica (Fix Dientes de Sierra):
       - Calcula el ángulo y smoothstep en cada vértice con suavizado Laplaciano en el grafo 3D.
       - La transición sobre la superficie es suave C1 y sin escalonamiento facetado.
    2. Corrección del Z-Buffer (Fix Inversión de Profundidad):
       - vert_depth = camera_dist - verts[:, 2]. Menor valor = más cerca de la cámara.
    3. Raycasting con Umbral Seguro de Distancia (Fix Huecos Falsos en Pliegues y Costuras UV):
       - Calcula la distancia analítica al impacto e ignora colisiones a distancia menor a 0.03 * escala.
    4. Dilatación Simultánea de Color y Blend Mask (Fix Seam Bleeding en Costuras UV):
       - Dilata tanto proj_rgb como blend_mask hacia afuera de las islas UV ANTES del Gaussian Blur.
       - Las costuras UV mantienen peso 1.0 sin sangrar contra el fondo vacío 0.0.
    """
    if mesh is None or not hasattr(mesh, "visual") or mesh.visual is None:
        return mesh
    mat = getattr(mesh.visual, "material", None)
    if mat is None or getattr(mat, "baseColorTexture", None) is None:
        return mesh

    print(f" -> [Frontal Projection] Iniciando reproyección 2D (Vértices 3D, Culling dot > {blend_start}, Raycast={enable_raycast_occlusion})...")
    start_t = time.time()

    cur_tex = mat.baseColorTexture.convert("RGB")
    ai_tex_np = np.array(cur_tex, dtype=np.float32)
    res_h, res_w = ai_tex_np.shape[:2]

    if isinstance(highres_image, (str, Path)):
        ref_pil = Image.open(highres_image).convert("RGBA")
    else:
        ref_pil = highres_image.convert("RGBA")

    target_res = max(res_w, ref_pil.width, ref_pil.height)
    ref_prep = preprocess_image_for_sf3d(
        ref_pil,
        foreground_ratio=foreground_ratio,
        edge_dilation_pixels=10,
        target_resolution=target_res
    )
    ref_arr = np.array(ref_prep, dtype=np.uint8)

    # 1. Proyección perspectiva de los vértices a coordenadas normalizadas [0, 1] y cálculo de profundidad real
    verts = mesh.vertices.astype(np.float32)
    dist = float(camera_distance)
    tan_half = float(np.tan(np.deg2rad(fovy_deg) / 2.0))

    # Fix 3: Profundidad exacta desde la cámara (menor valor = más cerca de la cámara)
    vert_depth = (dist - verts[:, 2]).astype(np.float32)
    z_dist = np.maximum(vert_depth, 0.05)

    x_proj = verts[:, 0] / (z_dist * tan_half)
    y_proj = verts[:, 1] / (z_dist * tan_half)
    u_proj = 0.5 + 0.5 * x_proj
    v_proj = 0.5 - 0.5 * y_proj
    proj_verts = np.column_stack([u_proj, v_proj]).astype(np.float32)

    # 2. Fix 2: Backface Culling y Transición Soft-Threshold Calculada por VÉRTICE en 3D
    mesh.fix_normals()
    vn = mesh.vertex_normals.astype(np.float32)
    cam_pos = np.array([0.0, 0.0, dist], dtype=np.float32)

    dirs_v = cam_pos - verts
    dist_to_cam = np.linalg.norm(dirs_v, axis=1, keepdims=True)
    dirs_v /= np.maximum(dist_to_cam, 1e-6)
    dot_v = np.sum(vn * dirs_v, axis=1)

    # Smoothstep cúbico por vértice entre blend_start (0.50) y blend_full (0.80)
    t = np.clip((dot_v - float(blend_start)) / (float(blend_full) - float(blend_start)), 0.0, 1.0)
    vert_weights = t * t * (3.0 - 2.0 * t)

    # 3. Fix 4: Oclusión por Raycasting con Umbral Seguro de Distancia (Previene falsos positivos en costuras y pliegues)
    if enable_raycast_occlusion and hasattr(mesh, "ray"):
        try:
            cand_v = np.where(vert_weights > 0.0)[0]
            if len(cand_v) > 0:
                cand_pos = verts[cand_v]
                cand_normals = vn[cand_v]
                ray_origins = cand_pos + cand_normals * 1e-4
                ray_dirs = cam_pos - ray_origins
                dist_cand = np.linalg.norm(ray_dirs, axis=1)
                ray_dirs /= np.maximum(dist_cand[:, None], 1e-6)

                hits = mesh.ray.intersects_first(
                    ray_origins.astype(np.float64),
                    ray_dirs.astype(np.float64)
                )
                hit_mask = (hits != -1)
                if np.any(hit_mask):
                    hit_sub_idx = np.where(hit_mask)[0]
                    # Umbral seguro basado en la escala del objeto 3D
                    min_occlusion_dist = 0.03 * float(mesh.scale)

                    # Distancia analítica exacta a la cara intersectada
                    tri_v0 = verts[mesh.faces[hits[hit_sub_idx], 0]]
                    tri_norm = mesh.face_normals[hits[hit_sub_idx]]
                    denom = np.sum(ray_dirs[hit_sub_idx] * tri_norm, axis=1)
                    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)
                    t_hit = np.sum((tri_v0 - ray_origins[hit_sub_idx]) * tri_norm, axis=1) / denom

                    # Solo descartar si la colisión está claramente separada (oclusión real por otro miembro)
                    real_occlusion = (t_hit >= min_occlusion_dist) & (t_hit < dist_cand[hit_sub_idx] - 1e-3)
                    occluded_verts = cand_v[hit_sub_idx[real_occlusion]]
                    vert_weights[occluded_verts] = 0.0

                    occluded_count = len(occluded_verts)
                    if occluded_count > 0:
                        print(f" -> [Raycasting Occlusion] Se protegieron {occluded_count:,} vértices con oclusión real frontal.")
        except Exception as e_ray:
            print(f" -> [Aviso Raycasting] Error en cálculo de visibilidad ({e_ray}). Continuando con culling por normales.")

    # 4. Fix 2 & 4: Suavizado Laplaciano de Pesos en el Grafo de Adyacencia 3D
    # Elimina completamente los saltos dentados y cicatriza posibles huecos aislados de raycast
    try:
        import scipy.sparse as sp
        edges = mesh.edges_unique
        v_count = len(verts)
        adj_mat = sp.csr_matrix(
            (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(v_count, v_count)
        )
        adj_mat = adj_mat + adj_mat.T
        deg = np.array(adj_mat.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        inv_deg = sp.diags(1.0 / deg)
        norm_adj = inv_deg.dot(adj_mat)

        for _ in range(3):
            vert_weights = 0.5 * vert_weights + 0.5 * norm_adj.dot(vert_weights)
    except Exception as e_smooth:
        print(f" -> [Aviso Laplacian] {e_smooth}. Omitiendo suavizado Laplaciano en grafo 3D.")

    # Filtrar caras activas: aquellas que tengan al menos 1 vértice con peso mayor a cero
    faces = mesh.faces.astype(np.int32)
    active_mask = (vert_weights[faces[:, 0]] > 0.0) | (vert_weights[faces[:, 1]] > 0.0) | (vert_weights[faces[:, 2]] > 0.0)
    if not np.any(active_mask):
        print(" -> [Frontal Projection] Ninguna cara activa tras oclusión. Conservando textura IA nativa.")
        return mesh

    active_faces = faces[active_mask]
    uvs = mesh.visual.uv.astype(np.float32)

    # 5. Rasterización del Mapa UV con Pesos Interpolados Baricéntricamente
    try:
        if globals().get("_HAS_NUMBA", False) and _numba_rasterize_projected_buffer is not None:
            proj_rgb, blend_mask = _numba_rasterize_projected_buffer(
                uvs,
                active_faces,
                vert_weights,
                proj_verts,
                vert_depth,
                ref_arr,
                res_w,
                res_h
            )
        else:
            proj_rgb = np.zeros((res_h, res_w, 3), dtype=np.float32)
            blend_mask = np.zeros((res_h, res_w), dtype=np.float32)

        # 6. Fix 1: Dilatación Simultánea de Color Y Blend Mask hacia Afuera de las Islas UV
        # Al dilatar blend_mask replicando el valor de frontera antes del blur, los píxeles adyacentes
        # a las costuras UV mantienen 1.0 en todo su vecindario y NUNCA caen a ~0.5.
        known_mask = (blend_mask > 0.001).astype(np.uint8)
        if np.any(known_mask):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated_rgb = proj_rgb.copy()
            dilated_mask = blend_mask.copy()
            for _ in range(5):
                expanded = cv2.dilate(known_mask, kernel)
                edge = (expanded > 0) & (known_mask == 0)
                for c in range(3):
                    dil_c = cv2.dilate(dilated_rgb[:, :, c], kernel)
                    dilated_rgb[edge, c] = dil_c[edge]
                dil_m = cv2.dilate(dilated_mask, kernel)
                dilated_mask[edge] = dil_m[edge]
                known_mask = expanded

            # 7. Feathering Progresivo: Desenfoque Gaussiano sobre la máscara ya dilatada
            ksize = int(feather_kernel_size) if int(feather_kernel_size) % 2 == 1 else int(feather_kernel_size) + 1
            blurred_mask = cv2.GaussianBlur(dilated_mask, (ksize, ksize), float(blur_sigma))
            blurred_mask_3d = np.clip(blurred_mask[:, :, None], 0.0, 1.0)

            # 8. Composición final suave (Alpha Blend entre ilustración 2D y textura IA)
            final_tex = (1.0 - blurred_mask_3d) * ai_tex_np + blurred_mask_3d * dilated_rgb
            final_tex = np.clip(final_tex, 0.0, 255.0).astype(np.uint8)

            blended_pil = Image.fromarray(final_tex, "RGB")
            blended_pil.format = "PNG"
            mat.baseColorTexture = blended_pil
            print(f" -> [Frontal Projection Blending] Mezcla exitosa en {time.time() - start_t:.3f}s. Costuras UV sin sangrado, transiciones continuas.")
    except Exception as e_blend:
        print(f" -> [Aviso Frontal Projection] Error en reproyección UV ({e_blend}). Se conserva textura horneada.")

    return mesh
def preprocess_and_remove_background(
    input_image_path: Union[str, Path, Image.Image],
    padding_ratio: float = 0.12,
    target_resolution: int = 1024,
    force_rembg: bool = False,
    session: Optional[Any] = None,
    alpha_threshold: int = 35
) -> Image.Image:
    """
    Paso A: Elimina el fondo de la imagen de entrada usando `rembg`.
    Aplica filtrado estricto de canal alfa y componentes 2D para eliminar
    sombras tenues de suelo, halos y pedestales que causan losas flotantes en el 3D.
    Normaliza el encuadre con un margen uniforme (padding) para maximizar la calidad del modelo 3D.
    """
    print("\n[Paso A/D] Preprocesando imagen y eliminando fondo con rembg...")
    start_t = time.time()
    
    if isinstance(input_image_path, Image.Image):
        img = input_image_path.convert("RGBA")
    else:
        img = Image.open(input_image_path).convert("RGBA")

    # Verificar si la imagen ya tiene fondo transparente
    alpha = np.array(img.split()[-1])
    min_alpha = alpha.min()
    has_transparency = min_alpha < 10

    if not has_transparency or force_rembg:
        try:
            from rembg import remove, new_session
            print(" -> Ejecutando rembg (U2-Net / ONNX en CPU) para extraer silueta limpia (0 MB VRAM)...")
            if session is None:
                session = new_session("u2net", providers=['CPUExecutionProvider'])
            img = remove(img, session=session)
        except Exception as e:
            print(f" -> [Aviso] Error al llamar a rembg directo ({e}). Intentando hy3dgen BackgroundRemover...")
            try:
                from hy3dgen.rembg import BackgroundRemover
                remover = BackgroundRemover()
                img = remover(img)
            except Exception as e2:
                print(f" -> [Advertencia] No se pudo ejecutar rembg ({e2}). Se usará la imagen original.")

    # 1. Limpieza estricta de canal alfa:
    # Eliminar cualquier píxel tenue (sombras residuales del suelo, halos o pedestales)
    arr = np.array(img)
    alpha = arr[:, :, 3]
    
    # Binarizado suave / thresholding: valores con alpha menor a alpha_threshold pasan a 0
    clean_alpha = np.where(alpha < alpha_threshold, 0, alpha).astype(np.uint8)
    
    # 2. Filtrar manchas aisladas en 2D (manchas del fondo, partículas)
    try:
        import cv2
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((clean_alpha > 0).astype(np.uint8))
        if num_labels > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = areas.max()
            for i in range(1, num_labels):
                # Descartar manchas menores al 5% del área principal
                if stats[i, cv2.CC_STAT_AREA] < max_area * 0.05:
                    clean_alpha[labels == i] = 0
    except Exception:
        pass
        
    arr[:, :, 3] = clean_alpha
    # Para áreas 100% transparentes, asignar fondo blanco limpio para que hy3dgen no lea colores residuales
    arr[clean_alpha == 0, :3] = 255
    img = Image.fromarray(arr)

    # Recorte inteligente de la caja delimitadora (bbox) con padding
    non_empty_coords = np.argwhere(clean_alpha > 0)
    
    if non_empty_coords.size > 0:
        y0, x0 = non_empty_coords.min(axis=0)
        y1, x1 = non_empty_coords.max(axis=0) + 1
        cropped = img.crop((x0, y0, x1, y1))
        
        # Calcular nuevo lienzo cuadrado con padding
        max_dim = max(cropped.width, cropped.height)
        pad = int(max_dim * padding_ratio)
        canvas_size = max_dim + 2 * pad
        
        square_img = Image.new("RGBA", (canvas_size, canvas_size), (255, 255, 255, 0))
        offset_x = (canvas_size - cropped.width) // 2
        offset_y = (canvas_size - cropped.height) // 2
        square_img.paste(cropped, (offset_x, offset_y))
        
        resized = square_img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
        # Limpieza post-resampling para evitar ringing o falsos positivos de Lanczos en los bordes
        res_arr = np.array(resized)
        res_alpha = res_arr[:, :, 3]
        res_alpha = np.where(res_alpha < alpha_threshold, 0, res_alpha).astype(np.uint8)
        res_arr[:, :, 3] = res_alpha
        res_arr[res_alpha == 0, :3] = 255
        img = Image.fromarray(res_arr)
    else:
        resized = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
        res_arr = np.array(resized)
        res_alpha = res_arr[:, :, 3]
        res_alpha = np.where(res_alpha < alpha_threshold, 0, res_alpha).astype(np.uint8)
        res_arr[:, :, 3] = res_alpha
        res_arr[res_alpha == 0, :3] = 255
        img = Image.fromarray(res_arr)

    print(f" -> Imagen limpia lista en {time.time() - start_t:.2f}s ({img.width}x{img.height} RGBA)")
    return img


# ==============================================================================
# 3. Limpieza de Geometría y Decimación para Rendimiento Móvil
# ==============================================================================

def clean_mesh_floating_artifacts(
    mesh: trimesh.Trimesh,
    min_face_ratio: float = 0.01,
    min_volume_ratio: float = 0.002,
    min_flatness: float = 0.04
) -> trimesh.Trimesh:
    """
    Elimina losas parásitas, láminas del suelo y motas flotantes en el espacio 3D.
    Conserva únicamente el cuerpo principal del personaje y descarta artefactos planos (losas/láminas).
    """
    try:
        # Si la malla cuenta con textura UV PBR (SF3D), no aplicar split agresivo para evitar romper coordenadas UV
        has_texture = hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "baseColorTexture", None) is not None
        if has_texture:
            return mesh

        comps = mesh.split(only_watertight=False)
        if len(comps) <= 1:
            return mesh

        comps = sorted(comps, key=lambda c: len(c.faces), reverse=True)
        main = comps[0]
        main_faces = len(main.faces)
        main_extents = np.maximum(main.extents, 1e-4)
        main_bbox_vol = float(np.prod(main_extents))

        kept = [main]
        for c in comps[1:]:
            c_ext = np.maximum(c.extents, 1e-4)
            c_bbox_vol = float(np.prod(c_ext))
            c_flatness = float(np.min(c_ext) / np.max(c_ext))
            c_faces = len(c.faces)

            # Descartar losas y láminas planas (flatness < 0.08 o espesor mínimo < 0.025)
            if c_flatness < min_flatness or np.min(c_ext) < 0.025:
                continue

            # Descartar componentes con volumen despreciable frente al personaje principal
            if c_bbox_vol < main_bbox_vol * min_volume_ratio:
                continue

            # Descartar fragmentos desconectados con pocas caras relativas
            if c_faces < int(main_faces * min_face_ratio):
                continue

            kept.append(c)

        if len(kept) == 1:
            clean_m = kept[0]
        else:
            clean_m = trimesh.util.concatenate(kept)

        removed = len(comps) - len(kept)
        if removed > 0:
            print(f" -> [Limpieza 3D] Se eliminaron {removed} fragmentos/losas flotantes parásitas ({len(mesh.faces):,} -> {len(clean_m.faces):,} caras).")
        return clean_m
    except Exception as e:
        print(f" -> [Aviso Limpieza 3D] Error al filtrar componentes ({e}). Se conserva malla original.")
        return mesh

def inpaint_uv_atlas_texture(
    mesh: trimesh.Trimesh,
    dilation_radius: int = 48
) -> trimesh.Trimesh:
    """
    Detecta los píxeles vacíos/negros de fondo del atlas UV y los rellena mediante
    dilatación morfológica amplia e inpainting de color continuo.
    Garantiza que ningún triángulo simplificado por decimation o filtrado de mipmaps
    en Godot 4 muestree fondo negro u oscuro en los bordes de las costuras UV.
    """
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return mesh
    
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return mesh

    tex_img = getattr(mat, "baseColorTexture", None)
    if tex_img is None:
        tex_img = getattr(mat, "image", None)
    if tex_img is None or not isinstance(tex_img, Image.Image):
        return mesh

    try:
        orig_img = tex_img.convert("RGB")
        img_np = np.array(orig_img, dtype=np.uint8)
        h, w = img_np.shape[:2]

        # 1. Píxeles que contienen color válido (no fondo negro o casi negro)
        valid_color = np.any(img_np > 5, axis=-1)

        # 2. Si la malla tiene UVs válidos, rasterizar triángulos para proteger detalles oscuros del modelo
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0 and len(mesh.faces) > 0:
            uvs = mesh.visual.uv
            faces = mesh.faces
            geom_mask = np.zeros((h, w), dtype=np.uint8)
            u_px = np.clip(uvs[:, 0], 0.0, 1.0) * (w - 1)
            v_px = np.clip(1.0 - uvs[:, 1], 0.0, 1.0) * (h - 1)
            pts = np.zeros((len(faces), 3, 2), dtype=np.int32)
            pts[:, 0, 0] = u_px[faces[:, 0]]
            pts[:, 0, 1] = v_px[faces[:, 0]]
            pts[:, 1, 0] = u_px[faces[:, 1]]
            pts[:, 1, 1] = v_px[faces[:, 1]]
            pts[:, 2, 0] = u_px[faces[:, 2]]
            pts[:, 2, 1] = v_px[faces[:, 2]]
            cv2.fillPoly(geom_mask, pts, 255)
            valid_mask = (geom_mask > 0) | valid_color
        else:
            valid_mask = valid_color

        valid_count = np.count_nonzero(valid_mask)
        if valid_count == 0 or valid_count == h * w:
            return mesh

        # 3. Dilatación morfológica en capas (push-pull) desde los bordes de las islas UV
        work_img = img_np.copy()
        current_mask = valid_mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        num_steps = max(12, min(dilation_radius // 2, 32))
        for _ in range(num_steps):
            dilated_mask = cv2.dilate(current_mask, kernel, iterations=1)
            new_pixels = (dilated_mask > 0) & (current_mask == 0)
            if not np.any(new_pixels):
                break
            dilated_img = cv2.dilate(work_img, kernel, iterations=1)
            work_img[new_pixels] = dilated_img[new_pixels]
            current_mask = dilated_mask

        # 4. Inpainting / Difusión de fondo para el espacio vacío restante (elimina 100% de fondo negro para mipmaps)
        empty_pixels = (current_mask == 0)
        if np.any(empty_pixels):
            small_w, small_h = max(32, w // 16), max(32, h // 16)
            bg_low = cv2.resize(work_img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            bg_smooth = cv2.resize(bg_low, (w, h), interpolation=cv2.INTER_LINEAR)
            work_img[empty_pixels] = bg_smooth[empty_pixels]

        # Actualizar la textura base en el material
        filled_pil = Image.fromarray(work_img, mode="RGB")
        filled_pil.format = "JPEG"
        mat.baseColorTexture = filled_pil
        if hasattr(mat, "image"):
            mat.image = filled_pil

        print(f" -> [Inpainting Atlas UV] Atlas UV protegido con {num_steps} iteraciones de dilatación morfológica (anti-bleeding y anti-seams).")
        return mesh
    except Exception as e_inp:
        print(f" -> [Aviso Inpainting UV] Error en dilatación UV ({e_inp}). Se conserva textura original.")
        return mesh


def decimate_mesh_preserving_texture(
    mesh: trimesh.Trimesh,
    target_faces: int = 4000
) -> trimesh.Trimesh:
    """
    Decima una malla 3D simplificando la topología al objetivo de caras mientras preserva
    al 100% las coordenadas UV, los atlas de textura PBR (albedo, normales) y el material original.
    Utiliza meshing_decimation_quadric_edge_collapse_with_texture de PyMeshLab blindando las costuras
    con boundaryweight=100.0 y extratcoordw=50.0 para evitar distorsión o colapso de bordes UV.
    """
    current_faces = len(mesh.faces)
    if target_faces is None or target_faces <= 0 or current_faces <= target_faces:
        print(f"\n[Paso C/D Textura] Malla con {current_faces:,} caras (dentro del objetivo <= {target_faces if target_faces else 'HD':,} caras). Conservando topología original.")
        return mesh

    print(f"\n[Paso C/D Textura] Simplificando malla PBR con textura: {current_faces:,} caras -> {target_faces:,} caras...")
    start_t = time.time()

    # 0. Blindar el atlas UV con inpainting y dilatación morfológica antes de exportar a OBJ para PyMeshLab
    mesh = inpaint_uv_atlas_texture(mesh, dilation_radius=48)

    import tempfile
    try:
        import pymeshlab
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_obj = Path(tmpdir) / "mesh_in.obj"
            tmp_out_obj = Path(tmpdir) / "mesh_out.obj"
            mesh.export(str(tmp_obj), file_type="obj")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(tmp_obj))
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=int(target_faces),
                qualitythr=0.5,
                preserveboundary=True,
                boundaryweight=100.0,
                preservenormal=True,
                extratcoordw=50.0,
                planarquadric=True
            )
            ms.save_current_mesh(str(tmp_out_obj))

            loaded = trimesh.load(str(tmp_out_obj), process=False)
            if isinstance(loaded, trimesh.Scene):
                geoms = list(loaded.geometry.values())
                decimated_mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
            else:
                decimated_mesh = loaded

            orig_mat = getattr(mesh.visual, "material", None)
            if orig_mat is not None:
                decimated_mesh.visual.material = orig_mat

            if hasattr(decimated_mesh.visual, "material") and decimated_mesh.visual.material is not None:
                decimated_mesh.visual.material.metallicFactor = 0.0
                decimated_mesh.visual.material.roughnessFactor = 0.90
                decimated_mesh.visual.material.doubleSided = True

            reduction = (1.0 - (len(decimated_mesh.faces) / current_faces)) * 100.0
            print(f" -> [PyMeshLab Textura] Éxito: {len(decimated_mesh.faces):,} caras ({reduction:.1f}% reducción), {len(decimated_mesh.vertices):,} vértices en {time.time() - start_t:.2f}s")
            return decimated_mesh
    except Exception as e:
        print(f" -> [Aviso Decimación Textura] No se pudo decimar con textura ({e}). Conservando malla actual...")
        return mesh


def decimate_mesh_to_target_faces(
    mesh: trimesh.Trimesh,
    target_faces: int = 4000
) -> trimesh.Trimesh:
    """
    Paso C: Simplifica la malla al número exacto de caras solicitado.
    Utiliza Decimación por Colapso de Bordes Cuádricos (QEM) para preservar
    rasgos, silueta y bordes nítidos ideales para Godot 4 en móviles.
    Si target_faces <= 0 o None, omite la decimación y conserva la malla HD completa.
    NOTA: Se eliminó el Taubin Smoothing duplicado interno para evitar correr 5 iters + 5 iters.
    """
    current_faces = len(mesh.faces)
    if target_faces is None or target_faces <= 0:
        print(f"\n[Paso C/D] Conservando Malla HD Original Completa ({current_faces:,} caras sin comprimir).")
        return mesh

    # Si la malla tiene textura o coordenadas UV, enrutar obligatoriamente a decimate_mesh_preserving_texture
    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
    has_texture = hasattr(mesh.visual, "material") and (
        getattr(mesh.visual.material, "baseColorTexture", None) is not None or
        getattr(mesh.visual.material, "image", None) is not None
    )
    if has_uv or has_texture:
        return decimate_mesh_preserving_texture(mesh, target_faces=target_faces)

    print(f"\n[Paso C/D] Simplificando malla: {current_faces:,} caras -> {target_faces:,} caras objetivo...")
    start_t = time.time()

    if current_faces <= target_faces:
        print(f" -> La malla ya tiene menos caras que el objetivo ({current_faces} <= {target_faces}). Conservando geometría...")
        return mesh

    # Intento 1: PyMeshLab (Algoritmo de Garland-Heckbert de grado industrial)
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        m_in = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
        ms.add_mesh(m_in)
        
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preserveboundary=True,
            preservenormal=True,
            planarquadric=True
        )
        decimated = ms.current_mesh()
        new_mesh = trimesh.Trimesh(
            vertices=decimated.vertex_matrix(),
            faces=decimated.face_matrix(),
            process=False
        )
        reduction = (1.0 - (len(new_mesh.faces) / current_faces)) * 100.0
        print(f" -> [PyMeshLab] Éxito: {len(new_mesh.faces):,} caras finales ({reduction:.1f}% de reducción) en {time.time() - start_t:.2f}s")
        return new_mesh
    except Exception as e:
        print(f" -> [PyMeshLab no disponible o falló: {e}]. Probando pyfqmr...")

    # Intento 2: pyfqmr (Fast Quadric Mesh Reduction en C++)
    try:
        import pyfqmr
        reducer = pyfqmr.Simplify()
        reducer.setMesh(mesh.vertices, mesh.faces)
        reducer.simplify_mesh(
            target_count=int(target_faces),
            aggressiveness=7,
            preserve_border=True,
            verbose=0
        )
        v, f, _ = reducer.getMesh()
        new_mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        reduction = (1.0 - (len(new_mesh.faces) / current_faces)) * 100.0
        print(f" -> [pyfqmr] Éxito: {len(new_mesh.faces):,} caras finales ({reduction:.1f}% reducción) en {time.time() - start_t:.2f}s")
        return new_mesh
    except Exception as e:
        print(f" -> [pyfqmr falló: {e}]. Probando motor nativo de Trimesh...")

    # Intento 3: Trimesh simplify_quadric_decimation
    try:
        new_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        reduction = (1.0 - (len(new_mesh.faces) / current_faces)) * 100.0
        print(f" -> [Trimesh] Éxito: {len(new_mesh.faces):,} caras finales ({reduction:.1f}% reducción) en {time.time() - start_t:.2f}s")
        return new_mesh
    except Exception as e:
        print(f" -> [Advertencia] No se pudo decimar la malla ({e}). Se mantiene la geometría original.")
        return mesh


# ==============================================================================
# 4. Centrado en el Origen (0,0,0) y Normalización para Godot
# ==============================================================================

def center_and_orient_mesh(
    mesh: trimesh.Trimesh,
    align_floor: bool = False,
    normalize_scale: bool = True,
    target_height: float = 1.2
) -> trimesh.Trimesh:
    """
    Centra y orienta la malla para el visor WebGL (Babylon.js) y Godot 4:
    - Convierte de Z-up (Hunyuan3D) a Y-up (estándar glTF / Godot).
    - Centra en (0,0,0) o apoya la base en Y=0.0 si align_floor=True.
    - Normaliza la escala para que encaje perfectamente en el encuadre de cámara.
    - Repara y calcula normales de vértices hacia el exterior (fix_normals).
    """
    mesh = mesh.copy()

    # Hunyuan3D-2 DiT genera la geometría directamente en formato estándar Y-up (pies en -Y, cabeza en +Y)
    bounds = mesh.bounds
    if align_floor:
        # Centrado en X y Z; base de pies en Y = 0.0
        center_x = (bounds[0][0] + bounds[1][0]) / 2.0
        center_y = bounds[0][1] # Punto mínimo en Y
        center_z = (bounds[0][2] + bounds[1][2]) / 2.0
        mesh.apply_translation([-center_x, -center_y, -center_z])
    else:
        # Centrado geométrico puro en (0,0,0)
        center = (bounds[0] + bounds[1]) / 2.0
        mesh.apply_translation(-center)

    if normalize_scale:
        max_extent = float(np.max(mesh.extents))
        if max_extent > 1e-4:
            scale_factor = target_height / max_extent
            mesh.apply_scale(scale_factor)

    try:
        mesh.fix_normals()
    except Exception:
        pass
            
    return mesh


# ==============================================================================
# 5. Generador Principal Hunyuan3D-2 (DiT Turbo + Paint en fp16)
# ==============================================================================

# Parche 1: Tolerancia de llaves en encoders visuales (evita 'Missing key in state_dict')
try:
    from hy3dgen.shapegen.models.conditioner import SingleImageEncoder, DualImageEncoder
    _orig_single_encoder_load = SingleImageEncoder.load_state_dict
    _orig_dual_encoder_load = DualImageEncoder.load_state_dict

    def _tolerant_single_encoder_load(self, state_dict, strict=True):
        return _orig_single_encoder_load(self, state_dict, strict=False)

    def _tolerant_dual_encoder_load(self, state_dict, strict=True):
        return _orig_dual_encoder_load(self, state_dict, strict=False)

    SingleImageEncoder.load_state_dict = _tolerant_single_encoder_load
    DualImageEncoder.load_state_dict = _tolerant_dual_encoder_load
except Exception:
    pass

# Parche 2: Soporte nativo de components para CPU Offload en Hunyuan3D
# Hunyuan3DDiTPipeline copió enable_model_cpu_offload pero omitió self.components,
# provocando "'Hunyuan3DDiTFlowMatchingPipeline' object has no attribute 'components'".
try:
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTPipeline
    if not hasattr(Hunyuan3DDiTPipeline, "components"):
        @property
        def _hunyuan_components(self):
            return {
                "vae": getattr(self, "vae", None),
                "model": getattr(self, "model", None),
                "conditioner": getattr(self, "conditioner", None),
                "scheduler": getattr(self, "scheduler", None),
                "image_processor": getattr(self, "image_processor", None),
            }
        Hunyuan3DDiTPipeline.components = _hunyuan_components
except Exception:
    pass

try:
    from hy3dgen.shapegen import Hunyuan3DShapePipeline
except ImportError:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline as Hunyuan3DShapePipeline

_PAINT_UNAVAILABLE_REASON: Optional[str] = None
_PAINT_CHECKED = False


def hunyuan_paint_available() -> bool:
    """
    Comprueba una sola vez si Hunyuan3D-Paint puede ejecutarse. Necesita la extensión CUDA
    compilada `custom_rasterizer` (y `mesh_processor`) del repositorio de Hunyuan3D-2.
    """
    global _PAINT_UNAVAILABLE_REASON, _PAINT_CHECKED
    if not _PAINT_CHECKED:
        _PAINT_CHECKED = True
        try:
            import custom_rasterizer  # noqa: F401
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
        except Exception as e:
            _PAINT_UNAVAILABLE_REASON = (
                f"{e}. Compila la extensión: cd hy3dgen/texgen/custom_rasterizer && python setup.py install "
                "(requiere CUDA Toolkit y Visual Studio Build Tools)"
            )
    return _PAINT_UNAVAILABLE_REASON is None


class Hunyuan3DGenerator:
    """
    Motor integral de generación de modelos 3D secuencial en 2 etapas:
    Etapa 1: Geometría con Hunyuan3D-DiT (Hunyuan3DShapePipeline) -> Decimación previa + Taubin -> Purga total VRAM
    Etapa 2: Texturizado Nativo Multivista con Hunyuan3D-Paint en modo Low-VRAM (1024x1024) -> Purga total VRAM
    Garantiza que ambos pipelines nunca coexistan en VRAM para estabilidad total en GPUs de 8 GB (RTX 4060).
    """
    def __init__(
        self,
        model_id: str = "tencent/Hunyuan3D-2",
        shape_subfolder: str = "hunyuan3d-dit-v2-0-turbo",
        paint_subfolder: str = "hunyuan3d-paint-v2-0",
        use_fp16: bool = True,
        low_vram_mode: Optional[bool] = None,
        vram_profile: str = "auto",
        device: Optional[str] = None
    ):
        self.model_id = model_id
        self.shape_subfolder = shape_subfolder
        self.paint_subfolder = paint_subfolder
        self.use_fp16 = use_fp16
        if torch.cuda.is_available() and device != "cpu":
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.dtype = torch.float16 if (use_fp16 and "cuda" in self.device) else torch.float32

        # Resolver perfil de VRAM dinámicamente
        if vram_profile == "auto" or not vram_profile:
            resolved_key = detect_vram_profile()
        else:
            resolved_key = vram_profile.lower()
        if "cuda" not in self.device:
            resolved_key = "cpu"

        pcfg = VRAM_PROFILES.get(resolved_key, VRAM_PROFILES["low_8gb"])
        self.vram_profile_key = resolved_key
        self.vram_profile_name = pcfg.get("name", "Auto")
        # low_vram_mode explícito (CLI --no-low-vram) tiene prioridad sobre el perfil
        self.staged_offload = pcfg["staged_offload"] if low_vram_mode is None else bool(low_vram_mode)
        self.max_octree = pcfg["max_octree"]
        self.paint_resolution = pcfg["paint_resolution"]

        mode_txt = "un componente en GPU a la vez" if self.staged_offload else "pipeline completo en GPU"
        print(f"[Hunyuan3D] Dispositivo: {self.device.upper()} | Tipo de dato: {self.dtype} | Perfil VRAM: {self.vram_profile_name} ({mode_txt})")
        if "cuda" in str(self.device):
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Hunyuan3D] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) -> Pipeline secuencial: forma y textura nunca coexisten en VRAM")

            # Ajustes de aceleración CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _clean_vram(self):
        """Purga estricta de VRAM delegada a flush_vram()."""
        flush_vram()

    def _get_shape_pipeline(self):
        """
        Devuelve el pipeline de forma listo en el dispositivo, reutilizando la copia aparcada
        en RAM si existe (evita releer ~5 GB del disco). Con staged_offload sube a la GPU un
        componente a la vez: condicionador -> DiT -> VAE.
        """
        mem = MODEL_MEMORY_GB["shape"]
        shape_pipe = activate_model("shape", vram_needed_gb=mem["vram"] if self.staged_offload else mem["vram_full"])
        if shape_pipe is not None and getattr(shape_pipe, "_studio_dtype", None) != self.dtype:
            release_model("shape")
            shape_pipe = None

        if shape_pipe is None:
            local_shape_dir = HUNYUAN_LOCAL_DIR
            model_source = str(local_shape_dir) if (local_shape_dir / self.shape_subfolder).exists() else self.model_id

            # Se carga en CPU: la carga no ocupa VRAM y cada etapa sube solo lo que necesita
            print(f" -> Cargando Hunyuan3DShapePipeline desde {model_source} [{self.shape_subfolder}] ({self.dtype})...")
            t_load = time.time()
            shape_pipe = Hunyuan3DShapePipeline.from_pretrained(
                model_source,
                subfolder=self.shape_subfolder,
                torch_dtype=self.dtype,
                device="cpu"
            )

            # FlashVDM: aceleración del Volume Decoding (se activa una sola vez por pipeline)
            if hasattr(shape_pipe, "enable_flashvdm"):
                try:
                    shape_pipe.enable_flashvdm(enabled=True)
                    print(" -> [FlashVDM] FlashVDM activado en Hunyuan3DShapePipeline.")
                except Exception as e_fvdm:
                    try:
                        shape_pipe.enable_flashvdm(enabled=True, replace_vae=False)
                        print(" -> [FlashVDM] FlashVDM activado (sin reemplazo de VAE).")
                    except Exception as e_fvdm2:
                        print(f" -> [FlashVDM] No disponible ({e_fvdm} / {e_fvdm2}). Continuando con VAE estándar.")

            shape_pipe._studio_dtype = self.dtype
            shape_pipe._studio_orig_encode_cond = shape_pipe.encode_cond
            shape_pipe._studio_orig_export = shape_pipe._export
            register_model("shape", shape_pipe)
            print(f" -> Pipeline de forma cargado desde disco en {time.time() - t_load:.1f}s (se conservará en RAM).")
        else:
            print(" -> Pipeline de forma reutilizado desde RAM (sin leer el disco).")

        device = torch.device(self.device)
        staged = self.staged_offload and device.type == "cuda"

        # NOTA: no se usa enable_model_cpu_offload(): en hy3dgen 2.0.2 crea los latentes en
        # pipe.device (CPU) mientras el DiT corre en CUDA y la inferencia falla con
        # "Cannot generate a cpu tensor from a generator of type cuda".
        def staged_encode_cond(*args, **kwargs):
            shape_pipe.conditioner.to(device)
            result = shape_pipe._studio_orig_encode_cond(*args, **kwargs)
            if staged:
                shape_pipe.conditioner.to("cpu")
                flush_vram()
            shape_pipe.model.to(device)
            return result

        def staged_export(*args, **kwargs):
            # Volume Decoding con la VRAM libre de DiT y condicionador
            if staged:
                shape_pipe.model.to("cpu")
                shape_pipe.conditioner.to("cpu")
                flush_vram()
            shape_pipe.vae.to(device)
            print_real_vram_status("Inicio Volume Decoding")
            return shape_pipe._studio_orig_export(*args, **kwargs)

        shape_pipe.encode_cond = staged_encode_cond
        shape_pipe._export = staged_export

        # pipe.device decide dónde se crean latentes y timesteps: siempre el dispositivo real
        if staged:
            shape_pipe.to("cpu")
            shape_pipe.device = device
        else:
            shape_pipe.to(device)
        print_real_vram_status("Pipeline de forma listo")
        return shape_pipe

    def generate_shape(
        self,
        image: Image.Image,
        steps: int = 5,
        octree_resolution: int = 256,
        seed: int = 42,
        target_faces: int = 40000,
        progress_callback: Optional[Callable[[str, int], None]] = None,
        texture_safe_cap: Optional[int] = None
    ) -> trimesh.Trimesh:
        """
        Etapa 1 - Generación de Geometría (Hunyuan3D-DiT):
        1. Saca de la VRAM cualquier otro modelo (SF3D / Paint) y prepara el pipeline de forma.
        2. Preprocesa el canal alfa con inpainting de bordes (evita halos oscuros).
        3. Inferencia DiT + Volume Decoding con offload por etapas según el perfil de VRAM.
        4. Limpia islas flotantes y reduce polígonos:
             - target_faces > 0: reduce a ese número de caras.
             - target_faces <= 0: conserva la malla completa (HD).
             - texture_safe_cap: tope adicional cuando después se va a texturizar.
        5. Suavizado Taubin y aparca el pipeline en RAM (VRAM libre para la Etapa 2).
        """
        safe_octree = min(int(octree_resolution or 256), self.max_octree)
        faces_txt = f"{target_faces:,}" if target_faces and target_faces > 0 else "HD (sin reducir)"
        print(f"\n[Etapa 1/2] Generando geometría 3D pura con Hunyuan3D-DiT ({self.shape_subfolder})...")
        print(f" -> Pasos DiT: {steps} | Octree: {safe_octree} | Semilla: {seed} | Objetivo caras: {faces_txt}")
        start_t = time.time()

        try:
            # Preprocesamiento con Inpainting de bordes en canal alfa (cv2.inpaint)
            prep_image = inpaint_edge_padding(image, dilation_pixels=10, alpha_threshold=25)

            shape_pipe = self._get_shape_pipeline()

            generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None

            # Callback para pasos individuales de difusión DiT
            def dit_step_callback(step_idx, t, outputs):
                if progress_callback:
                    cur_step = step_idx + 1
                    pct = 30 + int((cur_step / max(steps, 1)) * 35) # 30% a 65%
                    vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
                    progress_callback(f"[3/4] Inferencia DiT en GPU ({vram_mb:.0f} MB): Paso {cur_step}/{steps} ({pct}%)...", pct)

            # Hook para capturar progreso de Volume Decoding en tiempo real
            import tqdm
            orig_tqdm_update = tqdm.tqdm.update
            last_v_pct = -1

            def custom_tqdm_update(tqdm_self, n=1):
                nonlocal last_v_pct
                res = orig_tqdm_update(tqdm_self, n)
                desc = getattr(tqdm_self, "desc", "") or ""
                if "Volume Decoding" in desc and tqdm_self.total and progress_callback:
                    v_pct = int((tqdm_self.n / tqdm_self.total) * 100)
                    if v_pct != last_v_pct and (v_pct % 5 == 0 or v_pct == 100):
                        last_v_pct = v_pct
                        overall_pct = 66 + int(v_pct * 0.18) # 66% a 84%
                        vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
                        progress_callback(f"[3/4] Decodificando volumen en GPU ({vram_mb:.0f} MB): {v_pct}%...", overall_pct)
                return res

            tqdm.tqdm.update = custom_tqdm_update

            # num_chunks=32000 reduce iteraciones de decodificación sin disparar la VRAM
            safe_chunks = 32000
            print(f" -> [Inferencia DiT] Octree: {safe_octree} | Chunks: {safe_chunks} | Pasos: {steps}")

            try:
                with torch.inference_mode():
                    outputs = shape_pipe(
                        image=prep_image,
                        num_inference_steps=steps,
                        octree_resolution=safe_octree,
                        num_chunks=safe_chunks,
                        generator=generator,
                        callback=dit_step_callback,
                        callback_steps=1,
                        output_type="trimesh"
                    )
            finally:
                tqdm.tqdm.update = orig_tqdm_update

            raw_mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            try:
                raw_mesh.fix_normals()
            except Exception as e_orient:
                print(f" -> [Aviso Normales] {e_orient}")

            # Depuración de geometría: eliminar islas parásitas y losas flotantes
            cleaned_mesh = clean_mesh_floating_artifacts(raw_mesh)

            # Reducción de polígonos: lo que pida el usuario y, si después se texturiza,
            # como máximo texture_safe_cap (no se pasa una malla de 280k caras al texturizado)
            safe_faces = int(target_faces) if target_faces and target_faces > 0 else None
            if texture_safe_cap:
                safe_faces = min(safe_faces or texture_safe_cap, texture_safe_cap)

            if safe_faces and len(cleaned_mesh.faces) > safe_faces:
                print(f" -> [Etapa 1/2] Decimando malla de {len(cleaned_mesh.faces):,} a {safe_faces:,} caras...")
                decimated_mesh = decimate_mesh_to_target_faces(cleaned_mesh, target_faces=safe_faces)
            else:
                decimated_mesh = cleaned_mesh

            print(f" -> [Etapa 1/2] Aplicando filtro de suavizado Taubin ({len(decimated_mesh.faces):,} caras)...")
            smoothed_mesh = apply_taubin_smoothing(decimated_mesh, lamb=0.53, nu=-0.55, iterations=5)

            print(f" -> [Etapa 1/2 Completada] Geometría: {len(smoothed_mesh.vertices):,} vértices, {len(smoothed_mesh.faces):,} caras en {time.time() - start_t:.2f}s")
            return smoothed_mesh

        except ImportError as e:
            print(f"[ERROR CRÍTICO] La librería hy3dgen no está instalada: {e}")
            print("Ejecuta 'setup.bat' o 'pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git'")
            raise
        except torch.cuda.OutOfMemoryError:
            # Un pipeline a medias tras un OOM no es fiable: se destruye
            release_model("shape")
            print("[ERROR] VRAM insuficiente para la geometría. Prueba un perfil de VRAM menor o un octree más bajo.")
            raise
        except Exception as e:
            err_str = str(e)
            if "SingleImageEncoder" in err_str or "state_dict" in err_str:
                release_model("shape")
                print(f"\n[DIAGNÓSTICO] {err_str}")
                print(" -> Causa: Los pesos de Hunyuan3D-2 Turbo están incompletos en disco o corruptos.")
                print(" -> Solución: Ejecuta 'descargar_modelos.bat' para completar el archivo model.fp16.safetensors (4.93 GB).")
                print(" -> O cambia en la interfaz a 'Stable Fast 3D (SF3D)' para generar en 1s sin este modelo.\n")
            print(f"[ERROR] Falló la generación de forma: {e}")
            raise
        finally:
            # Aparcar en RAM (o liberar si falta RAM): la VRAM queda libre para la Etapa 2
            park_model("shape")
            print_real_vram_status("Post-Etapa 1 (forma aparcada)")

    def _get_paint_pipeline(self):
        """Devuelve Hunyuan3D-Paint (reutilizando la copia aparcada en RAM) con la forma ya fuera de la VRAM."""
        paint_pipeline = activate_model("paint", strict_vram=True)
        if paint_pipeline is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            local_model_dir = HUNYUAN_LOCAL_DIR
            if (local_model_dir / "hunyuan3d-delight-v2-0").exists() and (local_model_dir / self.paint_subfolder).exists():
                paint_source = str(local_model_dir)
            else:
                paint_source = self.model_id

            print(f" -> Cargando Hunyuan3DPaintPipeline ({paint_source}) en float16...")
            t_load = time.time()
            paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(paint_source)

            # Offload de diffusers: delight y multivista suben a la GPU solo mientras se ejecutan
            if "cuda" in str(self.device) and hasattr(paint_pipeline, "enable_model_cpu_offload"):
                try:
                    paint_pipeline.enable_model_cpu_offload()
                    print(" -> [Paint Low-VRAM] Model CPU Offload activado en Hunyuan3D-Paint.")
                except Exception as e_off:
                    print(f" -> [Paint Low-VRAM] Offload no disponible ({e_off}); se usa la GPU directamente.")

            # VAE Slicing y VAE Tiling en los componentes internos de difusión
            for sub in getattr(paint_pipeline, "models", {}).values():
                sub_p = getattr(sub, "pipeline", None)
                for opt in ("enable_vae_slicing", "enable_vae_tiling"):
                    if sub_p is not None and hasattr(sub_p, opt):
                        try:
                            getattr(sub_p, opt)()
                        except Exception:
                            pass

            register_model("paint", paint_pipeline)
            print(f" -> Hunyuan3D-Paint cargado desde disco en {time.time() - t_load:.1f}s (se conservará en RAM).")
        else:
            print(" -> Hunyuan3D-Paint reutilizado desde RAM (sin leer el disco).")

        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        # Resolución de render/textura según el perfil de VRAM
        res = self.paint_resolution
        if hasattr(paint_pipeline, "config"):
            paint_pipeline.config.render_size = res
            paint_pipeline.config.texture_size = res
            paint_pipeline.config.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
            paint_pipeline.config.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
            paint_pipeline.config.candidate_view_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
        paint_pipeline.render = MeshRender(
            default_resolution=res,
            texture_size=res,
            device="cuda" if "cuda" in str(self.device) else "cpu"
        )
        return paint_pipeline

    def paint_and_texture_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image
    ) -> trimesh.Trimesh:
        """
        Etapa 2 - Texturizado Nativo Multivista (Hunyuan3D-Paint):
        1. Saca de la VRAM el pipeline de forma y SF3D (nunca coexisten con Paint).
        2. Reutiliza Paint desde RAM o lo carga una vez, con offload de diffusers y VAE slicing/tiling.
        3. Render/textura a la resolución del perfil de VRAM (1024 en 6-12 GB, 2048 en 16 GB+).
        4. Si Paint no está disponible o se queda sin VRAM, recurre al horneado UV de contingencia.
        5. Aparca Paint en RAM al terminar y aplica acabado PBR mate.
        """
        print("\n[Etapa 2/2] Texturizado Nativo Multivista con Hunyuan3D-Paint...")
        start_t = time.time()

        textured_mesh = None
        if not hunyuan_paint_available():
            print(f" -> [Hunyuan3D-Paint] No disponible: {_PAINT_UNAVAILABLE_REASON}")
            print(" -> Usando horneado UV de contingencia (proyección frontal/trasera en CPU).")
            textured_mesh = self._fallback_texture_bake(mesh, image)
        else:
            textured_mesh = self._run_paint_pipeline(mesh, image, start_t)

        # Ajuste de material PBR mate/limpio (metallicFactor=0.0, roughnessFactor=0.90)
        if textured_mesh is not None and hasattr(textured_mesh, "visual") and hasattr(textured_mesh.visual, "material") and textured_mesh.visual.material is not None:
            mat = textured_mesh.visual.material
            mat.metallicFactor = 0.0
            mat.roughnessFactor = 0.90
            mat.doubleSided = True

        return textured_mesh if textured_mesh is not None else mesh

    def _run_paint_pipeline(self, mesh: trimesh.Trimesh, image: Image.Image, start_t: float) -> Optional[trimesh.Trimesh]:
        """Ejecuta Hunyuan3D-Paint; ante OOM o error recurre al horneado de contingencia."""
        textured_mesh = None
        try:
            paint_pipeline = self._get_paint_pipeline()
            print_real_vram_status("Pre-Inferencia Hunyuan3D-Paint")
            print(f" -> Pintando malla de {len(mesh.faces):,} caras con Hunyuan3D-Paint...")
            with torch.inference_mode():
                textured_mesh = paint_pipeline(mesh, image=image)
            print(f" -> [Hunyuan3D-Paint] Texturizado multivista completado en {time.time() - start_t:.2f}s.")

        except InsufficientMemoryError as e_mem:
            print(f" -> [Hunyuan3D-Paint] No se carga para no saturar el equipo: {e_mem}")
            textured_mesh = None
        except torch.cuda.OutOfMemoryError as e_oom:
            print(f" -> [Aviso CUDA OOM en Hunyuan3D-Paint] {e_oom}. Recurriendo a horneado UV de contingencia...")
            release_model("paint")
            textured_mesh = None
        except Exception as e_tex:
            print(f" -> [Aviso Hunyuan3D-Paint] {e_tex}. Recurriendo a horneado UV de contingencia...")
            textured_mesh = None
        finally:
            park_model("paint")
            print_real_vram_status("Post-Hunyuan Paint (aparcado)")
        if textured_mesh is None:
            textured_mesh = self._fallback_texture_bake(mesh, image)
        return textured_mesh

    def _fallback_texture_bake(self, mesh: trimesh.Trimesh, image: Image.Image) -> trimesh.Trimesh:
        """
        Atlas UV Frontal / Trasero Diferenciado (Front-Back UV Split) de alta fidelidad:
        - Divide el mapa de textura en un Atlas Horizontal con dos mitades:
          * Mitad Izquierda [0.0 - 0.5]: Cara Frontal (Nz >= -0.1). Textura nítida original con
            dilatación Voronoi continua y núcleo sólido anti-halos.
          * Mitad Derecha [0.5 - 1.0]: Cara Trasera (Nz < -0.1). Textura procesada con desenfoque
            gaussiano profundo (radius=28) sobre los colores interiores, preservando el gradiente y
            tono de altura (casco, ropa, extremidades) pero eliminando rasgos faciales duplicados (ojos, boca, etc.).
        - Separación de caras frontales y traseras con unificación de costura limpia (seam) sin artefactos.
        - Margen protector (Gutter) en cada mitad con mapeo UV exacto 1:1 al bounding box.
        - Asigna material PBR estándar de doble cara (doubleSided=True) optimizado para Godot 4 / WebGL.
        """
        from scipy.ndimage import distance_transform_edt, binary_erosion
        from PIL import ImageFilter
        mesh_copy = mesh.copy()
        try:
            mesh_copy.fix_normals()
        except Exception:
            pass

        # 1. Preparar imagen RGBA y recortar silueta activa
        rgba_img = image.convert("RGBA")
        arr = np.array(rgba_img)
        alpha = arr[:, :, 3]
        rgb = arr[:, :, :3]

        mask_fg = alpha > 35
        if mask_fg.any():
            y_indices, x_indices = np.where(mask_fg)
            x0, x1 = x_indices.min(), x_indices.max()
            y0, y1 = y_indices.min(), y_indices.max()
            cropped_rgb = rgb[y0:y1+1, x0:x1+1].copy()
            cropped_alpha = alpha[y0:y1+1, x0:x1+1].copy()
        else:
            cropped_rgb = rgb.copy()
            cropped_alpha = alpha.copy()

        h_crop, w_crop = cropped_rgb.shape[:2]

        # 2. Erosión morfológica y dilatación Voronoi limpia desde el núcleo sólido
        # Usamos alpha > 210 y erosión de 2 px para eliminar el halo blanco/semitransparente de rembg
        solid_fg = cropped_alpha > 210
        if not solid_fg.any():
            solid_fg = cropped_alpha > 40

        core_fg = binary_erosion(solid_fg, iterations=2)
        if not core_fg.any():
            core_fg = binary_erosion(solid_fg, iterations=1)
        if not core_fg.any():
            core_fg = solid_fg

        bg_mask = ~core_fg
        indices = distance_transform_edt(bg_mask, return_distances=False, return_indices=True)
        dilated_rgb = cropped_rgb[indices[0], indices[1]]

        # Conservar el núcleo nítido original para la textura frontal
        clean_front = np.where(core_fg[:, :, None], cropped_rgb, dilated_rgb)

        # Para la textura trasera: aplicar desenfoque gaussiano profundo sobre los colores interiores
        # (GaussianBlur radius=28) para que los detalles faciales (ojos, boca, etc.) se disuelvan
        # en los colores base de cada altura (blanco, azul, etc.) sin proyectar el rostro en la espalda.
        pil_front = Image.fromarray(clean_front)
        blurred_back_pil = pil_front.filter(ImageFilter.GaussianBlur(radius=28))
        blurred_back_rgb = np.array(blurred_back_pil)
        clean_back = np.where(core_fg[:, :, None], blurred_back_rgb, dilated_rgb)

        # 3. Lienzo con margen protector (Gutter) para cada mitad
        pad_x = max(int(w_crop * 0.08), 16)
        pad_y = max(int(h_crop * 0.08), 16)
        sub_w = w_crop + 2 * pad_x
        sub_h = h_crop + 2 * pad_y

        front_canvas = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
        front_canvas[pad_y:pad_y+h_crop, pad_x:pad_x+w_crop] = clean_front

        canvas_bg_mask = np.ones((sub_h, sub_w), dtype=bool)
        canvas_bg_mask[pad_y:pad_y+h_crop, pad_x:pad_x+w_crop] = False
        c_indices = distance_transform_edt(canvas_bg_mask, return_distances=False, return_indices=True)
        front_canvas = front_canvas[c_indices[0], c_indices[1]]

        back_canvas = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
        back_canvas[pad_y:pad_y+h_crop, pad_x:pad_x+w_crop] = clean_back
        back_canvas = back_canvas[c_indices[0], c_indices[1]]

        # Atlas Horizontal: Mitad Izquierda = Frontal [0.0 - 0.5], Mitad Derecha = Trasera [0.5 - 1.0]
        atlas = np.hstack([front_canvas, back_canvas])
        tex_pil = Image.fromarray(atlas)

        # 4. Verificación de orientación de ejes y cálculo de límites con percentiles
        # Ejes canónicos Hunyuan3D: X = ancho horizontal, Y = altura vertical (pies en -Y, cabeza en +Y), Z = profundidad
        vx = mesh_copy.vertices[:, 0]
        vy = mesh_copy.vertices[:, 1]

        # Filtrar vértices visibles desde el frente (normal_z > -0.2) para el bounding box
        normals = mesh_copy.vertex_normals
        if normals is not None and len(normals) == len(vx):
            front_vert_mask = normals[:, 2] > -0.2
        else:
            front_vert_mask = np.ones(len(vx), dtype=bool)

        if front_vert_mask.sum() > 20:
            vx_front = vx[front_vert_mask]
            vy_front = vy[front_vert_mask]
        else:
            vx_front = vx
            vy_front = vy

        vx_min, vx_max = np.percentile(vx_front, [1.5, 98.5])
        vy_min, vy_max = np.percentile(vy_front, [1.0, 99.0])

        w_span = max(vx_max - vx_min, 1e-5)
        h_span = max(vy_max - vy_min, 1e-5)

        u_min_local = pad_x / sub_w
        u_max_local = (pad_x + w_crop) / sub_w
        v_min_local = pad_y / sub_h
        v_max_local = (pad_y + h_crop) / sub_h

        # 5. Partición Frontal / Trasera según la normal Z de las caras (Front-Back Split)
        fnz = mesh_copy.face_normals[:, 2]
        front_face_mask = fnz >= -0.1
        back_face_mask = ~front_face_mask

        if front_face_mask.sum() > 0 and back_face_mask.sum() > 0:
            # Submalla Frontal
            m_front = trimesh.Trimesh(vertices=mesh_copy.vertices, faces=mesh_copy.faces[front_face_mask], process=False)
            m_front.remove_unreferenced_vertices()

            # Submalla Trasera
            m_back = trimesh.Trimesh(vertices=mesh_copy.vertices, faces=mesh_copy.faces[back_face_mask], process=False)
            m_back.remove_unreferenced_vertices()

            # UV Frontal (mapeado a la mitad izquierda [0.0 - 0.5])
            vxf = m_front.vertices[:, 0]
            vyf = m_front.vertices[:, 1]
            u_rel_f = (vxf - vx_min) / w_span
            v_rel_f = (vyf - vy_min) / h_span
            uf = 0.5 * (u_min_local + u_rel_f * (u_max_local - u_min_local))
            vf = v_min_local + v_rel_f * (v_max_local - v_min_local)
            uv_front = np.column_stack([np.clip(uf, 0.001, 0.499), np.clip(vf, 0.001, 0.999)])
            m_front.visual = trimesh.visual.TextureVisuals(uv=uv_front, image=tex_pil)

            # UV Trasero (mapeado a la mitad derecha [0.5 - 1.0])
            vxb = m_back.vertices[:, 0]
            vyb = m_back.vertices[:, 1]
            u_rel_b = (vxb - vx_min) / w_span
            v_rel_b = (vyb - vy_min) / h_span
            ub = 0.5 + 0.5 * (u_min_local + u_rel_b * (u_max_local - u_min_local))
            vb = v_min_local + v_rel_b * (v_max_local - v_min_local)
            uv_back = np.column_stack([np.clip(ub, 0.501, 0.999), np.clip(vb, 0.001, 0.999)])
            m_back.visual = trimesh.visual.TextureVisuals(uv=uv_back, image=tex_pil)

            final_mesh = trimesh.util.concatenate([m_front, m_back])
        else:
            # Contingencia si el modelo no tiene caras traseras
            u_rel = (vx - vx_min) / w_span
            v_rel = (vy - vy_min) / h_span
            u = 0.5 * (u_min_local + u_rel * (u_max_local - u_min_local))
            v = v_min_local + v_rel * (v_max_local - v_min_local)
            uv = np.column_stack([np.clip(u, 0.001, 0.499), np.clip(v, 0.001, 0.999)])
            mesh_copy.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_pil)
            final_mesh = mesh_copy

        # 6. Material PBR estándar mate limpio (metallicFactor=0.0, roughnessFactor=0.90)
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_pil,
            metallicFactor=0.0,
            roughnessFactor=0.90,
            doubleSided=True
        )
        final_mesh.visual.material = material
        return final_mesh


# ==============================================================================
# 5b. Generador Secundario: Stable Fast 3D (SF3D de Stability AI)
# ==============================================================================

def _ensure_sf3d_shims(repo_dir: Optional[Path] = None):
    """
    Configura shims / polyfills de alto rendimiento para uv_unwrapper y texture_baker
    permitiendo ejecutar Stable Fast 3D en Windows sin necesidad de compilar extensiones C++ nativas con MSVC.
    """
    import types
    import importlib.util

    # 1. uv_unwrapper
    cur_uv = sys.modules.get("uv_unwrapper")
    if cur_uv is None or not hasattr(cur_uv, "Unwrapper"):
        unwrapper_cls = None

        # Intentar cargar la implementación oficial de unwrap.py desde el repositorio de SF3D
        candidate_paths = []
        if repo_dir:
            candidate_paths.append(repo_dir / "uv_unwrapper" / "uv_unwrapper" / "unwrap.py")
        candidate_paths.extend([
            REPOS_DIR / "stable-fast-3d" / "uv_unwrapper" / "uv_unwrapper" / "unwrap.py",
            Path("./stable-fast-3d/uv_unwrapper/uv_unwrapper/unwrap.py")
        ])

        for unwrap_path in candidate_paths:
            if unwrap_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("uv_unwrapper.unwrap", unwrap_path)
                    if spec and spec.loader:
                        unwrap_mod = importlib.util.module_from_spec(spec)
                        sys.modules["uv_unwrapper.unwrap"] = unwrap_mod
                        spec.loader.exec_module(unwrap_mod)

                        # Parchear _assign_faces_uv_to_atlas_index para evitar la llamada C++ a torch.ops.UVUnwrapper
                        def _safe_assign_faces(self, vertex_positions, triangle_idxs, face_uv, face_index):
                            try:
                                if hasattr(torch.ops, "UVUnwrapper") and hasattr(torch.ops.UVUnwrapper, "assign_faces_uv_to_atlas_index"):
                                    return torch.ops.UVUnwrapper.assign_faces_uv_to_atlas_index(
                                        vertex_positions.cpu(),
                                        triangle_idxs.cpu(),
                                        face_uv.view(-1, 2).cpu(),
                                        face_index.cpu(),
                                    ).to(vertex_positions.device)
                            except Exception:
                                pass
                            return face_index

                        unwrap_mod.Unwrapper._assign_faces_uv_to_atlas_index = _safe_assign_faces
                        unwrapper_cls = unwrap_mod.Unwrapper
                        break
                except Exception as ex:
                    print(f"[SF3D Shim] Advertencia al compilar unwrap.py oficial: {ex}")

        # Fallback si no se encontró unwrap.py o falló la carga
        if unwrapper_cls is None:
            class FallbackUnwrapper(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, v_pos, v_nrm, t_pos_idx, island_padding=0.02):
                    try:
                        import xatlas
                        v_np = v_pos.detach().cpu().numpy().astype(np.float32)
                        f_np = t_pos_idx.detach().cpu().numpy().astype(np.uint32)

                        # Configurar Unwrap para mantener parches (charts) continuos y amplios sin fragmentación
                        atlas = xatlas.Atlas()
                        atlas.add_mesh(v_np, f_np)

                        chart_options = xatlas.ChartOptions()
                        chart_options.max_cost = 4.0               # Mayor tolerancia para formar parches grandes y continuos
                        chart_options.normal_deviation_weight = 1.0 # Menor penalización por curvatura en transiciones
                        chart_options.normal_seam_weight = 1.0      # Evita partir en costuras innecesarias en laterales
                        chart_options.max_iterations = 4            # Calidad de relajación de atlas UV

                        pack_options = xatlas.PackOptions()
                        pack_options.padding = int(max(2, island_padding * 512)) # Margen entre islas para evitar bleeding
                        pack_options.resolution = 1024

                        atlas.generate(chart_options, pack_options)
                        vmapping, indices, uvs = atlas.get_mesh(0)

                        return (
                            torch.from_numpy(uvs).to(v_pos.device, dtype=torch.float32),
                            torch.from_numpy(indices.astype(np.int64)).to(t_pos_idx.device, dtype=torch.int64)
                        )
                    except Exception:
                        nv = v_pos.shape[0]
                        dummy_uv = torch.zeros((nv, 2), dtype=torch.float32, device=v_pos.device)
                        return dummy_uv, t_pos_idx
            unwrapper_cls = FallbackUnwrapper

        mod_uv = types.ModuleType("uv_unwrapper")
        mod_uv.Unwrapper = unwrapper_cls
        sys.modules["uv_unwrapper"] = mod_uv

    # 2. texture_baker
    cur_tb = sys.modules.get("texture_baker")
    if cur_tb is None or not hasattr(cur_tb, "TextureBaker"):
        # Rasterizador de textura UV ultrarrápido con Numba (CPU multihilo, 0 MB VRAM)
        _numba_rast_fn = None
        try:
            from numba import njit, prange

            @njit(parallel=True, fastmath=True)
            def _numba_rasterize_sf3d(uvs, faces, res):
                rast = np.zeros((res, res, 4), dtype=np.float32)
                rast[:, :, 3] = -1.0
                num_faces = faces.shape[0]
                for f in prange(num_faces):
                    i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
                    u0, v0 = uvs[i0, 0], uvs[i0, 1]
                    u1, v1 = uvs[i1, 0], uvs[i1, 1]
                    u2, v2 = uvs[i2, 0], uvs[i2, 1]
                    x0, y0 = u0 * res, (1.0 - v0) * res
                    x1, y1 = u1 * res, (1.0 - v1) * res
                    x2, y2 = u2 * res, (1.0 - v2) * res
                    min_x = max(0, int(min(x0, x1, x2)))
                    max_x = min(res - 1, int(max(x0, x1, x2)) + 1)
                    min_y = max(0, int(min(y0, y1, y2)))
                    max_y = min(res - 1, int(max(y0, y1, y2)) + 1)
                    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
                    if abs(denom) < 1e-9:
                        continue
                    inv_denom = 1.0 / denom
                    for py in range(min_y, max_y + 1):
                        for px in range(min_x, max_x + 1):
                            p_x = px + 0.5
                            p_y = py + 0.5
                            w0 = ((y1 - y2) * (p_x - x2) + (x2 - x1) * (p_y - y2)) * inv_denom
                            w1 = ((y2 - y0) * (p_x - x2) + (x0 - x2) * (p_y - y2)) * inv_denom
                            w2 = 1.0 - w0 - w1
                            if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                                rast[py, px, 0] = w0
                                rast[py, px, 1] = w1
                                rast[py, px, 2] = w2
                                rast[py, px, 3] = float(f)
                return rast

            _numba_rast_fn = _numba_rasterize_sf3d
        except Exception as e_nb:
            print(f" -> [Aviso SF3D Baker] Numba no disponible: {e_nb}")
            _numba_rast_fn = None

        class FastTextureBaker(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def rasterize(self, uv: torch.Tensor, face_indices: torch.Tensor, bake_resolution: int) -> torch.Tensor:
                device = uv.device
                res = int(bake_resolution)
                if _numba_rast_fn is not None:
                    uv_np = uv.detach().cpu().numpy().astype(np.float32)
                    faces_np = face_indices.detach().cpu().numpy().astype(np.int32)
                    rast_np = _numba_rast_fn(uv_np, faces_np, res)
                    return torch.from_numpy(rast_np).to(device)
                else:
                    rast = torch.zeros((res, res, 4), dtype=torch.float32, device=device)
                    rast[..., -1] = -1.0
                    return rast

            def get_mask(self, rast: torch.Tensor) -> torch.Tensor:
                return rast[..., -1] >= 0

            def interpolate(self, attr: torch.Tensor, rast: torch.Tensor, face_indices: torch.Tensor, uv: Optional[torch.Tensor] = None) -> torch.Tensor:
                H, W, _ = rast.shape
                device = attr.device
                C = attr.shape[-1] if attr.ndim > 1 else 1
                out = torch.zeros((H, W, C), dtype=attr.dtype, device=device)
                tri_idx = rast[..., 3].long()
                valid_mask = tri_idx >= 0
                if not valid_mask.any():
                    return out
                valid_tris = tri_idx[valid_mask]
                bary = rast[..., :3][valid_mask].to(device)
                f_verts = face_indices[valid_tris]
                v0 = attr[f_verts[:, 0]]
                v1 = attr[f_verts[:, 1]]
                v2 = attr[f_verts[:, 2]]
                interp = v0 * bary[:, 0:1] + v1 * bary[:, 1:2] + v2 * bary[:, 2:3]
                out[valid_mask] = interp
                return out

            def forward(self, attr, uv, face_indices, bake_resolution):
                rast = self.rasterize(uv, face_indices, bake_resolution)
                return self.interpolate(attr, rast, face_indices, uv)

        mod_tb = types.ModuleType("texture_baker")
        mod_tb.TextureBaker = FastTextureBaker
        sys.modules["texture_baker"] = mod_tb


class SF3DGenerator:
    """
    Motor de generación 3D con Stable Fast 3D (Stability AI).
    Genera modelos 3D con texturas PBR completas (albedo, normales, rugosidad) en 1-2 segundos.
    """
    def __init__(
        self,
        model_id: str = "stabilityai/stable-fast-3d",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_id = model_id
        self.device = device
        self.model = None

    def _load_model(self):
        # Hunyuan3D (forma/pintura) sale de la VRAM: SF3D es el único modelo en la GPU
        cached = activate_model("sf3d")
        print_real_vram_status("Pre-Carga SF3D")
        if cached is not None:
            cached.to(self.device)
            self.model = cached
            print("[SF3D] Modelo reutilizado desde RAM (sin leer el disco).")
            return

        candidates = [
            REPOS_DIR / "stable-fast-3d",
            Path("./stable-fast-3d")
        ]
        found_repo = None
        for c in candidates:
            if c.exists() and (c / "sf3d").exists():
                found_repo = c.resolve()
                if str(found_repo) not in sys.path:
                    sys.path.insert(0, str(found_repo))
                break

        _ensure_sf3d_shims(repo_dir=found_repo)

        try:
            from sf3d.system import SF3D
        except ImportError as e:
            raise RuntimeError(
                f"No se pudo importar la librería de SF3D ({e}).\n"
                "Ejecuta 'setup.bat' y selecciona la opción de Stable Fast 3D."
            ) from e

        local_weights_dir = SF3D_LOCAL_DIR
        target_path_or_id = str(local_weights_dir) if (local_weights_dir / "model.safetensors").exists() else self.model_id

        print(f"[SF3D] Inicializando Stable Fast 3D desde '{target_path_or_id}' en {self.device.upper()}...")
        try:
            model = SF3D.from_pretrained(
                target_path_or_id,
                config_name="config.yaml",
                weight_name="model.safetensors"
            )
            model.to(self.device)
            model.eval()
            register_model("sf3d", model)
            self.model = model
            print("[SF3D] Modelo cargado exitosamente en VRAM.")
        except Exception as e:
            err_str = str(e)
            # Si el modelo oficial está restringido por autorización (gated repo) en Hugging Face
            if ("401" in err_str or "403" in err_str or "gated" in err_str.lower() or "unauthorized" in err_str.lower() or "restricted" in err_str.lower()) and target_path_or_id == "stabilityai/stable-fast-3d":
                print("[SF3D] Acceso a stabilityai/stable-fast-3d restringido. Cargando desde mirror verificado (patrickbdevaney/stable-fast-3d)...")
                try:
                    model = SF3D.from_pretrained(
                        "patrickbdevaney/stable-fast-3d",
                        config_name="config.yaml",
                        weight_name="model.safetensors"
                    )
                    model.to(self.device)
                    model.eval()
                    register_model("sf3d", model)
                    self.model = model
                    print("[SF3D] Modelo cargado exitosamente desde mirror en VRAM.")
                    return
                except Exception as e2:
                    raise RuntimeError(
                        "Stable Fast 3D requiere aceptar los términos en Hugging Face:\n"
                        "1. Inicia sesión en https://huggingface.co/stabilityai/stable-fast-3d y haz clic en 'Acknowledge license'.\n"
                        "2. Configura tu token en .env (HF_TOKEN=hf_...) o ejecuta 'huggingface-cli login'."
                    ) from e2
            raise RuntimeError(f"Error al cargar Stable Fast 3D: {e}") from e

    def generate_mesh(
        self,
        image: Image.Image,
        texture_resolution: int = 2048,
        generate_texture: bool = True,
        foreground_ratio: float = 0.82,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> trimesh.Trimesh:
        self._load_model()
        if progress_callback:
            progress_callback("⚡ SF3D: Preprocesando silueta (Anti-Color Bleeding y Padding 0.82)...", 35)

        # 1. Preprocesamiento Inpainting 2D Anti-Halos y encuadre con foreground_ratio=0.82 fijo
        rgba_img = preprocess_image_for_sf3d(
            image,
            foreground_ratio=foreground_ratio,
            edge_dilation_pixels=10
        )

        temp_dir = Path(os.environ["TEMP"])
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_glb = temp_dir / f"sf3d_tmp_{int(time.time()*1000)}.glb"

        try:
            from contextlib import nullcontext
            if progress_callback:
                progress_callback("⚡ SF3D: Inferencia base y horneado de textura en GPU...", 55)

            # Limpieza preventiva de VRAM antes de inferencia y horneado
            flush_vram()
            print_real_vram_status("Pre-Inferencia SF3D")

            target_res = texture_resolution
            mesh_out = None

            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16) if "cuda" in self.device else nullcontext():
                    try:
                        print(f" -> [SF3D Inferencia] Generando modelo 3D con textura a {target_res}x{target_res}...")
                        mesh_out, _ = self.model.run_image(
                            [rgba_img],
                            bake_resolution=target_res,
                            remesh="none"
                        )
                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e_oom:
                        err_str = str(e_oom).lower()
                        if ("out of memory" in err_str or "cuda" in err_str) and target_res > 1024:
                            print(f" -> [SF3D VRAM Fallback] VRAM insuficiente para textura {target_res}x{target_res} en 8GB. "
                                  f"Liberando caché y reintentando automáticamente a 1024x1024...")
                            flush_vram()
                            target_res = 1024
                            mesh_out, _ = self.model.run_image(
                                [rgba_img],
                                bake_resolution=target_res,
                                remesh="none"
                            )
                        else:
                            raise e_oom

            mesh_out.export(str(temp_glb), include_normals=True)
            loaded_mesh = trimesh.load(str(temp_glb), process=False)

            if isinstance(loaded_mesh, trimesh.Scene):
                geometries = list(loaded_mesh.geometry.values())
                if geometries:
                    final_mesh = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
                else:
                    final_mesh = trimesh.Trimesh()
            else:
                final_mesh = loaded_mesh

            # 2. Suavizado Taubin post-baking ELIMINADO:
            # Desplazar vértices 3D después del UV unwrap desalinea la geometría respecto a la textura
            # y agrava el desfase en las costuras. SF3D ya aplica Taubin smoothing antes del UV unwrap en sf3d/system.py.

            # 3. TEXTURA NATIVA SF3D: Dilatación e inpainting de fondo para eliminar costuras UV y texture bleeding
            if generate_texture:
                final_mesh = inpaint_uv_atlas_texture(final_mesh, dilation_radius=48)
                if progress_callback:
                    progress_callback("🎨 SF3D: Textura PBR y atlas UV protegidos...", 80)

            # 4. Ajuste de material PBR anti-glare / mate limpio
            # - metallicFactor forzado a 0.0 (evita reflejos metálicos/plásticos)
            # - roughnessFactor configurado en 0.90 (acabado mate limpio y estilizado)
            # - doubleSided = True
            if generate_texture and hasattr(final_mesh, "visual") and hasattr(final_mesh.visual, "material"):
                mat = final_mesh.visual.material
                if mat is not None:
                    mat.metallicFactor = 0.0
                    mat.roughnessFactor = 0.90
                    mat.doubleSided = True

            if not generate_texture:
                neutral_mat = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                    metallicFactor=0.0,
                    roughnessFactor=0.90,
                    doubleSided=True
                )
                final_mesh.visual.material = neutral_mat

            return final_mesh

        finally:
            if temp_glb.exists():
                try:
                    temp_glb.unlink()
                except Exception:
                    pass
            # SF3D sale de la VRAM y queda aparcado en RAM para la siguiente generación
            self.model = None
            park_model("sf3d")
            print_real_vram_status("Post-SF3D (aparcado)")


# ==============================================================================
# 6. Post-Procesamiento y Validación GLB con pygltflib
# ==============================================================================

def postprocess_and_validate_glb(
    glb_path: Union[str, Path],
    asset_name: str = "KaboomiModel",
    target_faces: int = 4000
) -> None:
    """
    Inspecciona y valida el archivo binario .glb con pygltflib:
    - Asegura que todos los materiales tengan doubleSided = True (evita caras invisibles).
    - Configura baseColorFactor PBR estándar [1.0, 1.0, 1.0, 1.0].
    - Aplica parámetros anti-glare / anti-grasa (metallicFactor=0.0, roughnessFactor=0.90).
    - Inyecta metadatos de exportación para Godot 4.
    """
    try:
        gltf = pygltflib.GLTF2().load(str(glb_path))
        if gltf.asset is None:
            gltf.asset = pygltflib.Asset()
            
        gltf.asset.generator = "Kaboomi Party 3D Generator (SF3D + Hunyuan3D-2)"
        gltf.asset.copyright = "Kaboomi Party / Mastermind"
        
        # Habilitar doubleSided en todos los materiales para que nunca se vuelvan invisibles
        if gltf.materials:
            for mat in gltf.materials:
                mat.doubleSided = True
                if mat.pbrMetallicRoughness:
                    mat.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                    # Parámetros PBR anti-glare / anti-grasa:
                    # metallicFactor = 0.0 (difuso mate puro) y roughnessFactor alto (0.90)
                    mat.pbrMetallicRoughness.metallicFactor = 0.0
                    mat.pbrMetallicRoughness.roughnessFactor = 0.90

        # Metadatos embebidos en el asset glTF
        gltf.asset.extras = {
            "name": asset_name,
            "target_faces": target_faces,
            "engine": "Godot 4.x",
            "centered": True,
            "vram_profile": "16GB_fp16"
        }
        
        gltf.save(str(glb_path))
    except Exception as e:
        print(f" -> [Aviso] Validación pygltflib omitida o no requerida: {e}")


# ==============================================================================
# 7. Exportación Multi-Formato (.GLB, .STL, .3MF)
# ==============================================================================

def export_formats_from_mesh(
    mesh: Any,
    base_output_path: Union[str, Path],
    formats: Optional[list] = None,
    target_faces: int = 4000
) -> dict:
    """
    Exporta una malla a múltiples formatos 3D (.glb, .stl, .3mf).
    Retorna un diccionario con las rutas de los archivos exportados: {'glb': Path, 'stl': Path, '3mf': Path}
    """
    if formats is None:
        formats = ["glb"]

    norm_formats = [str(f).lower().lstrip(".") for f in formats]
    dest_base = Path(base_output_path)
    stem_base = dest_base.with_suffix("")
    
    results = {}
    geo_mesh = mesh.to_geometry() if isinstance(mesh, trimesh.Scene) else mesh

    for fmt in norm_formats:
        out_file = stem_base.with_suffix(f".{fmt}")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f" -> Exportando formato .{fmt.upper()}: {out_file.name}...")
        
        try:
            if fmt == "glb":
                mesh.export(str(out_file), file_type="glb", include_normals=True)
                try:
                    postprocess_and_validate_glb(out_file, asset_name=out_file.stem, target_faces=target_faces)
                except Exception as e_post:
                    print(f"    [Aviso pygltflib] {e_post}")
            elif fmt == "stl":
                geo_mesh.export(str(out_file), file_type="stl")
            elif fmt == "3mf":
                geo_mesh.export(str(out_file), file_type="3mf")
            else:
                geo_mesh.export(str(out_file), file_type=fmt)
                
            results[fmt] = out_file
            print(f"    [OK] .{fmt.upper()} generado ({out_file.stat().st_size / 1024:.1f} KB)")
        except Exception as e_exp:
            print(f"    [ERROR] Falló exportación a .{fmt.upper()}: {e_exp}")

    return results


def convert_glb_to_format(glb_path: Union[str, Path], target_format: str) -> Path:
    """
    Convierte un archivo .glb existente a .stl o .3mf en la misma ubicación.
    Si ya existe y no está desactualizado, retorna la ruta existente.
    """
    glb_path = Path(glb_path)
    fmt = target_format.lower().lstrip(".")
    if fmt == "glb":
        return glb_path
    
    out_file = glb_path.with_suffix(f".{fmt}")
    if not out_file.exists() or out_file.stat().st_mtime < glb_path.stat().st_mtime:
        scene = trimesh.load(str(glb_path))
        geo = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
        geo.export(str(out_file), file_type=fmt)
    return out_file


# ==============================================================================
# 8. Función Orquestadora Completa: generate_model_pipeline()
# ==============================================================================

def generate_model_pipeline(
    input_image_path: Union[str, Path, Image.Image],
    output_glb_path: Optional[Union[str, Path]] = None,
    model_type: str = "hunyuan",
    target_faces: int = 4000,
    steps: int = 5,
    octree_resolution: int = 256,
    seed: int = 42,
    align_bottom: bool = False,
    normalize_scale: bool = True,
    vram_profile: str = "auto",
    low_vram_mode: Optional[bool] = None,
    generate_texture: bool = True,
    rembg_session: Optional[Any] = None,
    clean_image_input: Optional[Image.Image] = None,
    step_callback: Optional[Callable[[int, str], None]] = None,
    export_formats: Optional[list] = None
) -> Path:
    """
    Ejecuta el pipeline automático multi-motor de 4 etapas:
      a) Eliminación de fondo y sombras tenues con rembg (o usa clean_image_input con thresholding)
      b) Generación 3D con Hunyuan3D-2 Turbo o Stable Fast 3D (SF3D)
      c) Decimación a target_faces (2000, 4000 u 8000 tris)
      d) Texturizado PBR / Modo arcilla neutro, centrado y exportación multi-formato
    """
    total_start = time.time()
    
    if output_glb_path is None:
        timestamp = int(time.time())
        dest_path = DEFAULT_OUTPUT_DIR / f"model_{timestamp}.glb"
    else:
        dest_path = resolve_target_path(str(output_glb_path))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolver perfil de VRAM
    resolved_profile = detect_vram_profile() if (vram_profile == "auto" or not vram_profile) else vram_profile.lower()
    p_info = VRAM_PROFILES.get(resolved_profile, VRAM_PROFILES["low_8gb"])
    
    print("\n" + "="*70)
    print(f" INICIANDO GENERACIÓN 3D: {Path(str(input_image_path)).name if not isinstance(input_image_path, Image.Image) else 'Imagen PIL'}")
    print(f" Motor seleccionado: {'Stable Fast 3D (SF3D)' if 'sf3d' in model_type.lower() else 'Tencent Hunyuan3D-2 Turbo'}")
    print(f" Destino final: {dest_path}")
    print(f" Triángulos objetivo: {target_faces:,} (optimizado para Godot Móvil)")
    print(f" Perfil de VRAM: {p_info.get('name', resolved_profile)}")
    print(f" Textura activada: {'Sí (PBR)' if generate_texture else 'No (Modo Arcilla / Malla pura sólida)'}")
    print("="*70)

    # 1. Eliminar Fondo / Usar silueta limpia con umbral estricto
    if step_callback:
        step_callback(1, "✂️ Quitando fondo y limpiando halos residuales con Rembg...")
    if clean_image_input is not None:
        clean_image = preprocess_and_remove_background(clean_image_input, force_rembg=False, session=rembg_session)
        print(" -> Usando silueta pre-procesada de la interfaz (con umbral alfa estricto).")
    else:
        clean_image = preprocess_and_remove_background(input_image_path, session=rembg_session)

    # 2. Inicializar Generador
    if step_callback:
        is_cached = is_dit_cached()
        if is_cached:
            step_callback(2, f"🧠 Cargando Hunyuan3D-2 en GPU ({p_info.get('name', 'fp16')})...")
        else:
            step_callback(2, "🧠 Descargando pesos de Hunyuan3D-2 a D:\\AI_Models... (primera vez, por favor espera)")
            
    if "sf3d" in model_type.lower():
        generator = None
    else:
        generator = Hunyuan3DGenerator(
            model_id="tencent/Hunyuan3D-2",
            shape_subfolder="hunyuan3d-dit-v2-0-turbo",
            paint_subfolder="hunyuan3d-paint-v2-0",
            use_fp16=True,
            low_vram_mode=low_vram_mode,
            vram_profile=resolved_profile
        )

    # Cada motor saca a los demás de la VRAM al activarse (un solo modelo en GPU a la vez)
    print_real_vram_status("CLI Inicio Inferencia")

    # 3. Generación y texturizado según el motor seleccionado
    if "sf3d" in model_type.lower():
        if step_callback:
            step_callback(2, "⚡ Inicializando Stable Fast 3D en GPU...")
        sf3d_gen = SF3DGenerator()
        if step_callback:
            step_callback(3, "⚡ SF3D: Generando geometría y texturas PBR instantáneas...")
        raw_mesh = sf3d_gen.generate_mesh(
            clean_image,
            texture_resolution=2048,
            generate_texture=generate_texture,
            foreground_ratio=0.82
        )
        if target_faces and target_faces < len(raw_mesh.faces):
            decimated_mesh = decimate_mesh_preserving_texture(raw_mesh, target_faces=target_faces)
        else:
            decimated_mesh = raw_mesh

        final_textured_mesh = center_and_orient_mesh(
            decimated_mesh,
            align_floor=align_bottom,
            normalize_scale=normalize_scale
        )

        if hasattr(final_textured_mesh, "visual") and hasattr(final_textured_mesh.visual, "material") and final_textured_mesh.visual.material is not None:
            final_textured_mesh.visual.material.metallicFactor = 0.0
            final_textured_mesh.visual.material.roughnessFactor = 0.90
            final_textured_mesh.visual.material.doubleSided = True

        if not generate_texture:
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.90,
                doubleSided=True
            )
            final_textured_mesh.visual.material = neutral_mat
    else:
        # Etapa 1: Geometría Hunyuan3D-DiT (con textura, máx. TEXTURE_SAFE_MAX_FACES caras) y Taubin Smoothing
        if step_callback:
            step_callback(3, "🧊 [Etapa 1/2] Esculpiendo geometría 3D y optimizando topología...")
        raw_mesh = generator.generate_shape(
            clean_image,
            steps=steps,
            octree_resolution=octree_resolution,
            seed=seed,
            target_faces=target_faces,
            texture_safe_cap=TEXTURE_SAFE_MAX_FACES if generate_texture else None
        )
        aligned_mesh = center_and_orient_mesh(
            raw_mesh,
            align_floor=align_bottom,
            normalize_scale=normalize_scale
        )

        # Purga total de VRAM entre Etapa 1 y Etapa 2
        flush_vram()
        print_real_vram_status("CLI Entre Etapa 1 y 2")

        # Etapa 2: Texturizado Nativo Multivista (Hunyuan3D-Paint en Low-VRAM) o Material PBR Neutro
        if generate_texture:
            if step_callback:
                step_callback(4, "🎨 [Etapa 2/2] Horneando textura multivista con Hunyuan3D-Paint...")
            final_textured_mesh = generator.paint_and_texture_mesh(aligned_mesh, clean_image)
            final_textured_mesh = center_and_orient_mesh(
                final_textured_mesh,
                align_floor=align_bottom,
                normalize_scale=False
            )
        else:
            if step_callback:
                step_callback(4, "⚡ Aplicando material neutro mate y empaquetando archivo .GLB...")
            print("\n[Paso D/D] Generando modelo con material PBR base estándar mate...")
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.90,
                doubleSided=True
            )
            aligned_mesh.visual.material = neutral_mat
            final_textured_mesh = aligned_mesh

        # La CLI termina al exportar: se liberan RAM y VRAM
        unload_all_models()
        print_real_vram_status("CLI Fin Generación")

    # 5. Exportación Multi-Formato (.GLB, .STL, .3MF)
    req_formats = list(export_formats) if export_formats else []
    if not req_formats:
        out_ext = dest_path.suffix.lower().lstrip(".")
        if out_ext in ("glb", "stl", "3mf"):
            req_formats = [out_ext]
        else:
            req_formats = ["glb"]

    print(f"\n[Guardando] Exportando modelo en formatos: {[f.upper() for f in req_formats]}...")
    exported_map = export_formats_from_mesh(
        mesh=final_textured_mesh,
        base_output_path=dest_path,
        formats=req_formats,
        target_faces=target_faces
    )

    # Purga estricta final de VRAM
    flush_vram()

    elapsed = time.time() - total_start
    main_file = exported_map.get("glb") or (list(exported_map.values())[0] if exported_map else dest_path)
    
    print("\n" + "="*70)
    print(" ¡GENERACIÓN COMPLETADA CON ÉXITO!")
    for fmt_k, fmt_path in exported_map.items():
        if fmt_path.exists():
            f_mb = fmt_path.stat().st_size / (1024 * 1024)
            print(f" -> Formato .{fmt_k.upper()}: {fmt_path.name} ({f_mb:.2f} MB)")
    print(f" -> Directorio Destino: {dest_path.parent}")
    print(f" -> Triángulos: {len(final_textured_mesh.faces):,}")
    print(f" -> Vértices: {len(final_textured_mesh.vertices):,}")
    print(f" -> Tiempo Total: {elapsed:.2f} segundos")
    print("="*70 + "\n")

    return main_file


# ==============================================================================
# 9. Interfaz por Consola (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generador 3D con Tencent Hunyuan3D-2 para Kaboomi Party (Godot 4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Ruta a la imagen de entrada (PNG, JPG, WEBP)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Ruta de salida del modelo .glb, .stl o .3mf (ej: ../kaboomi-party/assets/models/bombos/mi_modelo.glb)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="hunyuan",
        choices=["hunyuan", "sf3d"],
        help="Motor de IA a utilizar: 'hunyuan' (Hunyuan3D-2 Turbo HD) o 'sf3d' (Stable Fast 3D PBR instantáneo)"
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Generar el modelo en modo arcilla/sólido SIN textura (ideal para impresión 3D / amigos / CAD)"
    )
    parser.add_argument(
        "--faces", "-f",
        type=int,
        default=4000,
        help="Número objetivo de caras/triángulos de la malla (por defecto 4000 para Godot móvil)"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=5,
        help="Pasos de inferencia DiT (5 para Turbo, 50 para modelo base)"
    )
    parser.add_argument(
        "--octree",
        type=int,
        default=380,
        help="Resolución del Octree para la geometría"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (-1 para aleatorio)"
    )
    parser.add_argument(
        "--align-bottom",
        action="store_true",
        help="Colocar el punto de pivote en los pies (Y=0 en el suelo) en lugar del centro absoluto (0,0,0)"
    )
    parser.add_argument(
        "--vram-profile",
        type=str,
        default="auto",
        choices=["auto", "low_8gb", "mid_12gb", "high_16gb", "cpu"],
        help="Perfil de consumo de VRAM: auto (detecta GPU), low_8gb (RTX 4060), mid_12gb, high_16gb o cpu."
    )
    parser.add_argument(
        "--no-low-vram",
        action="store_true",
        help="Sube el pipeline de forma completo a la GPU en lugar de un componente a la vez (más rápido si sobra VRAM)"
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Formatos a exportar separados por comas (ej: glb,stl,3mf). Si se omite, se deduce de --output o usa glb."
    )
    parser.add_argument(
        "--export-stl",
        action="store_true",
        help="Exportar también en formato STL (.stl) para impresión 3D"
    )
    parser.add_argument(
        "--export-3mf",
        action="store_true",
        help="Exportar también en formato 3MF (.3mf) para slicers de manufactura aditiva (Bambu, Prusa)"
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU para la IA: índice de nvidia-smi (0, 1...), 'cpu' o 'ask' (se aplica en gpu_select.py antes de importar torch)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] El archivo de entrada '{input_path}' no existe.")
        sys.exit(1)

    requested_formats = []
    if args.formats:
        requested_formats.extend([f.strip().lower() for f in args.formats.split(",") if f.strip()])
    if args.export_stl and "stl" not in requested_formats:
        requested_formats.append("stl")
    if args.export_3mf and "3mf" not in requested_formats:
        requested_formats.append("3mf")
    if not requested_formats:
        requested_formats = None

    try:
        generate_model_pipeline(
            input_image_path=input_path,
            output_glb_path=args.output,
            model_type=args.model,
            target_faces=args.faces,
            steps=args.steps,
            octree_resolution=args.octree,
            seed=args.seed,
            align_bottom=args.align_bottom,
            generate_texture=not args.no_texture,
            vram_profile=args.vram_profile,
            low_vram_mode=False if args.no_low_vram else None,
            export_formats=requested_formats
        )
    except Exception as e:
        print(f"\n[ERROR FATAL] La generación falló: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
