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

from mesh_detail import estimate_relative_depth, depth_guided_warp, follow_frame, bake_normal_map, silhouette_iou

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
# Gestor de Residencia de Modelos: UN SOLO modelo en la GPU a la vez y MÍNIMA RAM
# ==============================================================================
# Hay tres modelos pesados: "shape" (Hunyuan3D-DiT), "paint" (Hunyuan3D-Paint) y
# "sf3d" (Stable Fast 3D). Flujo garantizado para cada carga:
#   1. Se sacan de la VRAM todos los demás modelos.
#   2. Se VALIDA que haya VRAM libre suficiente; si no, se destruyen los demás y se revalida.
#   3. Si el modelo no está ya en RAM, se VALIDA que haya RAM libre para cargarlo (más un
#      margen para Windows); si no, se liberan de la RAM los otros modelos y se revalida.
#   4. Si aun así no cabe, NO se carga (InsufficientMemoryError) para no saturar el equipo.
# Al terminar, park_model() lo conserva en RAM solo si cabe dejando RAM_PARK_MIN_FREE_GB
# libres; si no, lo destruye directamente (sin copiarlo antes a la RAM).

RAM_SAFETY_MARGIN_GB = float(os.environ.get("RAM_SAFETY_MARGIN_GB", "2"))
RAM_PARK_MIN_FREE_GB = float(os.environ.get("RAM_PARK_MIN_FREE_GB", "4"))
KEEP_MODELS_IN_RAM = os.environ.get("KEEP_MODELS_IN_RAM", "1").strip().lower() not in ("0", "false", "no")

# Modo de carga del modelo de forma (ver Hunyuan3DGenerator._get_shape_pipeline):
#   auto (recomendado) | full | stream | staged
SHAPE_LOAD_MODE = os.environ.get("SHAPE_LOAD_MODE", "auto").strip().lower()

# Memoria estimada por modelo (fp16), medida sobre los pesos en disco:
#   ram_load: pico de RAM al cargarlo desde disco
#   vram:     VRAM mínima durante su etapa
# Forma (condicionador 2.1 GB + DiT 2.1 GB + VAE 0.4 GB): con la carga directa a GPU la RAM
# apenas se usa; "ram_staged" es lo que ocupa si se guarda entero en RAM.
MODEL_MEMORY_GB = {
    "shape": {"ram_load": 1.0, "ram_staged": 5.0, "vram": 3.2, "vram_full": 6.0},
    # Paint: delight (~2.1 GB) y multivista (~4.3 GB) se cargan directo a la GPU de uno en uno;
    # el multivista baja hasta ~3.2 GB transmitiendo sus bloques (ver PAINT_MV_TIERS)
    "paint": {"ram_load": 3.0, "ram_load_opt": 1.5, "vram": 3.5, "vram_full": 9.0},
    # SF3D: 3.8 GB fp32 en disco; en GPU ~2.6 GB (codificador y backbone en bf16, ver
    # _sf3d_prepare_inference) + ~1 GB de horneado por lotes
    "sf3d":  {"ram_load": 4.5, "vram": 4.0},
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


def cap_vram_to_physical(margin_gb: float = 0.25) -> None:
    """
    Impide que PyTorch pida más VRAM de la que queda libre físicamente. En Windows, el driver
    de NVIDIA desborda en silencio a la RAM compartida (muy lento) en lugar de dar error; con
    este límite se produce un OutOfMemoryError real y los modelos bajan de ubicación
    (lean / bloques / stream) en vez de arrastrarse durante minutos.
    """
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info(0)
    limit = torch.cuda.memory_reserved(0) + free - int(margin_gb * 1024 ** 3)
    fraction = max(0.1, min(1.0, limit / total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(f" -> [VRAM Guard] Límite de VRAM para PyTorch: {limit / 1024 ** 3:.1f} GB (sin desbordar a memoria compartida)")


def _vram_held_mb() -> float:
    """VRAM ocupada por tensores de ESTE proceso."""
    return torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0.0


def _model_modules(kind: str, model: Any) -> list:
    if kind == "paint":
        mods = []
        for sub in getattr(model, "models", {}).values():
            pipe = getattr(sub, "pipeline", None)
            mods += [getattr(pipe, n, None) for n in ("unet", "vae", "text_encoder")]
        return mods
    if kind == "shape":
        return [getattr(model, n, None) for n in ("model", "conditioner", "vae")]
    return [model]


def _model_size_gb(kind: str, model: Any) -> float:
    """Tamaño real de los pesos (los marcadores "meta" de la carga por partes no cuentan)."""
    total = 0
    for m in _model_modules(kind, model):
        if isinstance(m, torch.nn.Module):
            total += sum(t.numel() * t.element_size() for t in m.parameters() if t.device.type != "meta")
    return total / (1024 ** 3)


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
    """
    Destruye un modelo liberando VRAM y RAM, SIN copiarlo antes a la RAM. Si algo aún lo
    referencia (p. ej. el traceback de una excepción), sus pesos pasan al dispositivo "meta":
    se libera la memoria sin copiar nada.
    """
    import weakref
    model = _MODEL_CACHE.pop(kind, None)
    if model is None:
        return
    ram_before = _system_ram_free_gb()
    try:
        ref = weakref.ref(model)
    except TypeError:
        ref = None
    del model
    flush_vram()
    survivor = ref() if ref is not None else None
    if survivor is not None:
        for module in _model_modules(kind, survivor):
            if isinstance(module, torch.nn.Module):
                try:
                    module.to("meta")
                except Exception:
                    pass
        if kind == "paint" and hasattr(survivor, "render"):
            survivor.render = None
        del survivor
        flush_vram()
    print(f" -> [Memoria] '{kind}' liberado (RAM libre: {ram_before:.1f} -> {_system_ram_free_gb():.1f} GB, VRAM del proceso: {_vram_held_mb():.0f} MB)")


def park_model(kind: str) -> None:
    """
    Saca un modelo de la VRAM. Lo conserva en RAM solo si KEEP_MODELS_IN_RAM está activo y,
    tras guardarlo, siguen libres RAM_PARK_MIN_FREE_GB; si no, lo destruye directamente.
    """
    model = _MODEL_CACHE.get(kind)
    if model is None:
        return
    if kind == "shape" and getattr(model, "_studio_mode", "") == "stream":
        release_model(kind)  # La carga por partes no guarda nada en RAM
        return

    size_gb = _model_size_gb(kind, model)
    ram_after = _system_ram_free_gb() - size_gb
    if not KEEP_MODELS_IN_RAM or ram_after < RAM_PARK_MIN_FREE_GB:
        if KEEP_MODELS_IN_RAM:
            print(f" -> [RAM Guard] Guardar '{kind}' ({size_gb:.1f} GB) dejaría {ram_after:.1f} GB de RAM libre: se libera.")
        del model
        release_model(kind)
        return

    _move_model_to_cpu(kind, model)
    flush_vram()
    held_mb = _vram_held_mb()
    if held_mb > 256:
        print(f" -> [VRAM Guard] Tras sacar '{kind}' siguen {held_mb:.0f} MB en VRAM: se destruye el modelo.")
        del model
        release_model(kind)
        return
    print(f" -> [Memoria] '{kind}' ({size_gb:.1f} GB) fuera de la VRAM y conservado en RAM ({_system_ram_free_gb():.1f} GB libres)")


def park_all_models() -> None:
    """Deja la VRAM vacía conservando en RAM los modelos que quepan."""
    for kind in list(_MODEL_CACHE):
        park_model(kind)
    flush_vram()


def activate_model(
    kind: str,
    vram_needed_gb: Optional[float] = None,
    ram_needed_gb: Optional[float] = None,
    strict_vram: bool = False
) -> Optional[Any]:
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
    cap_vram_to_physical()

    # 2. Validar VRAM
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
        ram_need = (ram_needed_gb if ram_needed_gb is not None else est.get("ram_load", 0.0)) + RAM_SAFETY_MARGIN_GB
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
                "Cierra programas o revisa el diagnóstico: python preparar_modelos.py memoria"
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

# Campos con efecto real (el modo de carga de la forma se decide midiendo la memoria libre,
# ver SHAPE_LOAD_MODE):
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
        "max_octree": 256,
        "paint_resolution": 1024,
    },
    "low_6gb": {
        "name": "6GB VRAM (Optimizado RTX 4050 Laptop / RTX 3050)",
        "desc": "6GB: carga por partes directa a GPU (mínima RAM) y octree 256.",
        "use_fp16": True,
        "max_octree": 256,
        "paint_resolution": 1024,
    },
    "low_8gb": {
        "name": "8GB VRAM (Optimizado RTX 4060 / Laptops)",
        "desc": "8GB: carga directa a GPU (mínima RAM) y octree 256.",
        "use_fp16": True,
        "max_octree": 256,
        "paint_resolution": 1024,
    },
    "mid_12gb": {
        "name": "12GB VRAM (Equilibrado RTX 3060 / 4070)",
        "desc": "12GB: carga directa a GPU y octree hasta 320.",
        "use_fp16": True,
        "max_octree": 320,
        "paint_resolution": 1024,
    },
    "high_16gb": {
        "name": "16GB+ VRAM (Máximo Rendimiento RTX 4080 / 4090 / 3090)",
        "desc": "16GB+: carga directa a GPU, octree hasta 384 y textura 2048.",
        "use_fp16": True,
        "max_octree": 384,
        "paint_resolution": 2048,
    },
    "cpu": {
        "name": "CPU Seguro (Sin GPU)",
        "desc": "Inferencia forzada en memoria RAM del sistema (muy lenta).",
        "use_fp16": False,
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
        # trimesh aplica el paso inverso como "vertices -= nu * laplaciano": con nu negativo los dos
        # pasos suavizan, así que en la práctica es un Laplaciano. Es intencionado: quita los bultos
        # de pocos vóxeles del generador mucho mejor que un Taubin real (nu positivo) y apenas
        # adelgaza (~1 % de fondo en 5 iteraciones). No cambiar el signo sin volver a medirlo.
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

            # Se conserva el material original con su acabado (mate, satinado o metal)
            if hasattr(decimated_mesh.visual, "material") and decimated_mesh.visual.material is not None:
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

# ==============================================================================
# Profundidad del modelo: medición, refuerzo y ajustes de "Profundidad mejorada" / "Máxima calidad"
# ==============================================================================
# Tras center_and_orient_mesh el modelo queda con Y hacia arriba y Z como eje de fondo
# (el eje de la cámara de la imagen de entrada).
# Solo se engorda lo que sale casi de papel: una moneda o una placa miden ~8-12 % de fondo de
# verdad y forzar más las infla. Para personajes que salen tipo cartón, subirlo en .env (~0.30).
DEPTH_MIN_RATIO = float(os.environ.get("DEPTH_MIN_RATIO", "0.08"))
DEPTH_FAST_STEPS = 30
# Semillas que se prueban con "Profundidad mejorada"; gana la que más se parece a la silueta de la imagen
DEPTH_SEED_CANDIDATES = max(1, int(os.environ.get("DEPTH_SEED_CANDIDATES", "3")))
# Octree con el que se decodifica cada semilla candidata solo para medir su fondo (rápido)
DEPTH_SCORE_OCTREE = 128
# "Máxima calidad": modelo completo (sin destilar) con guía; cada pasada cuesta ~3x la Fast
MAX_QUALITY_SUBFOLDER = "hunyuan3d-dit-v2-0"
MAX_QUALITY_STEPS = 50
MAX_QUALITY_GUIDANCE = float(os.environ.get("MAX_QUALITY_GUIDANCE", "6.0"))
MAX_QUALITY_SEED_CANDIDATES = 2


def measure_depth_ratio(mesh: trimesh.Trimesh) -> float:
    """Fondo (Z) respecto a la mayor dimensión frontal (X/Y): ~0.15 = cartón, >0.4 = volumen normal."""
    ext = np.asarray(mesh.extents, dtype=np.float64)
    front = max(ext[0], ext[1])
    return float(ext[2] / front) if front > 1e-9 else 0.0


def reinforce_depth(
    mesh: trimesh.Trimesh,
    min_ratio: float = DEPTH_MIN_RATIO,
    image: Optional[Image.Image] = None,
    companions: tuple = ()
) -> Tuple[trimesh.Trimesh, float, float, str]:
    """
    Si el modelo salió plano (fondo < min_ratio), lo engorda en Z hasta min_ratio:
      - Con `image`: guiado por la profundidad estimada (Depth Anything V2): crece más donde la
        imagen sobresale y se redondea hacia los bordes, en vez de estirar todo por igual.
      - Lo que falte (o sin estimador) se completa estirando solo en Z alrededor del centro.
    `companions` (p. ej. la malla HD del normal map) reciben exactamente la misma deformación.
    Devuelve (malla, proporción antes, proporción después, método: "", "guiado" o "uniforme").
    """
    before = measure_depth_ratio(mesh)
    if before <= 0 or before >= min_ratio:
        return mesh, before, before, ""
    method = "uniforme"
    estimate = estimate_relative_depth(image) if image is not None else None
    if estimate is not None:
        front = float(max(mesh.extents[0], mesh.extents[1]))
        warp = depth_guided_warp(mesh, *estimate, target_depth=min_ratio * front)
        if warp is not None:
            for m in (mesh, *companions):
                m.vertices = warp(m.vertices)
            method = "guiado"

    current = measure_depth_ratio(mesh)
    if 0 < current < min_ratio:
        factor = min_ratio / current
        center_z = float(mesh.bounds.mean(axis=0)[2])
        transform = np.eye(4)
        transform[2, 2] = factor
        transform[2, 3] = center_z * (1.0 - factor)
        for m in (mesh, *companions):
            m.apply_transform(transform)
    return mesh, before, measure_depth_ratio(mesh), method


def _dit_variant_ready(subfolder: str) -> bool:
    folder = HUNYUAN_LOCAL_DIR / subfolder
    return (folder / "model.fp16.safetensors").exists() and (folder / "config.yaml").exists()


def shape_quality_settings(octree_requested: int, improved_depth: bool = False, max_quality: bool = False) -> dict:
    """
    Ajustes de la geometría Hunyuan3D según las casillas (sin ninguna: Turbo, como siempre):
      Profundidad mejorada: variante Fast (30 pasos, reconstruye mejor el volumen que no se ve),
        DEPTH_SEED_CANDIDATES semillas (gana la que más se parece a la silueta), suavizado ligero (2) y refuerzo guiado.
      Máxima calidad: modelo completo hunyuan3d-dit-v2-0 (50 pasos, guía MAX_QUALITY_GUIDANCE)
        y suavizado ligero (2). Con las dos casillas, MAX_QUALITY_SEED_CANDIDATES semillas.
      Cualquiera de las dos: octree 384 (si la decodificación no cabe se reintenta con 256) y el
        detalle de la malla HD se hornea en un normal map de la malla final.
    Si falta un modelo se usa el mejor descargado y "missing" dice cuál descargar.
    """
    settings = {
        "shape_subfolder": "hunyuan3d-dit-v2-0-turbo",
        "steps": None,
        "guidance": None,
        "octree": int(octree_requested or 256),
        "taubin_iterations": 5,
        "seeds": 1,
        "normal_map": False,
        "missing": None,
    }
    if not (improved_depth or max_quality):
        return settings
    has_fast = _dit_variant_ready("hunyuan3d-dit-v2-0-fast")
    fast = {"shape_subfolder": "hunyuan3d-dit-v2-0-fast", "steps": DEPTH_FAST_STEPS} if has_fast else {}
    settings.update(octree=max(settings["octree"], 384), normal_map=True)
    if improved_depth:
        settings.update(taubin_iterations=2, seeds=DEPTH_SEED_CANDIDATES, **fast)
        if not has_fast:
            settings["missing"] = ("dit_fast", "Hunyuan3D-2 Fast")
    if max_quality:
        settings["taubin_iterations"] = 2
        if _dit_variant_ready(MAX_QUALITY_SUBFOLDER):
            settings.update(shape_subfolder=MAX_QUALITY_SUBFOLDER, steps=MAX_QUALITY_STEPS,
                            guidance=MAX_QUALITY_GUIDANCE,
                            seeds=min(settings["seeds"], MAX_QUALITY_SEED_CANDIDATES))
        else:
            settings.update(**fast)
            settings["missing"] = ("dit_full", "Hunyuan3D-2 completo")
    return settings


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

    # *args/**kwargs: reenvía assign=True (carga directa a GPU sin pasar por RAM)
    def _tolerant_single_encoder_load(self, state_dict, strict=True, *args, **kwargs):
        return _orig_single_encoder_load(self, state_dict, False, *args, **kwargs)

    def _tolerant_dual_encoder_load(self, state_dict, strict=True, *args, **kwargs):
        return _orig_dual_encoder_load(self, state_dict, False, *args, **kwargs)

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
PAINT_RASTERIZER = "ninguno"


def hunyuan_paint_available() -> bool:
    """
    Comprueba una sola vez si Hunyuan3D-Paint puede ejecutarse. Si la extensión CUDA compilada
    `custom_rasterizer` no está instalada, se usa el rasterizador PyTorch equivalente
    (torch_rasterizer.py): no hace falta Visual Studio ni CUDA Toolkit.
    """
    global _PAINT_UNAVAILABLE_REASON, _PAINT_CHECKED, PAINT_RASTERIZER
    if not _PAINT_CHECKED:
        _PAINT_CHECKED = True
        try:
            import custom_rasterizer  # noqa: F401
            PAINT_RASTERIZER = "custom_rasterizer (CUDA compilado)"
        except Exception:
            import torch_rasterizer
            sys.modules["custom_rasterizer"] = torch_rasterizer
            PAINT_RASTERIZER = "PyTorch en GPU (torch_rasterizer.py, sin compilar)"
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
        except Exception as e:
            _PAINT_UNAVAILABLE_REASON = f"no se pudo importar hy3dgen.texgen ({e})"
        print(f" -> [Hunyuan3D-Paint] Rasterizador: {PAINT_RASTERIZER}")
        # Relleno de huecos por vértice: la versión Python de hy3dgen tarda 40-80 s; si no está
        # la versión C++ compilada se usa la de Numba (mismo algoritmo y resultado, <1 s)
        try:
            import hy3dgen.texgen.differentiable_renderer.mesh_processor as mesh_processor_mod
            import hy3dgen.texgen.differentiable_renderer.mesh_render as mesh_render_mod
            if str(getattr(mesh_processor_mod, "__file__", "")).endswith(".py"):
                import fast_mesh_inpaint
                mesh_render_mod.meshVerticeInpaint = fast_mesh_inpaint.meshVerticeInpaint
                print(" -> [Hunyuan3D-Paint] Relleno de huecos: Numba (compilado al vuelo)")
            else:
                print(" -> [Hunyuan3D-Paint] Relleno de huecos: mesh_processor nativo C++")
        except Exception as e_mp:
            print(f" -> [Hunyuan3D-Paint] Relleno de huecos: versión Python ({e_mp})")
    return _PAINT_UNAVAILABLE_REASON is None


# Dispositivo destino del UNet multivista durante su carga (lo fija _paint_lowmem_loading)
_PAINT_UNET_TARGET = "cuda"


def _lowmem_unet2p5d_from_pretrained(unet_cls):
    """
    Sustituye UNet2p5DConditionModel.from_pretrained (el modules.py que trae el modelo) por una
    carga sin pico de RAM. El original crea el UNet en fp32 (~7 GB), lee el .bin entero (3.5 GB)
    y luego convierte a fp16. Aquí se crea sin memoria (meta) y el .bin se lee con mmap:
      - destino "cuda": cada tensor va directo a la GPU.
      - destino "cpu" : los tensores siguen respaldados por el archivo (caché de disco de Windows,
                        no RAM reservada); se usa para transmitir los bloques a la GPU.
    Mismo código de modelo y mismos pesos: resultado idéntico.
    """
    import json
    original = unet_cls.from_pretrained
    module = sys.modules[unet_cls.__module__]

    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", torch.float32)
        target = _PAINT_UNET_TARGET if torch.cuda.is_available() else "cpu"
        try:
            with open(os.path.join(pretrained_model_name_or_path, "config.json"), "r", encoding="utf-8") as fh:
                config = json.load(fh)
            with torch.device("meta"):
                unet = unet_cls(module.UNet2DConditionModel(**config))
            bin_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.bin")
            st_path = os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors")
            if not os.path.exists(bin_path) and os.path.exists(st_path):
                # Caché optimizada (fp16 safetensors): lectura directa al dispositivo destino
                from safetensors import safe_open
                state = {}
                with safe_open(st_path, framework="pt", device=str(target)) as f:
                    for k in f.keys():
                        v = f.get_tensor(k)
                        state[k] = v.to(torch_dtype) if v.is_floating_point() else v
            else:
                ckpt = torch.load(bin_path, map_location="cpu", weights_only=True, mmap=True)
                state = {k: (v.to(target, torch_dtype) if v.is_floating_point() else v.to(target)) for k, v in ckpt.items()}
                del ckpt
            unet.load_state_dict(state, strict=True, assign=True)
            del state
            if any(t.is_meta for t in list(unet.parameters()) + list(unet.buffers())):
                raise RuntimeError("quedaron pesos sin cargar")
            print(f" -> [Carga] UNet multivista -> {target} sin pasar por la RAM")
            return unet
        except Exception as e:
            print(f" -> [Carga] Carga directa del UNet no disponible ({e}); se usa la carga original.")
            flush_vram()
            return original(pretrained_model_name_or_path, torch_dtype=torch_dtype, **kwargs)

    return staticmethod(from_pretrained)


class _paint_lowmem_loading:
    """Contexto: mientras diffusers monta el pipeline multivista, su UNet se carga con poca RAM."""

    def __init__(self, unet_target: str = "cuda"):
        self.unet_target = unet_target

    def __enter__(self):
        global _PAINT_UNET_TARGET
        import diffusers.pipelines.pipeline_loading_utils as plu
        self._plu = plu
        self._orig = plu.get_class_obj_and_candidates
        self._prev_target = _PAINT_UNET_TARGET
        _PAINT_UNET_TARGET = self.unet_target

        def patched(*args, **kwargs):
            class_obj, candidates = self._orig(*args, **kwargs)
            if getattr(class_obj, "__name__", "") == "UNet2p5DConditionModel" and not getattr(class_obj, "_studio_lowmem", False):
                class_obj.from_pretrained = _lowmem_unet2p5d_from_pretrained(class_obj)
                class_obj._studio_lowmem = True
            return class_obj, candidates

        plu.get_class_obj_and_candidates = patched
        return self

    def __exit__(self, *exc):
        global _PAINT_UNET_TARGET
        self._plu.get_class_obj_and_candidates = self._orig
        _PAINT_UNET_TARGET = self._prev_target
        return False


# ------------------------------------------------------------------------------
# Ubicación del modelo multivista según la VRAM libre (de más rápido a menos VRAM).
# Si una se queda sin VRAM durante la inferencia, se reintenta con la siguiente.
# El UNet multivista tiene dos copias: "unet" (1.8 GB, 30 pasos) y "unet_dual" (1.6 GB, usada
# UNA vez por generación para la imagen de referencia). Bloque más grande: 0.56 GB.
#   gpu     : todo en la GPU (~4.2 GB de pesos).
#   lean    : "unet" fijo en GPU; "unet_dual" se transmite por bloques (solo se usa una vez) y
#             el text encoder (0.65 GB) sube solo al codificar el prompt. Prácticamente igual
#             de rápido que "gpu" con ~2.3 GB menos de VRAM.
#   bloques : también "unet" se transmite por bloques en cada uno de los 30 pasos: solo el
#             bloque en curso (<= 0.56 GB) está en la GPU. Cabe en GPUs de 6 GB con holgura;
#             añade unos segundos por generación. Misma calidad en los tres.
# Los pesos transmitidos se leen del archivo (mmap): Windows los mantiene en su caché de disco
# si sobra RAM, sin reservarla.
# ------------------------------------------------------------------------------
PAINT_MV_VRAM_GB = {"gpu": 7.0, "lean": 5.2, "bloques": 3.2}
PAINT_MV_TIERS = ("gpu", "lean", "bloques")


def _on_demand_gpu(module: torch.nn.Module, device: str) -> None:
    """El módulo vive en CPU y sube a la GPU solo mientras se ejecuta su forward."""
    module.to("cpu")

    def pre(mod, args):
        mod.to(device)

    def post(mod, args, output):
        mod.to("cpu")
        return output

    module.register_forward_pre_hook(pre)
    module.register_forward_hook(post)


def _pin_pipeline_device(pipe, device: str) -> None:
    """
    diffusers deduce pipe.device / _execution_device del "primer" componente, en el orden de un
    set (cambia en cada arranque de Python). Con el text encoder en CPU a veces devolvía "cpu":
    la imagen se quedaba en CPU con el VAE en la GPU. Se fija al dispositivo de cálculo.
    """
    dev = torch.device(device)
    cls = type(pipe)
    pipe.__class__ = type(cls.__name__, (cls,), {
        "device": property(lambda self: dev),
        "_execution_device": property(lambda self: dev),
    })


def _stream_unet_blocks(unet2p5d: torch.nn.Module, device: str, stream=("unet", "unet_dual")) -> int:
    """
    Los bloques grandes del UNet conservan sus pesos en CPU (respaldados por el archivo) y se
    copian a la GPU al entrar en su forward; al salir se suelta la copia. Todo lo demás
    (embeddings, conv_in/out, parámetros directos como learned_text_clip_*) queda fijo en GPU,
    porque el código del modelo los lee fuera de cualquier forward.
    Las copias que no están en `stream` quedan enteras y fijas en la GPU.
    Devuelve el número de bloques transmitidos.
    """
    streamed = 0
    for inner_name in ("unet", "unet_dual"):
        inner = getattr(unet2p5d, inner_name, None)
        if inner is None:
            continue
        if inner_name not in stream:
            inner.to(device)
            continue
        blocks = [*inner.down_blocks, inner.mid_block, *inner.up_blocks]

        # Lo pequeño, a la GPU de forma fija
        for name, child in inner.named_children():
            if name not in ("down_blocks", "up_blocks", "mid_block"):
                child.to(device)
        for name, p in inner._parameters.items():
            if p is not None:
                p.data = p.data.to(device)
        for name, b in list(inner._buffers.items()):
            if b is not None:
                inner._buffers[name] = b.to(device)

        for block in blocks:
            if block is None:
                continue
            slots = []
            for m in block.modules():
                for n, p in m._parameters.items():
                    if p is not None:
                        slots.append((m, n, True, p.data))
                for n, b in m._buffers.items():
                    if b is not None:
                        slots.append((m, n, False, b))

            def pre(mod, args, _slots=slots):
                # Hereda el modo de quien llama: forzar inference_mode(False) aquí rompe el
                # conv2d si los pesos se crearon dentro de inference_mode
                with torch.no_grad():
                    for m, n, is_param, cpu in _slots:
                        gpu = cpu.to(device, non_blocking=True)
                        if is_param:
                            m._parameters[n].data = gpu
                        else:
                            m._buffers[n] = gpu

            def post(mod, args, output, _slots=slots):
                for m, n, is_param, cpu in _slots:
                    if is_param:
                        m._parameters[n].data = cpu
                    else:
                        m._buffers[n] = cpu
                return output

            block.register_forward_pre_hook(pre)
            block.register_forward_hook(post)
            streamed += 1

    # Otros hijos del envoltorio (si los hay) también fijos en GPU
    for name, child in unet2p5d.named_children():
        if name not in ("unet", "unet_dual"):
            child.to(device)
    return streamed


def _build_multiview_net(config, tier: str):
    """Monta Multiview_Diffusion_Net (mismo comportamiento que su __init__) con la ubicación `tier`."""
    import hy3dgen.texgen.utils.multiview_utils as mvu
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    device = config.device
    net = mvu.Multiview_Diffusion_Net.__new__(mvu.Multiview_Diffusion_Net)
    net.device = device
    net.view_size = 512
    custom_pipeline_path = os.path.join(os.path.dirname(os.path.abspath(mvu.__file__)), "..", "hunyuanpaint")

    # gpu: directo a la GPU. lean/bloques: respaldados por el archivo; luego se reparten
    unet_target = device if tier == "gpu" else "cpu"
    with _paint_lowmem_loading(unet_target):
        pipe = DiffusionPipeline.from_pretrained(
            config.multiview_ckpt_path, custom_pipeline=custom_pipeline_path, torch_dtype=torch.float16
        )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.set_progress_bar_config(disable=True)

    # El VAE siempre en GPU: pipeline.device (y el generador de ruido) se toman de él
    pipe.vae.to(device)
    if tier == "gpu" or "cuda" not in device:
        pipe.to(device)
    else:
        _on_demand_gpu(pipe.text_encoder, device)
        _pin_pipeline_device(pipe, device)
        stream = ("unet", "unet_dual") if tier == "bloques" else ("unet_dual",)
        n = _stream_unet_blocks(pipe.unet, device, stream=stream)
        print(f" -> [Carga] UNet multivista: {n} bloques ({', '.join(stream)}) se transmiten a la GPU al usarse")
    net.pipeline = pipe
    return net


class _PaintProgress:
    """Cronómetro y avance en vivo de las etapas de Hunyuan3D-Paint (consola + WebUI)."""

    def __init__(self, callback: Optional[Callable[[str, int], None]] = None):
        self.callback = callback
        self.times: list = []

    def report(self, msg: str, pct: int) -> None:
        if self.callback:
            try:
                self.callback(f"[4/4] {msg}", int(pct))
            except Exception:
                pass

    def stage(self, label: str, fn: Callable, pct: int) -> Callable:
        def timed(*args, **kwargs):
            print(f" -> [Paint] {label}...")
            self.report(f"{label}...", pct)
            t0 = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                self.times.append((label, time.time() - t0))
        return timed

    def submodel(self, label: str, pct_start: int, pct_end: int, steps: int) -> "_SubmodelProgress":
        return _SubmodelProgress(self, label, pct_start, pct_end, steps)

    def summary(self, total_s: float) -> str:
        merged: dict = {}
        for label, secs in self.times:
            merged[label] = merged.get(label, 0.0) + secs
        parts = [f"{label.split(' (')[0]} {secs:.0f}s" for label, secs in merged.items()]
        return " | ".join(parts + [f"total {total_s:.0f}s"])


class _SubmodelProgress:
    """Avance de un submodelo de Paint: carga y cada paso de difusión (cuenta llamadas al UNet)."""

    def __init__(self, tracker: _PaintProgress, label: str, pct_start: int, pct_end: int, steps: int):
        self.tracker = tracker
        self.label = label
        self.pct_start = pct_start
        self.pct_end = pct_end
        self.steps = steps
        self.count = 0
        self.tier: Optional[str] = None
        self.t0 = 0.0

    def begin(self, tier: Optional[str]) -> None:
        self.count = 0
        self.t0 = time.time()
        self.tier = tier
        self.tracker.report(f"{self.label}: cargando modelo en GPU...", self.pct_start)

    def step(self) -> None:
        self.count += 1
        done = min(self.count, self.steps)
        pct = self.pct_start + (self.pct_end - self.pct_start) * done / max(self.steps, 1)
        where = f" ({self.tier})" if self.tier else ""
        self.tracker.report(f"{self.label}{where}: paso {done}/{self.steps}", int(pct))

    def end(self) -> None:
        self.tracker.times.append((self.label, time.time() - self.t0))


def _paint_vram_stats_file() -> Path:
    return MODELS_ROOT / "paint_vram_stats.json"


def _load_paint_vram_stats() -> dict:
    import json
    try:
        return json.loads(_paint_vram_stats_file().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_paint_vram_stat(key: str, peak_gb: float) -> None:
    import json
    stats = _load_paint_vram_stats()
    stats[key] = round(peak_gb, 2)
    try:
        _paint_vram_stats_file().write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except Exception:
        pass


class _LazyPaintModel:
    """
    Submodelo de Hunyuan3D-Paint (delight o multivista) que se carga directo a la GPU solo
    cuando el pipeline lo llama, en la ubicación que quepa en la VRAM libre. Si se queda sin
    VRAM, se reintenta con la siguiente ubicación. Con keep=False (modo "stream") no coincide
    en la VRAM con el otro submodelo: al terminar se libera o, en ubicación "gpu" y si sobra
    RAM, se conserva en RAM para no releerlo del disco en la siguiente generación.
    La VRAM real de cada ubicación se mide y se guarda (paint_vram_stats.json): las siguientes
    generaciones eligen con datos reales en lugar de estimaciones.
    """

    def __init__(self, name: str, factory: Callable[[str], Any], device: str, tiers: tuple, vram_gb: dict):
        self.name = name
        self.factory = factory
        self.device = device
        self.tiers = tiers
        self.vram_gb = vram_gb
        self.keep = False
        self.instance = None
        self.tier = None
        self.parked = False
        self.progress: Optional[_SubmodelProgress] = None

    @property
    def pipeline(self):
        return getattr(self.instance, "pipeline", None)

    def _need_gb(self, tier: str) -> float:
        measured = _load_paint_vram_stats().get(f"{self.name}:{tier}")
        return measured * 1.05 + 0.2 if measured else self.vram_gb.get(tier, 0.0)

    def _pick_tiers(self) -> list:
        free = _vram_free_gb() if "cuda" in self.device else float("inf")
        fitting = [t for t in self.tiers if free >= self._need_gb(t)]
        start = self.tiers.index(fitting[0]) if fitting else len(self.tiers) - 1
        return list(self.tiers[start:])

    def _on_unet_forward(self, *_):
        if self.progress:
            self.progress.step()

    def _load(self, tier: str) -> None:
        t0 = time.time()
        flush_vram()
        cap_vram_to_physical()
        self.instance = self.factory(tier)
        self.tier = tier
        self.parked = False
        for opt in ("enable_vae_slicing", "enable_vae_tiling"):
            if hasattr(self.pipeline, opt):
                try:
                    getattr(self.pipeline, opt)()
                except Exception:
                    pass
        unet = getattr(self.pipeline, "unet", None)
        if isinstance(unet, torch.nn.Module):
            unet.register_forward_hook(self._on_unet_forward)
        print(f" -> [Carga] Paint '{self.name}' (ubicación '{tier}') en {time.time() - t0:.1f}s")
        print_real_vram_status(f"Paint '{self.name}' cargado")

    def _drop(self) -> None:
        self.instance = None
        self.tier = None
        self.parked = False
        flush_vram()

    def _size_gb(self) -> float:
        mods = (getattr(self.pipeline, n, None) for n in ("unet", "text_encoder", "vae"))
        return sum(_model_size_gb("sf3d", m) for m in mods if isinstance(m, torch.nn.Module))

    def _park_or_drop(self) -> None:
        """Tras usarse: a la RAM si es ubicación "gpu" y sobra memoria; si no, se libera."""
        size = self._size_gb()
        if KEEP_MODELS_IN_RAM and self.tier == "gpu" and _system_ram_free_gb() - size >= RAM_PARK_MIN_FREE_GB:
            self.pipeline.to("cpu")
            self.parked = True
            flush_vram()
            print(f" -> [Memoria] Paint '{self.name}' ({size:.1f} GB) conservado en RAM para la próxima generación")
        else:
            self._drop()
            print_real_vram_status(f"Paint '{self.name}' liberado")

    def _start_measure(self) -> float:
        """VRAM del proceso antes de cargar/subir el submodelo; el pico medido incluye sus pesos."""
        if not torch.cuda.is_available():
            return 0.0
        flush_vram()
        torch.cuda.reset_peak_memory_stats(0)
        return torch.cuda.memory_reserved(0) / (1024 ** 3)

    def __call__(self, *args, **kwargs):
        start_gb = self._start_measure()
        if self.instance is not None and self.parked:
            if self.tier in self._pick_tiers():
                cap_vram_to_physical()
                self.pipeline.to(self.device)  # Venía aparcado en RAM: sin leer el disco
                self.parked = False
                print(f" -> [Carga] Paint '{self.name}' reutilizado desde RAM")
            else:
                self._drop()
        tiers = [self.tier] if self.instance is not None else self._pick_tiers()
        if self.progress:
            self.progress.begin(tiers[0])
        try:
            for i, tier in enumerate(tiers):
                if i > 0:
                    start_gb = self._start_measure()
                if self.instance is None:
                    # Paint se llama dentro de inference_mode: los pesos se crean fuera para que
                    # sean tensores normales (mover/transmitir por bloques/aparcar sin errores)
                    with torch.inference_mode(False), torch.no_grad():
                        self._load(tier)
                if self.progress:
                    self.progress.tier = tier
                    self.progress.count = 0
                try:
                    result = self.instance(*args, **kwargs)
                except torch.cuda.OutOfMemoryError:
                    self._drop()
                    if i + 1 >= len(tiers):
                        raise
                    print(f" -> [Memoria] Paint '{self.name}' sin VRAM en '{tier}': reintentando con '{tiers[i + 1]}'...")
                    continue
                if torch.cuda.is_available():
                    # Pesos + cálculo de este submodelo, por encima de lo que ya ocupaba el proceso
                    used_gb = torch.cuda.max_memory_reserved(0) / (1024 ** 3) - start_gb
                    _save_paint_vram_stat(f"{self.name}:{tier}", max(used_gb, 0.0))
                    print(f" -> [Memoria] Paint '{self.name}' en '{tier}': usó {used_gb:.2f} GB de VRAM (medido)")
                return result
        finally:
            if self.progress:
                self.progress.end()
            if not self.keep and self.instance is not None and not self.parked:
                self._park_or_drop()


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
        # Modo de carga de la forma: SHAPE_LOAD_MODE (.env); --no-low-vram en la CLI fuerza "full"
        self.shape_mode_pref = "full" if low_vram_mode is False else SHAPE_LOAD_MODE
        self.max_octree = pcfg["max_octree"]
        self.paint_resolution = pcfg["paint_resolution"]
        # Última geometría: malla HD sin reducir (para el normal map) y semillas probadas
        self.last_hd_mesh = None
        self.last_seed_report = None

        print(f"[Hunyuan3D] Dispositivo: {self.device.upper()} | Tipo de dato: {self.dtype} | Perfil VRAM: {self.vram_profile_name} | Carga: {self.shape_mode_pref}")
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

    # --------------------------------------------------------------------------
    # Carga del pipeline de forma con la MÍNIMA RAM posible
    # --------------------------------------------------------------------------
    # hy3dgen.from_pretrained lee el archivo entero a la RAM y crea el modelo vacío en fp32
    # antes de copiar los pesos: pico de ~14 GB de RAM. Aquí cada componente se crea sin
    # memoria (dispositivo "meta") y recibe sus pesos fp16 leídos directamente al dispositivo
    # destino, sin pasar por la RAM. Modos:
    #   full   : condicionador + DiT + VAE en la GPU (VRAM ~6 GB, RAM ~0). El más rápido.
    #   stream : por partes; cada componente se lee del disco a la GPU en su etapa y se libera
    #            al terminarla (VRAM ~3 GB, RAM ~0). Relee ~4.6 GB del disco por generación.
    #   staged : los 3 componentes en RAM y suben a la GPU de uno en uno (RAM ~4.6 GB).
    # auto: "full" si cabe en la VRAM; si no, "staged" solo si el pipeline ya está en RAM
    # (no cuesta RAM extra) y "stream" en cualquier otro caso. La calidad es idéntica.

    def _shape_files(self) -> Tuple[Path, Path]:
        from hy3dgen.shapegen.utils import smart_load_model
        config_path, ckpt_path = smart_load_model(
            self.model_id, subfolder=self.shape_subfolder, use_safetensors=True, variant="fp16"
        )
        return Path(config_path), Path(ckpt_path)

    def _meta_shape_component(self, cfg: dict, comp: str) -> torch.nn.Module:
        """Componente sin memoria: conserva atributos (p. ej. guidance_embed) sin ocupar RAM ni VRAM."""
        from hy3dgen.shapegen.pipelines import instantiate_from_config
        with torch.device("meta"):
            return instantiate_from_config(cfg[comp])

    def _load_shape_component(self, cfg: dict, ckpt: Path, comp: str, device: torch.device) -> torch.nn.Module:
        """Crea un componente y carga sus pesos tensor a tensor directamente en `device`."""
        from safetensors import safe_open
        from hy3dgen.shapegen.pipelines import instantiate_from_config
        t0 = time.time()
        if comp == "vae":
            # El VAE calcula un buffer en su constructor (fourier_embedder): se crea normal (0.4 GB)
            module = instantiate_from_config(cfg[comp])
        else:
            module = self._meta_shape_component(cfg, comp)
        prefix = comp + "."
        state = {}
        with safe_open(str(ckpt), framework="pt", device=str(device)) as f:
            for key in f.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = f.get_tensor(key).to(self.dtype)
        module.load_state_dict(state, strict=True, assign=(comp != "vae"))
        del state
        # El parche tolerante de los encoders usa strict=False: validar que no quedó nada vacío
        empty = [n for n, t in list(module.named_parameters()) + list(module.named_buffers()) if t.is_meta]
        if empty:
            raise RuntimeError(f"state_dict incompleto para '{comp}': faltan {len(empty)} pesos (ej. {empty[:3]})")
        module.to(device=device, dtype=self.dtype).eval()
        print(f" -> [Carga] '{comp}' -> {device} en {time.time() - t0:.1f}s ({_model_size_gb('sf3d', module):.2f} GB)")
        return module

    def _build_shape_pipeline(self, mode: str):
        """Monta el pipeline de hy3dgen sin su from_pretrained (que dispara la RAM)."""
        import yaml
        from hy3dgen.shapegen.pipelines import instantiate_from_config
        config_path, ckpt = self._shape_files()
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        device = torch.device(self.device)
        load_device = torch.device("cpu") if mode == "staged" else device

        pipe = Hunyuan3DShapePipeline.__new__(Hunyuan3DShapePipeline)
        pipe.scheduler = instantiate_from_config(cfg["scheduler"])
        pipe.image_processor = instantiate_from_config(cfg["image_processor"])
        pipe.kwargs = {"from_pretrained_kwargs": {
            "model_path": str(HUNYUAN_LOCAL_DIR), "subfolder": self.shape_subfolder,
            "use_safetensors": True, "variant": "fp16", "dtype": self.dtype, "device": str(device),
        }}
        pipe.device = device
        pipe.dtype = self.dtype
        pipe.vae = self._load_shape_component(cfg, ckpt, "vae", load_device)
        if mode == "stream":
            pipe.conditioner = self._meta_shape_component(cfg, "conditioner")
            pipe.model = self._meta_shape_component(cfg, "model")
        else:
            pipe.conditioner = self._load_shape_component(cfg, ckpt, "conditioner", load_device)
            pipe.model = self._load_shape_component(cfg, ckpt, "model", load_device)

        # FlashVDM: aceleración del Volume Decoding (solo afecta al VAE)
        try:
            pipe.enable_flashvdm(enabled=True)
            print(" -> [FlashVDM] FlashVDM activado.")
        except Exception as e_fvdm:
            print(f" -> [FlashVDM] No disponible ({e_fvdm}). Continuando con VAE estándar.")

        pipe._studio_cfg = cfg
        pipe._studio_ckpt = ckpt
        pipe._studio_dtype = self.dtype
        pipe._studio_subfolder = self.shape_subfolder
        pipe._studio_orig_encode_cond = pipe.encode_cond
        pipe._studio_orig_export = pipe._export
        return pipe

    def _choose_shape_mode(self, cached) -> str:
        if self.shape_mode_pref in ("full", "stream", "staged"):
            return self.shape_mode_pref
        if "cuda" not in str(self.device):
            return "full"
        if _vram_free_gb() >= MODEL_MEMORY_GB["shape"]["vram_full"]:
            return "full"
        return "staged" if cached is not None else "stream"

    def _get_shape_pipeline(self):
        """Devuelve el pipeline de forma listo, con el modo de carga que menos RAM necesita."""
        mem = MODEL_MEMORY_GB["shape"]
        cached = activate_model("shape", vram_needed_gb=mem["vram"], ram_needed_gb=mem["ram_load"])
        if cached is not None and (getattr(cached, "_studio_dtype", None) != self.dtype
                                   or getattr(cached, "_studio_subfolder", None) != self.shape_subfolder):
            release_model("shape")
            cached = None

        mode = self._choose_shape_mode(cached)
        if mode == "staged" and cached is None and _system_ram_free_gb() < mem["ram_staged"] + RAM_SAFETY_MARGIN_GB:
            print(" -> [Memoria] No hay RAM para el modo 'staged': se usa la carga por partes ('stream').")
            mode = "stream"
        print(f" -> [Carga] Modo de carga de la forma: '{mode}'")

        if cached is not None:
            shape_pipe = cached
            print(" -> Pipeline de forma reutilizado desde RAM (sin leer el disco).")
        else:
            t_load = time.time()
            shape_pipe = self._build_shape_pipeline(mode)
            register_model("shape", shape_pipe)
            print(f" -> Pipeline de forma preparado en {time.time() - t_load:.1f}s")
        shape_pipe._studio_mode = mode

        device = torch.device(self.device)
        cfg, ckpt = shape_pipe._studio_cfg, shape_pipe._studio_ckpt

        # NOTA: no se usa enable_model_cpu_offload(): en hy3dgen 2.0.2 crea los latentes en
        # pipe.device (CPU) mientras el DiT corre en CUDA y la inferencia falla con
        # "Cannot generate a cpu tensor from a generator of type cuda".
        def staged_encode_cond(*args, **kwargs):
            # Varias semillas de la misma imagen: la condición se calcula una sola vez
            if getattr(shape_pipe, "_studio_cond_cache", None) is not None:
                return shape_pipe._studio_cond_cache
            if mode == "stream":
                shape_pipe.conditioner = self._load_shape_component(cfg, ckpt, "conditioner", device)
            else:
                shape_pipe.conditioner.to(device)
            result = shape_pipe._studio_orig_encode_cond(*args, **kwargs)
            if mode == "stream":
                shape_pipe.conditioner = self._meta_shape_component(cfg, "conditioner")
                flush_vram()
                shape_pipe.model = self._load_shape_component(cfg, ckpt, "model", device)
            elif mode == "staged":
                shape_pipe.conditioner.to("cpu")
                flush_vram()
                shape_pipe.model.to(device)
            if getattr(shape_pipe, "_studio_cache_cond", False):
                shape_pipe._studio_cond_cache = result
            return result

        def staged_export(*args, **kwargs):
            # Semillas candidatas (output_type="latent"): sin decodificar; el DiT sigue en VRAM
            if (args[1] if len(args) > 1 else kwargs.get("output_type")) == "latent":
                return shape_pipe._studio_orig_export(*args, **kwargs)
            # Volume Decoding con la VRAM libre de DiT y condicionador
            if mode == "stream":
                shape_pipe.model = self._meta_shape_component(cfg, "model")
                flush_vram()
            elif mode == "staged":
                shape_pipe.model.to("cpu")
                shape_pipe.conditioner.to("cpu")
                flush_vram()
            shape_pipe.vae.to(device)
            print_real_vram_status("Inicio Volume Decoding")
            octree = args[5] if len(args) > 5 else None
            try:
                return shape_pipe._studio_orig_export(*args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                if not (octree and octree > 256):
                    raise
            # Reintento FUERA del except: dentro, la traza del error retiene los tensores del intento
            # fallido y la VRAM no se libera. Los latentes ya están calculados: basta con menos detalle.
            print(f" -> [Memoria] Sin VRAM para decodificar con octree {octree}: reintentando con 256...")
            flush_vram()
            args = list(args)
            args[5] = 256
            return shape_pipe._studio_orig_export(*args, **kwargs)

        shape_pipe.encode_cond = staged_encode_cond
        shape_pipe._export = staged_export

        # pipe.device decide dónde se crean latentes y timesteps: siempre el dispositivo real
        if mode == "full":
            shape_pipe.to(device)
        elif mode == "staged":
            shape_pipe.to("cpu")
        shape_pipe.device = device
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
        texture_safe_cap: Optional[int] = None,
        taubin_iterations: int = 5,
        allow_high_octree: bool = False,
        guidance_scale: Optional[float] = None,
        seed_candidates: int = 1,
        keep_hd: bool = False
    ) -> trimesh.Trimesh:
        """
        Etapa 1 - Generación de Geometría (Hunyuan3D-DiT):
        1. Saca de la VRAM cualquier otro modelo (SF3D / Paint) y prepara el pipeline de forma.
        2. Preprocesa el canal alfa con inpainting de bordes (evita halos oscuros).
        3. Inferencia DiT + Volume Decoding con offload por etapas según el perfil de VRAM.
           Con seed_candidates > 1 prueba varias semillas y se queda la que más se parece a la silueta.
           guidance_scale solo afecta al modelo completo (Turbo y Fast vienen destilados de guía).
        4. Limpia islas flotantes y reduce polígonos:
             - target_faces > 0: reduce a ese número de caras.
             - target_faces <= 0: conserva la malla completa (HD).
             - texture_safe_cap: tope adicional cuando después se va a texturizar.
           Con keep_hd la malla sin reducir queda en self.last_hd_mesh (para el normal map).
        5. Suavizado Taubin y aparca el pipeline en RAM (VRAM libre para la Etapa 2).
        """
        # "Profundidad mejorada" / "Máxima calidad" permiten 384 en cualquier perfil: si la
        # decodificación no cabe, staged_export reintenta con 256 (los latentes ya están calculados)
        octree_cap = max(self.max_octree, 384) if allow_high_octree else self.max_octree
        safe_octree = min(int(octree_resolution or 256), octree_cap)
        candidates = max(1, int(seed_candidates or 1))
        faces_txt = f"{target_faces:,}" if target_faces and target_faces > 0 else "HD (sin reducir)"
        print(f"\n[Etapa 1/2] Generando geometría 3D pura con Hunyuan3D-DiT ({self.shape_subfolder})...")
        print(f" -> Pasos DiT: {steps} | Octree: {safe_octree} | Semilla: {seed}"
              + (f" (+{candidates - 1} candidatas)" if candidates > 1 else "")
              + (f" | Guía: {guidance_scale}" if guidance_scale is not None else "")
              + f" | Objetivo caras: {faces_txt}")
        start_t = time.time()
        shape_pipe = None
        discard_shape = False
        self.last_hd_mesh = None
        self.last_seed_report = None

        try:
            # Preprocesamiento con Inpainting de bordes en canal alfa (cv2.inpaint)
            prep_image = inpaint_edge_padding(image, dilation_pixels=10, alpha_threshold=25)

            shape_pipe = self._get_shape_pipeline()

            generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None

            # Callback para pasos individuales de difusión DiT (con varias semillas, la barra las recorre todas)
            run = {"index": 0, "label": ""}

            def dit_step_callback(step_idx, t, outputs):
                if progress_callback:
                    cur_step = step_idx + 1
                    pct = 30 + int(((run["index"] + cur_step / max(steps, 1)) / candidates) * 35) # 30% a 65%
                    vram_mb = torch.cuda.memory_allocated(0) / (1024 ** 2) if torch.cuda.is_available() else 0
                    progress_callback(f"[3/4] Inferencia DiT{run['label']} en GPU ({vram_mb:.0f} MB): Paso {cur_step}/{steps} ({pct}%)...", pct)

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

            call_kwargs = dict(
                image=prep_image,
                num_inference_steps=steps,
                octree_resolution=safe_octree,
                num_chunks=safe_chunks,
                callback=dit_step_callback,
                callback_steps=1
            )
            if guidance_scale is not None:
                call_kwargs["guidance_scale"] = float(guidance_scale)

            try:
                with torch.inference_mode():
                    if candidates > 1:
                        outputs = self._best_seed_shape(shape_pipe, call_kwargs, seed, candidates, run, image, progress_callback)
                    else:
                        outputs = shape_pipe(generator=generator, output_type="trimesh", **call_kwargs)
            finally:
                tqdm.tqdm.update = orig_tqdm_update

            raw_mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            try:
                raw_mesh.fix_normals()
            except Exception as e_orient:
                print(f" -> [Aviso Normales] {e_orient}")

            # Depuración de geometría: eliminar islas parásitas y losas flotantes
            cleaned_mesh = clean_mesh_floating_artifacts(raw_mesh)
            if keep_hd:
                # Sin reducir ni suavizar: es la fuente del detalle del normal map
                self.last_hd_mesh = cleaned_mesh.copy()

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
            smoothed_mesh = apply_taubin_smoothing(decimated_mesh, lamb=0.53, nu=-0.55, iterations=taubin_iterations)

            print(f" -> [Etapa 1/2 Completada] Geometría: {len(smoothed_mesh.vertices):,} vértices, {len(smoothed_mesh.faces):,} caras en {time.time() - start_t:.2f}s")
            return smoothed_mesh

        except ImportError as e:
            print(f"[ERROR CRÍTICO] La librería hy3dgen no está instalada: {e}")
            print("Ejecuta 'setup.bat' o 'pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git'")
            raise
        except torch.cuda.OutOfMemoryError:
            # Un pipeline a medias tras un OOM no es fiable: se destruye en el finally
            discard_shape = True
            print("[ERROR] VRAM insuficiente para la geometría. Prueba un perfil de VRAM menor o un octree más bajo.")
            raise
        except Exception as e:
            err_str = str(e)
            if "SingleImageEncoder" in err_str or "state_dict" in err_str:
                discard_shape = True
                print(f"\n[DIAGNÓSTICO] {err_str}")
                print(" -> Causa: Los pesos de Hunyuan3D-2 Turbo están incompletos en disco o corruptos.")
                print(" -> Solución: Ejecuta 'descargar_modelos.bat' para completar el archivo model.fp16.safetensors (4.93 GB).")
                print(" -> O cambia en la interfaz a 'Stable Fast 3D (SF3D)' para generar en 1s sin este modelo.\n")
            print(f"[ERROR] Falló la generación de forma: {e}")
            raise
        finally:
            # Soltar la referencia local: si no, liberar el modelo obligaría a copiarlo a la RAM
            shape_pipe = None
            # Aparcar en RAM (solo si sobra) o liberar: la VRAM queda libre para la Etapa 2
            if discard_shape:
                release_model("shape")
            else:
                park_model("shape")
            print_real_vram_status("Post-Etapa 1 (forma aparcada)")

    def _best_seed_shape(
        self,
        shape_pipe,
        call_kwargs: dict,
        seed: int,
        count: int,
        run: dict,
        image: Image.Image,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ):
        """
        Prueba `count` semillas de la misma imagen y devuelve la malla cuya vista frontal se parece
        más a la silueta de la imagen (IoU). No se elige por fondo: eso inflaría los objetos planos.
          1. DiT por semilla: la condición se calcula una vez y el DiT sigue en VRAM entre semillas.
          2. Cada candidata se decodifica a octree DEPTH_SCORE_OCTREE (rápido) solo para compararla.
          3. Solo la ganadora se decodifica con el octree pedido.
        """
        mask = np.asarray(image.convert("RGBA"))[..., 3] > 127
        base = seed if seed >= 0 else int(time.time()) % 100000
        seeds = [base + i * 1000 for i in range(count)]
        latents = []
        shape_pipe._studio_cache_cond = True
        try:
            for i, s in enumerate(seeds):
                run["index"], run["label"] = i, f" · semilla {i + 1}/{count}"
                generator = torch.Generator(device=self.device).manual_seed(s)
                latents.append(shape_pipe(generator=generator, output_type="latent", **call_kwargs))
        finally:
            shape_pipe._studio_cache_cond = False
            shape_pipe._studio_cond_cache = None

        num_chunks = call_kwargs["num_chunks"]
        scores, depths = [], []
        for i, candidate in enumerate(latents):
            if progress_callback:
                progress_callback(f"[3/4] Comparando cada semilla con la imagen ({i + 1}/{count})...", 66)
            preview = shape_pipe._export(candidate, "trimesh", 1.01, 0.0, num_chunks, DEPTH_SCORE_OCTREE, None, False)
            preview = preview[0] if isinstance(preview, (list, tuple)) else preview
            if preview is None or len(preview.faces) == 0:
                scores.append(0.0)
                depths.append(0.0)
                continue
            preview = clean_mesh_floating_artifacts(preview)
            scores.append(silhouette_iou(preview, mask))
            depths.append(measure_depth_ratio(preview))
            print(f" -> [Semillas] Semilla {seeds[i]}: silueta {scores[-1]:.1%} · fondo {depths[-1]:.0%}")

        best = int(np.argmax(scores))
        self.last_seed_report = {"seeds": seeds, "scores": scores, "depths": depths, "best": best}
        print(f" -> [Semillas] Elegida la semilla {seeds[best]} (silueta {scores[best]:.1%} · fondo {depths[best]:.0%})")
        return shape_pipe._export(latents[best], "trimesh", 1.01, 0.0, num_chunks, call_kwargs["octree_resolution"], None)

    def _get_paint_pipeline(self):
        """
        Prepara Hunyuan3D-Paint con la forma ya fuera de la VRAM. Los submodelos (delight y
        multivista) se cargan directo a la GPU solo cuando se usan:
          stream: uno a la vez; cada uno se libera al terminar. El multivista elige su
                  ubicación (gpu / lean / bloques) según la VRAM libre: cabe en 6 GB.
          full  : se quedan cargados durante la pintura (si sobra VRAM, ~9 GB).
        """
        from preparar_modelos import paint_opt_ready
        mem = MODEL_MEMORY_GB["paint"]
        # Con la caché fp16 la carga es directa y la RAM solo hace de paso (~1.5 GB transitorios)
        ram_need = mem["ram_load_opt"] if paint_opt_ready() else mem["ram_load"]
        paint_pipeline = activate_model("paint", vram_needed_gb=mem["vram"], ram_needed_gb=ram_need, strict_vram=True)
        if paint_pipeline is None:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            from hy3dgen.texgen.utils.dehighlight_utils import Light_Shadow_Remover

            from preparar_modelos import paint_opt_ready, PAINT_OPT_DIR
            local_model_dir = HUNYUAN_LOCAL_DIR
            if paint_opt_ready():
                paint_source = str(PAINT_OPT_DIR)  # Caché fp16 safetensors: carga más rápida
            elif (local_model_dir / "hunyuan3d-delight-v2-0").exists() and (local_model_dir / self.paint_subfolder).exists():
                paint_source = str(local_model_dir)
            else:
                paint_source = self.model_id
                print(" -> [Hunyuan3D-Paint] Consejo: crea la caché optimizada (una vez) para cargar más rápido:"
                      " python preparar_modelos.py optimize paint")

            # from_pretrained resuelve/descarga las rutas; se evita que cargue los dos submodelos
            # a la vez (load_models) y se sustituyen por cargas perezosas
            original_load_models = Hunyuan3DPaintPipeline.load_models
            Hunyuan3DPaintPipeline.load_models = lambda pipe_self: None
            try:
                paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(paint_source)
            finally:
                Hunyuan3DPaintPipeline.load_models = original_load_models

            config = paint_pipeline.config
            config.device = "cuda" if "cuda" in str(self.device) else "cpu"

            paint_pipeline.models = {
                "delight_model": _LazyPaintModel(
                    "delight", lambda tier: Light_Shadow_Remover(config), config.device,
                    tiers=("gpu",), vram_gb={"gpu": 3.2}
                ),
                "multiview_model": _LazyPaintModel(
                    "multiview", lambda tier: _build_multiview_net(config, tier), config.device,
                    tiers=PAINT_MV_TIERS, vram_gb=PAINT_MV_VRAM_GB
                ),
            }
            register_model("paint", paint_pipeline)
            print(f" -> Hunyuan3D-Paint preparado ({paint_source}); los submodelos se cargan al usarse.")
        else:
            print(" -> Hunyuan3D-Paint reutilizado.")

        mode = "full" if _vram_free_gb() >= mem["vram_full"] else "stream"
        for sub in paint_pipeline.models.values():
            sub.keep = mode == "full"
        print(f" -> [Carga] Modo de carga de la pintura: '{mode}'")

        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        # Resolución de render/textura según el perfil de VRAM
        res = self.paint_resolution
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
        image: Image.Image,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> trimesh.Trimesh:
        """
        Etapa 2 - Texturizado Nativo Multivista (Hunyuan3D-Paint):
        1. Saca de la VRAM el pipeline de forma y SF3D (nunca coexisten con Paint).
        2. Carga delight y multivista directo a la GPU, de uno en uno, con VAE slicing/tiling.
           Sin la extensión custom_rasterizer usa el rasterizador PyTorch (torch_rasterizer.py).
        3. Render/textura a la resolución del perfil de VRAM (1024 en 6-12 GB, 2048 en 16 GB+).
        4. Si Paint no está disponible o se queda sin VRAM, recurre al horneado UV de contingencia.
        5. Aparca Paint en RAM al terminar y aplica acabado PBR mate.
        """
        print("\n[Etapa 2/2] Texturizado Nativo Multivista con Hunyuan3D-Paint...")
        start_t = time.time()

        textured_mesh = None
        self.last_paint_fallback = None
        self.last_paint_summary = None
        if not hunyuan_paint_available():
            self.last_paint_fallback = f"Hunyuan3D-Paint no disponible ({_PAINT_UNAVAILABLE_REASON})"
            print(f" -> [Hunyuan3D-Paint] No disponible: {_PAINT_UNAVAILABLE_REASON}")
            print(" -> Usando horneado UV de contingencia (proyección frontal/trasera en CPU).")
            textured_mesh = self._fallback_texture_bake(mesh, image)
        else:
            textured_mesh = self._run_paint_pipeline(mesh, image, start_t, progress_callback)

        # Ajuste de material PBR mate/limpio (metallicFactor=0.0, roughnessFactor=0.90)
        if textured_mesh is not None and hasattr(textured_mesh, "visual") and hasattr(textured_mesh.visual, "material") and textured_mesh.visual.material is not None:
            mat = textured_mesh.visual.material
            mat.metallicFactor = 0.0
            mat.roughnessFactor = 0.90
            mat.doubleSided = True

        return textured_mesh if textured_mesh is not None else mesh

    def _run_paint_pipeline(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        start_t: float,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Optional[trimesh.Trimesh]:
        """Ejecuta Hunyuan3D-Paint por etapas medidas; ante OOM o error recurre al horneado de contingencia."""
        import hy3dgen.texgen.pipelines as texgen_pipelines
        textured_mesh = None
        paint_pipeline = None
        out_of_memory = False
        tracker = _PaintProgress(progress_callback)
        original_uv_wrap = texgen_pipelines.mesh_uv_wrap
        try:
            tracker.report("Preparando Hunyuan3D-Paint...", 88)
            paint_pipeline = self._get_paint_pipeline()
            print_real_vram_status("Pre-Inferencia Hunyuan3D-Paint")
            print(f" -> Pintando malla de {len(mesh.faces):,} caras con Hunyuan3D-Paint...")

            # Cronómetro y avance en vivo por etapa (se restauran en el finally)
            delight, multiview = paint_pipeline.models["delight_model"], paint_pipeline.models["multiview_model"]
            delight.progress = tracker.submodel("Delight (quitando luces/sombras)", 88, 90, steps=50)
            multiview.progress = tracker.submodel("Multivista (pintando 6 vistas)", 91, 96, steps=30)
            texgen_pipelines.mesh_uv_wrap = tracker.stage("Desplegando UV (xatlas)", original_uv_wrap, 90)
            for attr, label, pct in (
                ("render_normal_multiview", "Mapas de normales", 91),
                ("render_position_multiview", "Mapas de posición", 91),
                ("bake_from_multiview", "Horneando textura", 96),
                ("texture_inpaint", "Rellenando huecos de la textura", 97),
            ):
                setattr(paint_pipeline, attr, tracker.stage(label, getattr(paint_pipeline, attr), pct))

            with torch.inference_mode():
                textured_mesh = paint_pipeline(mesh, image=image)
            print(f" -> [Hunyuan3D-Paint] Texturizado multivista completado en {time.time() - start_t:.2f}s.")
            self.last_paint_summary = tracker.summary(time.time() - start_t)
            print(f" -> [Paint] Tiempos: {self.last_paint_summary}")

        except InsufficientMemoryError as e_mem:
            print(f" -> [Hunyuan3D-Paint] No se carga para no saturar el equipo: {e_mem}")
            self.last_paint_fallback = str(e_mem)
            textured_mesh = None
        except torch.cuda.OutOfMemoryError as e_oom:
            print(f" -> [Aviso CUDA OOM en Hunyuan3D-Paint] {e_oom}. Recurriendo a horneado UV de contingencia...")
            self.last_paint_fallback = "VRAM insuficiente incluso transmitiendo el modelo por bloques"
            out_of_memory = True
            textured_mesh = None
        except Exception as e_tex:
            import traceback
            traceback.print_exc()
            print(f" -> [Aviso Hunyuan3D-Paint] {e_tex}. Recurriendo a horneado UV de contingencia...")
            self.last_paint_fallback = f"error en Hunyuan3D-Paint: {type(e_tex).__name__}: {e_tex}"
            textured_mesh = None
        finally:
            texgen_pipelines.mesh_uv_wrap = original_uv_wrap
            if paint_pipeline is not None:
                for attr in ("render_normal_multiview", "render_position_multiview", "bake_from_multiview", "texture_inpaint"):
                    paint_pipeline.__dict__.pop(attr, None)
                for sub in paint_pipeline.models.values():
                    sub.progress = None
            paint_pipeline = None  # Soltar la referencia local antes de liberar/aparcar
            if out_of_memory:
                release_model("paint")  # Tras un OOM el pipeline no es fiable
            else:
                park_model("paint")
            print_real_vram_status("Post-Hunyuan Paint")
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

_sf3d_overlap_kernel = None


def _get_sf3d_overlap_kernel():
    """
    Compila (una vez) el detector de solapes UV del unwrapper de SF3D en Numba.
    Réplica de UVUnwrapper::perform_intersection_check (uv_unwrapper/csrc/unwrapper.cpp):
    dos triángulos de la misma proyección se solapan si su intersección tiene área >= 1e-10;
    el que queda detrás según el centroide 3D en el eje de la proyección se marca para moverlo.
    """
    global _sf3d_overlap_kernel
    if _sf3d_overlap_kernel is not None:
        return _sf3d_overlap_kernel
    from numba import njit, prange

    @njit(cache=False, fastmath=False)
    def _overlap_area(t1, t2):
        # Recorte Sutherland-Hodgman de t2 contra t1 (ambos en sentido antihorario)
        bx = np.empty(12); by = np.empty(12)
        nx = np.empty(12); ny = np.empty(12)
        n = 3
        for k in range(3):
            bx[k] = t2[k, 0]; by[k] = t2[k, 1]
        for e in range(3):
            ax0, ay0 = t1[(e + 2) % 3, 0], t1[(e + 2) % 3, 1]
            ax1, ay1 = t1[e, 0], t1[e, 1]
            m = 0
            for j in range(n):
                jp = (j - 1 + n) % n
                pcx, pcy = bx[j], by[j]
                ppx, ppy = bx[jp], by[jp]
                oc = ax0 * (ay1 - pcy) + ax1 * (pcy - ay0) + pcx * (ay0 - ay1)
                op = ax0 * (ay1 - ppy) + ax1 * (ppy - ay0) + ppx * (ay0 - ay1)
                in_c = oc >= -1e-7
                in_p = op >= -1e-7
                if in_c != in_p:
                    t = op / (op - oc)
                    nx[m] = ppx + (pcx - ppx) * t
                    ny[m] = ppy + (pcy - ppy) * t
                    m += 1
                if in_c:
                    nx[m] = pcx; ny[m] = pcy
                    m += 1
            n = m
            if n < 3:
                return 0.0
            for j in range(n):
                bx[j] = nx[j]; by[j] = ny[j]
        area = 0.0
        for j in range(n):
            jn = (j + 1) % n
            area += bx[j] * by[jn] - bx[jn] * by[j]
        return abs(area) * 0.5

    @njit(parallel=True, cache=False)
    def _find_occluded(tri_uv, centroids, slot, active, grid):
        nf = tri_uv.shape[0]
        # Triángulos en sentido antihorario y caja envolvente
        tri = tri_uv.copy()
        bmin = np.empty((nf, 2)); bmax = np.empty((nf, 2))
        valid = np.zeros(nf, dtype=np.bool_)
        for i in range(nf):
            det = (tri[i, 0, 0] * (tri[i, 1, 1] - tri[i, 2, 1]) + tri[i, 1, 0] * (tri[i, 2, 1] - tri[i, 0, 1])
                   + tri[i, 2, 0] * (tri[i, 0, 1] - tri[i, 1, 1]))
            if det < 0:
                for c in range(2):
                    tmp = tri[i, 1, c]; tri[i, 1, c] = tri[i, 2, c]; tri[i, 2, c] = tmp
            valid[i] = active[i] and abs(det) >= 1e-7  # degenerados se ignoran (isTriDegenerated)
            for c in range(2):
                bmin[i, c] = min(tri[i, 0, c], tri[i, 1, c], tri[i, 2, c])
                bmax[i, c] = max(tri[i, 0, c], tri[i, 1, c], tri[i, 2, c])

        # Rejilla uniforme por ranura: celda = slot * grid^2 + cy * grid + cx (formato CSR)
        n_cells = 13 * grid * grid
        counts = np.zeros(n_cells + 1, dtype=np.int64)
        for i in range(nf):
            if not valid[i]:
                continue
            x0 = min(grid - 1, max(0, int(bmin[i, 0] * grid))); x1 = min(grid - 1, max(0, int(bmax[i, 0] * grid)))
            y0 = min(grid - 1, max(0, int(bmin[i, 1] * grid))); y1 = min(grid - 1, max(0, int(bmax[i, 1] * grid)))
            for cy in range(y0, y1 + 1):
                for cx in range(x0, x1 + 1):
                    counts[slot[i] * grid * grid + cy * grid + cx + 1] += 1
        for k in range(n_cells):
            counts[k + 1] += counts[k]
        fill = counts[:-1].copy()
        items = np.empty(counts[-1], dtype=np.int64)
        for i in range(nf):
            if not valid[i]:
                continue
            x0 = min(grid - 1, max(0, int(bmin[i, 0] * grid))); x1 = min(grid - 1, max(0, int(bmax[i, 0] * grid)))
            y0 = min(grid - 1, max(0, int(bmin[i, 1] * grid))); y1 = min(grid - 1, max(0, int(bmax[i, 1] * grid)))
            for cy in range(y0, y1 + 1):
                for cx in range(x0, x1 + 1):
                    cell = slot[i] * grid * grid + cy * grid + cx
                    items[fill[cell]] = i
                    fill[cell] += 1

        occluded = np.zeros(nf, dtype=np.bool_)
        for i in prange(nf):
            if not valid[i]:
                continue
            s = slot[i]
            x0 = min(grid - 1, max(0, int(bmin[i, 0] * grid))); x1 = min(grid - 1, max(0, int(bmax[i, 0] * grid)))
            y0 = min(grid - 1, max(0, int(bmin[i, 1] * grid))); y1 = min(grid - 1, max(0, int(bmax[i, 1] * grid)))
            axis = 0 if (s % 6) < 2 else (1 if (s % 6) < 4 else 2)
            use_max = (s % 2) == 1
            for cy in range(y0, y1 + 1):
                for cx in range(x0, x1 + 1):
                    cell = s * grid * grid + cy * grid + cx
                    for p in range(counts[cell], counts[cell + 1]):
                        j = items[p]
                        if j <= i:
                            continue
                        if bmax[i, 0] < bmin[j, 0] or bmax[j, 0] < bmin[i, 0] or bmax[i, 1] < bmin[j, 1] or bmax[j, 1] < bmin[i, 1]:
                            continue
                        # Cada par se evalúa solo en la celda de la esquina mínima de la intersección de cajas
                        rx = min(grid - 1, max(0, int(max(bmin[i, 0], bmin[j, 0]) * grid)))
                        ry = min(grid - 1, max(0, int(max(bmin[i, 1], bmin[j, 1]) * grid)))
                        if rx != cx or ry != cy:
                            continue
                        if _overlap_area(tri[i], tri[j]) < 1e-10:
                            continue
                        pa = centroids[i, axis]; pb = centroids[j, axis]
                        # Igual que el C++: el "segundo" del par (tapado) es el que se mueve
                        if (use_max and pa < pb) or ((not use_max) and pa > pb):
                            occluded[i] = True
                        else:
                            occluded[j] = True
        return occluded

    _sf3d_overlap_kernel = _find_occluded
    return _sf3d_overlap_kernel


def _assign_faces_uv_to_atlas_index_numba(vertex_positions, triangle_idxs, face_uv, face_index):
    """
    Sustituto de torch.ops.UVUnwrapper.assign_faces_uv_to_atlas_index (extensión C++ de SF3D).
    Sin este paso, los triángulos que se tapan entre sí en una misma proyección de la caja
    (p. ej. un brazo delante del torso) comparten los mismos texeles y la textura se mezcla.
    Dos pasadas como el original: solapes de las ranuras 0-5 pasan a 6-11 y los de 6-11 a 12.
    """
    kernel = _get_sf3d_overlap_kernel()
    v = vertex_positions.detach().cpu().double().numpy()
    f = triangle_idxs.detach().cpu().long().numpy()
    tri_uv = face_uv.detach().cpu().double().reshape(-1, 3, 2).numpy()
    assign = face_index.detach().cpu().long().numpy().copy()
    centroids = v[f].mean(axis=1)
    grid = int(np.clip(np.sqrt(len(f) / 6.0) / 2.0, 8, 512))
    for offset in (0, 6):
        active = (assign >= offset) & (assign < offset + 6)
        if not active.any():
            continue
        occluded = kernel(tri_uv, centroids, assign, active, grid)
        assign[occluded] = np.minimum(assign[occluded] + 6, 12)
    return torch.from_numpy(assign).to(vertex_positions.device)


# Hasta este número de caras SF3D despliega las UV con xatlas (malla de juego); por encima, proyección en caja
SF3D_XATLAS_MAX_FACES = 60000
SF3D_XATLAS_RESOLUTION = 2048
SF3D_XATLAS_PADDING_PX = 4


def _xatlas_unwrap(vertex_positions, triangle_idxs):
    """Atlas UV con xatlas en el formato del Unwrapper de SF3D: (uv únicas, índice uv por cara)."""
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(
        vertex_positions.detach().cpu().numpy().astype(np.float32),
        triangle_idxs.detach().cpu().numpy().astype(np.uint32),
    )
    pack = xatlas.PackOptions()
    pack.resolution = SF3D_XATLAS_RESOLUTION
    pack.padding = SF3D_XATLAS_PADDING_PX
    pack.bilinear = True
    atlas.generate(xatlas.ChartOptions(), pack)
    _, indices, uvs = atlas.get_mesh(0)  # xatlas conserva el orden de las caras
    return (
        torch.from_numpy(uvs.astype(np.float32)).to(vertex_positions.device),
        torch.from_numpy(indices.astype(np.int64)).to(triangle_idxs.device),
    )


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
                            try:
                                return _assign_faces_uv_to_atlas_index_numba(
                                    vertex_positions, triangle_idxs, face_uv, face_index
                                )
                            except Exception as e_assign:
                                print(f" -> [Aviso SF3D UV] Detección de solapes UV no disponible ({e_assign}); "
                                      "las piezas que se tapan compartirán textura.")
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

        class AtlasFirstUnwrapper(unwrapper_cls):
            """
            Mallas de juego (remalladas): atlas xatlas, que aprovecha ~65-75% de la textura frente al
            ~25% de la proyección en caja de SF3D (2.5x más texeles por triángulo y menos costuras).
            Mallas HD enormes: proyección en caja original (xatlas tardaría minutos).
            """
            def forward(self, vertex_positions, vertex_normals, triangle_idxs, island_padding=0.02):
                if triangle_idxs.shape[0] <= SF3D_XATLAS_MAX_FACES:
                    try:
                        return _xatlas_unwrap(vertex_positions, triangle_idxs)
                    except Exception as e_xa:
                        print(f" -> [Aviso SF3D UV] xatlas falló ({e_xa}); se usa la proyección en caja.")
                return super().forward(vertex_positions, vertex_normals, triangle_idxs, island_padding)

        mod_uv = types.ModuleType("uv_unwrapper")
        mod_uv.Unwrapper = AtlasFirstUnwrapper
        sys.modules["uv_unwrapper"] = mod_uv

    # 2. texture_baker
    cur_tb = sys.modules.get("texture_baker")
    if cur_tb is None or not hasattr(cur_tb, "TextureBaker"):
        # Rasterizador de textura UV ultrarrápido con Numba (CPU multihilo, 0 MB VRAM)
        _numba_rast_fn = None
        try:
            from numba import njit

            # Secuencial a propósito: en paralelo, dos triángulos que tocan el mismo píxel en una
            # costura podían escribir a la vez y dejar baricéntricas de uno con el índice del otro.
            @njit(fastmath=True)
            def _numba_rasterize_sf3d(uvs, faces, res):
                rast = np.zeros((res, res, 4), dtype=np.float32)
                rast[:, :, 3] = -1.0
                num_faces = faces.shape[0]
                for f in range(num_faces):
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


def _sf3d_load_lowmem(sf3d_cls, source: str, device: str):
    """
    Carga SF3D con la mínima RAM. SF3D.from_pretrained crea el modelo en fp32 (3.8 GB) y además
    lee el archivo entero a la RAM (otros 3.8 GB). Aquí el modelo se sube a la GPU recién creado
    y los pesos se copian tensor a tensor desde el archivo: pico de RAM ~4 GB en vez de ~8 GB.
    """
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load_model
    from huggingface_hub import hf_hub_download

    if os.path.isdir(source):
        config_path = os.path.join(source, "config.yaml")
        weight_path = os.path.join(source, "model.safetensors")
    else:
        config_path = hf_hub_download(repo_id=source, filename="config.yaml")
        weight_path = hf_hub_download(repo_id=source, filename="model.safetensors")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    model = sf3d_cls(cfg)
    model.to(device)
    gc.collect()

    targets = model.state_dict()
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        # Las claves del modelo que no están en el archivo deben ser alias (pesos compartidos)
        # de tensores que sí están: safetensors guarda cada tensor compartido una sola vez.
        loaded_ptrs = {targets[k].data_ptr() for k in keys if k in targets}
        aliases_only = keys <= set(targets) and all(
            targets[k].data_ptr() in loaded_ptrs for k in set(targets) - keys
        )
        if aliases_only:
            with torch.no_grad():
                for key in keys:
                    targets[key].copy_(f.get_tensor(key))
            del targets
            return _sf3d_prepare_inference(model.eval(), device)

    # Claves distintas: carga estándar de safetensors. Se sueltan antes las referencias a los
    # tensores de la GPU para que model.to("cpu") libere de verdad la VRAM.
    print(" -> [SF3D] Claves no coinciden; usando la carga estándar de safetensors.")
    del targets, loaded_ptrs
    model.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    load_model(model, weight_path)
    return _sf3d_prepare_inference(model.to(device).eval(), device)


def _sf3d_dilate_fill_lowmem(img, mask, iterations=10):
    """
    Igual que sf3d.models.utils.dilate_fill (relleno de los bordes de las islas UV con la media
    de los vecinos válidos), resultado idéntico, pero con convoluciones en vez de unfold/fold:
    unfold crea tensores 9x la textura (~450 MB cada uno a 2048) y agotaba la VRAM al hornear.
    """
    import torch.nn.functional as F
    c = img.shape[1]
    ones_c = torch.ones((c, 1, 3, 3), dtype=img.dtype, device=img.device)
    ones_1 = torch.ones((1, 1, 3, 3), dtype=img.dtype, device=img.device)
    old_mask = mask.to(img.dtype)
    old_img = img
    for _ in range(iterations):
        new_mask = F.max_pool2d(old_mask, 3, 1, 1)
        # Media de los válidos en cada ventana 3x3 y reparto a los píxeles nuevos que cubre
        win_mean = F.conv2d(old_img, ones_c, groups=c) / F.conv2d(old_mask, ones_1).clip(1)
        spread = F.conv_transpose2d(win_mean, ones_c, groups=c)
        new_img = spread * new_mask / F.conv2d(new_mask, ones_1, padding=1).clamp(1)
        old_img = torch.lerp(old_img, new_img, new_mask - old_mask)
        old_mask = new_mask
    return old_img


# Puntos por lote al consultar el triplano y el decodificador de SF3D (~0.5 GB de pico por lote)
SF3D_POINT_CHUNK = 262144


def _sf3d_chunk_pointwise(model) -> None:
    """
    El horneado consulta el triplano y el decodificador con TODOS los texeles a la vez: a 2048x2048
    son millones de puntos y varios GB de activaciones (OOM en 8 GB). Ambos operan punto a punto,
    así que procesarlos por lotes da el mismo resultado con un pico de memoria acotado.
    """
    if getattr(model, "_chunked_pointwise", False):
        return
    query, decode = model.query_triplane, model.decoder.forward

    def chunked_query(positions, triplanes):
        n = positions.shape[-2]
        if n <= SF3D_POINT_CHUNK:
            return query(positions, triplanes)
        return torch.cat([query(positions[..., s:s + SF3D_POINT_CHUNK, :], triplanes)
                          for s in range(0, n, SF3D_POINT_CHUNK)], dim=-2)

    def chunked_decode(x, include=None, exclude=None):
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] <= SF3D_POINT_CHUNK:
            return decode(x, include=include, exclude=exclude)
        parts = [decode(flat[s:s + SF3D_POINT_CHUNK], include=include, exclude=exclude)
                 for s in range(0, flat.shape[0], SF3D_POINT_CHUNK)]
        return {k: torch.cat([p[k] for p in parts], dim=0).reshape(*x.shape[:-1], -1) for k in parts[0]}

    model.query_triplane = chunked_query
    model.decoder.forward = chunked_decode
    model._chunked_pointwise = True


def _sf3d_prepare_inference(model, device: str):
    """
    Reduce la VRAM de SF3D sin cambiar el resultado:
    - Sin gradientes: autocast no guarda copias bf16 de cada peso durante la inferencia (~1.5 GB).
    - Pesos Linear/Conv del codificador de imagen (DINOv2) y del backbone en bf16: autocast ya los
      ejecuta en bf16, así que la salida es la misma y se ahorran ~1.5 GB fijos.
    El decodificador y el meshing (que corren sin autocast) se quedan en fp32.
    """
    model.requires_grad_(False)
    _sf3d_chunk_pointwise(model)
    if "cuda" in str(device) and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        for part in (model.image_tokenizer, model.backbone):
            for mod in part.modules():
                if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d)):
                    mod.to(torch.bfloat16)
        gc.collect()
        torch.cuda.empty_cache()
    return model


# Margen sobre target_faces que se acepta del remallado de SF3D sin volver a decimar
SF3D_FACE_TOLERANCE = 1.3


def _mark_textures_lossless(mesh: Any) -> None:
    """Hace que trimesh exporte las texturas del material en PNG (sin pérdida) en vez de JPEG."""
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    for attr in ("baseColorTexture", "normalTexture", "metallicRoughnessTexture", "emissiveTexture", "occlusionTexture"):
        tex = getattr(mat, attr, None) if mat is not None else None
        if isinstance(tex, Image.Image):
            tex.format = "PNG"


# Acabado del material (metallicFactor, roughnessFactor). "mate" es el acabado cartoon de siempre
MATERIAL_PRESETS = {
    "mate": (0.0, 0.90),
    "satinado": (0.0, 0.50),
    "metal": (1.0, 0.30),
}

# Relieve de detalle sacado del albedo: pendiente que alcanza el 3% de texels con más detalle.
# Por encima de ~0.6 las zonas lisas de los personajes empiezan a verse granuladas.
TEXTURE_RELIEF_LEVELS = {
    "none": 0.0,
    "suave": 0.30,
    "marcado": 0.60,
}


def apply_material_preset(mesh: Any, preset: str = "mate") -> None:
    """Aplica el acabado elegido (mate, satinado o metal) al material PBR de la malla."""
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    if mat is None:
        return
    metallic, roughness = MATERIAL_PRESETS.get(preset, MATERIAL_PRESETS["mate"])
    mat.metallicFactor = metallic
    mat.roughnessFactor = roughness
    mat.doubleSided = True


def _rasterize_uv_faces(uv: np.ndarray, faces: np.ndarray, width: int, height: int) -> np.ndarray:
    """Máscara (bool, filas de la imagen) de los texels que cubren las caras en el atlas UV."""
    mask = np.zeros((height, width), np.uint8)
    if len(faces) == 0:
        return mask > 0
    px = np.empty((len(uv), 2), np.float64)
    px[:, 0] = uv[:, 0] * width
    px[:, 1] = (1.0 - uv[:, 1]) * height
    # 4 bits de subpíxel con el centro del texel en (0.5, 0.5)
    pts = np.round(px[faces] * 16.0 - 8.0).astype(np.int32)
    cv2.fillPoly(mask, list(pts), 255, lineType=cv2.LINE_8, shift=4)
    return mask > 0


def fix_sf3d_normal_convention(mesh: Any) -> bool:
    """
    Pasa el normal map de SF3D a la convención glTF/Godot (+X derecha, +Y arriba en la imagen).

    SF3D hornea con su propia base tangente: recorta el determinante UV a positivo (en las islas
    espejadas la tangente sale invertida) y al exportar invierte el sentido de las caras. En la
    malla exportada eso deja Y invertida en todas las islas y además X en las espejadas, y la luz
    entra al revés en el relieve. Comprobado en las costuras UV: tras corregir, la normal coincide
    a ambos lados (~0.9° de media frente a 2-4° sin corregir).
    """
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    tex = getattr(mat, "normalTexture", None) if mat is not None else None
    uv = getattr(mesh.visual, "uv", None) if mat is not None else None
    if not isinstance(tex, Image.Image) or uv is None or len(uv) != len(mesh.vertices):
        return False
    nmap = np.array(tex.convert("RGB"))
    h, w = nmap.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    d1 = uv[faces[:, 1]] - uv[faces[:, 0]]
    d2 = uv[faces[:, 2]] - uv[faces[:, 0]]
    mirrored_faces = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) < 0
    mirrored = _rasterize_uv_faces(uv, faces[mirrored_faces], w, h)
    regular = _rasterize_uv_faces(uv, faces[~mirrored_faces], w, h)
    # El relleno entre islas toma la orientación de la isla más cercana
    dist_mirrored = cv2.distanceTransform((~mirrored).astype(np.uint8), cv2.DIST_L2, 3)
    dist_regular = cv2.distanceTransform((~regular).astype(np.uint8), cv2.DIST_L2, 3)
    flip_x = dist_mirrored < dist_regular
    nmap[..., 1] = 255 - nmap[..., 1]
    nmap[..., 0] = np.where(flip_x, 255 - nmap[..., 0], nmap[..., 0])
    fixed = Image.fromarray(nmap, mode="RGB")
    fixed.format = "PNG"
    mat.normalTexture = fixed
    return True


def _extend_islands(img: np.ndarray, covered: np.ndarray) -> None:
    """
    Rellena todo el espacio entre islas UV con el texel cubierto más cercano. Si ese margen
    conserva valores antiguos, el filtrado bilineal y los mipmaps los mezclan en el borde de las
    islas y la costura se ve (más cuanto más lejos está el modelo de la cámara).
    """
    _, labels = cv2.distanceTransformWithLabels((~covered).astype(np.uint8), cv2.DIST_L2, 5,
                                                labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(covered)
    fill = ~covered
    nearest = labels[fill] - 1
    img[fill] = img[ys[nearest], xs[nearest]]


def add_texture_relief(mesh: Any, strength: float) -> bool:
    """
    Relieve de detalle: saca un mapa de alturas del albedo (paso banda de la luminancia: las
    líneas oscuras se hunden y los brillos suben) y lo suma al normal map con Reoriented Normal
    Mapping. El normal map de entrada debe estar ya en convención glTF.
    """
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    albedo = getattr(mat, "baseColorTexture", None) if mat is not None else None
    uv = getattr(mesh.visual, "uv", None) if mat is not None else None
    if strength <= 0 or not isinstance(albedo, Image.Image) or uv is None or len(uv) != len(mesh.vertices):
        return False
    rgb = np.asarray(albedo.convert("RGB"), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    base_img = getattr(mat, "normalTexture", None)
    if isinstance(base_img, Image.Image):
        if base_img.size != (w, h):
            base_img = base_img.resize((w, h), Image.LANCZOS)
        encoded = np.array(base_img.convert("RGB"))
    else:
        encoded = np.empty((h, w, 3), np.uint8)
        encoded[...] = (128, 128, 255)

    mask = _rasterize_uv_faces(np.asarray(uv, dtype=np.float64), np.asarray(mesh.faces), w, h)
    if not mask.any():
        return False
    m = mask.astype(np.float32)
    lum = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    scale = w / 2048.0

    def masked_blur(x, sigma):
        # Desenfoque normalizado: el relleno entre islas no entra en el relieve
        num = cv2.GaussianBlur(x * m, (0, 0), sigma)
        return num / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-4)

    height = masked_blur(lum, 1.0 * scale) - masked_blur(lum, 8.0 * scale)

    # Derivadas centrales sin cruzar el borde de la isla (de un solo lado si falta el vecino)
    hp = np.pad(height, 1, mode="edge")
    mp = np.pad(mask, 1, mode="constant")

    def deriv(a_plus, a_minus, m_plus, m_minus):
        return np.where(m_plus & m_minus, (a_plus - a_minus) * 0.5,
                        np.where(m_plus, a_plus - height, np.where(m_minus, height - a_minus, 0.0)))

    gx = deriv(hp[1:-1, 2:], hp[1:-1, :-2], mp[1:-1, 2:], mp[1:-1, :-2])
    gy = deriv(hp[2:, 1:-1], hp[:-2, 1:-1], mp[2:, 1:-1], mp[:-2, 1:-1])
    grad = np.hypot(gx, gy)
    ref = float(np.percentile(grad[mask][::4], 97))
    if ref < 1e-5:
        return False  # textura lisa: no hay detalle que realzar
    # Solo los trazos claros (contornos, bordes) dan relieve: el ruido suave del albedo de SF3D
    # quedaría como una superficie agrietada en las zonas lisas
    t = np.clip((grad / ref - 0.25) / 0.35, 0.0, 1.0)
    active = mask & (t > 0)
    if not active.any():
        return False
    t = t[active]
    keep = t * t * (3.0 - 2.0 * t) * (strength / ref)
    # glTF: +X hacia la derecha de la imagen, +Y hacia arriba (filas decrecientes)
    detail = np.stack([-gx[active] * keep, gy[active] * keep, np.ones_like(keep)], axis=-1).astype(np.float32)
    detail /= np.linalg.norm(detail, axis=-1, keepdims=True)

    # Reoriented Normal Mapping solo en los texels con trazo (el resto conserva el mapa tal cual)
    base = encoded[active].astype(np.float32) / 127.5 - 1.0
    base /= np.maximum(np.linalg.norm(base, axis=-1, keepdims=True), 1e-6)
    tb = base + np.array([0.0, 0.0, 1.0], np.float32)
    u = detail * np.array([-1.0, -1.0, 1.0], np.float32)
    combined = tb * (np.sum(tb * u, axis=-1, keepdims=True) / np.maximum(tb[..., 2:3], 1e-4)) - u
    combined /= np.maximum(np.linalg.norm(combined, axis=-1, keepdims=True), 1e-6)
    encoded[active] = np.clip(np.round((combined * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    _extend_islands(encoded, mask)

    out = Image.fromarray(encoded, mode="RGB")
    out.format = "PNG"
    mat.normalTexture = out
    return True


# Cámara con la que SF3D ve la imagen de entrada (sf3d/system.py: default_fovy_deg, default_distance)
SF3D_COND_FOVY_DEG = 40.0
SF3D_COND_DISTANCE = 1.6
# Proyección HD: zona de mezcla según el ángulo entre la normal y la cámara (coseno)
SF3D_PROJECTION_COS = (0.25, 0.60)
# Enfoque de la textura de SF3D (máscara de enfoque) en las zonas que no cubre la proyección
SF3D_SHARPEN_AMOUNT = 0.5

_sf3d_projection_kernels = None


def _get_sf3d_projection_kernels():
    """Compila (una vez) los rasterizadores Numba de la proyección HD: atlas UV y profundidad."""
    global _sf3d_projection_kernels
    if _sf3d_projection_kernels is not None:
        return _sf3d_projection_kernels
    from numba import njit

    @njit(cache=False)
    def _uv_raster(pts, faces, width, height):
        fid = np.full((height, width), -1, np.int32)
        bary = np.zeros((height, width, 3), np.float32)
        for f in range(faces.shape[0]):
            a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
            x0, y0, x1, y1, x2, y2 = pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1], pts[c, 0], pts[c, 1]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            for py in range(max(0, int(min(y0, y1, y2))), min(height - 1, int(max(y0, y1, y2)) + 1) + 1):
                for px in range(max(0, int(min(x0, x1, x2))), min(width - 1, int(max(x0, x1, x2)) + 1) + 1):
                    X = px + 0.5
                    Y = py + 0.5
                    w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
                    w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
                    w2 = 1.0 - w0 - w1
                    if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                        fid[py, px] = f
                        bary[py, px, 0] = w0
                        bary[py, px, 1] = w1
                        bary[py, px, 2] = w2
        return fid, bary

    @njit(cache=False)
    def _depth_raster(pts, depth, faces, width, height):
        zbuf = np.full((height, width), np.inf, np.float32)
        for f in range(faces.shape[0]):
            a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
            x0, y0, x1, y1, x2, y2 = pts[a, 0], pts[a, 1], pts[b, 0], pts[b, 1], pts[c, 0], pts[c, 1]
            den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(den) < 1e-12:
                continue
            for py in range(max(0, int(min(y0, y1, y2))), min(height - 1, int(max(y0, y1, y2)) + 1) + 1):
                for px in range(max(0, int(min(x0, x1, x2))), min(width - 1, int(max(x0, x1, x2)) + 1) + 1):
                    X = px + 0.5
                    Y = py + 0.5
                    w0 = ((y1 - y2) * (X - x2) + (x2 - x1) * (Y - y2)) / den
                    w1 = ((y2 - y0) * (X - x2) + (x0 - x2) * (Y - y2)) / den
                    w2 = 1.0 - w0 - w1
                    if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
                        if z < zbuf[py, px]:
                            zbuf[py, px] = z
        return zbuf

    @njit(cache=False)
    def _uv_interp(fid, bary, faces, attr):
        # Atributo por vértice interpolado en cada texel cubierto (0 fuera de las islas)
        h, w = fid.shape
        out = np.zeros((h, w, attr.shape[1]), np.float32)
        for y in range(h):
            for x in range(w):
                f = fid[y, x]
                if f < 0:
                    continue
                for c in range(attr.shape[1]):
                    out[y, x, c] = (bary[y, x, 0] * attr[faces[f, 0], c] + bary[y, x, 1] * attr[faces[f, 1], c]
                                    + bary[y, x, 2] * attr[faces[f, 2], c])
        return out

    @njit(cache=False)
    def _rebase_texels(fid, bary, faces, face_n, face_t, face_sign, vnorm, nmap):
        # Normal del mapa: base plana de la cara -> base con la normal suave interpolada
        h, w = fid.shape
        out = nmap.copy()
        for y in range(h):
            for x in range(w):
                f = fid[y, x]
                if f < 0:
                    continue
                nfx, nfy, nfz = face_n[f, 0], face_n[f, 1], face_n[f, 2]
                tx, ty, tz = face_t[f, 0], face_t[f, 1], face_t[f, 2]
                s = face_sign[f]
                # Base plana (tangente ya ortogonal a la normal de la cara)
                bfx = (nfy * tz - nfz * ty) * s
                bfy = (nfz * tx - nfx * tz) * s
                bfz = (nfx * ty - nfy * tx) * s
                a, b, c = nmap[y, x, 0], nmap[y, x, 1], nmap[y, x, 2]
                wx = a * tx + b * bfx + c * nfx
                wy = a * ty + b * bfy + c * nfy
                wz = a * tz + b * bfz + c * nfz
                # Normal suave interpolada y tangente reortogonalizada contra ella
                nsx = nsy = nsz = 0.0
                for k in range(3):
                    v = faces[f, k]
                    bk = bary[y, x, k]
                    nsx += bk * vnorm[v, 0]
                    nsy += bk * vnorm[v, 1]
                    nsz += bk * vnorm[v, 2]
                ln = np.sqrt(nsx * nsx + nsy * nsy + nsz * nsz)
                if ln < 1e-12:
                    continue
                nsx /= ln
                nsy /= ln
                nsz /= ln
                d = tx * nsx + ty * nsy + tz * nsz
                tsx, tsy, tsz = tx - d * nsx, ty - d * nsy, tz - d * nsz
                lt = np.sqrt(tsx * tsx + tsy * tsy + tsz * tsz)
                if lt < 1e-12:
                    continue
                tsx /= lt
                tsy /= lt
                tsz /= lt
                bsx = (nsy * tsz - nsz * tsy) * s
                bsy = (nsz * tsx - nsx * tsz) * s
                bsz = (nsx * tsy - nsy * tsx) * s
                ox = wx * tsx + wy * tsy + wz * tsz
                oy = wx * bsx + wy * bsy + wz * bsz
                oz = max(wx * nsx + wy * nsy + wz * nsz, 0.05)
                lo = np.sqrt(ox * ox + oy * oy + oz * oz)
                out[y, x, 0] = ox / lo
                out[y, x, 1] = oy / lo
                out[y, x, 2] = oz / lo
        return out

    _sf3d_projection_kernels = (_uv_raster, _depth_raster, _uv_interp, _rebase_texels)
    return _sf3d_projection_kernels


def _smooth_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Normales suaves por posición (SF3D separa los vértices en cada costura UV)."""
    _, pos_id = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
    pos_id = pos_id.ravel()
    fn = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    acc = np.zeros((pos_id.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, pos_id[faces[:, k]], fn)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    return acc[pos_id]


def _sf3d_camera_project(points: np.ndarray, distance: float, fovy_deg: float):
    """
    Proyecta puntos del GLB de SF3D a la imagen de entrada (u, v en [0, 1], v hacia abajo).
    SF3D pone la cámara en +X interno; al exportar rota (x, y, z) -> (-y, z, -x), así que en el
    GLB la cámara está en (0, 0, -distancia) mirando a +Z, con la derecha de la imagen en -X.
    """
    depth = points[:, 2] + distance
    k = 0.5 / (np.maximum(depth, 1e-3) * np.tan(np.deg2rad(fovy_deg) / 2.0))
    u = 0.5 - points[:, 0] * k
    v = 0.5 - points[:, 1] * k
    return np.stack([u, v], axis=1), depth


def _fit_silhouette(sil: np.ndarray, alpha: np.ndarray, max_scale_dev: float):
    """
    Ajuste 2D (escala uniforme + desplazamiento) de la silueta proyectada sobre el alfa de la
    imagen: corrige el pequeño desfase entre la forma de SF3D y el dibujo. Devuelve (escala,
    du, dv, IoU) en coordenadas normalizadas, con la escala alrededor del centro (0.5, 0.5).
    """
    n = sil.shape[0]
    ys, xs = np.nonzero(sil)
    ya, xa = np.nonzero(alpha)
    if len(xs) == 0 or len(xa) == 0:
        return 1.0, 0.0, 0.0, 0.0
    c = n / 2.0
    s0 = ((xa.max() - xa.min()) + (ya.max() - ya.min())) / max((xs.max() - xs.min()) + (ys.max() - ys.min()), 1)
    s0 = float(np.clip(s0, 1.0 - max_scale_dev, 1.0 + max_scale_dev))
    cx0 = (xa.min() + xa.max()) / 2.0 - ((xs.min() + xs.max()) / 2.0 - c) * s0 - c
    cy0 = (ya.min() + ya.max()) / 2.0 - ((ys.min() + ys.max()) / 2.0 - c) * s0 - c
    sil_u8 = sil.astype(np.uint8)
    alpha_b = alpha.astype(bool)

    def iou(s, tx, ty):
        m = np.float32([[s, 0, c - s * c + tx], [0, s, c - s * c + ty]])
        w = cv2.warpAffine(sil_u8, m, (n, n), flags=cv2.INTER_NEAREST).astype(bool)
        union = np.count_nonzero(w | alpha_b)
        return np.count_nonzero(w & alpha_b) / union if union else 0.0

    best = (iou(s0, cx0, cy0), s0, cx0, cy0)
    for step_s, step_t in ((0.02, 2.0), (0.01, 1.0), (0.005, 0.5)):
        _, bs, bx, by = best
        for ds in (-step_s, 0.0, step_s):
            s = bs + ds
            if abs(s - 1.0) > max_scale_dev:
                continue
            for dx in (-step_t, 0.0, step_t):
                for dy in (-step_t, 0.0, step_t):
                    val = iou(s, bx + dx, by + dy)
                    if val > best[0]:
                        best = (val, s, bx + dx, by + dy)
    score, s, tx, ty = best
    return s, tx / n, ty / n, score


def project_input_onto_texture(
    mesh: Any,
    rgba_image: Image.Image,
    distance: float = SF3D_COND_DISTANCE,
    fovy_deg: float = SF3D_COND_FOVY_DEG,
    sharpen: float = SF3D_SHARPEN_AMOUNT,
    max_scale_dev: float = 0.08,
    min_iou: float = 0.85,
) -> bool:
    """
    Proyección HD: pinta la imagen de entrada (a resolución completa) sobre las caras del GLB de
    SF3D que la cámara ve de frente, y enfoca la textura de SF3D en el resto. SF3D solo ve la
    imagen a 512 px y guarda el color en un triplano de ~384 muestras, por eso su albedo sale suave.
    rgba_image: la imagen ya encuadrada para SF3D (preprocess_image_for_sf3d), con alfa.
    """
    t_start = time.time()
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    albedo_img = getattr(mat, "baseColorTexture", None) if mat is not None else None
    uv = getattr(mesh.visual, "uv", None) if mat is not None else None
    if not isinstance(albedo_img, Image.Image) or uv is None or len(uv) != len(mesh.vertices):
        return False
    uv_raster, depth_raster, uv_interp, _ = _get_sf3d_projection_kernels()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    ref = np.asarray(rgba_image.convert("RGBA"))
    ref_res = ref.shape[0]

    # 1. Silueta de la malla vista desde la cámara de SF3D y ajuste fino contra el alfa del dibujo
    proj_v, depth_v = _sf3d_camera_project(verts, distance, fovy_deg)
    fit_res = 256
    sil = np.isfinite(depth_raster(proj_v * fit_res, depth_v, faces, fit_res, fit_res))
    alpha_small = cv2.resize(ref[..., 3], (fit_res, fit_res), interpolation=cv2.INTER_AREA) > 127
    scale, du, dv, iou = _fit_silhouette(sil, alpha_small, max_scale_dev)
    if iou < min_iou:
        print(f" -> [Proyección HD] La forma no encaja con el dibujo (IoU {iou:.2f}); se conserva la textura de SF3D.")
        return False

    # 2. Posición, normal, coordenadas en el dibujo y visibilidad de cada texel del atlas
    tex_h, tex_w = albedo_img.height, albedo_img.width
    uv_px = np.stack([uv[:, 0] * tex_w, (1.0 - uv[:, 1]) * tex_h], axis=1)
    fid, bary = uv_raster(uv_px, faces, tex_w, tex_h)
    covered = fid >= 0
    attr = np.concatenate([proj_v, depth_v[:, None], _smooth_vertex_normals(verts, faces), verts], axis=1)
    texel = uv_interp(fid, bary, faces, attr.astype(np.float32))[covered]
    proj_t, depth_t, normals, pos = texel[:, :2], texel[:, 2], texel[:, 3:6], texel[:, 6:9]
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    to_cam = np.array([0.0, 0.0, -distance], np.float32) - pos
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-12)
    cos_view = np.sum(normals * to_cam, axis=1)

    z_res = 1024
    zbuf = depth_raster(proj_v * z_res, depth_v, faces, z_res, z_res)
    # Mínimo en 3x3: cerca de un borde de profundidad gana la superficie de delante
    zbuf = cv2.erode(zbuf, np.ones((3, 3), np.uint8))
    zi = np.clip((proj_t * z_res).astype(np.int64), 0, z_res - 1)
    visible = depth_t <= zbuf[zi[:, 1], zi[:, 0]] + 0.02 * distance

    # 3. Coordenadas en el dibujo (con el ajuste de silueta) y su núcleo de alfa (sin el borde)
    img_uv = (proj_t - 0.5) * scale + 0.5 + np.array([du, dv])
    alpha = (ref[..., 3] > 127).astype(np.uint8)
    erode_px = max(2, int(round(ref_res * 0.008)))
    core = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2))
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), erode_px)
    # Mapas de muestreo a tamaño de la textura (cv2.remap no admite más de 32767 filas)
    map_x = np.full((tex_h, tex_w), -1.0, np.float32)
    map_y = np.full((tex_h, tex_w), -1.0, np.float32)
    map_x[covered] = img_uv[:, 0] * ref_res - 0.5
    map_y[covered] = img_uv[:, 1] * ref_res - 0.5
    sample = lambda img: cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    projected = sample(np.ascontiguousarray(ref[..., :3])).astype(np.float32)
    img_core = sample(core)[covered]

    lo, hi = SF3D_PROJECTION_COS
    t = np.clip((cos_view - lo) / (hi - lo), 0.0, 1.0)
    weight = np.zeros((tex_h, tex_w), np.float32)
    weight[covered] = (t * t * (3.0 - 2.0 * t)) * visible * np.clip(img_core, 0.0, 1.0)

    # Suavizado de la máscara dentro de las islas: sin escalones entre texels ni costuras
    m = covered.astype(np.float32)
    weight = cv2.GaussianBlur(weight, (0, 0), 1.5) / np.maximum(cv2.GaussianBlur(m, (0, 0), 1.5), 1e-4)
    weight = np.clip(weight, 0.0, 1.0) * m

    # 4. Enfoque de la textura de SF3D y mezcla con el dibujo
    out = np.asarray(albedo_img.convert("RGB"), dtype=np.float32)
    if sharpen > 0:
        # El relleno entre islas también se enfoca: inpaint_uv_atlas_texture lo rehace después
        sigma = 1.6 * tex_w / 2048.0
        blur = cv2.GaussianBlur(out * m[..., None], (0, 0), sigma)
        blur /= np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-4)[..., None]
        out += sharpen * (out - blur)
    projected -= out
    projected *= weight[..., None]
    out += projected
    out += 0.5
    np.clip(out, 0, 255, out=out)
    out = out.astype(np.uint8)
    _extend_islands(out, covered)
    result = Image.fromarray(out, mode="RGB")
    result.format = "PNG"
    mat.baseColorTexture = result
    share = float(np.count_nonzero(weight > 0.5)) / max(int(covered.sum()), 1)
    print(f" -> [Proyección HD] Dibujo proyectado en el {share:.0%} de la textura "
          f"(encaje IoU {iou:.2f}, escala {scale:.3f}) en {time.time() - t_start:.1f}s.")
    return True


def rebase_normal_map_to_vertex_normals(mesh: Any) -> bool:
    """
    SF3D hornea el normal map respecto a la normal plana de cada triángulo (malla sin vértices
    compartidos), pero el GLB sale con vértices unidos y normales suaves: en Godot cada triángulo
    se ve como una celda (superficie "agrietada"). Une los vértices por posición y UV y vuelve a
    expresar el normal map respecto a las normales suaves que trimesh exporta para esa malla.
    Espera el mapa ya en convención glTF (fix_sf3d_normal_convention).
    """
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    tex = getattr(mat, "normalTexture", None) if mat is not None else None
    if not isinstance(tex, Image.Image) or getattr(mesh.visual, "uv", None) is None:
        return False
    mesh.merge_vertices(merge_tex=False, merge_norm=True)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    if len(uv) != len(verts):
        return False
    # Las normales que trimesh calcula para esta topología: center_and_orient_mesh (fix_normals)
    # y cada recarga del GLB las vuelven a calcular igual, así que son las que llegan a Godot
    smooth = np.asarray(trimesh.geometry.weighted_vertex_normals(
        len(verts), faces, mesh.face_normals, mesh.face_angles), dtype=np.float64)

    uv_raster, _, _, rebase_texels = _get_sf3d_projection_kernels()
    h, w = tex.height, tex.width
    fid, bary = uv_raster(np.stack([uv[:, 0] * w, (1.0 - uv[:, 1]) * h], axis=1), faces, w, h)
    covered = fid >= 0

    # Base tangente plana de cada cara (la que usó SF3D, ya en convención glTF)
    p0, p1, p2 = (verts[faces[:, k]] for k in range(3))
    t0, t1, t2 = (uv[faces[:, k]] for k in range(3))
    e1, e2, d1, d2 = p1 - p0, p2 - p0, t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    tangent = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / det[:, None]
    face_n = np.cross(e1, e2)
    face_n /= np.maximum(np.linalg.norm(face_n, axis=1, keepdims=True), 1e-12)
    tangent -= np.sum(tangent * face_n, axis=1, keepdims=True) * face_n
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    sign = np.where(det < 0, -1.0, 1.0)

    nmap = np.asarray(tex.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    out = rebase_texels(fid, bary, faces, face_n, tangent, sign, smooth, nmap)
    enc = np.clip(np.round((out * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    _extend_islands(enc, covered)
    fixed = Image.fromarray(enc, mode="RGB")
    fixed.format = "PNG"
    mat.normalTexture = fixed
    return True


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
            import sf3d.system as sf3d_system
            from sf3d.system import SF3D
        except ImportError as e:
            raise RuntimeError(
                f"No se pudo importar la librería de SF3D ({e}).\n"
                "Ejecuta 'setup.bat' y selecciona la opción de Stable Fast 3D."
            ) from e
        sf3d_system.dilate_fill = _sf3d_dilate_fill_lowmem

        local_weights_dir = SF3D_LOCAL_DIR
        target_path_or_id = str(local_weights_dir) if (local_weights_dir / "model.safetensors").exists() else self.model_id

        print(f"[SF3D] Inicializando Stable Fast 3D desde '{target_path_or_id}' en {self.device.upper()}...")
        try:
            model = _sf3d_load_lowmem(SF3D, target_path_or_id, self.device)
            register_model("sf3d", model)
            self.model = model
            print("[SF3D] Modelo cargado exitosamente en VRAM.")
        except Exception as e:
            err_str = str(e)
            # Si el modelo oficial está restringido por autorización (gated repo) en Hugging Face
            if ("401" in err_str or "403" in err_str or "gated" in err_str.lower() or "unauthorized" in err_str.lower() or "restricted" in err_str.lower()) and target_path_or_id == "stabilityai/stable-fast-3d":
                print("[SF3D] Acceso a stabilityai/stable-fast-3d restringido. Cargando desde mirror verificado (patrickbdevaney/stable-fast-3d)...")
                try:
                    model = _sf3d_load_lowmem(SF3D, "patrickbdevaney/stable-fast-3d", self.device)
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
        progress_callback: Optional[Callable[[str, int], None]] = None,
        target_faces: Optional[int] = None,
        remesh: str = "triangle",
        material: str = "mate",
        relief: str = "suave",
        project_input: bool = True
    ) -> trimesh.Trimesh:
        """
        target_faces > 0: SF3D remalla la geometría a ~target_faces triángulos ANTES de desplegar
        las UV y hornear, así la textura y el normal map se hornean sobre la malla final del juego
        (decimar después de hornear deforma las UV y el normal map deja de corresponder).
        remesh: "triangle" (isotrópico, robusto) o "quad" (Instant Meshes, topología más limpia).
        material: clave de MATERIAL_PRESETS. relief: clave de TEXTURE_RELIEF_LEVELS.
        project_input: Proyección HD (pinta la imagen de entrada a resolución completa sobre las
        caras que se ven de frente y enfoca la textura de SF3D en el resto).
        """
        self._load_model()
        if progress_callback:
            progress_callback("⚡ SF3D: Preprocesando silueta (Anti-Color Bleeding y Padding 0.82)...", 35)

        if target_faces and target_faces > 0 and remesh in ("triangle", "quad"):
            # Una malla cerrada de triángulos tiene ~2 caras por vértice
            remesh_mode, vertex_count = remesh, max(500, int(target_faces) // 2)
        else:
            remesh_mode, vertex_count = "none", -1

        # 1. Preprocesamiento Inpainting 2D Anti-Halos y encuadre con foreground_ratio=0.82 fijo
        rgba_img = preprocess_image_for_sf3d(
            image,
            foreground_ratio=foreground_ratio,
            edge_dilation_pixels=10
        )

        try:
            from contextlib import nullcontext
            if progress_callback:
                progress_callback("⚡ SF3D: Inferencia base y horneado de textura en GPU...", 55)

            # Limpieza preventiva de VRAM antes de inferencia y horneado
            flush_vram()
            print_real_vram_status("Pre-Inferencia SF3D")

            target_res = texture_resolution
            mesh_out = None

            def _run(res: int, mode: str, count: int):
                with torch.inference_mode():
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16) if "cuda" in self.device else nullcontext():
                        return self.model.run_image([rgba_img], bake_resolution=res, remesh=mode, vertex_count=count)[0]

            print(f" -> [SF3D Inferencia] Generando modelo 3D con textura a {target_res}x{target_res}"
                  + (f", remallado '{remesh_mode}' a ~{vertex_count * 2:,} caras..." if remesh_mode != "none" else "..."))
            # Los reintentos se hacen FUERA del except: dentro, la traza de la excepción mantiene vivos
            # los tensores del intento fallido (GB de VRAM) y el reintento vuelve a quedarse sin memoria
            attempt = (target_res, remesh_mode, vertex_count)
            while mesh_out is None:
                failure = None
                try:
                    mesh_out = _run(*attempt)
                except Exception as e_run:
                    err_str = str(e_run)
                    is_vram = isinstance(e_run, torch.cuda.OutOfMemoryError) or "out of memory" in err_str.lower()
                    failure = (type(e_run).__name__, err_str, is_vram)
                if failure is None:
                    break
                flush_vram()  # la excepción ya se soltó: aquí sí se libera la VRAM del intento fallido
                name, err_str, is_vram = failure
                res, mode, count = attempt
                if is_vram and res > 1024:
                    print(f" -> [SF3D VRAM Fallback] VRAM insuficiente para textura {res}x{res}; reintentando a 1024x1024...")
                    attempt = (1024, mode, count)
                elif not is_vram and mode != "none":
                    # Remallado (gpytoolbox / Instant Meshes) fallido: malla completa, el pipeline decima
                    print(f" -> [SF3D Remesh] El remallado falló ({err_str}); se genera la malla completa.")
                    attempt = (res, "none", -1)
                else:
                    raise RuntimeError(f"SF3D falló ({name}): {err_str}")
            target_res = attempt[0]

            # SF3D marca las texturas como JPEG y trimesh las guarda con calidad 75 y croma
            # submuestreado: en PNG el GLB final no pierde calidad (el normal map en JPEG mete
            # bloques en la iluminación). SF3D ya entrega un trimesh: no hace falta pasar por un
            # GLB temporal (codificar y leer las texturas costaba ~1 s por generación)
            _mark_textures_lossless(mesh_out)
            final_mesh = mesh_out

            # 2. Suavizado Taubin post-baking ELIMINADO:
            # Desplazar vértices 3D después del UV unwrap desalinea la geometría respecto a la textura
            # y agrava el desfase en las costuras. SF3D ya aplica Taubin smoothing antes del UV unwrap en sf3d/system.py.

            # 3. TEXTURA NATIVA SF3D: Dilatación e inpainting de fondo para eliminar costuras UV y texture bleeding
            if generate_texture:
                # Normal map a convención glTF y relieve de detalle (antes del relleno del atlas,
                # que solo toca el albedo)
                if fix_sf3d_normal_convention(final_mesh):
                    print(" -> [SF3D Normal map] Convertido a la convención glTF/Godot.")
                try:
                    if rebase_normal_map_to_vertex_normals(final_mesh):
                        print(f" -> [SF3D Normal map] Expresado respecto a las normales suaves del GLB "
                              f"({len(final_mesh.vertices):,} vértices unidos).")
                except Exception as e_rebase:
                    print(f" -> [Aviso Normal map] {e_rebase}")
                if project_input:
                    try:
                        # El dibujo con el mismo encuadre que vio SF3D, a la resolución de la entrada
                        proj_res = int(np.clip(max(image.size) / foreground_ratio, 1024, 2048))
                        proj_img = rgba_img if proj_res == rgba_img.width else preprocess_image_for_sf3d(
                            image, foreground_ratio=foreground_ratio, edge_dilation_pixels=10,
                            target_resolution=proj_res)
                        project_input_onto_texture(final_mesh, proj_img)
                    except Exception as e_proj:
                        print(f" -> [Aviso Proyección HD] {e_proj}. Se conserva la textura de SF3D.")
                relief_strength = TEXTURE_RELIEF_LEVELS.get(relief, 0.0)
                if relief_strength > 0 and add_texture_relief(final_mesh, relief_strength):
                    print(f" -> [SF3D Relieve] Relieve de detalle '{relief}' añadido al normal map.")
                final_mesh = inpaint_uv_atlas_texture(final_mesh, dilation_radius=48)
                _mark_textures_lossless(final_mesh)
                if progress_callback:
                    progress_callback("🎨 SF3D: Textura PBR y atlas UV protegidos...", 80)

            # 4. Acabado del material elegido (mate, satinado o metal), doble cara
            if generate_texture:
                apply_material_preset(final_mesh, material)

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
    - Conserva el acabado elegido; solo si el material llega sin él (valores por defecto de glTF,
      metal 1.0 / rugosidad 1.0) aplica el mate anti-glare (metallicFactor=0.0, roughnessFactor=0.90).
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
                    pbr = mat.pbrMetallicRoughness
                    pbr.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                    # Sin acabado explícito glTF asume metal 1.0 / rugosidad 1.0 (se ve oscuro y
                    # sucio): ahí se pone el mate anti-glare. Ningún preset usa rugosidad 1.0.
                    metallic = 1.0 if pbr.metallicFactor is None else pbr.metallicFactor
                    roughness = 1.0 if pbr.roughnessFactor is None else pbr.roughnessFactor
                    if metallic >= 1.0 and roughness >= 1.0:
                        pbr.metallicFactor = 0.0
                        pbr.roughnessFactor = 0.90

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
    export_formats: Optional[list] = None,
    improved_depth: bool = False,
    max_quality: bool = False,
    material: str = "mate",
    relief: str = "suave",
    project_input: bool = True
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
            
    quality = shape_quality_settings(octree_resolution, improved_depth, max_quality)
    shape_subfolder, taubin_iters, octree_resolution = quality["shape_subfolder"], quality["taubin_iterations"], quality["octree"]
    steps = quality["steps"] or steps
    if (improved_depth or max_quality) and "sf3d" not in model_type.lower():
        print(f" -> [Calidad] {shape_subfolder} · {steps} pasos · octree {octree_resolution} · "
              f"Taubin {taubin_iters} · semillas {quality['seeds']}")
        if quality["missing"]:
            print(f" -> [Calidad] Para mejor resultado: python preparar_modelos.py download {quality['missing'][0]}")

    if "sf3d" in model_type.lower():
        generator = None
    else:
        generator = Hunyuan3DGenerator(
            model_id="tencent/Hunyuan3D-2",
            shape_subfolder=shape_subfolder,
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
            foreground_ratio=0.82,
            target_faces=target_faces,
            relief=relief,
            project_input=project_input
        )
        # SF3D ya remalla a ~target_faces antes de hornear; solo se decima si el remallado falló
        if target_faces and len(raw_mesh.faces) > target_faces * SF3D_FACE_TOLERANCE:
            decimated_mesh = decimate_mesh_preserving_texture(raw_mesh, target_faces=target_faces)
        else:
            decimated_mesh = raw_mesh

        final_textured_mesh = center_and_orient_mesh(
            decimated_mesh,
            align_floor=align_bottom,
            normalize_scale=normalize_scale
        )

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
            texture_safe_cap=TEXTURE_SAFE_MAX_FACES if generate_texture else None,
            taubin_iterations=taubin_iters,
            allow_high_octree=improved_depth or max_quality,
            guidance_scale=quality["guidance"],
            seed_candidates=quality["seeds"],
            keep_hd=quality["normal_map"]
        )
        hd_mesh, generator.last_hd_mesh = generator.last_hd_mesh, None
        aligned_mesh = center_and_orient_mesh(
            raw_mesh,
            align_floor=align_bottom,
            normalize_scale=normalize_scale
        )
        if hd_mesh is not None:
            hd_mesh.vertices = follow_frame(hd_mesh.vertices, raw_mesh, aligned_mesh)
        if improved_depth:
            aligned_mesh, d_before, d_after, _ = reinforce_depth(
                aligned_mesh, image=clean_image, companions=(hd_mesh,) if hd_mesh is not None else ()
            )
        else:
            d_before = d_after = measure_depth_ratio(aligned_mesh)
        print(f" -> [Profundidad] Fondo/alto: {d_before:.2f}" + (f" → reforzada a {d_after:.2f}" if d_after > d_before else ""))

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

        # Detalle de la malla HD horneado en un normal map (la malla final conserva sus caras)
        if hd_mesh is not None:
            hd_mesh.vertices = follow_frame(hd_mesh.vertices, aligned_mesh, final_textured_mesh)
            try:
                final_textured_mesh = bake_normal_map(final_textured_mesh, hd_mesh) or final_textured_mesh
            except Exception as e_bake:
                print(f" -> [Aviso Normal map] {e_bake}")
            hd_mesh = None

    # Acabado elegido sobre el material existente (solo factores: conserva texturas y normal map)
    if generate_texture:
        apply_material_preset(final_textured_mesh, material)

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
        "--profundidad",
        action="store_true",
        help="Profundidad mejorada (Hunyuan): variante Fast de 30 pasos, 3 semillas (gana la más fiel a la silueta), "
             "octree 384, refuerzo guiado si sale plano y normal map del detalle HD"
    )
    parser.add_argument(
        "--maxima",
        action="store_true",
        help="Máxima calidad (Hunyuan): modelo completo de 50 pasos con guía, octree 384, sin suavizado y "
             "normal map del detalle HD (más lento; descarga: preparar_modelos.py download dit_full)"
    )
    parser.add_argument(
        "--material",
        choices=list(MATERIAL_PRESETS),
        default="mate",
        help="Acabado del material: mate (cartoon), satinado (plástico/barniz) o metal (oro, plata, monedas)"
    )
    parser.add_argument(
        "--relieve",
        choices=list(TEXTURE_RELIEF_LEVELS),
        default="suave",
        help="Relieve de detalle sacado de la textura (SF3D): none, suave o marcado"
    )
    parser.add_argument(
        "--sin-proyeccion",
        action="store_true",
        help="SF3D: no proyectar la imagen de entrada en HD sobre el frente del modelo"
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
            export_formats=requested_formats,
            improved_depth=args.profundidad,
            max_quality=args.maxima,
            material=args.material,
            relief=args.relieve,
            project_input=not args.sin_proyeccion
        )
    except Exception as e:
        print(f"\n[ERROR FATAL] La generación falló: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
