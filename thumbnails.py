#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Miniaturas de modelos .glb para el panel lateral de modelos generados
==============================================================================
Render por software sin dependencias extra (sin OpenGL): se muestrean puntos
sobre la superficie, se proyectan en vista 3/4 con z-buffer y se sombrean con
luz Lambert usando el color de la textura. Las miniaturas se cachean en disco y
solo se regeneran si el .glb es más nuevo.
==============================================================================
"""

import os
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh
from PIL import Image

THUMB_SIZE = 192
_SUPERSAMPLE = 2
_MAX_SAMPLES = 2_500_000
_CLAY_RGB = np.array([0.82, 0.84, 0.88])
_LOCK = threading.Lock()


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Vista 3/4: giro en Y (yaw) y luego inclinación en X (pitch)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return rx @ ry


def _texture_array(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    img = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    if img is None or getattr(mesh.visual, "uv", None) is None:
        return None
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _base_color(mesh: trimesh.Trimesh) -> np.ndarray:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    factor = getattr(material, "baseColorFactor", None)
    if factor is not None and len(factor) >= 3:
        rgb = np.asarray(factor[:3], dtype=np.float32)
        return rgb / 255.0 if rgb.max() > 1.0 else rgb
    return _CLAY_RGB


def _fill_gaps(depth: np.ndarray, color: np.ndarray, threshold: float, iterations: int = 2) -> tuple:
    """
    Corrige el muestreo por puntos: un píxel vacío o claramente más lejano que su vecino
    más cercano (un punto de la cara trasera que se coló por un poro) toma el color de ese vecino.
    """
    h, w = depth.shape
    for _ in range(iterations):
        pd = np.pad(depth, 1, constant_values=-np.inf)
        pc = np.pad(color, ((1, 1), (1, 1), (0, 0)))
        best_d, best_c = depth.copy(), color.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                d = pd[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
                better = d > best_d
                best_d = np.where(better, d, best_d)
                best_c = np.where(better[..., None], pc[1 + dy:1 + dy + h, 1 + dx:1 + dx + w], best_c)
        replace = depth < best_d - threshold
        depth = np.where(replace, best_d, depth)
        color = np.where(replace[..., None], best_c, color)
    return depth, color


def render_thumbnail(glb_path: Union[str, Path], out_path: Union[str, Path], size: int = THUMB_SIZE) -> Path:
    """Renderiza la miniatura PNG (fondo transparente) del modelo."""
    mesh = trimesh.load(str(glb_path), force="mesh")
    res = size * _SUPERSAMPLE
    # Los modelos generados miran hacia -Z: cámara al frente, girada 30° (vista 3/4)
    rot = _rotation(210.0, 14.0)

    verts = (mesh.vertices - mesh.bounds.mean(axis=0)) @ rot.T
    extent = float(np.max(np.abs(verts[:, :2]))) or 1.0
    scale = (res * 0.44) / extent
    px = np.empty_like(verts)
    px[:, 0] = res / 2 + verts[:, 0] * scale
    px[:, 1] = res / 2 - verts[:, 1] * scale
    px[:, 2] = verts[:, 2]

    faces = mesh.faces
    a, b, c = px[faces[:, 0]], px[faces[:, 1]], px[faces[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    counts = np.clip(np.ceil(area * 2.5), 1, 900).astype(np.int64)
    if counts.sum() > _MAX_SAMPLES:
        counts = np.maximum(1, (counts * (_MAX_SAMPLES / counts.sum())).astype(np.int64))

    face_idx = np.repeat(np.arange(len(faces)), counts)
    rng = np.random.default_rng(7)
    r1, r2 = rng.random(len(face_idx)), rng.random(len(face_idx))
    flip = r1 + r2 > 1.0
    r1[flip], r2[flip] = 1.0 - r1[flip], 1.0 - r2[flip]
    pa, pb, pc = a[face_idx], b[face_idx], c[face_idx]
    pts = pa + r1[:, None] * (pb - pa) + r2[:, None] * (pc - pa)

    # Color: textura (UV interpolada por muestra), color de vértice o color base del material
    tex = _texture_array(mesh)
    if tex is not None:
        uv = np.asarray(mesh.visual.uv)
        ua, ub, uc = uv[faces[face_idx, 0]], uv[faces[face_idx, 1]], uv[faces[face_idx, 2]]
        suv = ua + r1[:, None] * (ub - ua) + r2[:, None] * (uc - ua)
        th, tw = tex.shape[:2]
        tx = np.clip((suv[:, 0] % 1.0) * (tw - 1), 0, tw - 1).astype(np.int64)
        ty = np.clip((1.0 - suv[:, 1] % 1.0) * (th - 1), 0, th - 1).astype(np.int64)
        color = tex[ty, tx]
    elif getattr(mesh.visual, "kind", None) == "vertex":
        vc = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
        color = vc[faces[face_idx]].mean(axis=1)
    else:
        color = np.broadcast_to(_base_color(mesh), (len(face_idx), 3))

    # Sombreado Lambert (doble cara) con luz principal + ambiente
    normals = mesh.face_normals @ rot.T
    light = np.array([-0.35, 0.55, 0.75])
    light /= np.linalg.norm(light)
    lambert = np.abs(normals @ light)[face_idx]
    shaded = np.clip(color * (0.38 + 0.72 * lambert[:, None]), 0, 1)

    # Z-buffer: se ordena por profundidad y la última escritura (la más cercana) gana
    ix, iy = pts[:, 0].astype(np.int64), pts[:, 1].astype(np.int64)
    inside = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
    ix, iy, z, shaded = ix[inside], iy[inside], pts[inside, 2], shaded[inside]
    order = np.argsort(z)
    pix = (iy * res + ix)[order]
    depth = np.full(res * res, -np.inf, dtype=np.float32)
    color_buf = np.zeros((res * res, 3), dtype=np.float32)
    depth[pix] = z[order]
    color_buf[pix] = shaded[order]
    depth, color_buf = _fill_gaps(depth.reshape(res, res), color_buf.reshape(res, res, 3),
                                  threshold=0.04 * extent * 2)
    alpha = np.isfinite(depth).astype(np.float32)[..., None]
    rgba = (np.concatenate([color_buf, alpha], axis=2) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, "RGBA").resize((size, size), Image.LANCZOS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp.png")
    img.save(tmp)
    os.replace(tmp, out_path)
    return out_path


def thumbnail_path(glb_path: Union[str, Path], thumbs_dir: Union[str, Path]) -> Path:
    glb_path = Path(glb_path)
    return Path(thumbs_dir) / f"{glb_path.parent.name}__{glb_path.stem}.png"


def get_thumbnail(glb_path: Union[str, Path], thumbs_dir: Union[str, Path], render_missing: bool = True) -> Optional[Path]:
    """Miniatura cacheada; la renderiza si falta o el .glb es más nuevo (si render_missing)."""
    glb_path = Path(glb_path)
    out = thumbnail_path(glb_path, thumbs_dir)
    try:
        if out.exists() and out.stat().st_mtime >= glb_path.stat().st_mtime:
            return out
        if not render_missing:
            return None
        with _LOCK:
            return render_thumbnail(glb_path, out)
    except Exception as e:
        print(f"[Aviso Miniatura] {glb_path.name}: {e}")
        return None


def make_placeholder(path: Union[str, Path], size: int = THUMB_SIZE) -> Path:
    """Miniatura neutra mientras se genera la real."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        dot = Image.new("RGBA", (size // 3, size // 3), (139, 92, 246, 60))
        img.paste(dot, (size // 3, size // 3))
        img.save(path)
    return path


def warm_thumbnails(glb_paths: list, thumbs_dir: Union[str, Path]) -> threading.Thread:
    """Genera en segundo plano las miniaturas que falten (no bloquea el arranque)."""
    def _run():
        for p in glb_paths:
            get_thumbnail(p, thumbs_dir)
    t = threading.Thread(target=_run, daemon=True, name="thumb-warmup")
    t.start()
    return t
