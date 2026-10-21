"""
Detalle y volumen para las mallas de Hunyuan3D ("Profundidad mejorada" / "Máxima calidad"):
  - estimate_relative_depth: profundidad relativa de la imagen con Depth Anything V2.
  - depth_guided_warp: engorda una malla plana siguiendo esa profundidad (no un estirado uniforme).
  - follow_frame: lleva la malla HD al mismo centrado/escala que la malla final.
  - bake_normal_map: hornea el detalle de la malla HD en un normal map de la malla ligera.
Todo corre en la CPU: no compite por la VRAM con los modelos de forma y textura.
"""
import os
import time
from typing import Callable, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image

DEPTH_ESTIMATOR_MODEL = os.environ.get("DEPTH_ESTIMATOR_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
# Lado del normal map cuando la malla no trae textura (con textura se usa el tamaño de la textura)
NORMAL_MAP_SIZE = int(os.environ.get("NORMAL_MAP_SIZE", "1024"))
# Si la malla HD no tiene al menos este múltiplo de caras, no hay detalle que hornear
NORMAL_BAKE_MIN_GAIN = 1.5
# La malla HD del generador trae ruido fino (bultos de pocos vóxeles) que el normal map volvería a
# pintar como escamas. Se suavizan sus NORMALES (promedio con los vecinos, sin mover la malla):
# medido en una moneda tipo octree 384, 10 pasadas reducen el error del ruido de ~15° a ~8°
# (Taubin sobre la geometría apenas lo bajaba a ~14.5°) a cambio de suavizar algo los relieves finos
NORMAL_DENOISE_ITERATIONS = int(os.environ.get("NORMAL_DENOISE_ITERATIONS", "10"))
# Intensidad del relieve horneado (1 = el de la malla HD ya filtrada)
NORMAL_MAP_STRENGTH = float(os.environ.get("NORMAL_MAP_STRENGTH", "1.0"))

_DEPTH_PIPE = None

try:
    from numba import njit

    @njit(fastmath=True, cache=True)
    def _rasterize_uv(uv_px, faces, size):
        """Por cada texel: cara que lo cubre (-1 = vacío) y sus coordenadas baricéntricas."""
        face_id = np.full((size, size), -1, dtype=np.int32)
        bary = np.zeros((size, size, 3), dtype=np.float32)
        for f in range(faces.shape[0]):
            i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
            x0, y0 = uv_px[i0, 0], uv_px[i0, 1]
            x1, y1 = uv_px[i1, 0], uv_px[i1, 1]
            x2, y2 = uv_px[i2, 0], uv_px[i2, 1]
            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-12 or max(x0, x1, x2) < 0 or max(y0, y1, y2) < 0:
                continue
            inv_denom = 1.0 / denom
            min_x = max(0, int(min(x0, x1, x2)))
            max_x = min(size - 1, int(max(x0, x1, x2)) + 1)
            min_y = max(0, int(min(y0, y1, y2)))
            max_y = min(size - 1, int(max(y0, y1, y2)) + 1)
            for py in range(min_y, max_y + 1):
                for px in range(min_x, max_x + 1):
                    p_x = px + 0.5
                    p_y = py + 0.5
                    w0 = ((y1 - y2) * (p_x - x2) + (x2 - x1) * (p_y - y2)) * inv_denom
                    w1 = ((y2 - y0) * (p_x - x2) + (x0 - x2) * (p_y - y2)) * inv_denom
                    w2 = 1.0 - w0 - w1
                    if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                        face_id[py, px] = f
                        bary[py, px, 0] = w0
                        bary[py, px, 1] = w1
                        bary[py, px, 2] = w2
        return face_id, bary

except Exception:
    _rasterize_uv = None


# ==============================================================================
# Profundidad estimada de la imagen y refuerzo guiado
# ==============================================================================

def _depth_pipeline():
    global _DEPTH_PIPE
    if _DEPTH_PIPE is None:
        from transformers import pipeline
        t0 = time.time()
        # En la CPU (~100 MB): no ocupa VRAM y tarda 1-3 s por imagen
        _DEPTH_PIPE = pipeline("depth-estimation", model=DEPTH_ESTIMATOR_MODEL, device=-1)
        print(f" -> [Profundidad] Depth Anything V2 cargado en {time.time() - t0:.1f}s ({DEPTH_ESTIMATOR_MODEL})")
    return _DEPTH_PIPE


def estimate_relative_depth(image: Image.Image) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Relieve relativo del objeto en la imagen: (relieve HxW en [0, 1], 1 = lo más cercano; máscara HxW).
    Devuelve None si no hay silueta o el estimador no está disponible.
    """
    import torch
    rgba = image.convert("RGBA")
    mask = np.asarray(rgba)[..., 3] > 127
    if mask.sum() < 64:
        return None
    # Fondo gris neutro: el estimador no confunde el fondo con el objeto
    background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
    rgb = Image.alpha_composite(background, rgba).convert("RGB")
    try:
        with torch.inference_mode():
            # Depth Anything da disparidad relativa (mayor = más cerca), a veces a otra resolución
            depth = torch.as_tensor(_depth_pipeline()(rgb)["predicted_depth"], dtype=torch.float32).cpu()
            while depth.ndim < 4:
                depth = depth.unsqueeze(0)
            depth = torch.nn.functional.interpolate(depth, size=mask.shape, mode="bicubic", align_corners=False)
            depth = depth[0, 0].numpy()
    except Exception as e:
        print(f" -> [Profundidad] Depth Anything V2 no disponible ({e}): se usa el refuerzo uniforme.")
        return None

    lo, hi = np.percentile(depth[mask], [2, 98])
    if hi - lo < 1e-6:
        return None
    relief = np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    relief[~mask] = 0.0
    return relief, mask


def depth_guided_warp(
    mesh: trimesh.Trimesh,
    relief: np.ndarray,
    mask: np.ndarray,
    target_depth: float
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """
    Devuelve warp(puntos) que engorda la malla en Z (frente +Z hacia la cámara) hasta ~target_depth:
    cada columna (x, y) crece según el relieve de la imagen y se redondea hacia los bordes de la
    silueta. El frente avanza y la espalda retrocede alrededor del plano medio de cada columna, de
    forma monótona en Z (no cruza superficies). Las coordenadas se toman de `mesh`, así que el mismo
    warp sirve para la malla HD que esté en su mismo marco.
    """
    import cv2
    from scipy import ndimage

    v = np.asarray(mesh.vertices, dtype=np.float64)
    lo = v.min(axis=0)
    ext = np.maximum(v.max(axis=0) - lo, 1e-9)
    growth = float(target_depth - ext[2])
    if growth <= 0:
        return None

    # Perfil de engorde: relieve de la imagen x caída suave hacia los bordes de la silueta
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    span = float(max(r1 - r0, c1 - c0, 1))
    edge = ndimage.distance_transform_edt(mask)
    falloff = np.sqrt(np.clip(edge / (0.08 * span), 0.0, 1.0))
    profile = (falloff * (0.35 + 0.65 * relief)).astype(np.float32)
    profile = cv2.GaussianBlur(profile, (0, 0), sigmaX=max(1.0, 0.01 * span))
    peak = float(profile.max())
    if peak < 1e-6:
        return None
    profile /= peak

    def sample_profile(p: np.ndarray) -> np.ndarray:
        # Encaje ortográfico: caja XY de la malla <-> caja de la silueta (X a la derecha, Y arriba)
        col = c0 + (p[:, 0] - lo[0]) / ext[0] * (c1 - c0)
        row = r1 - (p[:, 1] - lo[1]) / ext[1] * (r1 - r0)
        return ndimage.map_coordinates(profile, [row, col], order=1, mode="nearest")

    # Plano medio y medio grosor de cada columna (rejilla XY)
    grid = 96

    def cell(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = np.clip(((p[:, 0] - lo[0]) / ext[0] * grid).astype(np.int64), 0, grid - 1)
        gy = np.clip(((p[:, 1] - lo[1]) / ext[1] * grid).astype(np.int64), 0, grid - 1)
        return gy, gx

    gy, gx = cell(v)
    zmin = np.full((grid, grid), np.inf)
    zmax = np.full((grid, grid), -np.inf)
    np.minimum.at(zmin, (gy, gx), v[:, 2])
    np.maximum.at(zmax, (gy, gx), v[:, 2])
    empty = ~np.isfinite(zmin)
    if empty.any():
        nearest = ndimage.distance_transform_edt(empty, return_distances=False, return_indices=True)
        zmin, zmax = zmin[tuple(nearest)], zmax[tuple(nearest)]
    zmid = (zmin + zmax) / 2.0
    half = np.maximum((zmax - zmin) / 2.0, 1e-3 * float(ext.max()))

    def warp(points: np.ndarray) -> np.ndarray:
        p = np.array(points, dtype=np.float64, copy=True)
        cy, cx = cell(p)
        side = np.clip((p[:, 2] - zmid[cy, cx]) / half[cy, cx], -1.0, 1.0)
        p[:, 2] += side * (0.5 * growth) * sample_profile(p)
        return p

    return warp


def silhouette_iou(mesh: trimesh.Trimesh, mask: np.ndarray, res: int = 128) -> float:
    """
    Parecido entre la vista frontal de la malla (proyección en XY, frente +Z) y la silueta de la
    imagen: IoU de ambas encajadas en un cuadrado res x res conservando su proporción.
    """
    import cv2

    def fit(points_xy: np.ndarray, flip_y: bool) -> np.ndarray:
        lo, hi = points_xy.min(axis=0), points_xy.max(axis=0)
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-9))
        p = (points_xy - (lo + hi) / 2.0) / span * (res - 2) + res / 2.0
        if flip_y:
            p[:, 1] = res - p[:, 1]
        return p

    rows, cols = np.nonzero(mask)
    if len(rows) < 16 or len(mesh.faces) == 0:
        return 0.0
    # Silueta de la imagen (fila hacia abajo = Y imagen)
    target = np.zeros((res, res), np.uint8)
    pix = fit(np.column_stack([cols, rows]).astype(np.float64), flip_y=False)
    target[np.clip(pix[:, 1].astype(int), 0, res - 1), np.clip(pix[:, 0].astype(int), 0, res - 1)] = 1

    # Vista frontal de la malla (Y arriba): triángulos rellenos (frente y espalda se solapan: unión)
    projected = fit(np.asarray(mesh.vertices)[:, :2], flip_y=True)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if _rasterize_uv is not None:
        front = (_rasterize_uv(projected, faces, res)[0] >= 0).astype(np.uint8)
    else:
        front = np.zeros((res, res), np.uint8)
        for t in np.round(projected[faces]).astype(np.int32):
            cv2.fillConvexPoly(front, t, 1)

    union = np.logical_or(front, target).sum()
    return float(np.logical_and(front, target).sum() / union) if union else 0.0


# ==============================================================================
# Malla HD -> normal map de la malla ligera
# ==============================================================================

def follow_frame(points: np.ndarray, before: trimesh.Trimesh, after: trimesh.Trimesh) -> np.ndarray:
    """Aplica a `points` el centrado y la escala uniforme que llevaron `before` a `after`."""
    b0, b1 = before.bounds, after.bounds
    scale = float(np.max(b1[1] - b1[0]) / max(float(np.max(b0[1] - b0[0])), 1e-12))
    return (np.asarray(points, dtype=np.float64) - b0.mean(axis=0)) * scale + b1.mean(axis=0)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(length, 1e-12)


def _vertex_tangents(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, normals: np.ndarray):
    """Tangentes por vértice (dirección +U) y signo de la bitangente (dirección +V de la UV)."""
    p0, p1, p2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    w0, w1, w2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-12, 1.0 / np.where(det == 0, 1.0, det), 0.0)[:, None]
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    tan_u = np.zeros_like(vertices)
    tan_v = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(tan_u, faces[:, k], sdir)
        np.add.at(tan_v, faces[:, k], tdir)
    tangent = tan_u - normals * np.sum(normals * tan_u, axis=1, keepdims=True)
    # Vértices sin dirección U útil: cualquier perpendicular a la normal
    degenerate = np.linalg.norm(tangent, axis=1) < 1e-12
    if degenerate.any():
        helper = np.where(np.abs(normals[degenerate, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        tangent[degenerate] = np.cross(normals[degenerate], helper)
    tangent = _normalize(tangent)
    sign = np.where(np.sum(np.cross(normals, tangent) * tan_v, axis=1) < 0.0, -1.0, 1.0)
    return tangent, sign


def _smoothed_vertex_normals(mesh: trimesh.Trimesh, iterations: int) -> np.ndarray:
    """Normales de vértice promediadas `iterations` veces con sus vecinos (la geometría no se mueve)."""
    import scipy.sparse as sp
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64).copy()
    if iterations <= 0:
        return normals
    edges = mesh.edges_unique
    count = len(mesh.vertices)
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(count, count)
    ).tocsr() + sp.identity(count, format="csr")
    for _ in range(iterations):
        normals = _normalize(adjacency @ normals)
    return normals


def _hd_normals_at(
    points: np.ndarray,
    normals: np.ndarray,
    hd_mesh: trimesh.Trimesh,
    hd_vertex_normals: np.ndarray,
    reach: float
) -> np.ndarray:
    """
    Normal de la malla HD en cada punto de la malla ligera: rayo desde fuera (punto + normal*reach)
    hacia dentro con Embree; los fallos se resuelven con los vértices HD más cercanos que miran
    hacia el mismo lado. Devuelve ceros donde no hay una normal HD fiable.
    """
    result = np.zeros_like(points)
    found = np.zeros(len(points), dtype=bool)

    if trimesh.ray.has_embree:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        origins = points + normals * reach
        locations, ray_idx, tri_idx = RayMeshIntersector(hd_mesh).intersects_location(
            origins, -normals, multiple_hits=False
        )
        if len(ray_idx):
            close = np.linalg.norm(locations - origins[ray_idx], axis=1) <= 2.0 * reach
            ray_idx, tri_idx, locations = ray_idx[close], tri_idx[close], locations[close]
            bary = trimesh.triangles.points_to_barycentric(hd_mesh.triangles[tri_idx], locations)
            hit_normals = _normalize(np.einsum("nk,nkj->nj", bary, hd_vertex_normals[hd_mesh.faces[tri_idx]]))
            facing = np.sum(hit_normals * normals[ray_idx], axis=1) > 0.2
            result[ray_idx[facing]] = hit_normals[facing]
            found[ray_idx[facing]] = True

    rest = np.nonzero(~found)[0]
    if len(rest):
        from scipy.spatial import cKDTree
        dist, idx = cKDTree(hd_mesh.vertices).query(points[rest], k=8)
        candidates = hd_vertex_normals[idx]
        agree = np.einsum("mkj,mj->mk", candidates, normals[rest])
        weights = np.where(agree > 0.3, agree, 0.0) / (dist + 1e-9)
        result[rest] = np.einsum("mk,mkj->mj", weights, candidates)
    return result


def bake_normal_map(mesh: trimesh.Trimesh, hd_mesh: trimesh.Trimesh, size: Optional[int] = None) -> Optional[trimesh.Trimesh]:
    """
    Hornea el detalle de hd_mesh en un normal map de `mesh` (espacio tangente, convención glTF:
    X = +U, Y = +V de la imagen hacia arriba). `mesh` conserva sus polígonos; si no tiene UV se
    despliega con xatlas. Las normales de hd_mesh se filtran (NORMAL_DENOISE_ITERATIONS) para quitar
    el ruido del generador. Devuelve la malla con material PBR (color + normalTexture) o None si
    no aplica (sin Numba, o la malla HD no aporta detalle).
    """
    if _rasterize_uv is None:
        print(" -> [Normal map] Numba no disponible: se omite el horneado.")
        return None
    if len(hd_mesh.faces) < NORMAL_BAKE_MIN_GAIN * len(mesh.faces):
        print(f" -> [Normal map] La malla HD ({len(hd_mesh.faces):,} caras) no aporta detalle sobre "
              f"{len(mesh.faces):,} caras: se omite.")
        return None
    t0 = time.time()

    hd_normals = _smoothed_vertex_normals(hd_mesh, NORMAL_DENOISE_ITERATIONS)

    material = getattr(mesh.visual, "material", None)
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        print(f" -> [Normal map] Desplegando UV con xatlas ({len(mesh.faces):,} caras)...")
        unwrapped = mesh.unwrap()
        mesh, uv = unwrapped, unwrapped.visual.uv
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        pbr = material.copy()
    elif isinstance(material, trimesh.visual.material.SimpleMaterial):
        pbr = material.to_pbr()
    else:
        pbr = trimesh.visual.material.PBRMaterial(baseColorFactor=[0.82, 0.84, 0.88, 1.0])
    base_texture = pbr.baseColorTexture
    if size is None:
        size = max(base_texture.size) if base_texture is not None else NORMAL_MAP_SIZE
    size = int(size)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(uv, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tangents, signs = _vertex_tangents(vertices, faces, uv, normals)

    # Fila 0 = arriba de la imagen = V alta (trimesh guarda la V hacia arriba)
    uv_px = np.column_stack([uv[:, 0] * size, (1.0 - uv[:, 1]) * size]).astype(np.float64)
    face_id, bary = _rasterize_uv(uv_px, faces.astype(np.int32), size)
    covered = face_id >= 0
    if not covered.any():
        return None

    tri = faces[face_id[covered]]
    w = bary[covered].astype(np.float64)[:, :, None]
    points = np.sum(vertices[tri] * w, axis=1)
    n = _normalize(np.sum(normals[tri] * w, axis=1))
    t = np.sum(tangents[tri] * w, axis=1)
    t = _normalize(t - n * np.sum(n * t, axis=1, keepdims=True))
    b = np.where(np.sum(signs[tri] * w[:, :, 0], axis=1) < 0.0, -1.0, 1.0)[:, None] * np.cross(n, t)

    reach = 0.02 * float(np.max(mesh.extents))
    hd_n = _hd_normals_at(points, n, hd_mesh, hd_normals, reach)
    valid = np.sum(hd_n * n, axis=1) > 0.2
    hd_n = np.where(valid[:, None], _normalize(hd_n), n)

    local = np.column_stack([np.sum(hd_n * t, axis=1), np.sum(hd_n * b, axis=1), np.sum(hd_n * n, axis=1)])
    local[:, 2] = np.maximum(local[:, 2], 0.05)
    local = _normalize(local)
    noisy = float(np.mean(local[:, 2] < 0.9))
    local[:, :2] *= NORMAL_MAP_STRENGTH
    local = _normalize(local)

    normal_img = np.zeros((size, size, 3), dtype=np.float32)
    normal_img[..., 2] = 1.0
    normal_img[covered] = local
    # Márgenes de las islas UV: cada texel vacío copia el texel cubierto más cercano (sin costuras al filtrar)
    from scipy import ndimage
    nearest = ndimage.distance_transform_edt(~covered, return_distances=False, return_indices=True)
    normal_img = normal_img[nearest[0], nearest[1]]
    rgb = np.clip(np.round((normal_img * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)

    pbr.normalTexture = Image.fromarray(rgb)
    pbr.metallicFactor = 0.0
    pbr.roughnessFactor = 0.90
    pbr.doubleSided = True
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr)
    print(f" -> [Normal map] {size}x{size} horneado desde {len(hd_mesh.faces):,} caras HD sobre "
          f"{len(mesh.faces):,} caras ({valid.mean():.0%} texels con normal HD, {noisy:.0%} con relieve fuerte) "
          f"en {time.time() - t0:.1f}s")
    return mesh
