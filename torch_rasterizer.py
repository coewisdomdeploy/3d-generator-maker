#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Rasterizador de Triángulos en PyTorch (GPU) compatible con `custom_rasterizer`
==============================================================================
Sustituto sin compilar de la extensión CUDA `custom_rasterizer` de Hunyuan3D-2, que es
lo único que impide usar Hunyuan3D-Paint sin Visual Studio Build Tools + CUDA Toolkit.

Reproduce el kernel original (hy3dgen/texgen/custom_rasterizer):
  - Coordenadas de pantalla: x = (X/W*0.5+0.5)*(ancho-1)+0.5, y = (0.5+0.5*Y/W)*(alto-1)+0.5,
    z = Z/W*0.49999+0.5, muestreo en el centro del píxel (px+0.5, py+0.5).
  - Baricéntricas por cocientes de áreas con signo (sin descarte de caras traseras).
  - Z-buffer con token int64 = z_cuantizada * MAXINT + (cara + 1) y mínimo atómico
    (aquí scatter_reduce "amin"), z cuantizada con 2^18 como en el original.
  - Baricéntricas finales con corrección de perspectiva (divididas por W y normalizadas).
  - findices = índice de cara + 1 (0 = fondo).

Trabaja por bloques de píxeles candidatos para acotar la VRAM temporal (~200 MB).
"""

import torch

_MAXINT = 2147483647
_Z_SCALE = 2 << 17                      # 262144, igual que el kernel CUDA original
_EMPTY = 1 << 62                        # Valor inicial del z-buffer (fondo)
_CANDIDATE_BUDGET = 4_000_000           # Píxeles candidatos por bloque


def _screen_coords(V: torch.Tensor, width: int, height: int):
    w = V[:, 3]
    x = (V[:, 0] / w * 0.5 + 0.5) * (width - 1) + 0.5
    y = (0.5 + 0.5 * V[:, 1] / w) * (height - 1) + 0.5
    z = V[:, 2] / w * 0.49999 + 0.5
    return x, y, z, w


def _signed_area(x0, y0, x1, y1, x2, y2):
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def _barycentric(ax, ay, bx, by, cx, cy, px, py):
    """Pesos (a, b, c) del punto p; NaN/inf en triángulos degenerados (fallan el test)."""
    inv = 1.0 / _signed_area(ax, ay, bx, by, cx, cy)
    beta = _signed_area(ax, ay, px, py, cx, cy) * inv
    gamma = _signed_area(ax, ay, bx, by, px, py) * inv
    return 1.0 - beta - gamma, beta, gamma


def rasterize(pos: torch.Tensor, tri: torch.Tensor, resolution, clamp_depth=None, use_depth_prior=0):
    """
    pos: (1, N, 4) o (N, 4) posiciones en clip space; tri: (F, 3) índices; resolution: (alto, ancho).
    Devuelve findices (alto, ancho) int32 y barycentric (alto, ancho, 3) float32.
    """
    height, width = int(resolution[0]), int(resolution[1])
    V = (pos[0] if pos.dim() == 3 else pos).float()
    F = tri.long()
    device = V.device

    x, y, z, w = _screen_coords(V, width, height)
    fx, fy, fz = x[F], y[F], z[F]                       # (F, 3)

    # Rango de píxeles cuyo centro (p + 0.5) cae dentro de la caja del triángulo
    px0 = torch.ceil(fx.min(1).values - 0.5)
    px1 = torch.floor(fx.max(1).values - 0.5)
    py0 = torch.ceil(fy.min(1).values - 0.5)
    py1 = torch.floor(fy.max(1).values - 0.5)
    finite = torch.isfinite(px0) & torch.isfinite(px1) & torch.isfinite(py0) & torch.isfinite(py1)
    px0 = torch.where(finite, px0, torch.zeros_like(px0)).clamp(0, width - 1).long()
    py0 = torch.where(finite, py0, torch.zeros_like(py0)).clamp(0, height - 1).long()
    px1 = torch.where(finite, px1, torch.full_like(px1, -1)).clamp(-1, width - 1).long()
    py1 = torch.where(finite, py1, torch.full_like(py1, -1)).clamp(-1, height - 1).long()
    bw = (px1 - px0 + 1).clamp(min=0)
    bh = (py1 - py0 + 1).clamp(min=0)
    counts_all = bw * bh

    zbuf = torch.full((height * width,), _EMPTY, dtype=torch.int64, device=device)
    faces = torch.nonzero(counts_all > 0).squeeze(1)
    if faces.numel():
        cum = torch.cumsum(counts_all[faces], 0)
        total = int(cum[-1])
        start_face = 0
        start_count = 0
        while start_face < faces.numel():
            # Último índice cuyo acumulado no supera el presupuesto (al menos una cara por bloque)
            limit = start_count + _CANDIDATE_BUDGET
            end_face = int(torch.searchsorted(cum, torch.tensor(limit, device=device), right=True))
            end_face = max(end_face, start_face + 1)
            chunk = faces[start_face:end_face]
            counts = counts_all[chunk]

            rep = torch.repeat_interleave(chunk, counts)
            offsets = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            local = torch.arange(rep.numel(), device=device) - offsets
            rbw = bw[rep]
            px = px0[rep] + local % rbw
            py = py0[rep] + torch.div(local, rbw, rounding_mode="floor")
            pxc = px.float() + 0.5
            pyc = py.float() + 0.5

            a, b, c = _barycentric(fx[rep, 0], fy[rep, 0], fx[rep, 1], fy[rep, 1], fx[rep, 2], fy[rep, 2], pxc, pyc)
            inside = (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1) & (c >= 0) & (c <= 1)
            if inside.any():
                rep, px, py = rep[inside], px[inside], py[inside]
                depth = a[inside] * fz[rep, 0] + b[inside] * fz[rep, 1] + c[inside] * fz[rep, 2]
                zq = (depth * _Z_SCALE).to(torch.int64)          # Truncado hacia cero, como en C
                token = zq * _MAXINT + (rep + 1)
                zbuf.scatter_reduce_(0, py * width + px, token, reduce="amin", include_self=True)

            start_count = int(cum[end_face - 1])
            start_face = end_face
            if start_count >= total:
                break

    valid = zbuf != _EMPTY
    face1 = torch.remainder(zbuf, _MAXINT)
    findices = torch.where(valid, face1, torch.zeros_like(face1)).to(torch.int32).view(height, width)

    bary = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    pix = torch.nonzero(valid).squeeze(1)
    if pix.numel():
        f = face1[pix] - 1
        pxc = (pix % width).float() + 0.5
        pyc = torch.div(pix, width, rounding_mode="floor").float() + 0.5
        a, b, c = _barycentric(fx[f, 0], fy[f, 0], fx[f, 1], fy[f, 1], fx[f, 2], fy[f, 2], pxc, pyc)
        # Corrección de perspectiva
        wf = w[F[f]]
        a, b, c = a / wf[:, 0], b / wf[:, 1], c / wf[:, 2]
        norm = 1.0 / (a + b + c)
        bary[pix] = torch.stack((a * norm, b * norm, c * norm), dim=1)
    return findices, bary.view(height, width, 3)


def interpolate(col: torch.Tensor, findices: torch.Tensor, barycentric: torch.Tensor, tri: torch.Tensor) -> torch.Tensor:
    """Interpola atributos por vértice con las baricéntricas (misma semántica que custom_rasterizer)."""
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = torch.sum(barycentric.view(*barycentric.shape, 1) * vcol, dim=-2)
    return result.view(1, *result.shape)
