#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Perfiles de Exportación (estilo Meshy): Modelos 3D o Impresión 3D
==============================================================================
- Modelos 3D: formatos GLB / OBJ / PLY / STL, origen (base o centro), altura en
  metros y tamaño máximo de textura. Y-up (Godot, Unity, Unreal, Blender, web).
- Impresión 3D: impresora seleccionable (volumen de impresión), altura en mm,
  ajuste automático al volumen, reparación de malla, Z-up apoyado en la cama y
  apertura directa en el slicer instalado (Bambu Studio, PrusaSlicer, Orca...).
==============================================================================
"""

import collections
import json
import os
import subprocess
import sys
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import trimesh

# ==============================================================================
# 1. Catálogo de impresoras y slicers
# ==============================================================================

CUSTOM_PRINTER = "Personalizada"


def _printer(bed, tech="FDM", motion="CoreXY", enclosed=False, multicolor=None, slicer="orca"):
    """bed: (X, Y, Z) mm · motion: CoreXY / Cama móvil / CoreXZ / Resina · multicolor: sistema o None."""
    return {"bed": tuple(bed), "tech": tech, "motion": motion, "enclosed": enclosed,
            "multicolor": multicolor, "slicer": slicer}


def _resin(bed, slicer="chitubox"):
    return _printer(bed, tech="Resina", motion="Resina (LCD)", enclosed=True, slicer=slicer)


# Volúmenes de impresión según las fichas de los fabricantes (verificados en octubre 2026).
# Las velocidades no se incluyen: las fichas mezclan velocidad de impresión y de desplazamiento.
PRINTERS = {
    # Bambu Lab
    "Bambu Lab X1 Carbon": _printer((256, 256, 256), enclosed=True, multicolor="AMS (opcional)", slicer="bambu"),
    "Bambu Lab P1S": _printer((256, 256, 256), enclosed=True, multicolor="AMS (opcional)", slicer="bambu"),
    "Bambu Lab P1P": _printer((256, 256, 256), multicolor="AMS (opcional)", slicer="bambu"),
    "Bambu Lab P2S": _printer((256, 256, 256), enclosed=True, multicolor="AMS 2 Pro (opcional)", slicer="bambu"),
    "Bambu Lab A1": _printer((256, 256, 256), motion="Cama móvil", multicolor="AMS lite (opcional)", slicer="bambu"),
    "Bambu Lab A1 mini": _printer((180, 180, 180), motion="Cama móvil", multicolor="AMS lite (opcional)", slicer="bambu"),
    "Bambu Lab H2D": _printer((325, 320, 325), enclosed=True, multicolor="Doble boquilla + AMS", slicer="bambu"),
    "Bambu Lab H2S": _printer((340, 320, 340), enclosed=True, multicolor="AMS (opcional)", slicer="bambu"),
    # Prusa
    "Prusa MK4S": _printer((250, 210, 220), motion="Cama móvil", multicolor="MMU3 (opcional)", slicer="prusa"),
    "Prusa MK3S+": _printer((250, 210, 210), motion="Cama móvil", multicolor="MMU3 (opcional)", slicer="prusa"),
    "Prusa CORE One": _printer((250, 220, 270), enclosed=True, multicolor="MMU3 (opcional)", slicer="prusa"),
    "Prusa MINI+": _printer((180, 180, 180), motion="Cama móvil", slicer="prusa"),
    "Prusa XL": _printer((360, 360, 360), multicolor="Cambio de herramienta (hasta 5)", slicer="prusa"),
    # Creality
    "Creality Ender-3 V3": _printer((220, 220, 250), motion="CoreXZ", slicer="creality"),
    "Creality Ender-3 V3 SE": _printer((220, 220, 250), motion="Cama móvil", slicer="creality"),
    "Creality Ender-3 V3 KE": _printer((220, 220, 240), motion="Cama móvil", slicer="creality"),
    "Creality K1C": _printer((220, 220, 250), enclosed=True, slicer="creality"),
    "Creality K1 Max": _printer((300, 300, 300), enclosed=True, slicer="creality"),
    "Creality K2 Plus": _printer((350, 350, 350), enclosed=True, multicolor="CFS (opcional)", slicer="creality"),
    "Creality Hi": _printer((260, 260, 300), motion="Cama móvil", multicolor="CFS (versión Combo)", slicer="creality"),
    # Flashforge
    "Flashforge Adventurer 5M (AD5M)": _printer((220, 220, 220), slicer="flashforge"),
    "Flashforge Adventurer 5M Pro (AD5M Pro)": _printer((220, 220, 220), enclosed=True, slicer="flashforge"),
    "Flashforge AD5X": _printer((220, 220, 220), multicolor="IFS 4 colores (integrado)", slicer="flashforge"),
    # Elegoo
    "Elegoo Centauri Carbon": _printer((256, 256, 256), enclosed=True, slicer="elegoo"),
    "Elegoo Neptune 4 Pro": _printer((225, 225, 265), motion="Cama móvil", slicer="elegoo"),
    "Elegoo Neptune 4 Plus": _printer((320, 320, 385), motion="Cama móvil", slicer="elegoo"),
    "Elegoo Neptune 4 Max": _printer((420, 420, 480), motion="Cama móvil", slicer="elegoo"),
    "Elegoo Saturn 4 Ultra": _resin((218.88, 122.88, 220)),
    "Elegoo Saturn 4 Ultra 16K": _resin((211.68, 118.37, 220)),
    "Elegoo Mars 5 Ultra": _resin((153.36, 77.76, 165)),
    # Anycubic
    "Anycubic Kobra 3": _printer((250, 250, 260), motion="Cama móvil", multicolor="ACE Pro (versión Combo)", slicer="anycubic"),
    "Anycubic Kobra 2 Neo": _printer((220, 220, 250), motion="Cama móvil", slicer="anycubic"),
    "Anycubic Kobra S1": _printer((250, 250, 250), enclosed=True, multicolor="ACE Pro (versión Combo)", slicer="anycubic"),
    "Anycubic Photon Mono M7 Pro": _resin((223, 126, 230)),
    # QIDI
    "QIDI Q1 Pro": _printer((245, 245, 240), enclosed=True, slicer="qidi"),
    "QIDI Plus4": _printer((305, 305, 280), enclosed=True, slicer="qidi"),
    # Sovol
    "Sovol SV06": _printer((220, 220, 250), motion="Cama móvil", slicer="orca"),
    "Sovol SV08": _printer((350, 350, 345), slicer="orca"),
    CUSTOM_PRINTER: _printer((220, 220, 250), motion="—", slicer="orca"),
}

# clave: (nombre visible, carpetas de instalación con comodines, palabras del .exe)
# Se busca en Program Files, Program Files (x86) y AppData\Local\Programs
SLICERS = {
    "bambu": ("Bambu Studio", ["Bambu Studio*"], ["bambu-studio"]),
    "orca": ("OrcaSlicer", ["OrcaSlicer*"], ["orca-slicer"]),
    "prusa": ("PrusaSlicer", ["Prusa3D/PrusaSlicer*", "PrusaSlicer*"], ["prusa-slicer"]),
    "creality": ("Creality Print", ["Creality/Creality Print*", "Creality Print*"], ["crealityprint", "creality print"]),
    "flashforge": ("Orca-Flashforge", ["Orca-Flashforge*", "Flashforge/Orca-Flashforge*", "Flash Studio*",
                                       "Flashforge/Flash Studio*"], ["orca-flashforge", "flashstudio", "flash studio"]),
    "flashprint": ("FlashPrint 5", ["FlashForge/FlashPrint*", "FlashPrint*"], ["flashprint"]),
    "elegoo": ("ElegooSlicer", ["ElegooSlicer*", "ELEGOO/ElegooSlicer*", "Elegoo Slicer*"], ["elegoo"]),
    "anycubic": ("Anycubic Slicer Next", ["AnycubicSlicerNext*", "Anycubic/AnycubicSlicer*", "AnycubicSlicer*"], ["anycubic"]),
    "qidi": ("QIDI Studio", ["QIDIStudio*", "QIDI*/QIDIStudio*", "QIDI Studio*"], ["qidi"]),
    "cura": ("UltiMaker Cura", ["UltiMaker Cura*", "Ultimaker Cura*"], ["ultimaker-cura", "cura"]),
    "chitubox": ("CHITUBOX", ["CHITUBOX*", "ChituBox*"], ["chitubox"]),
}
_EXE_SKIP_WORDS = ("unins", "uninstall", "crash", "update", "console", "helper", "setup", "elevate")
SLICER_FALLBACK = {
    "flashforge": ["flashprint", "orca"],
    "elegoo": ["orca"],
    "anycubic": ["orca"],
    "qidi": ["orca"],
    "creality": ["orca"],
}
SLICER_FALLBACK_FDM = ["orca", "prusa", "bambu", "cura", "creality"]
SLICER_FALLBACK_RESIN = ["chitubox"]

MODEL_FORMATS = ["glb", "obj", "ply", "stl"]
PRINT_FORMATS = ["3mf", "stl", "obj"]
ALL_FORMATS = ["glb", "obj", "ply", "stl", "3mf"]

DEFAULT_SETTINGS = {
    "mode": "modelos",            # "modelos" | "impresion"
    "model_formats": ["glb"],
    "model_pivot": "base",        # "base" | "centro"
    "model_height_m": 1.0,
    "texture_max": 2048,
    "print_printer": "Bambu Lab P1S",
    "print_custom_bed": [220, 220, 250],
    "print_height_mm": 80.0,
    "print_fit_bed": True,
    "print_repair": True,
    "print_formats": ["3mf", "stl"],
    "print_slicer": "auto",       # "auto" | clave de SLICERS | "default"
    "print_slicer_path": "",      # ejecutable elegido a mano (tiene prioridad si existe)
    "print_colors": 0,            # filamentos/colores a exportar; 0 = automático según la impresora
}


# ==============================================================================
# 2. Persistencia de la configuración
# ==============================================================================

def normalize_settings(raw: Optional[dict]) -> dict:
    """
    Completa con los valores por defecto y descarta claves o valores inválidos
    (la configuración puede venir del localStorage del navegador).
    """
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    if not isinstance(raw, dict):
        return settings
    for key, default in DEFAULT_SETTINGS.items():
        value = raw.get(key)
        if value is None or type(value) is not type(default) and not (
            isinstance(default, float) and isinstance(value, int)
        ):
            continue
        settings[key] = float(value) if isinstance(default, float) else value
    if settings["mode"] not in ("modelos", "impresion"):
        settings["mode"] = DEFAULT_SETTINGS["mode"]
    settings["model_formats"] = [f for f in settings["model_formats"] if f in MODEL_FORMATS] or ["glb"]
    settings["print_formats"] = [f for f in settings["print_formats"] if f in PRINT_FORMATS] or ["3mf"]
    if settings["model_pivot"] not in ("base", "centro"):
        settings["model_pivot"] = "base"
    if settings["print_printer"] not in PRINTERS:
        settings["print_printer"] = DEFAULT_SETTINGS["print_printer"]
    if settings["print_slicer"] not in ("auto", "default", *SLICERS):
        settings["print_slicer"] = "auto"
    bed = settings["print_custom_bed"]
    if len(bed) != 3 or not all(isinstance(v, (int, float)) and v > 0 for v in bed):
        settings["print_custom_bed"] = list(DEFAULT_SETTINGS["print_custom_bed"])
    return settings


def load_settings(path: Path) -> dict:
    """Lee la configuración guardada en disco y la completa con los valores por defecto."""
    try:
        return normalize_settings(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return normalize_settings(None)


def save_settings(path: Path, settings: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def active_formats(settings: dict) -> list:
    """Formatos activos para el modo actual, en el orden canónico de ALL_FORMATS."""
    key = "print_formats" if settings["mode"] == "impresion" else "model_formats"
    chosen = set(settings.get(key) or [])
    return [f for f in ALL_FORMATS if f in chosen]


def printer_dims(settings: dict) -> tuple:
    """(X, Y, Z, tecnología, slicer preferido) de la impresora elegida."""
    name = settings.get("print_printer", CUSTOM_PRINTER)
    spec = PRINTERS.get(name, PRINTERS[CUSTOM_PRINTER])
    (x, y, z), tech, slicer = spec["bed"], spec["tech"], spec["slicer"]
    if name == CUSTOM_PRINTER:
        x, y, z = [float(v) for v in settings.get("print_custom_bed", [x, y, z])]
    return x, y, z, tech, slicer


def printer_features(name: str) -> list:
    """Características legibles de la impresora para mostrar en la configuración."""
    spec = PRINTERS.get(name, PRINTERS[CUSTOM_PRINTER])
    if name == CUSTOM_PRINTER:
        return []
    features = [spec["tech"]]
    if spec["tech"] != "Resina":
        features.append(spec["motion"])
        features.append("Cerrada" if spec["enclosed"] else "Abierta")
    if spec["multicolor"]:
        features.append(f"Multicolor: {spec['multicolor']}")
    return features


def profile_label(settings: dict) -> str:
    """Texto corto del perfil activo para el botón de configuración."""
    if settings["mode"] == "impresion":
        return f"Impresión · {settings.get('print_printer', CUSTOM_PRINTER)}"
    return "Modelos 3D · " + " ".join(f.upper() for f in active_formats(settings))


# ==============================================================================
# 3. Detección y apertura de slicers
# ==============================================================================

def _search_roots() -> list:
    roots = [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]
    if os.environ.get("LOCALAPPDATA"):
        roots.append(str(Path(os.environ["LOCALAPPDATA"]) / "Programs"))
    return [Path(r) for r in roots if r and Path(r).exists()]


@lru_cache(maxsize=1)
def detect_slicers() -> dict:
    """{clave: ruta del ejecutable} de los slicers instalados (se busca una sola vez)."""
    found = {}
    for key, (_, folders, words) in SLICERS.items():
        for root in _search_roots():
            for folder in folders:
                for install_dir in sorted(root.glob(folder), reverse=True):  # versión más reciente primero
                    exes = [e for e in install_dir.glob("*.exe")
                            if any(w in e.name.lower() for w in words)
                            and not any(s in e.name.lower() for s in _EXE_SKIP_WORDS)]
                    if exes:
                        found[key] = str(exes[0])
                        break
                if key in found:
                    break
            if key in found:
                break
    return found


def resolve_slicer(settings: dict) -> tuple:
    """(clave, nombre visible, ejecutable o None). None = programa predeterminado del sistema."""
    custom = str(settings.get("print_slicer_path") or "").strip().strip('"')
    if custom and Path(custom).is_file():
        return "custom", Path(custom).stem, custom
    choice = settings.get("print_slicer", "auto")
    installed = detect_slicers()
    if choice == "default":
        return "default", "programa predeterminado", None
    if choice != "auto":
        return choice, SLICERS.get(choice, (choice,))[0], installed.get(choice)
    _, _, _, tech, preferred = printer_dims(settings)
    fallback = SLICER_FALLBACK_RESIN if tech == "Resina" else SLICER_FALLBACK_FDM
    for key in [preferred] + SLICER_FALLBACK.get(preferred, []) + fallback:
        if key in installed:
            return key, SLICERS[key][0], installed[key]
    return "default", "programa predeterminado", None


def open_in_slicer(file_path: Union[str, Path], settings: dict) -> str:
    """Abre el archivo en el slicer configurado. Retorna el nombre del programa usado."""
    _, name, exe = resolve_slicer(settings)
    path = str(Path(file_path).resolve())
    if exe:
        subprocess.Popen([exe, path])
    elif sys.platform == "win32":
        os.startfile(path)
    else:
        subprocess.Popen(["xdg-open", path])
    return name


# ==============================================================================
# 4. Transformaciones por perfil
# ==============================================================================

def _load_mesh(source: Union[str, Path, trimesh.Trimesh, trimesh.Scene]) -> trimesh.Trimesh:
    if isinstance(source, (str, Path)):
        return trimesh.load(str(source), force="mesh")
    if isinstance(source, trimesh.Scene):
        return source.to_geometry()
    return source.copy()


def _limit_textures(mesh: trimesh.Trimesh, max_size: int) -> None:
    """Reduce las texturas del material a max_size px (lado mayor) si son más grandes."""
    material = getattr(getattr(mesh, "visual", None), "material", None)
    if material is None or not max_size:
        return
    for attr in ("baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                 "emissiveTexture", "occlusionTexture", "image"):
        img = getattr(material, attr, None)
        if img is not None and hasattr(img, "size") and max(img.size) > max_size:
            img = img.copy()
            img.thumbnail((max_size, max_size))
            setattr(material, attr, img)


def prepare_model_mesh(mesh: trimesh.Trimesh, settings: dict) -> trimesh.Trimesh:
    """Modelos 3D: Y-up, altura en metros y origen en la base o el centro."""
    mesh = mesh.copy()
    height = float(settings.get("model_height_m") or 0)
    if height > 0 and mesh.extents[1] > 1e-6:
        mesh.apply_scale(height / mesh.extents[1])
    b = mesh.bounds
    cx, cz = (b[0][0] + b[1][0]) / 2.0, (b[0][2] + b[1][2]) / 2.0
    cy = b[0][1] if settings.get("model_pivot") == "base" else (b[0][1] + b[1][1]) / 2.0
    mesh.apply_translation([-cx, -cy, -cz])
    _limit_textures(mesh, int(settings.get("texture_max") or 0))
    return mesh


# ------------------------------------------------------------------------------
# Color para impresión multicolor: textura -> colores por cara -> N filamentos
# ------------------------------------------------------------------------------

def print_color_count(settings: dict) -> int:
    """Número de colores (filamentos) a exportar. 0 = automático según la impresora."""
    n = int(settings.get("print_colors") or 0)
    if n > 0:
        return max(1, min(n, 16))
    spec = PRINTERS.get(settings.get("print_printer"), PRINTERS[CUSTOM_PRINTER])
    if not spec["multicolor"]:
        return 1
    return 5 if "hasta 5" in spec["multicolor"] else 4


def _face_colors(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    """Color RGB (0-1) de cada cara: textura (promedio de 4 muestras), color de vértice o de cara."""
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    img = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    uv = getattr(visual, "uv", None)
    faces = mesh.faces
    if img is not None and uv is not None and len(uv) == len(mesh.vertices):
        tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        th, tw = tex.shape[:2]
        fuv = np.asarray(uv)[faces]                                   # (F, 3, 2)
        samples = np.concatenate([fuv, fuv.mean(axis=1, keepdims=True)], axis=1)
        tx = np.clip((samples[..., 0] % 1.0) * (tw - 1), 0, tw - 1).astype(np.int64)
        ty = np.clip((1.0 - samples[..., 1] % 1.0) * (th - 1), 0, th - 1).astype(np.int64)
        rgb = tex[ty, tx].mean(axis=1)
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None and len(factor) >= 3:
            f = np.asarray(factor[:3], dtype=np.float32)
            rgb = rgb * (f / 255.0 if f.max() > 1.0 else f)
        return np.clip(rgb, 0, 1)
    kind = getattr(visual, "kind", None)
    if kind == "vertex":
        vc = np.asarray(visual.vertex_colors[:, :3], dtype=np.float32) / 255.0
        return vc[faces].mean(axis=1)
    if kind == "face":
        return np.asarray(visual.face_colors[:, :3], dtype=np.float32) / 255.0
    return None


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-1) -> CIELAB (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def _quantize_colors(colors: np.ndarray, weights: np.ndarray, k: int, iterations: int = 20) -> tuple:
    """
    k-means ponderado por área en espacio Lab con la luminosidad atenuada: la sombra horneada
    en la textura no debe partir un mismo color en dos filamentos.
    Retorna (etiqueta por cara, paleta RGB (k, 3)) con el filamento 1 = color dominante.
    """
    feats = _rgb_to_lab(colors) * np.array([0.45, 1.0, 1.0])
    rng = np.random.default_rng(0)
    sample, sw = feats, weights
    if len(feats) > 60000:
        idx = rng.choice(len(feats), 60000, replace=False, p=weights / weights.sum())
        sample, sw = feats[idx], np.ones(len(idx))
    centers = [sample[rng.choice(len(sample), p=sw / sw.sum())]]
    for _ in range(1, k):
        d = np.min(((sample[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1) * sw
        if d.sum() <= 1e-9:
            break
        centers.append(sample[rng.choice(len(sample), p=d / d.sum())])
    centers = np.array(centers)
    for _ in range(iterations):
        labels = np.argmin(((sample[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
        for c in range(len(centers)):
            m = labels == c
            if m.any():
                centers[c] = np.average(sample[m], axis=0, weights=sw[m])
    labels = np.argmin(((feats[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    area = np.bincount(labels, weights=weights, minlength=len(centers))
    order = [i for i in np.argsort(-area) if area[i] > 0]
    remap = np.zeros(len(centers), dtype=np.int64)
    remap[order] = np.arange(len(order))
    labels = remap[labels]
    palette = np.array([np.average(colors[labels == i], axis=0, weights=weights[labels == i])
                        for i in range(len(order))])
    return labels, palette


def _smooth_labels(mesh: trimesh.Trimesh, labels: np.ndarray, k: int,
                   iterations: int = 6, min_island: float = 0.004) -> np.ndarray:
    """
    Limpieza de regiones de color para imprimir:
    1) filtro de moda: una cara toma el color de 2+ vecinas que coinciden entre sí
    2) islas conectadas con menos de min_island del área total se funden con su vecino dominante
    Evita cambios de filamento por manchas sueltas.
    """
    adj = mesh.face_adjacency
    if len(adj) == 0:
        return labels
    labels = labels.copy()
    n = len(labels)
    rows = np.arange(n)
    for _ in range(iterations):
        counts = np.zeros((n, k), dtype=np.int32)
        np.add.at(counts, (adj[:, 0], labels[adj[:, 1]]), 1)
        np.add.at(counts, (adj[:, 1], labels[adj[:, 0]]), 1)
        best = counts.argmax(axis=1)
        switch = (counts[rows, best] >= 2) & (best != labels) & (counts[rows, best] > counts[rows, labels])
        if not switch.any():
            break
        labels[switch] = best[switch]

    areas = mesh.area_faces
    limit = float(areas.sum()) * min_island
    for _ in range(3):
        same = labels[adj[:, 0]] == labels[adj[:, 1]]
        comps = trimesh.graph.connected_components(adj[same], nodes=rows, min_len=1)
        comp_id = np.empty(n, dtype=np.int64)
        for ci, faces_in in enumerate(comps):
            comp_id[faces_in] = ci
        comp_area = np.bincount(comp_id, weights=areas, minlength=len(comps))
        small = comp_area < limit
        if not small.any():
            break
        # Vecino dominante de cada isla: color al otro lado del borde, ponderado por área
        border = ~same
        a, b = adj[border, 0], adj[border, 1]
        votes = collections.defaultdict(lambda: np.zeros(k))
        for fa, fb in ((a, b), (b, a)):
            sel = small[comp_id[fa]]
            for c, lab, w in zip(comp_id[fa][sel], labels[fb][sel], areas[fb][sel]):
                votes[c][lab] += w
        changed = False
        for c, v in votes.items():
            target = int(v.argmax())
            faces_in = comps[c]
            if target != labels[faces_in[0]]:
                labels[faces_in] = target
                changed = True
        if not changed:
            break
    return labels


def prepare_print_mesh(mesh: trimesh.Trimesh, settings: dict) -> tuple:
    """
    Impresión 3D: geometría reparada, Z-up, en milímetros, apoyada en la cama y centrada
    en el volumen de la impresora. Con 2+ colores, la textura se convierte en N colores
    por cara (uno por filamento). Retorna (malla, info) con info["labels"] / info["palette"].
    """
    k = print_color_count(settings)
    face_rgb = _face_colors(mesh) if k > 1 else None
    labels = palette = None
    if face_rgb is not None:
        labels, palette = _quantize_colors(face_rgb, np.maximum(mesh.area_faces, 1e-12), k)

    geo = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.faces), process=True)
    if labels is not None:
        # El color viaja como color de cara para sobrevivir a la reparación (que borra o añade caras)
        rgba = np.c_[(palette[labels] * 255).round(), np.full(len(labels), 255)].astype(np.uint8)
        geo.visual = trimesh.visual.ColorVisuals(geo, face_colors=rgba)
    if settings.get("print_repair", True):
        geo.update_faces(geo.nondegenerate_faces())
        geo.update_faces(geo.unique_faces())
        geo.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(geo)
        trimesh.repair.fix_normals(geo)
    if labels is not None:
        fc = np.asarray(geo.visual.face_colors[:, :3], dtype=np.float32) / 255.0
        labels = np.argmin(((fc[:, None, :] - palette[None]) ** 2).sum(-1), axis=1)
        labels = _smooth_labels(geo, labels, len(palette))

    # glTF es Y-up; las impresoras trabajan en Z-up (la cabeza pasa a +Z, el frente a -Y)
    geo.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))

    bx, by, bz, _, _ = printer_dims(settings)
    ex, ey, ez = [max(float(e), 1e-6) for e in geo.extents]
    scale = float(settings.get("print_height_mm") or 80.0) / ez
    fitted = False
    if settings.get("print_fit_bed", True):
        margin = 5.0  # mm libres a cada lado de la cama
        limit = min((bx - 2 * margin) / ex, (by - 2 * margin) / ey, (bz - 1.0) / ez)
        if scale > limit:
            scale, fitted = limit, True
    geo.apply_scale(scale)

    b = geo.bounds
    geo.apply_translation([
        bx / 2.0 - (b[0][0] + b[1][0]) / 2.0,
        by / 2.0 - (b[0][1] + b[1][1]) / 2.0,
        -b[0][2],
    ])
    info = {
        "size_mm": tuple(round(float(e), 1) for e in geo.extents),
        "fitted": fitted,
        "watertight": bool(geo.is_watertight),
        "labels": labels,
        "palette": palette,
        "colors_requested": k,
    }
    return geo, info


# ==============================================================================
# 5. Exportación
# ==============================================================================

def _export_obj(mesh: trimesh.Trimesh, out_base: Path) -> Path:
    """OBJ simple; si lleva textura se empaqueta .obj + .mtl + imágenes en un .zip."""
    mtl_name = f"{out_base.name}.mtl"
    text, extra = trimesh.exchange.obj.export_obj(
        mesh, include_texture=True, return_texture=True, mtl_name=mtl_name
    )
    if not extra:
        out_file = out_base.with_suffix(".obj")
        out_file.write_text(text, encoding="utf-8")
        return out_file
    out_file = out_base.parent / f"{out_base.name}_obj.zip"
    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{out_base.name}.obj", text)
        for name, data in extra.items():
            zf.writestr(name, data)
    return out_file


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(round(float(c) * 255)):02X}" for c in rgb)


def _paint_code(extruder: int) -> str:
    """
    Código de pintura por triángulo de PrusaSlicer/Bambu/Orca (TriangleSelector serializado)
    para un triángulo sin subdividir: filamento 1 -> "4", 2 -> "8", 3 -> "0C", 4 -> "1C"...
    """
    if extruder <= 2:
        return "4" if extruder == 1 else "8"
    return f"{extruder - 3:X}C"


def _export_3mf_colored(mesh: trimesh.Trimesh, labels: np.ndarray, palette: np.ndarray, out_file: Path) -> Path:
    """
    3MF en milímetros con cada triángulo pintado por filamento:
    - paint_color: lo leen Bambu Studio, OrcaSlicer y derivados (Orca-Flashforge, ElegooSlicer...)
    - slic3rpe:mmu_segmentation: lo lee PrusaSlicer
    - basematerials (estándar 3MF): colores de visualización para el resto de programas
    """
    codes = [_paint_code(i + 1) for i in range(len(palette))]
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
        'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
        ' <metadata name="Application">3D AI Studio</metadata>\n'
        ' <resources>\n  <basematerials id="1">\n'
    ]
    parts += [f'   <base name="Filamento {i + 1}" displaycolor="{_hex(c)}FF"/>\n' for i, c in enumerate(palette)]
    parts.append('  </basematerials>\n  <object id="2" type="model" pid="1" pindex="0">\n   <mesh>\n    <vertices>\n')
    parts += [f'     <vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>\n' for x, y, z in mesh.vertices]
    parts.append('    </vertices>\n    <triangles>\n')
    parts += [
        f'     <triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{l}" '
        f'paint_color="{codes[l]}" slic3rpe:mmu_segmentation="{codes[l]}"/>\n'
        for (a, b, c), l in zip(mesh.faces, labels)
    ]
    parts.append('    </triangles>\n   </mesh>\n  </object>\n </resources>\n'
                 ' <build>\n  <item objectid="2"/>\n </build>\n</model>\n')
    content_types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
                     '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", "".join(parts))
    return out_file


def _export_obj_colored(mesh: trimesh.Trimesh, labels: np.ndarray, palette: np.ndarray, out_base: Path) -> Path:
    """
    OBJ + MTL con un material por filamento (Bambu Studio y Orca muestran el diálogo de
    asignación de colores al importarlo). Quedan .obj/.mtl sueltos para abrir en el slicer
    y un .zip con ambos para la descarga.
    """
    mtl_name = f"{out_base.name}.mtl"
    mtl = "".join(f"newmtl filamento_{i + 1}\nKd {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n\n" for i, c in enumerate(palette))
    lines = [f"mtllib {mtl_name}\n"]
    lines += [f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in mesh.vertices]
    for i in range(len(palette)):
        lines.append(f"usemtl filamento_{i + 1}\n")
        lines += [f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in mesh.faces[labels == i]]
    obj_text = "".join(lines)
    out_base.with_suffix(".obj").write_text(obj_text, encoding="utf-8")
    (out_base.parent / mtl_name).write_text(mtl, encoding="utf-8")
    out_zip = out_base.parent / f"{out_base.name}_obj.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{out_base.name}.obj", obj_text)
        zf.writestr(mtl_name, mtl)
    return out_zip


def export_with_profile(
    source: Union[str, Path, trimesh.Trimesh, trimesh.Scene],
    out_dir: Union[str, Path],
    stem: str,
    settings: dict,
) -> tuple:
    """
    Exporta el modelo con el perfil activo. Retorna ({formato: Path}, resumen legible).
    Los archivos van a out_dir; los nombres llevan sufijo _print en modo impresión.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh = _load_mesh(source)
    formats = active_formats(settings)
    results = {}

    if settings["mode"] == "impresion":
        mesh, info = prepare_print_mesh(mesh, settings)
        out_base = out_dir / f"{stem}_print"
        sx, sy, sz = info["size_mm"]
        summary = f"{settings.get('print_printer')} · {sx:g} × {sy:g} × {sz:g} mm"
        if info["fitted"]:
            summary += " (ajustado a la cama)"
        summary += " · estanca" if info["watertight"] else " · no estanca"
        if info["palette"] is not None and len(info["palette"]) > 1:
            summary += f" · {len(info['palette'])} colores (el STL no guarda color: usa 3MF u OBJ)"
            if print_color_count(dict(settings, print_colors=0)) == 1:
                summary += " · ojo: esta impresora es de un solo color"
        elif info["colors_requested"] > 1:
            summary += " · el modelo no tiene textura: se exporta en un color"
    else:
        mesh = prepare_model_mesh(mesh, settings)
        out_base = out_dir / stem
        summary = f"Modelos 3D · {float(settings.get('model_height_m') or 0):g} m · origen {settings.get('model_pivot')}"

    print_labels = print_palette = None
    if settings["mode"] == "impresion" and info["palette"] is not None and len(info["palette"]) > 1:
        print_labels, print_palette = info["labels"], info["palette"]

    for fmt in formats:
        try:
            if fmt == "glb":
                out_file = out_base.with_suffix(".glb")
                mesh.export(str(out_file), file_type="glb", include_normals=True)
                try:
                    from img2glb import postprocess_and_validate_glb
                    postprocess_and_validate_glb(out_file, asset_name=stem, target_faces=len(mesh.faces))
                except Exception as e_post:
                    print(f"    [Aviso pygltflib] {e_post}")
            elif fmt == "3mf" and print_palette is not None:
                out_file = _export_3mf_colored(mesh, print_labels, print_palette, out_base.with_suffix(".3mf"))
            elif fmt == "obj" and print_palette is not None:
                out_file = _export_obj_colored(mesh, print_labels, print_palette, out_base)
            elif fmt == "obj":
                out_file = _export_obj(mesh, out_base)
            else:
                out_file = out_base.with_suffix(f".{fmt}")
                mesh.export(str(out_file), file_type=fmt)
            results[fmt] = out_file
            print(f"    [OK] .{fmt.upper()} ({out_file.stat().st_size / 1024:.0f} KB) -> {out_file}")
        except Exception as e_exp:
            print(f"    [ERROR] Falló exportación a .{fmt.upper()}: {e_exp}")

    return results, summary
