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


def prepare_print_mesh(mesh: trimesh.Trimesh, settings: dict) -> tuple:
    """
    Impresión 3D: solo geometría, reparada, Z-up, en milímetros, apoyada en la cama
    y centrada en el volumen de la impresora. Retorna (malla, info).
    """
    geo = trimesh.Trimesh(vertices=np.array(mesh.vertices), faces=np.array(mesh.faces), process=True)
    if settings.get("print_repair", True):
        geo.update_faces(geo.nondegenerate_faces())
        geo.update_faces(geo.unique_faces())
        geo.remove_unreferenced_vertices()
        trimesh.repair.fill_holes(geo)
        trimesh.repair.fix_normals(geo)

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
    else:
        mesh = prepare_model_mesh(mesh, settings)
        out_base = out_dir / stem
        summary = f"Modelos 3D · {float(settings.get('model_height_m') or 0):g} m · origen {settings.get('model_pivot')}"

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
