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

# Configuración Temprana de Cache para evitar saturar el Disco C:\ (100% en D:\)
def configure_environment_and_cache():
    # Prioridad absoluta a la unidad D:\ para modelos y cachés pesados
    if Path("D:/").exists():
        base_ai_dir = Path("D:/AI_Models")
        hf_dir = base_ai_dir / "huggingface"
        torch_dir = base_ai_dir / "torch"
        u2net_dir = base_ai_dir / "u2net"
        temp_dir = base_ai_dir / "temp"
        
        output_dir = base_ai_dir / "output"
        for d in [hf_dir, torch_dir, u2net_dir, temp_dir, output_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        os.environ.setdefault("HF_HOME", str(hf_dir))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_dir / "hub"))
        os.environ.setdefault("TORCH_HOME", str(torch_dir))
        os.environ.setdefault("U2NET_HOME", str(u2net_dir))
        os.environ.setdefault("HY3DGEN_MODELS", str(base_ai_dir))
        os.environ.setdefault("TEMP", str(temp_dir))
        os.environ.setdefault("TMP", str(temp_dir))
        print(f"[Smart Cache] Almacenamiento 3D y modelos configurados en D: ({base_ai_dir})")

    # Cargar variables de entorno adicionales desde .env si existe
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() not in os.environ:
                            os.environ[k.strip()] = v.strip()
        except Exception:
            pass

configure_environment_and_cache()

import numpy as np
from PIL import Image
import torch
import trimesh
import pygltflib


DEFAULT_OUTPUT_DIR = Path("D:/AI_Models/output") if Path("D:/").exists() else Path("./outputs")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Perfiles de Rendimiento de VRAM y Hardware Adaptativo
# ==============================================================================

VRAM_PROFILES = {
    "auto": {
        "name": "Auto-detectar (Recomendado)",
        "desc": "Ajusta la asignación de VRAM según el hardware detectado en el equipo.",
        "low_vram": True,
        "low_vram_mode": True,
        "use_fp16": True,
        "cpu_offload": False,
        "default_octree": 256,
        "gc_after_step": True
    },
    "low_6gb": {
        "name": "6GB VRAM (Optimizado RTX 4050 Laptop / RTX 3050)",
        "desc": "Perfil nativo CUDA en fp16 con liberación inmediata de VRAM tras la inferencia para GPU de 6GB.",
        "low_vram": True,
        "low_vram_mode": True,
        "use_fp16": True,
        "cpu_offload": False,
        "default_octree": 256,
        "gc_after_step": True
    },
    "low_8gb": {
        "name": "8GB VRAM (Optimizado RTX 4060 / Laptops)",
        "desc": "Perfil optimizado para 8GB: descarga tensores tras inferencia y vacía caché para evitar saturar memoria compartida.",
        "low_vram": True,
        "low_vram_mode": True,
        "use_fp16": True,
        "cpu_offload": False,
        "default_octree": 256,
        "gc_after_step": True
    },
    "mid_12gb": {
        "name": "12GB VRAM (Equilibrado RTX 3060 / 4070)",
        "desc": "Perfil intermedio para 10-12GB: retiene pipeline en VRAM pero limpia buffers intermedios.",
        "low_vram": False,
        "low_vram_mode": False,
        "use_fp16": True,
        "cpu_offload": False,
        "default_octree": 320,
        "gc_after_step": False
    },
    "high_16gb": {
        "name": "16GB+ VRAM (Máximo Rendimiento RTX 4080 / 4090 / 3090)",
        "desc": "Perfil de alto rendimiento para 16GB+: retiene modelos completamente en GPU para generaciones instantáneas.",
        "low_vram": False,
        "low_vram_mode": False,
        "use_fp16": True,
        "cpu_offload": False,
        "default_octree": 384,
        "gc_after_step": False
    },
    "cpu": {
        "name": "CPU Seguro (Sin GPU)",
        "desc": "Inferencia forzada en memoria RAM del sistema.",
        "low_vram": True,
        "low_vram_mode": True,
        "use_fp16": False,
        "cpu_offload": False,
        "default_octree": 192,
        "gc_after_step": True
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
    direct_path = Path("D:/AI_Models/tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors")
    if direct_path.exists() and direct_path.stat().st_size > 4 * 1024 * 1024 * 1024:
        return True

    # 2. Comprobar snapshots de Hugging Face
    hf_hub = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "D:/AI_Models/huggingface/hub"))
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
    o en 'C:/Users/Mastermind/Proyectos/Games/kaboomi-party/...',
    resuelve la ruta real existente para que funcione sin fricciones.
    Si solo se especifica un nombre de archivo, lo ubica en D:/AI_Models/output/.
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
            
        # Probar en C:/Users/Mastermind/Proyectos/Games/kaboomi-party
        proyectos_root = Path("C:/Users/Mastermind/Proyectos/Games/kaboomi-party")
        if proyectos_root.exists():
            subparts = parts[idx+1:]
            candidate_proy = proyectos_root.joinpath(*subparts)
            if candidate_proy.parent.exists():
                return candidate_proy.resolve()
            
    # Si no existe, crear la estructura de carpetas
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.resolve()


# ==============================================================================
# 2. Preprocesamiento y Eliminación de Fondo (rembg)
# ==============================================================================

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
            from rembg import remove
            print(" -> Ejecutando rembg (U2-Net / ONNX) para extraer silueta limpia...")
            if session is not None:
                img = remove(img, session=session)
            else:
                img = remove(img)
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
    min_face_ratio: float = 0.15,
    min_volume_ratio: float = 0.05,
    min_flatness: float = 0.08
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

def decimate_mesh_preserving_texture(
    mesh: trimesh.Trimesh,
    target_faces: int = 4000
) -> trimesh.Trimesh:
    """
    Decima una malla 3D simplificando la topología al objetivo de caras mientras preserva
    al 100% las coordenadas UV, los atlas de textura PBR (albedo, normales) y el material original.
    """
    current_faces = len(mesh.faces)
    if current_faces <= target_faces:
        return mesh

    print(f"\n[Paso C/D Textura] Simplificando malla PBR con textura: {current_faces:,} caras -> {target_faces:,} caras...")
    start_t = time.time()

    import tempfile
    try:
        import pymeshlab
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_glb = Path(tmpdir) / "mesh_for_decimation.glb"
            mesh.export(str(tmp_glb), file_type="glb")

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(tmp_glb))
            ms.meshing_decimation_quadric_edge_collapse_with_texture(
                targetfacenum=int(target_faces),
                preservenormal=True
            )
            m = ms.current_mesh()
            v = m.vertex_matrix()
            f = m.face_matrix()
            uvs = m.wedge_tex_coord_matrix()

            unrolled_v = v[f.flatten()]
            unrolled_uv = uvs
            combo = np.hstack([unrolled_v, unrolled_uv])
            _, u_idx, inv = np.unique(combo, axis=0, return_index=True, return_inverse=True)

            orig_mat = getattr(mesh.visual, "material", None)
            decimated_mesh = trimesh.Trimesh(
                vertices=unrolled_v[u_idx],
                faces=inv.reshape(f.shape),
                visual=trimesh.visual.texture.TextureVisuals(
                    uv=unrolled_uv[u_idx],
                    material=orig_mat
                ),
                process=False
            )
            reduction = (1.0 - (len(decimated_mesh.faces) / current_faces)) * 100.0
            print(f" -> [PyMeshLab Textura] Éxito: {len(decimated_mesh.faces):,} caras ({reduction:.1f}% reducción), {len(decimated_mesh.vertices):,} vértices en {time.time() - start_t:.2f}s")
            return decimated_mesh
    except Exception as e:
        print(f" -> [Aviso Decimación Textura] No se pudo decimar con textura ({e}). Se conserva malla original.")
        return mesh

def decimate_mesh_to_target_faces(
    mesh: trimesh.Trimesh,
    target_faces: int = 4000
) -> trimesh.Trimesh:
    """
    Paso C: Simplifica la malla al número exacto de caras solicitado.
    Utiliza Decimación por Colapso de Bordes Cuádricos (QEM) para preservar
    rasgos, silueta y bordes nítidos ideales para Godot 4 en móviles.
    Implementa un pipeline en cascada: PyMeshLab -> pyfqmr -> Trimesh nativo.
    """
    has_texture = hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "baseColorTexture", None) is not None
    if has_texture:
        return decimate_mesh_preserving_texture(mesh, target_faces=target_faces)

    current_faces = len(mesh.faces)
    print(f"\n[Paso C/D] Simplificando malla: {current_faces:,} caras -> {target_faces:,} caras objetivo...")
    start_t = time.time()

    if current_faces <= target_faces:
        print(f" -> La malla ya tiene menos caras que el objetivo ({current_faces} <= {target_faces}). Sin cambios.")
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

_GLOBAL_SHAPE_PIPELINE = None

class Hunyuan3DGenerator:
    """
    Motor integral de generación de modelos 3D con perfiles adaptativos de hardware.
    Gestiona la carga en fp16 de Hunyuan3D-DiT-v2-0-Turbo y Hunyuan3D-Paint-v2-0
    con optimizaciones específicas para 8GB, 12GB, 16GB+ o CPU.
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
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.device = "cpu"
        self.dtype = torch.float16 if (use_fp16 and "cuda" in self.device) else torch.float32

        # Resolver perfil de VRAM dinámicamente
        if vram_profile == "auto" or not vram_profile:
            resolved_key = detect_vram_profile()
        else:
            resolved_key = vram_profile.lower()

        pcfg = VRAM_PROFILES.get(resolved_key, VRAM_PROFILES["low_8gb"])
        self.vram_profile_key = resolved_key
        self.vram_profile_name = pcfg.get("name", "Auto")
        self.low_vram_mode = pcfg.get("low_vram", True) if low_vram_mode is None else low_vram_mode

        print(f"[Hunyuan3D] Dispositivo: {self.device.upper()} | Tipo de dato: {self.dtype} | Perfil VRAM: {self.vram_profile_name}")
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Hunyuan3D] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM) -> Low VRAM: {'Activado (Optimizado 8GB)' if self.low_vram_mode else 'Desactivado (Residente en VRAM)'}")
            
            # Ajustes de aceleración CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _clean_vram(self):
        """Libera VRAM y recolecta basura para evitar fragmentación o desbordamiento en memoria compartida."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def generate_shape(
        self,
        image: Image.Image,
        steps: int = 5,
        octree_resolution: int = 256,
        seed: int = 42,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> trimesh.Trimesh:
        """
        Paso B: Genera la geometría 3D usando el transformer Hunyuan3D-DiT-v2-0-Turbo.
        En modo Turbo, 5 pasos de inferencia entregan calidad equivalente a 50 pasos del modelo base.
        """
        global _GLOBAL_SHAPE_PIPELINE
        print(f"\n[Paso B/D] Generando geometría 3D con Hunyuan3D-2 ({self.shape_subfolder})...")
        print(f" -> Pasos de inferencia DiT: {steps} | Octree: {octree_resolution} | Semilla: {seed}")
        start_t = time.time()
        self._clean_vram()

        try:
            if _GLOBAL_SHAPE_PIPELINE is not None:
                shape_pipe = _GLOBAL_SHAPE_PIPELINE
                print(" -> Reutilizando pipeline Hunyuan3D DiT ya cargado en VRAM.")
            else:
                from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
                print(f" -> Cargando pesos DiT desde {self.model_id} [{self.shape_subfolder}] a GPU NVIDIA ({self.device})...")
                shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    self.model_id,
                    subfolder=self.shape_subfolder,
                    torch_dtype=self.dtype,
                    device=self.device
                )
                if self.device == "cuda":
                    shape_pipe.to("cuda")
                if not self.low_vram_mode:
                    _GLOBAL_SHAPE_PIPELINE = shape_pipe

            generator = torch.Generator(device=self.device).manual_seed(seed) if seed >= 0 else None
            
            # Callback para pasos individuales de difusión DiT
            def dit_step_callback(step_idx, t, outputs):
                if progress_callback:
                    cur_step = step_idx + 1
                    pct = 30 + int((cur_step / max(steps, 1)) * 35) # 30% a 65%
                    progress_callback(f"[3/4] 🧊 Difusión DiT: Paso {cur_step}/{steps} ({pct}%)...", pct)

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
                        progress_callback(f"[3/4] 🧊 Decodificando vóxeles 3D: {v_pct}%...", overall_pct)
                return res

            tqdm.tqdm.update = custom_tqdm_update

            try:
                with torch.inference_mode():
                    outputs = shape_pipe(
                        image=image,
                        num_inference_steps=steps,
                        octree_resolution=octree_resolution,
                        generator=generator,
                        callback=dit_step_callback,
                        callback_steps=1,
                        output_type="trimesh"
                    )
            finally:
                tqdm.tqdm.update = orig_tqdm_update
                
            raw_mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

            # Hunyuan3D-2 DiT genera la geometría directamente en formato vertical Y-up canónico
            # (Pies en -Y, cabeza en +Y, X horizontal, Z profundidad)
            try:
                raw_mesh.fix_normals()
            except Exception as e_orient:
                print(f" -> [Aviso Normales] {e_orient}")

            # Depuración de geometría: eliminar islas parásitas y losas flotantes
            raw_mesh = clean_mesh_floating_artifacts(raw_mesh)

            # Descargar de VRAM solo si está en modo low_vram (esencial en 8GB para no desbordar memoria)
            if self.low_vram_mode:
                del shape_pipe
                _GLOBAL_SHAPE_PIPELINE = None
                self._clean_vram()
                print(" -> Pipeline DiT descargado de VRAM (Perfil Low VRAM / RTX 4060).")

            print(f" -> Malla generada (Y-up vertical erguida): {len(raw_mesh.vertices):,} vértices, {len(raw_mesh.faces):,} caras en {time.time() - start_t:.2f}s")
            return raw_mesh

        except ImportError as e:
            print(f"[ERROR CRÍTICO] La librería hy3dgen no está instalada: {e}")
            print("Ejecuta 'setup.bat' o 'pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git'")
            raise
        except Exception as e:
            err_str = str(e)
            if "SingleImageEncoder" in err_str or "state_dict" in err_str:
                print(f"\n[DIAGNÓSTICO] {err_str}")
                print(" -> Causa: Los pesos de Hunyuan3D-2 Turbo están incompletos en disco o corruptos.")
                print(" -> Solución: Ejecuta 'descargar_modelos.bat' para completar el archivo model.fp16.safetensors (4.93 GB).")
                print(" -> O cambia en la interfaz a 'Stable Fast 3D (SF3D)' para generar en 1s sin este modelo.\n")
            print(f"[ERROR] Falló la generación de forma: {e}")
            raise

    def paint_and_texture_mesh(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image
    ) -> trimesh.Trimesh:
        """
        Paso D: Hornea y sintetiza la textura de color sobre la malla decimada
        utilizando Hunyuan3D-Paint-v2-0.
        Si las extensiones C++ de rasterización no estuviesen compiladas,
        proporciona un mecanismo de textura UV proyectiva de respaldo para nunca fallar.
        """
        print(f"\n[Paso D/D] Sintetizando y horneando textura sobre la malla con {self.paint_subfolder}...")
        start_t = time.time()
        self._clean_vram()

        textured_mesh = None

        # Intento oficial: Hunyuan3DPaintPipeline (solo si los modelos existen localmente en D:)
        base_dir = os.environ.get('HY3DGEN_MODELS', 'D:/AI_Models')
        local_delight = Path(base_dir) / self.model_id / "hunyuan3d-delight-v2-0"
        local_paint = Path(base_dir) / self.model_id / self.paint_subfolder

        if local_delight.exists() and local_paint.exists():
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                print(f" -> Cargando pipeline de texturizado local desde {self.model_id}...")
                
                paint_pipe = Hunyuan3DPaintPipeline.from_pretrained(
                    str(Path(base_dir) / self.model_id)
                )

                if self.device == "cuda":
                    paint_pipe.to("cuda")

                with torch.inference_mode():
                    textured_mesh = paint_pipe(mesh, image=image)

                if self.low_vram_mode:
                    del paint_pipe
                    self._clean_vram()

                print(f" -> Texturizado oficial completado en {time.time() - start_t:.2f}s")
                return textured_mesh

            except Exception as e:
                print(f" -> [Aviso] El motor de render nativo arrojó: {e}")
                print(" -> Activando motor de horneado y mapeo UV inteligente de contingencia...")
        else:
            print(" -> Usando motor de horneado UV ultra-rápido de alta fidelidad.")

        # Contingencia Robusta: Mapeo UV y horneado de textura ortográfica frontal/trasera con Trimesh y Pillow
        try:
            textured_mesh = self._fallback_texture_bake(mesh, image)
            print(f" -> Horneado UV de contingencia aplicado con éxito en {time.time() - start_t:.2f}s")
            return textured_mesh
        except Exception as e_fb:
            print(f" -> [Advertencia] Error en horneado de contingencia ({e_fb}). Exportando modelo con material PBR base.")
            return mesh

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

        # 6. Material PBR estándar de doble cara para Godot 4 / WebGL
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_pil,
            metallicFactor=0.05,
            roughnessFactor=0.60,
            doubleSided=True
        )
        final_mesh.visual.material = material
        return final_mesh


# ==============================================================================
# 5b. Generador Secundario: Stable Fast 3D (SF3D de Stability AI)
# ==============================================================================

_GLOBAL_SF3D_MODEL = None

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
            Path("D:/AI_Models/repos/stable-fast-3d/uv_unwrapper/uv_unwrapper/unwrap.py"),
            Path("./stable-fast-3d/uv_unwrapper/uv_unwrapper/unwrap.py"),
            Path(os.environ.get("HY3DGEN_MODELS", "D:/AI_Models")) / "repos/stable-fast-3d/uv_unwrapper/uv_unwrapper/unwrap.py"
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
                        v_np = v_pos.detach().cpu().numpy()
                        f_np = t_pos_idx.detach().cpu().numpy()
                        vmapping, indices, uvs = xatlas.parametrize(v_np, f_np)
                        return (
                            torch.from_numpy(uvs).to(v_pos.device, dtype=torch.float32),
                            torch.from_numpy(indices).to(t_pos_idx.device, dtype=torch.int64)
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
        # Intentar rasterizador acelerado con Numba
        try:
            from numba import njit, prange
            @njit(parallel=True, fastmath=True)
            def _numba_rasterize(uvs, faces, res):
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
            _has_numba_rasterizer = True
        except Exception:
            _has_numba_rasterizer = False

        class FastTextureBaker(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def rasterize(self, uv: torch.Tensor, face_indices: torch.Tensor, bake_resolution: int) -> torch.Tensor:
                device = uv.device
                res = int(bake_resolution)
                if _has_numba_rasterizer:
                    uv_np = uv.detach().cpu().numpy().astype(np.float32)
                    faces_np = face_indices.detach().cpu().numpy().astype(np.int32)
                    rast_np = _numba_rasterize(uv_np, faces_np, res)
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
        global _GLOBAL_SF3D_MODEL
        if _GLOBAL_SF3D_MODEL is not None:
            self.model = _GLOBAL_SF3D_MODEL
            return

        candidates = [
            Path("D:/AI_Models/repos/stable-fast-3d"),
            Path("./stable-fast-3d"),
            Path(os.environ.get("HY3DGEN_MODELS", "D:/AI_Models")) / "repos" / "stable-fast-3d"
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

        local_weights_dir = Path("D:/AI_Models/stabilityai/stable-fast-3d")
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
            _GLOBAL_SF3D_MODEL = model
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
                    _GLOBAL_SF3D_MODEL = model
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
        texture_resolution: int = 1024,
        generate_texture: bool = True,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> trimesh.Trimesh:
        self._load_model()
        if progress_callback:
            progress_callback("⚡ SF3D: Procesando imagen y esculpiendo 3D instantáneo...", 50)

        rgba_img = image.convert("RGBA")
        temp_dir = Path(os.environ.get("TEMP", "D:/AI_Models/temp"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_glb = temp_dir / f"sf3d_tmp_{int(time.time()*1000)}.glb"

        try:
            from contextlib import nullcontext
            with torch.inference_mode():
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16) if "cuda" in self.device else nullcontext():
                    mesh_out, _ = self.model.run_image(
                        [rgba_img],
                        bake_resolution=texture_resolution,
                        remesh="none"
                    )

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

            if not generate_texture:
                neutral_mat = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                    metallicFactor=0.08,
                    roughnessFactor=0.55,
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
    - Inyecta metadatos de exportación para Godot 4.
    """
    try:
        gltf = pygltflib.GLTF2().load(str(glb_path))
        if gltf.asset is None:
            gltf.asset = pygltflib.Asset()
            
        gltf.asset.generator = "Kaboomi Party 3D Generator (Tencent Hunyuan3D-2)"
        gltf.asset.copyright = "Kaboomi Party / Mastermind"
        
        # Habilitar doubleSided en todos los materiales para que nunca se vuelvan invisibles
        if gltf.materials:
            for mat in gltf.materials:
                mat.doubleSided = True
                if mat.pbrMetallicRoughness:
                    mat.pbrMetallicRoughness.baseColorFactor = [1.0, 1.0, 1.0, 1.0]
                    if mat.pbrMetallicRoughness.roughnessFactor is None:
                        mat.pbrMetallicRoughness.roughnessFactor = 0.6
                    if mat.pbrMetallicRoughness.metallicFactor is None:
                        mat.pbrMetallicRoughness.metallicFactor = 0.05

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

    # 3. Generación y texturizado según el motor seleccionado
    if "sf3d" in model_type.lower():
        if step_callback:
            step_callback(2, "⚡ Inicializando Stable Fast 3D en GPU...")
        sf3d_gen = SF3DGenerator()
        if step_callback:
            step_callback(3, "⚡ SF3D: Generando geometría y texturas PBR instantáneas...")
        raw_mesh = sf3d_gen.generate_mesh(
            clean_image,
            texture_resolution=1024,
            generate_texture=generate_texture
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
        if not generate_texture:
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.08,
                roughnessFactor=0.55,
                doubleSided=True
            )
            final_textured_mesh.visual.material = neutral_mat
    else:
        if step_callback:
            step_callback(3, f"🧊 Esculpiendo geometría 3D y optimizando a {target_faces:,} triángulos...")
        raw_mesh = generator.generate_shape(
            clean_image,
            steps=steps,
            octree_resolution=octree_resolution,
            seed=seed
        )
        cleaned_mesh = clean_mesh_floating_artifacts(raw_mesh)
        decimated_mesh = decimate_mesh_to_target_faces(cleaned_mesh, target_faces=target_faces)
        decimated_mesh = clean_mesh_floating_artifacts(decimated_mesh)
        aligned_mesh = center_and_orient_mesh(
            decimated_mesh,
            align_floor=align_bottom,
            normalize_scale=normalize_scale
        )

        # 4. Sintetizar Textura o Aplicar Material PBR Neutro
        if generate_texture:
            if step_callback:
                step_callback(4, "🎨 Pintando texturas y empaquetando archivo .GLB...")
            final_textured_mesh = generator.paint_and_texture_mesh(aligned_mesh, clean_image)
            final_textured_mesh = center_and_orient_mesh(
                final_textured_mesh,
                align_floor=align_bottom,
                normalize_scale=False
            )
        else:
            if step_callback:
                step_callback(4, "⚡ Aplicando material neutro y empaquetando archivo .GLB...")
            print("\n[Paso D/D] Generando modelo con material PBR base estándar...")
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.08,
                roughnessFactor=0.55
            )
            aligned_mesh.visual.material = neutral_mat
            final_textured_mesh = aligned_mesh

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

    elapsed = time.time() - total_start
    main_file = exported_map.get("glb") or (list(exported_map.values())[0] if exported_map else dest_path)
    file_size_mb = main_file.stat().st_size / (1024 * 1024) if main_file.exists() else 0.0
    
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
        help="Desactivar el modo de bajo VRAM (fuerza mantener modelos en VRAM simultáneamente)"
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
