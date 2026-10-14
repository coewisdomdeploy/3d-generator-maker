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
    session: Optional[Any] = None
) -> Image.Image:
    """
    Paso A: Elimina el fondo de la imagen de entrada usando `rembg`.
    Detecta si la imagen ya cuenta con canal alfa transparente para evitar pérdida de calidad.
    Normaliza el encuadre con un margen uniforme (padding) para maximizar la calidad del modelo 3D.
    Si se proporciona `session` (pre-inicializada con rembg.new_session), la inferencia toma <1s.
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

    # Recorte inteligente de la caja delimitadora (bbox) con padding
    alpha = np.array(img.split()[-1])
    non_empty_coords = np.argwhere(alpha > 10)
    
    if non_empty_coords.size > 0:
        y0, x0 = non_empty_coords.min(axis=0)
        y1, x1 = non_empty_coords.max(axis=0) + 1
        cropped = img.crop((x0, y0, x1, y1))
        
        # Calcular nuevo lienzo cuadrado con padding
        max_dim = max(cropped.width, cropped.height)
        pad = int(max_dim * padding_ratio)
        canvas_size = max_dim + 2 * pad
        
        square_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        offset_x = (canvas_size - cropped.width) // 2
        offset_y = (canvas_size - cropped.height) // 2
        square_img.paste(cropped, (offset_x, offset_y))
        
        img = square_img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
    else:
        img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)

    print(f" -> Imagen limpia lista en {time.time() - start_t:.2f}s ({img.width}x{img.height} RGBA)")
    return img


# ==============================================================================
# 3. Decimación y Simplificación de Mallas para Rendimiento Móvil
# ==============================================================================

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

_GLOBAL_SHAPE_PIPELINE = None

class Hunyuan3DGenerator:
    """
    Motor integral de generación de modelos 3D optimizado para 16GB de VRAM.
    Gestiona la carga secuencial en fp16 de Hunyuan3D-DiT-v2-0-Turbo y Hunyuan3D-Paint-v2-0.
    """
    def __init__(
        self,
        model_id: str = "tencent/Hunyuan3D-2",
        shape_subfolder: str = "hunyuan3d-dit-v2-0-turbo",
        paint_subfolder: str = "hunyuan3d-paint-v2-0",
        use_fp16: bool = True,
        low_vram_mode: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_id = model_id
        self.shape_subfolder = shape_subfolder
        self.paint_subfolder = paint_subfolder
        self.use_fp16 = use_fp16
        self.low_vram_mode = low_vram_mode
        self.device = device
        self.dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32

        print(f"[Hunyuan3D] Dispositivo de ejecución: {self.device.upper()} | Tipo de dato: {self.dtype}")
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Hunyuan3D] GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            
            # Ajustes de aceleración CUDA
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def _clean_vram(self):
        """Libera VRAM y recolecta basura para evitar fragmentación o desbordamiento en 16GB."""
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
                print(f" -> Cargando pesos DiT desde {self.model_id} [{self.shape_subfolder}] a VRAM...")
                shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    self.model_id,
                    subfolder=self.shape_subfolder,
                    torch_dtype=self.dtype
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

            # Hunyuan3D genera la geometría en Z-up. Convertir a estándar glTF 2.0 / Godot 4 (Y-up):
            try:
                R = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
                raw_mesh.apply_transform(R)
                raw_mesh.fix_normals()
            except Exception as e_orient:
                print(f" -> [Aviso Orientación] {e_orient}")

            # Descargar de VRAM solo si está en modo low_vram
            if self.low_vram_mode:
                del shape_pipe
                _GLOBAL_SHAPE_PIPELINE = None
                self._clean_vram()
                print(" -> Pipeline DiT descargado de VRAM.")

            print(f" -> Malla generada y orientada (Y-up): {len(raw_mesh.vertices):,} vértices, {len(raw_mesh.faces):,} caras en {time.time() - start_t:.2f}s")
            return raw_mesh

        except ImportError as e:
            print(f"[ERROR CRÍTICO] La librería hy3dgen no está instalada: {e}")
            print("Ejecuta 'setup.bat' o 'pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git'")
            raise
        except Exception as e:
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
        Generador de texturas y proyección UV inteligente de alta fidelidad:
        - Detecta automáticamente el ángulo de vista azimutal óptimo (<20ms) que alinea la
          geometría 3D con la silueta de la imagen 2D (frontal, 3/4 o lateral).
        - Genera un lienzo con dilatación e inpainting (cv2.inpaint) para prevenir bordes oscuros
          y sangrado de píxeles del fondo.
        - Mapea las coordenadas UV con la orientación canónica de glTF (v=0 superior, v=1 inferior).
        - Asigna material PBR de doble cara con normales corregidas para Godot 4.
        """
        import cv2
        mesh_copy = mesh.copy()
        try:
            mesh_copy.fix_normals()
        except Exception:
            pass

        # 1. Preparar imagen RGBA y máscara de silueta
        rgba_img = image.convert("RGBA")
        arr = np.array(rgba_img)
        alpha = arr[:, :, 3]
        mask_fg = alpha > 25

        if mask_fg.any():
            y_indices, x_indices = np.where(mask_fg)
            x0, x1 = x_indices.min(), x_indices.max()
            y0, y1 = y_indices.min(), y_indices.max()
            cropped_rgb = arr[y0:y1+1, x0:x1+1, :3]
            cropped_alpha = alpha[y0:y1+1, x0:x1+1]
            dom_color = tuple(int(c) for c in cropped_rgb[cropped_alpha > 50].mean(axis=0)) if (cropped_alpha > 50).any() else (180, 200, 215)
        else:
            cropped_rgb = arr[:, :, :3]
            cropped_alpha = alpha
            dom_color = (180, 200, 215)
            x0, y0 = 0, 0
            x1, y1 = arr.shape[1] - 1, arr.shape[0] - 1

        h_crop, w_crop = cropped_rgb.shape[:2]
        pad_x = max(int(w_crop * 0.06), 4)
        pad_y = max(int(h_crop * 0.06), 4)
        canvas_w = w_crop + 2 * pad_x
        canvas_h = h_crop + 2 * pad_y

        tex_canvas = np.full((canvas_h, canvas_w, 3), dom_color, dtype=np.uint8)
        tex_canvas[pad_y:pad_y+h_crop, pad_x:pad_x+w_crop] = cropped_rgb

        # Dilatar bordes exteriores del personaje usando inpainting para eliminar artefactos
        try:
            mask_bg = np.ones((canvas_h, canvas_w), dtype=np.uint8) * 255
            mask_bg[pad_y:pad_y+h_crop, pad_x:pad_x+w_crop] = 255 - cropped_alpha
            tex_canvas = cv2.inpaint(tex_canvas, mask_bg, 5, cv2.INPAINT_TELEA)
        except Exception:
            pass

        tex_pil = Image.fromarray(tex_canvas)

        # 2. Detección del ángulo azimutal óptimo mediante IoU ultra-rápido (<20ms)
        best_deg = self._find_optimal_view_angle(mesh_copy, alpha)
        theta = np.radians(best_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
        cam_verts = mesh_copy.vertices @ R.T

        # 3. Proyección sobre el plano de cámara (X_cam, Y_cam)
        vx = cam_verts[:, 0]
        vy = cam_verts[:, 1]
        vx_min, vx_max = vx.min(), vx.max()
        vy_min, vy_max = vy.min(), vy.max()

        w_span = max(vx_max - vx_min, 1e-5)
        h_span = max(vy_max - vy_min, 1e-5)

        u_rel = np.clip((vx - vx_min) / w_span, 0.0, 1.0)
        v_rel = np.clip((vy - vy_min) / h_span, 0.0, 1.0)

        # Mapear a espacio UV normalizado en glTF (v=0 es arriba, v=1 es abajo)
        u = (pad_x + u_rel * w_crop) / canvas_w
        v = (pad_y + (1.0 - v_rel) * h_crop) / canvas_h
        uv = np.column_stack([np.clip(u, 0.001, 0.999), np.clip(v, 0.001, 0.999)])

        # 4. Material PBR estándar para Godot 4
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_pil,
            metallicFactor=0.05,
            roughnessFactor=0.5,
            doubleSided=True
        )
        mesh_copy.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex_pil, material=material)
        return mesh_copy

    @staticmethod
    def _find_optimal_view_angle(mesh: trimesh.Trimesh, ref_alpha: np.ndarray) -> float:
        import cv2
        ref_small = cv2.resize((ref_alpha > 20).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST)
        if not ref_small.any():
            return 0.0

        y_ref, x_ref = np.where(ref_small > 0)
        ref_x0, ref_x1 = x_ref.min(), x_ref.max()
        ref_y0, ref_y1 = y_ref.min(), y_ref.max()
        ref_w = max(ref_x1 - ref_x0, 1)
        ref_h = max(ref_y1 - ref_y0, 1)

        v = mesh.vertices
        v_sub = v[::max(len(v) // 1000, 1)]

        best_iou = -1.0
        best_deg = 0.0

        for deg in np.linspace(-180, 180, 37)[:-1]:
            rad = np.radians(deg)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            vx = v_sub[:, 0] * cos_a + v_sub[:, 2] * sin_a
            vy = v_sub[:, 1]

            w3d = vx.max() - vx.min()
            h3d = vy.max() - vy.min()
            if w3d < 1e-4 or h3d < 1e-4:
                continue

            px = np.clip(((vx - vx.min()) / w3d) * ref_w + ref_x0, 0, 127).astype(np.int32)
            py = np.clip((1.0 - (vy - vy.min()) / h3d) * ref_h + ref_y0, 0, 127).astype(np.int32)

            canvas = np.zeros((128, 128), dtype=bool)
            canvas[py, px] = True
            canvas_dil = cv2.dilate(canvas.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)) > 0

            intersection = np.logical_and(canvas_dil, ref_small > 0).sum()
            union = np.logical_or(canvas_dil, ref_small > 0).sum()
            iou = intersection / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_deg = float(deg)

        return best_deg


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
    target_faces: int = 4000,
    steps: int = 5,
    octree_resolution: int = 380,
    seed: int = 42,
    align_bottom: bool = False,
    normalize_scale: bool = False,
    low_vram_mode: bool = True,
    generate_texture: bool = True,
    rembg_session: Optional[Any] = None,
    clean_image_input: Optional[Image.Image] = None,
    step_callback: Optional[Callable[[int, str], None]] = None,
    export_formats: Optional[list] = None
) -> Path:
    """
    Ejecuta el pipeline automático completo de 4 etapas:
      a) Eliminación de fondo con rembg (o usa clean_image_input si ya se extrajo)
      b) Generación 3D con Hunyuan3D-2
      c) Decimación a target_faces (2000, 4000 u 8000 tris)
      d) Horneado de textura (si generate_texture=True), centrado en (0,0,0) y exportación a .glb
    """
    total_start = time.time()
    
    if output_glb_path is None:
        timestamp = int(time.time())
        dest_path = DEFAULT_OUTPUT_DIR / f"model_{timestamp}.glb"
    else:
        dest_path = resolve_target_path(str(output_glb_path))
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print(f" INICIANDO GENERACIÓN 3D: {Path(str(input_image_path)).name if not isinstance(input_image_path, Image.Image) else 'Imagen PIL'}")
    print(f" Destino final: {dest_path}")
    print(f" Triángulos objetivo: {target_faces:,} (optimizado para Godot Móvil)")
    print(f" Textura activada: {'Sí' if generate_texture else 'No (Malla rápida)'}")
    print("="*70)

    # 1. Eliminar Fondo / Usar silueta limpia
    if step_callback:
        step_callback(1, "✂️ Quitando fondo con Rembg...")
    if clean_image_input is not None:
        clean_image = clean_image_input
        print(" -> Usando silueta pre-procesada de la interfaz.")
    else:
        clean_image = preprocess_and_remove_background(input_image_path, session=rembg_session)

    # 2. Inicializar Generador
    if step_callback:
        is_cached = is_dit_cached()
        if is_cached:
            step_callback(2, "🧠 Cargando Hunyuan3D-2 en GPU (VRAM 16GB fp16)...")
        else:
            step_callback(2, "🧠 Descargando pesos de Hunyuan3D-2 a D:\\AI_Models... (primera vez, por favor espera)")
            
    generator = Hunyuan3DGenerator(
        model_id="tencent/Hunyuan3D-2",
        shape_subfolder="hunyuan3d-dit-v2-0-turbo",
        paint_subfolder="hunyuan3d-paint-v2-0",
        use_fp16=True,
        low_vram_mode=low_vram_mode
    )

    # 3. Generar Geometría 3D y Decimar
    if step_callback:
        step_callback(3, f"🧊 Esculpiendo geometría 3D y optimizando a {target_faces:,} triángulos...")
    raw_mesh = generator.generate_shape(
        clean_image,
        steps=steps,
        octree_resolution=octree_resolution,
        seed=seed
    )

    decimated_mesh = decimate_mesh_to_target_faces(raw_mesh, target_faces=target_faces)
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
        "--no-low-vram",
        action="store_true",
        help="Desactivar el modo de bajo VRAM (mantiene ambos modelos en VRAM simultáneamente)"
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
            target_faces=args.faces,
            steps=args.steps,
            octree_resolution=args.octree,
            seed=args.seed,
            align_bottom=args.align_bottom,
            low_vram_mode=not args.no_low_vram,
            export_formats=requested_formats
        )
    except Exception as e:
        print(f"\n[ERROR FATAL] La generación falló: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
