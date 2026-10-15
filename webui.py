import gc
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==============================================================================
Kaboomi 3D Generator (Tencent Hunyuan3D-2) - Dashboard Profesional
Generador Local Ilimitado de Modelos .glb optimizados para Godot 4
Acelerado por GPU NVIDIA (16GB VRAM en fp16) | Almacenamiento 100% en Disco D:
Diseño Industrial Ultra-Limpio con Iconos SVG, Animaciones Fluidas y Cero Emojis
==============================================================================
"""

import os
import sys
import time
import shutil
import json
import argparse
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple, Generator, Union

# ==============================================================================
# 1. Configuración Estricta de Rutas y Almacenamiento en Disco D:
# ==============================================================================

if Path("D:/").exists():
    base_ai_dir = Path("D:/AI_Models")
    hf_dir = base_ai_dir / "huggingface"
    torch_dir = base_ai_dir / "torch"
    u2net_dir = base_ai_dir / "u2net"
    temp_dir = base_ai_dir / "temp"
    output_dir = base_ai_dir / "output"
    
    for d in [hf_dir, torch_dir, u2net_dir, temp_dir, output_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    os.environ["HF_HOME"] = str(hf_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_dir / "hub")
    os.environ["TORCH_HOME"] = str(torch_dir)
    os.environ["U2NET_HOME"] = str(u2net_dir)
    os.environ["HY3DGEN_MODELS"] = str(base_ai_dir)
    os.environ["TEMP"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)

OUTPUT_DIR = Path("D:/AI_Models/output") if Path("D:/").exists() else Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import gradio as gr
from PIL import Image
import rembg
import torch
import trimesh


from img2glb import (
    preprocess_and_remove_background,
    clean_mesh_floating_artifacts,
    decimate_mesh_to_target_faces,
    decimate_mesh_preserving_texture,
    center_and_orient_mesh,
    postprocess_and_validate_glb,
    export_formats_from_mesh,
    convert_glb_to_format,
    Hunyuan3DGenerator,
    is_dit_cached,
    VRAM_PROFILES,
    detect_vram_profile,
    DEFAULT_OUTPUT_DIR
)

# ==============================================================================
# 2. Pre-inicialización Inteligente de Rembg (U2-Net) al arrancar el servidor
# ==============================================================================

print("\n" + "="*70)
print(" [PRE-CARGA INTELIGENTE] Inicializando sesión de Rembg U2-Net en memoria...")
t_rembg_start = time.time()
try:
    REMBG_SESSION = rembg.new_session("u2net")
    print(f" -> Rembg U2-Net listo en {time.time() - t_rembg_start:.2f}s (Inferencia en <0.5s)")
except Exception as e:
    print(f" -> [Advertencia] Error al crear sesión rembg ({e}).")
    REMBG_SESSION = None
print("="*70 + "\n")

# ==============================================================================
# 3. Detección de Rutas y Configuración de Perfil (General vs Kaboomi Party)
# ==============================================================================

parser = argparse.ArgumentParser(description="3D AI Studio / Kaboomi 3D Studio WebUI")
parser.add_argument(
    "--mode", "--profile",
    choices=["general", "kambo"],
    default=os.environ.get("STUDIO_PROFILE", "general"),
    help="Perfil de ejecución: 'general' (estándar) o 'kambo' (Kaboomi Party)"
)
parser.add_argument("--port", type=int, default=7860, help="Puerto del servidor Gradio")
parser.add_argument("--share", action="store_true", help="Compartir enlace público Gradio")
cmd_args, _ = parser.parse_known_args()
MODE = cmd_args.mode.lower()

# Rutas específicas para el modo Kaboomi Party
POSSIBLE_KABOOMI_ROOTS = [
    Path("C:/Users/Mastermind/Proyectos/Games/kaboomi-party"),
    Path("C:/Users/Mastermind/Proyectos/kaboomi-party"),
    Path("../Games/kaboomi-party"),
    Path("../kaboomi-party")
]

def get_kaboomi_dir() -> Path:
    """Detecta la carpeta raíz del proyecto Kaboomi Party."""
    for p in POSSIBLE_KABOOMI_ROOTS:
        if p.exists() and (p / "project.godot").exists():
            return p.resolve()
        if p.exists() and (p / "assets").exists():
            return p.resolve()
    return Path("C:/Users/Mastermind/Proyectos/Games/kaboomi-party").resolve()

if MODE == "kambo":
    KABOOMI_ROOT = get_kaboomi_dir()
    MODELS_DIR = KABOOMI_ROOT / "assets" / "models"
    BOMBOS_DIR = MODELS_DIR / "bombos"
    BOZIS_DIR = MODELS_DIR / "bozis"
    COSMETICS_DIR = MODELS_DIR / "cosmetics"

    for folder in [BOMBOS_DIR, BOZIS_DIR, COSMETICS_DIR]:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
else:
    KABOOMI_ROOT = None
    MODELS_DIR = None
    BOMBOS_DIR = None
    BOZIS_DIR = None
    COSMETICS_DIR = None

CURRENT_GENERATED_GLB: Optional[Path] = None
CURRENT_GENERATED_FILES: dict[str, Optional[Path]] = {"glb": None, "stl": None, "3mf": None}
IS_GENERATING: bool = False
PLACEHOLDER_GLB = Path(__file__).parent / "assets" / "placeholder.glb"
EMPTY_GLB = Path(__file__).parent / "assets" / "empty.glb"

def get_initial_preview_model() -> Optional[str]:
    """Retorna la escena vacía para que el visor inicie completamente limpio y sin modelos cargados."""
    if EMPTY_GLB.exists():
        return str(EMPTY_GLB.resolve())
    return None

if torch.cuda.is_available():
    _gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "GeForce ").strip()
    _vram_gb = int(round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))
    GPU_STATUS_TEXT = f"{_gpu_name} ({_vram_gb}GB VRAM) • ONLINE"
    GPU_SYS_TEXT = f"NVIDIA {_gpu_name} ({_vram_gb}GB VRAM fp16)"
else:
    GPU_STATUS_TEXT = "CPU • ONLINE"
    GPU_SYS_TEXT = "CPU (Sin aceleración CUDA)"


PRESET_CONFIG_MAP = {
    "Móvil Rápido (4,000 caras • Balanceado)": {"faces": 4000, "steps": 5, "octree": 256},
    "Alta Calidad (8,000 caras • Detallado)": {"faces": 8000, "steps": 6, "octree": 320},
    "Ultra Ligero (2,000 caras • Máximo Rendimiento)": {"faces": 2000, "steps": 4, "octree": 256}
}

PRESET_FACES_MAP = {k: v["faces"] for k, v in PRESET_CONFIG_MAP.items()}

VRAM_PROFILE_CHOICES = [
    "Auto-detectar (Recomendado)",
    "6GB VRAM (Optimizado RTX 4050 Laptop / RTX 3050)",
    "8GB VRAM (Optimizado RTX 4060 / Laptops)",
    "12GB VRAM (Equilibrado RTX 3060 / 4070)",
    "16GB+ VRAM (Máximo Rendimiento)",
    "CPU Seguro (Sin GPU)"
]

VRAM_CHOICE_TO_KEY = {
    "Auto-detectar (Recomendado)": "auto",
    "6GB VRAM (Optimizado RTX 4050 Laptop / RTX 3050)": "low_6gb",
    "8GB VRAM (Optimizado RTX 4060 / Laptops)": "low_8gb",
    "12GB VRAM (Equilibrado RTX 3060 / 4070)": "mid_12gb",
    "16GB+ VRAM (Máximo Rendimiento)": "high_16gb",
    "CPU Seguro (Sin GPU)": "cpu"
}


def render_progress_html(step_text: str, percent: int) -> str:
    """Renderiza el widget de barra de progreso con diseño industrial y SVGs animados."""
    if percent == 100:
        icon_svg = """<svg class="status-svg status-done" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="16 9 10 15 7 12"/></svg>"""
    elif percent > 0:
        icon_svg = """<svg class="status-svg status-spin" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a855f7" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>"""
    else:
        icon_svg = """<svg class="status-svg status-idle" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="3" fill="#38bdf8"/></svg>"""
        
    return f"""
    <div class="live-status-container">
        <div class="live-status-header">
            <span class="status-icon-wrap">{icon_svg}</span>
            <span class="status-text">{step_text}</span>
        </div>
        <div class="progress-bar-row">
            <span class="progress-label">PROGRESO</span>
            <div class="progress-track">
                <div class="progress-fill" style="width: {percent}%;"></div>
            </div>
            <span class="progress-pct">{percent}%</span>
        </div>
    </div>
    """


# ==============================================================================
# 4. Lógica de Interfaz y Pipeline en Tiempo Real (con yield y streaming granular)
# ==============================================================================

def ui_remove_background(image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], str]:
    """Elimina el fondo usando la sesión pre-inicializada en menos de 1 segundo."""
    if image is None:
        return None, render_progress_html("Sube una imagen de entrada en el Paso 1.", 0)
    start_t = time.time()
    try:
        clean_img = preprocess_and_remove_background(
            image,
            force_rembg=True,
            session=REMBG_SESSION,
            alpha_threshold=35
        )
        elapsed = time.time() - start_t
        msg = render_progress_html(f"Silueta extraída con éxito en {elapsed:.2f}s (Canal alfa optimizado)", 0)
        return clean_img, msg
    except Exception as e:
        return None, render_progress_html(f"Error al procesar la imagen: {e}", 0)


def on_image_uploaded(image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """
    Se ejecuta automáticamente al subir o pegar una imagen en input_img.
    Extrae la silueta limpia con rembg U2-Net, actualiza el componente de entrada
    y almacena la imagen limpia en el State para su procesamiento 3D.
    """
    if image is None:
        return None, None, render_progress_html("Sube una imagen para comenzar.", 0)
    clean_img, msg = ui_remove_background(image)
    display_img = clean_img if clean_img is not None else image
    return display_img, clean_img, msg


def _ui_generate_3d_impl(
    image: Optional[Image.Image],
    clean_silhouette: Optional[Image.Image],
    preset_label: str,
    ai_model_choice: str,
    generate_texture: bool,
    model_name: str,
    vram_profile_choice: str = "Auto-detectar (Recomendado)"
) -> Generator[Tuple[Optional[str], any, any, any, str], None, None]:
    """
    Función generadora interna en streaming que emite actualizaciones en vivo etapa por etapa:
      - Etapa 1: Extracción alfa (15%)
      - Etapa 2: GPU VRAM init y Perfil de Hardware (25%)
      - Etapa 3: Síntesis DiT granular (30% -> 65%), Volume Decoding (66% -> 84%), Depuración de losas (86%), Decimación (88%)
      - Etapa 4: Texturizado UV (94%), Empaquetado Multi-Formato GLB/STL/3MF (97%), Finalizado (100%)
    """
    global CURRENT_GENERATED_GLB, CURRENT_GENERATED_FILES

    def step_yield(msg_html: str):
        return gr.skip(), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), msg_html
    
    if image is None and clean_silhouette is None:
        yield step_yield(render_progress_html("Sube una imagen en el Paso 1 antes de generar.", 0))
        return

    cfg = PRESET_CONFIG_MAP.get(preset_label, {"faces": 4000, "steps": 5, "octree": 256})
    target_faces = cfg["faces"]
    dit_steps = cfg["steps"]
    octree_res = cfg["octree"]
    total_start = time.time()
    
    # --------------------------------------------------------------------------
    # ETAPA 1/4: Silueta Limpia (15%)
    # --------------------------------------------------------------------------
    yield step_yield(render_progress_html("[1/4] Extracción Alfa: Procesando silueta con Rembg U2-Net...", 15))
    
    try:
        if clean_silhouette is not None:
            clean_img = clean_silhouette
            print(" -> Usando silueta pre-calculada de la interfaz.")
        else:
            clean_img = preprocess_and_remove_background(
                image,
                force_rembg=True,
                session=REMBG_SESSION,
                alpha_threshold=35
            )
    except Exception as e:
        yield step_yield(render_progress_html(f"Error en Rembg: {e}", 0))
        return

    # --------------------------------------------------------------------------
    # ETAPA 2 & 3: Selección de Motor de IA e Inferencia 3D
    # --------------------------------------------------------------------------
    is_sf3d = "sf3d" in str(ai_model_choice).lower() or "stable fast 3d" in str(ai_model_choice).lower()

    if is_sf3d:
        yield step_yield(render_progress_html("[2/4] Motor SF3D: Inicializando Stable Fast 3D en GPU...", 25))
        try:
            from img2glb import SF3DGenerator
            sf3d_gen = SF3DGenerator()
        except Exception as e:
            yield step_yield(render_progress_html(f"Error al inicializar Stable Fast 3D: {e}", 0))
            return

        yield step_yield(render_progress_html("[3/4] Inferencia Instantánea: Esculpiendo y horneando PBR en ~1 segundo...", 60))
        try:
            raw_mesh = sf3d_gen.generate_mesh(
                clean_img,
                texture_resolution=1024,
                generate_texture=generate_texture
            )
        except Exception as e:
            yield step_yield(render_progress_html(f"Error en inferencia SF3D: {e}", 0))
            return

        if target_faces and target_faces < len(raw_mesh.faces):
            yield step_yield(render_progress_html(f"[3/4] Optimización PBR: Decimando a {target_faces:,} caras conservando texturas...", 88))
            from img2glb import decimate_mesh_preserving_texture
            decimated_mesh = decimate_mesh_preserving_texture(raw_mesh, target_faces=target_faces)
        else:
            decimated_mesh = raw_mesh

        aligned_mesh = center_and_orient_mesh(
            decimated_mesh,
            align_floor=False,
            normalize_scale=True,
            target_height=1.2
        )

        if not generate_texture:
            import trimesh
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.84, 0.86, 0.90, 1.0],
                metallicFactor=0.08,
                roughnessFactor=0.55,
                doubleSided=True
            )
            aligned_mesh.visual.material = neutral_mat
            final_mesh = aligned_mesh
        else:
            final_mesh = aligned_mesh
    else:
        # Pipeline Tencent Hunyuan3D-2 Turbo
        cached = is_dit_cached()
        if not cached:
            status_file = Path("D:/AI_Models/download_status.json")
            script_dir = Path(__file__).resolve().parent
            download_script = script_dir / "download_model.py"
            print(f" -> [WebUI] Pesos DiT de Hunyuan3D-2 no encontrados en cache. Iniciando descarga automatica en segundo plano ({download_script})...")
            try:
                subprocess.Popen([sys.executable, str(download_script)], cwd=str(script_dir))
            except Exception as e_dl:
                print(f" -> [WebUI Error] No se pudo lanzar download_model.py: {e_dl}")

            last_console_time = 0
            while not is_dit_cached():
                if status_file.exists():
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            sdata = json.load(f)
                        if sdata.get("is_complete"):
                            print(" -> [WebUI] ¡Descarga de pesos DiT completada con exito!")
                            break
                        pct = sdata.get("percent", 0)
                        dl_mb = sdata.get("downloaded_mb", 0)
                        tot_mb = sdata.get("total_mb", 4702)
                        spd = sdata.get("speed_mb", 0)
                        eta = sdata.get("eta_seconds", 0)
                        eta_str = f"{int(eta//60)}m {int(eta%60):02d}s" if eta > 0 else "calculando..."
                        txt = f"[2/4] Descargando pesos DiT ({dl_mb:.0f}/{tot_mb:.0f} MB • {pct}% • {spd} MB/s • ETA: {eta_str})"
                        now_t = time.time()
                        if now_t - last_console_time >= 5.0:
                            print(f" -> {txt}")
                            last_console_time = now_t
                        yield step_yield(render_progress_html(txt, int(pct * 0.45 + 5)))
                    except Exception:
                        pass
                else:
                    now_t = time.time()
                    if now_t - last_console_time >= 5.0:
                        print(" -> [WebUI] Esperando sincronizacion de descarga de pesos...")
                        last_console_time = now_t
                    yield step_yield(render_progress_html("[2/4] Iniciando descarga de pesos Hunyuan3D-2 (~4.9GB) a Disco D:...", 25))
                
                time.sleep(1.5)

        # Determinar perfil VRAM activo
        selected_key = VRAM_CHOICE_TO_KEY.get(vram_profile_choice, "auto")
        active_profile_key = detect_vram_profile() if selected_key == "auto" else selected_key
        profile_info = VRAM_PROFILES.get(active_profile_key, VRAM_PROFILES["low_8gb"])
        print(f" -> [WebUI VRAM] Selección: '{vram_profile_choice}' | Perfil activo: '{active_profile_key}' ({profile_info['desc']})")

        yield step_yield(render_progress_html(f"[2/4] Perfil VRAM: {profile_info['desc']}...", 25))

        try:
            generator = Hunyuan3DGenerator(
                model_id="tencent/Hunyuan3D-2",
                shape_subfolder="hunyuan3d-dit-v2-0-turbo",
                paint_subfolder="hunyuan3d-paint-v2-0",
                use_fp16=profile_info.get("use_fp16", True),
                low_vram_mode=profile_info.get("low_vram_mode", profile_info.get("low_vram", True)),
                vram_profile=active_profile_key
            )
        except Exception as e:
            yield step_yield(render_progress_html(f"Error al cargar Hunyuan3D-2: {e}", 0))
            return

        # ETAPA 3/4: Geometría 3D y Decimación Hunyuan3D
        yield step_yield(render_progress_html("[3/4] Síntesis Geométrica: Iniciando inferencia en GPU...", 30))

        progress_q = queue.Queue()
        gen_result = {"mesh": None, "error": None}

        def on_shape_progress(step_text, pct):
            progress_q.put((step_text, pct))

        def run_shape_worker():
            try:
                m = generator.generate_shape(
                    clean_img,
                    steps=dit_steps,
                    octree_resolution=octree_res,
                    seed=42,
                    progress_callback=on_shape_progress
                )
                gen_result["mesh"] = m
            except Exception as err:
                gen_result["error"] = err
            finally:
                progress_q.put(("__DONE__", None))

        worker_thread = threading.Thread(target=run_shape_worker, daemon=True)
        worker_thread.start()

        while worker_thread.is_alive() or not progress_q.empty():
            try:
                item = progress_q.get(timeout=0.1)
                msg, pct = item
                if msg == "__DONE__":
                    break
                yield step_yield(render_progress_html(msg, pct))
            except queue.Empty:
                pass

        worker_thread.join()

        if gen_result["error"] is not None:
            yield step_yield(render_progress_html(f"Error en geometría: {gen_result['error']}", 0))
            return

        raw_mesh = gen_result["mesh"]

        yield step_yield(render_progress_html("[3/4] Depuración Geométrica: Eliminando losas e islas parásitas...", 86))
        cleaned_mesh = clean_mesh_floating_artifacts(raw_mesh)

        yield step_yield(render_progress_html(f"[3/4] Optimización de Malla: Decimando a {target_faces:,} caras...", 88))
        decimated_mesh = decimate_mesh_to_target_faces(cleaned_mesh, target_faces=target_faces)
        decimated_mesh = clean_mesh_floating_artifacts(decimated_mesh)
        aligned_mesh = center_and_orient_mesh(
            decimated_mesh,
            align_floor=False,
            normalize_scale=True,
            target_height=1.2
        )

        # ETAPA 4/4: Texturizado Hunyuan3D
        if generate_texture:
            yield step_yield(render_progress_html("[4/4] Texturizado PBR: Horneando mapas de color y proyección UV...", 94))
            try:
                final_mesh = generator.paint_and_texture_mesh(aligned_mesh, clean_img)
                final_mesh = center_and_orient_mesh(
                    final_mesh,
                    align_floor=False,
                    normalize_scale=False
                )
            except Exception as e:
                print(f" -> [Aviso Textura] {e}. Usando geometría base.")
                final_mesh = aligned_mesh
        else:
            yield step_yield(render_progress_html("[4/4] Empaquetado: Ensamblando malla base (sin textura)...", 94))
            import trimesh
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.84, 0.86, 0.90, 1.0],
                metallicFactor=0.08,
                roughnessFactor=0.55,
                doubleSided=True
            )
            aligned_mesh.visual.material = neutral_mat
            final_mesh = aligned_mesh

    yield step_yield(render_progress_html("[4/4] Exportación Multi-Formato: Generando archivos .GLB, .STL y .3MF...", 97))

    # Guardar archivos .glb, .stl y .3mf en D:\AI_Models\output\
    timestamp = int(time.time())
    raw_name = (model_name or "").strip()
    for ext in [".glb", ".stl", ".3mf"]:
        if raw_name.lower().endswith(ext):
            raw_name = raw_name[:-len(ext)]
    safe_base = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-")).strip()
    if not safe_base:
        safe_base = f"modelo_{timestamp}"

    base_dest = OUTPUT_DIR / safe_base
    if (OUTPUT_DIR / f"{safe_base}.glb").exists():
        base_dest = OUTPUT_DIR / f"{safe_base}_{timestamp}"

    try:
        exported_map = export_formats_from_mesh(
            mesh=final_mesh,
            base_output_path=base_dest,
            formats=["glb", "stl", "3mf"],
            target_faces=target_faces
        )

        glb_file = exported_map.get("glb")
        stl_file = exported_map.get("stl")
        threemf_file = exported_map.get("3mf")

        CURRENT_GENERATED_GLB = glb_file
        CURRENT_GENERATED_FILES["glb"] = glb_file
        CURRENT_GENERATED_FILES["stl"] = stl_file
        CURRENT_GENERATED_FILES["3mf"] = threemf_file

        elapsed = time.time() - total_start
        glb_size_mb = glb_file.stat().st_size / (1024 * 1024) if glb_file and glb_file.exists() else 0.0
        stl_size_kb = stl_file.stat().st_size / 1024 if stl_file and stl_file.exists() else 0.0
        threemf_size_kb = threemf_file.stat().st_size / 1024 if threemf_file and threemf_file.exists() else 0.0
        num_faces = len(final_mesh.faces)

        final_msg = render_progress_html(
            f"Modelo generado en {elapsed:.1f}s | {num_faces:,} caras | GLB: {glb_size_mb:.2f} MB • STL: {stl_size_kb:.0f} KB • 3MF: {threemf_size_kb:.0f} KB",
            100
        )

        yield (
            str(glb_file),
            gr.update(value=str(glb_file), interactive=True),
            gr.update(value=str(stl_file), interactive=True),
            gr.update(value=str(threemf_file), interactive=True),
            final_msg
        )

    except Exception as e:
        yield step_yield(render_progress_html(f"Error al exportar formatos: {e}", 0))


def ui_generate_3d(
    image: Optional[Image.Image],
    clean_silhouette: Optional[Image.Image],
    preset_label: str,
    ai_model_choice: str,
    generate_texture: bool,
    model_name: str,
    vram_profile_choice: str = "Auto-detectar (Recomendado)"
) -> Generator[Tuple[Optional[str], any, any, any, str], None, None]:
    """
    Función generadora externa con bloqueo atómico anti-doble clic (mutex).
    """
    global IS_GENERATING
    if IS_GENERATING:
        yield (
            gr.skip(),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            render_progress_html("Ya hay una generación en progreso. Por favor espera a que finalice.", 0)
        )
        return
    IS_GENERATING = True
    try:
        for out in _ui_generate_3d_impl(image, clean_silhouette, preset_label, ai_model_choice, generate_texture, model_name, vram_profile_choice):
            yield out
    finally:
        IS_GENERATING = False
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def ui_save_to_game(category: str, subcategory: str, custom_name: str) -> str:
    """Copia directamente el .glb generado a la carpeta de Kaboomi Party."""
    global CURRENT_GENERATED_GLB
    if CURRENT_GENERATED_GLB is None or not CURRENT_GENERATED_GLB.exists():
        return "Primero genera un modelo 3D antes de guardarlo en el juego."
    
    if BOMBOS_DIR is None:
        return "El modo Kaboomi Party no está activo."
    
    clean_cat = category.replace("/", "").strip().lower()
    folder_map = {
        "bombos": BOMBOS_DIR,
        "bozis": BOZIS_DIR,
        "cosmetics": COSMETICS_DIR
    }
    dest_folder = folder_map.get(clean_cat, BOMBOS_DIR)
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    raw_name = (custom_name or "").strip()
    if raw_name.lower().endswith(".glb"):
        raw_name = raw_name[:-4]
    safe_name = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-")).strip()
    if not safe_name:
        safe_name = f"bombo_{int(time.time())}"
    safe_name += ".glb"
        
    dest_file = dest_folder / safe_name
    try:
        shutil.copy2(CURRENT_GENERATED_GLB, dest_file)
        return (
            f"**Guardado con éxito en Kaboomi Party**\n\n"
            f"- **Destino:** `{dest_file.parent.name}/{safe_name}`\n"
            f"- **Ruta:** `{dest_file}`\n"
            f"- *Godot 4 lo detectará e importará automáticamente.*"
        )
    except Exception as e:
        return f"Error al guardar en el juego: {e}"


def ui_save_to_custom_dir(target_dir_str: str, custom_name: str) -> str:
    """Copia el .glb generado a una carpeta o proyecto local especificado."""
    global CURRENT_GENERATED_GLB
    if CURRENT_GENERATED_GLB is None or not CURRENT_GENERATED_GLB.exists():
        return "Primero genera un modelo 3D antes de exportarlo."
    
    if not target_dir_str or not target_dir_str.strip():
        return "Por favor especifica una ruta de carpeta de destino válida (ej. C:/MiProyecto/Assets)."
        
    dest_folder = Path(target_dir_str.strip().strip('"').strip("'"))
    try:
        dest_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return f"No se pudo crear o acceder a la carpeta destino: {e}"
        
    raw_name = (custom_name or "").strip()
    if raw_name.lower().endswith(".glb"):
        raw_name = raw_name[:-4]
    safe_name = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-")).strip()
    if not safe_name:
        safe_name = f"modelo_{int(time.time())}"
    safe_name += ".glb"
    
    dest_file = dest_folder / safe_name
    try:
        shutil.copy2(CURRENT_GENERATED_GLB, dest_file)
        return (
            f"**Modelo exportado con éxito**\n\n"
            f"- **Archivo:** `{safe_name}`\n"
            f"- **Ruta Completa:** `{dest_file}`\n"
            f"- *Tu motor de juego (Godot, Unity, Unreal) o editor 3D lo detectará.*"
        )
    except Exception as e:
        return f"Error al exportar modelo: {e}"


def list_existing_models():
    """Escanea los modelos disponibles según el modo activo."""
    choices = []
    if MODE == "kambo":
        if MODELS_DIR and MODELS_DIR.exists():
            categories = {"bombos": BOMBOS_DIR, "bozis": BOZIS_DIR, "cosmetics": COSMETICS_DIR}
            for cat_name, folder in categories.items():
                if folder and folder.exists():
                    for f in sorted(folder.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
                        choices.append(f"[{cat_name.upper()}] {f.name}")
        return choices or ["(No hay modelos exportados en Kaboomi Party aún)"]
    else:
        if OUTPUT_DIR.exists():
            for f in sorted(OUTPUT_DIR.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
                if not f.stem.endswith("_clay"):
                    choices.append(f.name)
        return choices or ["(No hay modelos generados en outputs aún)"]


def load_existing_model_preview(selection: str) -> Optional[str]:
    """Carga un modelo existente en el visor buscando en Kaboomi y Outputs."""
    if not selection or "(" in selection or selection.startswith("(No hay"):
        return None
    try:
        kab_dir = get_kaboomi_dir()
        models_dir = kab_dir / "assets" / "models"

        if "] " in selection:
            cat_part, name_part = selection.split("] ", 1)
            cat = cat_part.replace("[", "").lower().strip()
            target = models_dir / cat / name_part.strip()
            if target.exists():
                return str(target.resolve())
        else:
            clean_name = selection.strip()
            target = OUTPUT_DIR / clean_name
            if target.exists():
                return str(target.resolve())
            
            # Búsqueda en salidas de D:\AI_Models
            for f in OUTPUT_DIR.glob(f"**/{clean_name}"):
                if f.exists():
                    return str(f.resolve())
                    
            # Búsqueda en assets del juego
            for f in models_dir.glob(f"**/{clean_name}"):
                if f.exists():
                    return str(f.resolve())
    except Exception as e:
        print(f"[Aviso visor 3D] Error al buscar modelo: {e}")
    return None


def get_or_create_clay_glb(glb_path_str: Union[str, Path]) -> Optional[str]:
    """Genera o recupera una versión en arcilla neutra (clay) del modelo GLB para inspección sin textura."""
    try:
        glb_path = Path(glb_path_str)
        if not glb_path.exists():
            return None
        
        # Si ya es un archivo _clay.glb, retornarlo tal cual
        if glb_path.stem.endswith("_clay"):
            return str(glb_path.resolve())
            
        clay_path = glb_path.with_name(f"{glb_path.stem}_clay.glb")
        if clay_path.exists() and clay_path.stat().st_mtime >= glb_path.stat().st_mtime:
            return str(clay_path.resolve())
            
        scene = trimesh.load(str(glb_path))
        mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
        mesh_clay = mesh.copy()
        neutral_mat = trimesh.visual.material.PBRMaterial(
            name="ClayMaterial",
            baseColorFactor=[0.84, 0.86, 0.90, 1.0],
            metallicFactor=0.08,
            roughnessFactor=0.55,
            doubleSided=True
        )
        mesh_clay.visual = trimesh.visual.TextureVisuals(material=neutral_mat)
        mesh_clay.export(str(clay_path), file_type="glb")
        return str(clay_path.resolve())
    except Exception as e:
        print(f"[Aviso Clay 3D] Error al generar modelo arcilla: {e}")
        return str(glb_path_str)


def get_mesh_quick_stats(file_path_str: Optional[str]) -> dict:
    """Extrae rápidamente conteo de caras, vértices y presencia de textura."""
    if not file_path_str:
        return {"faces": 0, "vertices": 0, "has_texture": False, "size_kb": 0.0}
    try:
        p = Path(file_path_str)
        if not p.exists() or p.name in ["placeholder.glb", "empty.glb"]:
            return {"faces": 0, "vertices": 0, "has_texture": False, "size_kb": 0.0}
        
        scene = trimesh.load(str(p))
        mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
        faces = len(mesh.faces)
        verts = len(mesh.vertices)
        has_texture = hasattr(mesh.visual, "material") and getattr(mesh.visual.material, "baseColorTexture", None) is not None
        size_kb = p.stat().st_size / 1024
        return {
            "faces": faces,
            "vertices": verts,
            "has_texture": bool(has_texture),
            "size_kb": size_kb
        }
    except Exception:
        return {"faces": 0, "vertices": 0, "has_texture": False, "size_kb": 0.0}


def render_mesh_stats_html(file_path_str: Optional[str]) -> str:
    """Genera el bloque HTML con métricas técnicas del modelo 3D."""
    stats = get_mesh_quick_stats(file_path_str)
    if stats["faces"] == 0:
        return """
        <div class="mesh-stats-bar empty">
            <span class="mesh-stat-empty">Sin modelo 3D activo para inspeccionar</span>
        </div>
        """
    
    tex_status = "PBR Horneada" if stats["has_texture"] else "Sin Textura (Sólido)"
    tex_class = "active" if stats["has_texture"] else "inactive"
    faces_formatted = f"{stats['faces']:,}"
    verts_formatted = f"{stats['vertices']:,}"
    size_formatted = f"{stats['size_kb']:.1f} KB" if stats["size_kb"] < 1024 else f"{stats['size_kb']/1024:.2f} MB"
    
    return f"""
    <div class="mesh-stats-bar">
        <div class="mesh-stat-item">
            <span class="mesh-stat-label">POLÍGONOS (TRIS)</span>
            <span class="mesh-stat-value">{faces_formatted}</span>
        </div>
        <div class="mesh-stat-item">
            <span class="mesh-stat-label">VÉRTICES</span>
            <span class="mesh-stat-value">{verts_formatted}</span>
        </div>
        <div class="mesh-stat-item">
            <span class="mesh-stat-label">TEXTURA</span>
            <span class="mesh-stat-value {tex_class}">{tex_status}</span>
        </div>
        <div class="mesh-stat-item">
            <span class="mesh-stat-label">TAMAÑO GLB</span>
            <span class="mesh-stat-value">{size_formatted}</span>
        </div>
    </div>
    """


def on_change_shading_mode(mode: str, current_path: Optional[str]):
    """Alterna entre Textura PBR, Arcilla (Clay), Malla (Wireframe) y Nube de Puntos."""
    if not current_path or not Path(current_path).exists() or current_path.endswith("empty.glb") or current_path.endswith("placeholder.glb"):
        if mode in ["Malla", "Malla (Wireframe)"]:
            return gr.update(display_mode="wireframe")
        elif mode in ["Puntos", "Nube de Puntos"]:
            return gr.update(display_mode="point_cloud")
        else:
            return gr.update(display_mode="solid")
    
    orig_glb = str(current_path)
    if orig_glb.endswith("_clay.glb"):
        candidate = orig_glb.replace("_clay.glb", ".glb")
        if Path(candidate).exists():
            orig_glb = candidate
            
    if mode in ["Textura", "Textura PBR"]:
        return gr.update(value=orig_glb, display_mode="solid")
    elif mode in ["Arcilla", "Sin Textura (Clay)"]:
        clay_path = get_or_create_clay_glb(orig_glb)
        return gr.update(value=clay_path or orig_glb, display_mode="solid")
    elif mode in ["Malla", "Malla (Wireframe)"]:
        clay_path = get_or_create_clay_glb(orig_glb)
        return gr.update(value=clay_path or orig_glb, display_mode="wireframe")
    elif mode in ["Puntos", "Nube de Puntos"]:
        clay_path = get_or_create_clay_glb(orig_glb)
        return gr.update(value=clay_path or orig_glb, display_mode="point_cloud")
    return gr.update(display_mode="solid")


def on_change_background(bg_choice: str):
    """Cambia el color de fondo e iluminación del visor."""
    b = (bg_choice or "").lower()
    if "claro" in b:
        return gr.update(clear_color=[0.88, 0.90, 0.94, 1.0])
    elif "cad" in b or "neutral" in b:
        return gr.update(clear_color=[0.18, 0.19, 0.22, 1.0])
    else:
        return gr.update(clear_color=[0.10, 0.12, 0.18, 1.0])


def on_camera_preset(preset_name: str):
    """Reubica la cámara del visor a un ángulo ortogonal o perspectiva preestablecido."""
    p = (preset_name or "").lower()
    if "frontal" in p:
        return gr.update(camera_position=(0, 90, 2.4))
    elif "lateral" in p:
        return gr.update(camera_position=(90, 90, 2.4))
    elif "superior" in p:
        return gr.update(camera_position=(0, 0.1, 2.4))
    else:
        return gr.update(camera_position=(-90, 80, 2.4))


def on_generation_completed():
    """Se ejecuta al concluir la generación para actualizar métricas técnicas y reactivar controles."""
    glb_path = str(CURRENT_GENERATED_GLB) if CURRENT_GENERATED_GLB and CURRENT_GENERATED_GLB.exists() else None
    stats_html = render_mesh_stats_html(glb_path)
    return (
        gr.update(interactive=True, value="GENERAR MODELO 3D (GLB / STL / 3MF)"),
        glb_path,
        stats_html,
        "Textura"
    )


def on_select_library_model(selection: str):
    """Carga el modelo en el visor y prepara descargas en GLB, STL y 3MF instantáneamente."""
    glb_path_str = load_existing_model_preview(selection)
    if not glb_path_str:
        return (
            gr.skip(),
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value=None),
            gr.update(interactive=False, value=None),
            None,
            render_mesh_stats_html(None),
            "Textura"
        )
    glb_path = Path(glb_path_str)
    
    # 1. Comprobar si ya existen archivos hermanos .stl y .3mf para respuesta instantánea
    stl_file = glb_path.with_suffix(".stl")
    stl_val = str(stl_file.resolve()) if stl_file.exists() else None
    
    threemf_file = glb_path.with_suffix(".3mf")
    threemf_val = str(threemf_file.resolve()) if threemf_file.exists() else None

    # 2. Si no existen, convertir bajo demanda
    if not stl_val:
        try:
            s = convert_glb_to_format(glb_path, "stl")
            if s and s.exists():
                stl_val = str(s.resolve())
        except Exception:
            stl_val = None

    if not threemf_val:
        try:
            t = convert_glb_to_format(glb_path, "3mf")
            if t and t.exists():
                threemf_val = str(t.resolve())
        except Exception:
            threemf_val = None

    stats_html = render_mesh_stats_html(str(glb_path.resolve()))

    return (
        str(glb_path.resolve()),
        gr.update(interactive=True, value=str(glb_path.resolve())),
        gr.update(interactive=stl_val is not None, value=stl_val),
        gr.update(interactive=threemf_val is not None, value=threemf_val),
        str(glb_path.resolve()),
        stats_html,
        "Textura"
    )


# ==============================================================================
# 5. Estilos CSS Profesionales, Sin Emojis y con Animaciones de Alta Fidelidad
# ==============================================================================

CUSTOM_CSS = """
/* ==============================================================================
   1. Reset y Base Oscura Premium (#07080d / #0f111a)
   ============================================================================== */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background-color: #07080d !important;
    color: #f1f5f9 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

.gradio-container {
    max-width: 1760px !important;
    padding: 12px 20px !important;
    margin: 0 auto !important;
    background-color: #07080d !important;
}

/* ==============================================================================
   2. Etiquetas Flotantes Legibles (Badge Oscuro con Blur)
   ============================================================================== */
.block label,
.block span.label-text,
.block > label > span,
.form > label > span,
label > span,
.svelte-15lo0ep {
    background: rgba(11, 13, 23, 0.78) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    padding: 4px 10px !important;
    border-radius: 6px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
    display: inline-flex !important;
    align-items: center !important;
    width: fit-content !important;
    margin-bottom: 6px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

/* ==============================================================================
   3. Cabecera Ejecutiva Responsiva
   ============================================================================== */
.custom-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 14px 20px;
    margin-bottom: 14px;
    background: linear-gradient(135deg, rgba(17, 23, 38, 0.85), rgba(12, 17, 29, 0.85));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    backdrop-filter: blur(14px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-logo-icon {
    stroke: #a855f7;
    filter: drop-shadow(0 0 12px rgba(168, 85, 247, 0.6));
    flex-shrink: 0;
}
.header-titles {
    display: flex;
    flex-direction: column;
}
.header-title-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.custom-header-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.4px;
    margin: 0;
}
.header-badge-tag {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.18), rgba(99, 102, 241, 0.18));
    border: 1px solid rgba(168, 85, 247, 0.45);
    color: #c084fc;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.custom-header-sub1 {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 3px 0 0 0;
}

.header-meta {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
}
.system-status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.28);
    padding: 6px 14px;
    border-radius: 20px;
}
.status-pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    animation: pulseDot 2s infinite ease-in-out;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}
.status-pill-text {
    font-size: 0.76rem;
    font-weight: 700;
    color: #10b981;
    letter-spacing: 0.3px;
}

.storage-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #cbd5e1;
}

/* ==============================================================================
   4. Pestañas Superiores de Grado Estudio
   ============================================================================== */
.custom-tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
    gap: 8px !important;
    margin-bottom: 14px !important;
    padding-bottom: 4px !important;
}
.custom-tabs > .tab-nav > button {
    background: #111728 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 9px !important;
    color: #94a3b8 !important;
    font-size: 0.84rem !important;
    padding: 8px 16px !important;
    font-weight: 600 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.custom-tabs > .tab-nav > button:hover {
    background: #192238 !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
}
.custom-tabs > .tab-nav > button.selected {
    background: #1d253f !important;
    border-color: #8b5cf6 !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.3) !important;
}

/* ==============================================================================
   5. Layout Moderno SaaS 3D: 2 Áreas (Sidebar ~35% + Viewport ~65%)
   ============================================================================== */
.studio-main-row {
    display: flex !important;
    gap: 18px !important;
    align-items: stretch !important;
}

.studio-sidebar-card {
    background: #0f111a !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    display: flex !important;
    flex-direction: column !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
.studio-sidebar-card:hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
}

.studio-viewport-card {
    background: #0f111a !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5) !important;
    display: flex !important;
    flex-direction: column !important;
    transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
}
.studio-viewport-card:hover {
    border-color: rgba(255, 255, 255, 0.12) !important;
}

/* Eliminación de bordes anidados ("cajas dentro de cajas") */
.clean-controls-group,
.clean-group,
.geometry-box,
.studio-sidebar-card .gr-group,
.studio-sidebar-card .gr-form,
.studio-sidebar-card div[class*="group"],
.studio-viewport-card .gr-group,
.studio-viewport-card .gr-form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 8px !important;
}

@media (max-width: 1024px) {
    .studio-main-row {
        flex-direction: column !important;
    }
    .studio-sidebar-card,
    .studio-viewport-card {
        width: 100% !important;
        max-width: 100% !important;
    }
}

/* Cabeceras de Pasos / Secciones */
.step-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.step-badge {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
    border: 1px solid rgba(139, 92, 246, 0.45);
    color: #c084fc;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.8px;
    padding: 3px 8px;
    border-radius: 6px;
    text-transform: uppercase;
}
.step-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: 0.2px;
}

/* Radios de Topología (Acento Violeta / Índigo) */
.preset-radio-list .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}
.preset-radio-list label {
    background: #141a29 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: 1.5px solid #222d42 !important;
    border-radius: 9px !important;
    padding: 8px 12px !important;
    color: #f1f5f9 !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: auto !important;
    box-shadow: none !important;
    margin-bottom: 0 !important;
}
.preset-radio-list label:hover {
    border-color: #3b4d75 !important;
    background: #1a2337 !important;
    transform: translateX(2px) !important;
}
.preset-radio-list label:has(input:checked),
.preset-radio-list label.selected {
    border-color: #8b5cf6 !important;
    background: rgba(139, 92, 246, 0.12) !important;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.22) !important;
    color: #ffffff !important;
}

/* Botón Principal Generar 3D */
.btn-generate-glow {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    border: 1.5px solid #eab308 !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.45), 0 0 10px rgba(234, 179, 8, 0.3) !important;
    border-radius: 11px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
    min-height: 50px !important;
    height: auto !important;
    padding: 10px 16px !important;
    margin: 8px 0 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}
.btn-generate-glow:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 28px rgba(139, 92, 246, 0.65), 0 0 16px rgba(234, 179, 8, 0.5) !important;
}
.btn-generate-glow:active {
    transform: translateY(0px) !important;
}
.btn-generate-glow:disabled,
.btn-generate-glow[disabled] {
    opacity: 0.85 !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: 0 0 16px rgba(139, 92, 246, 0.25) !important;
    background: #1f1830 !important;
    border: 1.5px dashed #a855f7 !important;
    color: #e2e8f0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.btn-generate-glow:disabled .button-icon,
.btn-generate-glow[disabled] .button-icon,
.btn-generate-glow:disabled img,
.btn-generate-glow[disabled] img {
    display: none !important;
}
.btn-generate-glow:disabled::before,
.btn-generate-glow[disabled]::before {
    content: "" !important;
    display: inline-block !important;
    width: 18px !important;
    height: 18px !important;
    margin-right: 10px !important;
    border: 2.5px solid rgba(168, 85, 247, 0.25) !important;
    border-top: 2.5px solid #c084fc !important;
    border-right: 2.5px solid #a855f7 !important;
    border-radius: 50% !important;
    animation: spinBtn 0.75s linear infinite !important;
    vertical-align: middle !important;
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.5) !important;
}
@keyframes spinBtn {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

/* Widget de Progreso */
.live-status-container {
    background: #0b0f19;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 6px;
    box-shadow: inset 0 2px 6px rgba(0, 0, 0, 0.4);
}
.live-status-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.82rem;
    color: #e2e8f0;
    margin-bottom: 8px;
    font-weight: 500;
}
.status-spin {
    animation: spin 1s linear infinite;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.status-done {
    animation: popCheck 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes popCheck {
    0% { transform: scale(0.7); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
.progress-bar-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.progress-label {
    font-size: 0.72rem;
    color: #94a3b8;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    min-width: 68px;
}
.progress-track {
    flex-grow: 1;
    height: 8px;
    background: #172033;
    border-radius: 9999px;
    overflow: hidden;
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899, #6366f1);
    background-size: 200% 100%;
    animation: shimmerProgress 2s infinite linear;
    border-radius: 9999px;
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.5);
    transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes shimmerProgress {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.progress-pct {
    font-size: 0.78rem;
    color: #38bdf8;
    font-weight: 800;
    min-width: 38px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}

/* ==============================================================================
   6. Botones de Descarga: Tarjetas-Botón Interactivas 3D Unificadas
   ============================================================================== */
.download-card {
    background: rgba(15, 17, 26, 0.85) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    margin-top: 10px !important;
}

.download-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.74rem;
    font-weight: 800;
    color: #94a3b8;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.download-buttons-row {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 12px !important;
    width: 100% !important;
}
@media (max-width: 640px) {
    .download-buttons-row {
        grid-template-columns: 1fr !important;
    }
}

.btn-download-format {
    border-radius: 11px !important;
    min-height: 72px !important;
    height: auto !important;
    padding: 10px 12px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
    box-sizing: border-box !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
    text-decoration: none !important;
    overflow: visible !important;
}
.btn-download-format button,
.btn-download-format a {
    min-height: 72px !important;
    height: auto !important;
    padding: 10px 8px !important;
    display: inline-flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 4px !important;
    box-sizing: border-box !important;
    white-space: pre-line !important;
    text-align: center !important;
    line-height: 1.35 !important;
    background: transparent !important;
    border: none !important;
    color: rgba(255, 255, 255, 0.75) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    width: 100% !important;
}
.btn-download-format button::first-line,
.btn-download-format a::first-line {
    font-weight: 800 !important;
    font-size: 0.90rem !important;
    letter-spacing: 0.3px !important;
    color: #ffffff !important;
}
.btn-download-format svg,
.btn-download-format img {
    width: 22px !important;
    height: 22px !important;
    min-width: 22px !important;
    min-height: 22px !important;
    flex-shrink: 0 !important;
    display: inline-block !important;
    margin-bottom: 2px !important;
}

/* Botón GLB: Cyan Eléctrico */
.btn-download-glb {
    background: linear-gradient(135deg, rgba(2, 132, 199, 0.85) 0%, rgba(3, 105, 161, 0.85) 100%) !important;
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
}
.btn-download-glb:hover:not(:disabled) {
    background: linear-gradient(135deg, #0396e3 0%, #0284c7 100%) !important;
    box-shadow: 0 6px 22px rgba(2, 132, 199, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Botón STL: Verde Esmeralda */
.btn-download-stl {
    background: linear-gradient(135deg, rgba(5, 150, 105, 0.85) 0%, rgba(4, 120, 87, 0.85) 100%) !important;
    border: 1px solid #34d399 !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
}
.btn-download-stl:hover:not(:disabled) {
    background: linear-gradient(135deg, #0ba375 0%, #059669 100%) !important;
    box-shadow: 0 6px 22px rgba(16, 185, 129, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Botón 3MF: Violeta Real */
.btn-download-3mf {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.85) 0%, rgba(109, 40, 217, 0.85) 100%) !important;
    border: 1px solid #a78bfa !important;
    box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35) !important;
}
.btn-download-3mf:hover:not(:disabled) {
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
    box-shadow: 0 6px 22px rgba(139, 92, 246, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Estados Deshabilitados */
.btn-download-format:disabled,
.btn-download-format[disabled],
.btn-download-format.disabled {
    opacity: 0.38 !important;
    filter: grayscale(40%) !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
    border-style: dashed !important;
}

/* ==============================================================================
   7. Tarjeta Guardar en Proyecto / Juego
   ============================================================================== */
.game-save-card {
    background: #121828 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 11px !important;
    padding: 12px 14px !important;
    margin-top: 8px !important;
}
.game-save-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.86rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}
.btn-save-game {
    background: #192238 !important;
    border: 1px solid #2d3c5e !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
    min-height: 40px !important;
    height: auto !important;
    padding: 8px 14px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.btn-save-game:hover {
    background: #23304e !important;
    border-color: #4f6b9f !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* ==============================================================================
   8. Tab 2: Biblioteca de Modelos (Layout Responsivo y Botón de Recarga)
   ============================================================================== */
.library-layout-row {
    display: flex !important;
    gap: 18px !important;
    align-items: stretch !important;
}
.library-controls-card {
    background: linear-gradient(180deg, #101626 0%, #0c111d 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
}
.library-viewer-card {
    background: linear-gradient(180deg, #101626 0%, #0c111d 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
}

@media (max-width: 1024px) {
    .library-layout-row {
        flex-direction: column !important;
    }
    .library-controls-card,
    .library-viewer-card {
        width: 100% !important;
        max-width: 100% !important;
    }
}

.btn-refresh-library {
    background: #141c2f !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 9px !important;
    color: #e2e8f0 !important;
    font-weight: 700 !important;
    font-size: 0.86rem !important;
    height: 40px !important;
    margin-bottom: 8px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.btn-refresh-library:hover {
    background: #1c2742 !important;
    border-color: #6366f1 !important;
    color: #ffffff !important;
    box-shadow: 0 0 14px rgba(99, 102, 241, 0.3) !important;
    transform: translateY(-1px) !important;
}

.select-model-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
    margin-bottom: 8px !important;
}

.btn-load-model {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: 1px solid #818cf8 !important;
    border-radius: 9px !important;
    color: #ffffff !important;
    font-size: 0.86rem !important;
    font-weight: 700 !important;
    height: 42px !important;
    min-width: 120px !important;
    padding: 0 14px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}
.btn-load-model:hover {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    box-shadow: 0 0 16px rgba(99, 102, 241, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* ==============================================================================
   9. Componentes Nativos de Gradio (Dropdown, Inputs, Model3D)
   ============================================================================== */
.gradio-model3d,
.studio-model3d,
.studio-model3d > div,
.gradio-model3d > div,
.library-viewer-card .gradio-model3d {
    background: radial-gradient(circle at 50% 45%, #1a1d2e 0%, #090b12 100%) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: inset 0 2px 20px rgba(0, 0, 0, 0.6) !important;
}

.gradio-dropdown .wrap-inner,
.gradio-dropdown input,
.gradio-textbox input,
.gradio-textbox textarea {
    background-color: #0d121f !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #f8fafc !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.gradio-dropdown .wrap-inner:hover,
.gradio-dropdown input:focus,
.gradio-textbox input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.25) !important;
}

/* Tablero de ajedrez para silueta alfa */
.checkerboard-image {
    background-image: linear-gradient(45deg, #151a29 25%, transparent 25%), linear-gradient(-45deg, #151a29 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #151a29 75%), linear-gradient(-45deg, transparent 75%, #151a29 75%) !important;
    background-size: 16px 16px !important;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px !important;
    background-color: #0f1320 !important;
}

/* ==============================================================================
   10. Controles Avanzados del Visor 3D (Ultra-Minimalista)
   ============================================================================== */
.viewer-controls-card {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 6px !important;
    box-shadow: none !important;
}

/* Fila de Shading tipo Segmented Control / Apple Style */
.viewer-shading-mode-radio {
    margin-bottom: 4px !important;
    background: transparent !important;
    border: none !important;
}

.viewer-shading-mode-radio .wrap {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 3px !important;
    background: #090c15 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 7px !important;
    padding: 2px !important;
}

/* Ocultar el radio button nativo circular */
.viewer-shading-mode-radio input[type="radio"] {
    display: none !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    position: absolute !important;
}

.viewer-shading-mode-radio label {
    flex: 1 1 0% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 4px 2px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    text-align: center !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    white-space: nowrap !important;
    height: 24px !important;
    min-height: 24px !important;
}

.viewer-shading-mode-radio label:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05) !important;
}

.viewer-shading-mode-radio label.selected {
    background: #4f46e5 !important;
    color: #ffffff !important;
    box-shadow: 0 1px 6px rgba(79, 70, 229, 0.4) !important;
}

/* Fila de Subcontroles Fondo y Cámara */
.viewer-subcontrols-row {
    display: flex !important;
    gap: 6px !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
}

.viewer-subcontrols-row > div {
    flex: 1 1 50% !important;
    min-width: 0 !important;
    margin: 0 !important;
}

.minimal-dropdown {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

.minimal-dropdown .wrap-inner,
.minimal-dropdown input {
    background-color: #0b0f1a !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
    height: 26px !important;
    min-height: 26px !important;
    padding: 2px 8px !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
}

.minimal-dropdown .wrap-inner:hover {
    border-color: rgba(99, 102, 241, 0.5) !important;
}

/* Barra de Estadísticas de Malla */
.mesh-stats-container {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}

.mesh-stats-bar {
    display: grid !important;
    grid-template-columns: repeat(4, 1fr) !important;
    gap: 6px !important;
    background: #0a0d16 !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 8px !important;
    padding: 8px 10px !important;
}

.mesh-stats-bar.empty {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    padding: 10px !important;
    color: #64748b !important;
    font-size: 0.8rem !important;
}

.mesh-stat-item {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

.mesh-stat-item:last-child {
    border-right: none !important;
}

.mesh-stat-label {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    color: #64748b !important;
    text-transform: uppercase !important;
}

.mesh-stat-value {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    margin-top: 2px !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}

.mesh-stat-value.active {
    color: #10b981 !important;
}

.mesh-stat-value.inactive {
    color: #f59e0b !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#07080d",
    body_background_fill_dark="#07080d",
    block_background_fill="#0f111a",
    block_background_fill_dark="#0f111a",
    block_border_color="rgba(255, 255, 255, 0.06)",
    block_border_color_dark="rgba(255, 255, 255, 0.06)",
    block_radius="14px",
    block_label_background_fill="rgba(11, 13, 23, 0.78)",
    block_label_background_fill_dark="rgba(11, 13, 23, 0.78)",
    block_label_text_color="#cbd5e1",
    block_label_text_color_dark="#cbd5e1",
    block_label_text_size="*text_xs",
    block_label_text_weight="700",
    block_title_text_color="#f8fafc",
    block_title_text_color_dark="#f8fafc",
    input_background_fill="#0d121f",
    input_background_fill_dark="#0d121f",
    input_border_color="rgba(255, 255, 255, 0.08)",
    input_border_color_dark="rgba(255, 255, 255, 0.08)",
    input_border_color_focus="#8b5cf6",
    input_border_color_focus_dark="#8b5cf6",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff"
)

# ==============================================================================
# 6. Construcción del Dashboard con Gradio Blocks
# ==============================================================================

page_title = "Kaboomi 3D Studio" if MODE == "kambo" else "3D AI Studio"
header_title = "Kaboomi 3D Studio" if MODE == "kambo" else "3D AI Studio"
badge_tag = "Hunyuan3D-2 Turbo • Kaboomi Edition" if MODE == "kambo" else "Hunyuan3D-2 Turbo"
header_subtitle = (
    "Generador de Modelos 3D (.glb) Optimizados para Godot 4"
    if MODE == "kambo"
    else "Generador Local de Modelos 3D (.glb, .stl, .3mf) de Alta Fidelidad"
)

with gr.Blocks(title=page_title) as demo:
    
    # Cabecera Ejecutiva sin Emojis
    gr.HTML(
        f"""
        <div class="custom-header">
            <div class="header-brand">
                <svg class="header-logo-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.12 6.4-6.05-3.5a2 2 0 0 0-2.14 0L6.88 6.4A2 2 0 0 0 6 8.13v7.74a2 2 0 0 0 1 1.73l6.05 3.5a2 2 0 0 0 2.14 0l6.05-3.5a2 2 0 0 0 1-1.73V8.13a2 2 0 0 0-.88-1.73Z"/>
                    <path d="m12 22 9-5"/>
                    <path d="M12 12V2"/>
                    <path d="M3 8.5 12 13l9-4.5"/>
                </svg>
                <div class="header-titles">
                    <div class="header-title-row">
                        <span class="custom-header-title">{header_title}</span>
                        <span class="header-badge-tag">{badge_tag}</span>
                    </div>
                    <p class="custom-header-sub1">{header_subtitle}</p>
                </div>
            </div>
            <div class="header-meta">
                <div class="system-status-pill">
                    <span class="status-pulse-dot"></span>
                    <span class="status-pill-text">{GPU_STATUS_TEXT}</span>
                </div>
                <div class="storage-pill">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/></svg>
                    <span>DISCO D: ACTIVO</span>
                </div>
            </div>
        </div>
        """
    )

    with gr.Tabs(elem_classes=["custom-tabs"]):
        # TAB PRINCIPAL (Generador 3D)
        with gr.Tab("Generador 3D"):
            with gr.Row(elem_classes=["studio-main-row"]):
                
                # ==============================================================
                # SIDEBAR IZQUIERDO: 1. Entrada y Parámetros (scale=4, ~35%)
                # ==============================================================
                with gr.Column(scale=4, elem_classes=["studio-sidebar-card"]):
                    gr.HTML(
                        """
                        <div class="step-card-header">
                            <span class="step-badge">PASO 1</span>
                            <span class="step-title">IMAGEN Y SILUETA</span>
                        </div>
                        """
                    )
                    
                    input_img = gr.Image(
                        label="Imagen o Sprite 2D de Entrada",
                        height=250,
                        type="pil",
                        sources=["upload", "clipboard"],
                        elem_classes=["checkerboard-image"]
                    )
                    clean_silhouette_state = gr.State(value=None)

                    gr.HTML(
                        """
                        <div class="step-card-header" style="margin-top: 14px;">
                            <span class="step-badge">PASO 2</span>
                            <span class="step-title">TOPOLOGÍA Y MOTOR 3D</span>
                        </div>
                        """
                    )
                    
                    with gr.Group(elem_classes=["clean-controls-group"]):
                        gr.HTML('<div style="font-size:0.78rem;font-weight:700;color:#94a3b8;margin-bottom:6px;letter-spacing:0.5px;text-transform:uppercase;">Presets de Rendimiento:</div>')
                        
                        preset_radio = gr.Radio(
                            choices=list(PRESET_CONFIG_MAP.keys()),
                            value="Móvil Rápido (4,000 caras • Balanceado)",
                            label="",
                            show_label=False,
                            elem_classes=["preset-radio-list"]
                        )
                        
                        ai_model_dropdown = gr.Dropdown(
                            choices=[
                                "Hunyuan3D-2 Turbo (Geometría HD • Sólidos 3D)",
                                "Stable Fast 3D (SF3D • Texturas PBR Instantáneas)"
                            ],
                            value=(
                                "Stable Fast 3D (SF3D • Texturas PBR Instantáneas)"
                                if MODE == "kambo"
                                else "Hunyuan3D-2 Turbo (Geometría HD • Sólidos 3D)"
                            ),
                            label="Motor de Inteligencia Artificial 3D",
                            info="Hunyuan3D: Máxima definición sólida | SF3D: Texturas PBR en 1s",
                            interactive=True
                        )

                        vram_dropdown = gr.Dropdown(
                            choices=VRAM_PROFILE_CHOICES,
                            value="Auto-detectar (Recomendado)",
                            label="Perfil de Hardware / VRAM",
                            info="Adapta memoria y offloading a tu GPU",
                            interactive=True
                        )

                        chk_texture = gr.Checkbox(
                            value=True if MODE == "kambo" else False,
                            label="Generar y Proyectar Textura de Color",
                            info="Desmarcar para modo Arcilla / Geometría Pura (ideal para impresión 3D / amigos)"
                        )

                    btn_generate = gr.Button(
                        "GENERAR MODELO 3D (GLB / STL / 3MF)",
                        icon="assets/icons/wand.svg",
                        elem_classes=["btn-generate-glow"]
                    )
                    
                    status_widget = gr.HTML(
                        value=render_progress_html("Listo para generar. Sube una imagen y presiona generar.", 0)
                    )

                # ==============================================================
                # LIENZO DERECHO: 2. Viewport 3D y Exportación (scale=8, ~65%)
                # ==============================================================
                with gr.Column(scale=8, elem_classes=["studio-viewport-card"]):
                    gr.HTML(
                        """
                        <div class="step-card-header">
                            <span class="step-badge">VIEWPORT</span>
                            <span class="step-title">VISOR 3D Y EXPORTACIÓN</span>
                        </div>
                        """
                    )
                    
                    _initial_preview = get_initial_preview_model()
                    _has_real_model = bool(_initial_preview and not _initial_preview.endswith("placeholder.glb") and not _initial_preview.endswith("empty.glb"))
                    _init_stl = str(Path(_initial_preview).with_suffix(".stl")) if _has_real_model and Path(_initial_preview).with_suffix(".stl").exists() else None
                    _init_3mf = str(Path(_initial_preview).with_suffix(".3mf")) if _has_real_model and Path(_initial_preview).with_suffix(".3mf").exists() else None

                    with gr.Group(elem_classes=["viewer-controls-card"]):
                        viewer_shading_mode = gr.Radio(
                            choices=["Textura", "Arcilla", "Malla", "Puntos"],
                            value="Textura",
                            show_label=False,
                            elem_classes=["viewer-shading-mode-radio"]
                        )
                        with gr.Row(elem_classes=["viewer-subcontrols-row"]):
                            viewer_bg_choice = gr.Dropdown(
                                choices=["Fondo: Oscuro", "Fondo: CAD", "Fondo: Claro"],
                                value="Fondo: Oscuro",
                                show_label=False,
                                scale=1,
                                elem_classes=["minimal-dropdown"]
                            )
                            viewer_cam_choice = gr.Dropdown(
                                choices=["Cámara: 3D", "Cámara: Frontal", "Cámara: Lateral", "Cámara: Superior", "Cámara: Centrar"],
                                value="Cámara: 3D",
                                show_label=False,
                                scale=1,
                                elem_classes=["minimal-dropdown"]
                            )

                    viewer_3d = gr.Model3D(
                        value=_initial_preview,
                        label="Visor 3D Interactivo",
                        height=560,
                        camera_position=(-90, 80, 2.4),
                        clear_color=[0.10, 0.12, 0.18, 1.0],
                        elem_classes=["studio-model3d"]
                    )

                    mesh_stats_display = gr.HTML(
                        value=render_mesh_stats_html(_initial_preview if _has_real_model else None),
                        elem_classes=["mesh-stats-container"]
                    )
                    active_model_path = gr.State(value=_initial_preview if _has_real_model else None)
                    
                    with gr.Group(elem_classes=["download-card"]):
                        gr.HTML(
                            """
                            <div class="download-card-header">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7 10 12 15 17 10"/>
                                    <line x1="12" x2="12" y1="15" y2="3"/>
                                </svg>
                                <span>DESCARGAS POR FORMATO</span>
                            </div>
                            """
                        )
                        
                        with gr.Row(elem_classes=["download-buttons-row"]):
                            download_glb_btn = gr.DownloadButton(
                                label=".GLB (glTF)\nColor PBR · Godot / Web",
                                value=_initial_preview if _has_real_model else None,
                                icon="assets/icons/cube.svg",
                                elem_classes=["btn-download-format", "btn-download-glb"],
                                interactive=_has_real_model,
                                scale=1
                            )
                            download_stl_btn = gr.DownloadButton(
                                label=".STL (Malla)\nSólido 3D · Slicers / CAD",
                                value=_init_stl if _has_real_model else None,
                                icon="assets/icons/printer.svg",
                                elem_classes=["btn-download-format", "btn-download-stl"],
                                interactive=_has_real_model and bool(_init_stl),
                                scale=1
                            )
                            download_3mf_btn = gr.DownloadButton(
                                label=".3MF (Slicer)\nManufactura · Bambu / Prusa",
                                value=_init_3mf if _has_real_model else None,
                                icon="assets/icons/layers.svg",
                                elem_classes=["btn-download-format", "btn-download-3mf"],
                                interactive=_has_real_model and bool(_init_3mf),
                                scale=1
                            )
                    
                    if MODE == "kambo":
                        with gr.Group(elem_classes=["game-save-card"]):
                            gr.HTML(
                                """
                                <div class="game-save-header">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <line x1="6" x2="10" y1="12" y2="12"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="15" x2="15.01" y1="13" y2="13"/><line x1="18" x2="18.01" y1="11" y2="11"/><rect width="20" height="12" x="2" y="6" rx="6"/>
                                    </svg>
                                    <span>Integración Directa con Godot 4 (Kaboomi Party)</span>
                                </div>
                                """
                            )
                            with gr.Row():
                                dest_category = gr.Dropdown(
                                    choices=["/bombos/", "/bozis/", "/cosmetics/"],
                                    value="/bombos/",
                                    label="Destino:",
                                    scale=1
                                )
                                dest_subcategory = gr.Dropdown(
                                    choices=["/bozis/", "/bombos/", "/cosmetics/"],
                                    value="/bozis/",
                                    label="Sub-carpeta:",
                                    scale=1
                                )
                            txt_model_name = gr.Textbox(
                                placeholder="Nombre del modelo (ej. dragon_bombo)...",
                                label="Nombre de Archivo",
                                lines=1
                            )
                            btn_save_game = gr.Button(
                                "Guardar directo en el Juego",
                                icon="assets/icons/gamepad.svg",
                                elem_classes=["btn-save-game"]
                            )
                            export_feedback = gr.Markdown()
                    else:
                        with gr.Group(elem_classes=["game-save-card"]):
                            gr.HTML(
                                """
                                <div class="game-save-header">
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                                    </svg>
                                    <span>Exportar a Carpeta de Proyecto</span>
                                </div>
                                """
                            )
                            custom_dest_folder = gr.Textbox(
                                label="Ruta de Proyecto o Carpeta Destino",
                                placeholder="Ej. C:/Proyectos/Godot/assets o D:/Unity/Models...",
                                lines=1
                            )
                            txt_model_name = gr.Textbox(
                                placeholder="Nombre del modelo (ej. personaje_3d)...",
                                label="Nombre de Archivo",
                                lines=1
                            )
                            btn_save_custom = gr.Button(
                                "Copiar a Carpeta de Proyecto",
                                icon="assets/icons/cube.svg",
                                elem_classes=["btn-save-game"]
                            )
                            export_feedback = gr.Markdown()

        # TAB SECUNDARIA (Biblioteca de Modelos)
        library_title = "Biblioteca de Modelos 3D (.glb) en Kaboomi Party" if MODE == "kambo" else "Biblioteca de Modelos 3D (.glb) Generados"
        with gr.Tab("Biblioteca de Modelos"):
            gr.HTML(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin: 4px 0 12px 0;">
                    <span style="font-size:1.15rem; font-weight:700; color:#f8fafc;">{library_title}</span>
                </div>
                """
            )
            with gr.Row(elem_classes=["library-layout-row"]):
                with gr.Column(scale=5, elem_classes=["library-controls-card"]):
                    btn_refresh_library = gr.Button("Actualizar Lista", icon="assets/icons/refresh.svg", elem_classes=["btn-refresh-library"])
                    with gr.Row(elem_classes=["select-model-row"]):
                        dropdown_models = gr.Dropdown(
                            choices=list_existing_models(),
                            label="Selecciona un modelo",
                            scale=4,
                            allow_custom_value=True
                        )
                        btn_load_model = gr.Button(
                            "Visualizar",
                            icon="assets/icons/cube.svg",
                            elem_classes=["btn-load-model"],
                            scale=1
                        )
                    with gr.Group(elem_classes=["download-card"]):
                        gr.HTML(
                            """
                            <div class="download-card-header">
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                    <polyline points="7 10 12 15 17 10"/>
                                    <line x1="12" x2="12" y1="15" y2="3"/>
                                </svg>
                                <span>DESCARGAR MODELO SELECCIONADO</span>
                            </div>
                            """
                        )
                        with gr.Row(elem_classes=["download-buttons-row"]):
                            lib_download_glb = gr.DownloadButton(
                                label=".GLB (glTF)\nColor PBR · Godot / Web",
                                icon="assets/icons/cube.svg",
                                elem_classes=["btn-download-format", "btn-download-glb"],
                                interactive=False,
                                scale=1
                            )
                            lib_download_stl = gr.DownloadButton(
                                label=".STL (Malla)\nSólido 3D · Slicers / CAD",
                                icon="assets/icons/printer.svg",
                                elem_classes=["btn-download-format", "btn-download-stl"],
                                interactive=False,
                                scale=1
                            )
                            lib_download_3mf = gr.DownloadButton(
                                label=".3MF (Slicer)\nManufactura · Bambu / Prusa",
                                icon="assets/icons/layers.svg",
                                elem_classes=["btn-download-format", "btn-download-3mf"],
                                interactive=False,
                                scale=1
                            )
                with gr.Column(scale=7, elem_classes=["library-viewer-card"]):
                    with gr.Group(elem_classes=["viewer-controls-card"]):
                        lib_shading_mode = gr.Radio(
                            choices=["Textura", "Arcilla", "Malla", "Puntos"],
                            value="Textura",
                            show_label=False,
                            elem_classes=["viewer-shading-mode-radio"]
                        )
                        with gr.Row(elem_classes=["viewer-subcontrols-row"]):
                            lib_bg_choice = gr.Dropdown(
                                choices=["Fondo: Oscuro", "Fondo: CAD", "Fondo: Claro"],
                                value="Fondo: Oscuro",
                                show_label=False,
                                scale=1,
                                elem_classes=["minimal-dropdown"]
                            )
                            lib_cam_choice = gr.Dropdown(
                                choices=["Cámara: 3D", "Cámara: Frontal", "Cámara: Lateral", "Cámara: Superior", "Cámara: Centrar"],
                                value="Cámara: 3D",
                                show_label=False,
                                scale=1,
                                elem_classes=["minimal-dropdown"]
                            )

                    library_viewer = gr.Model3D(
                        value=_initial_preview,
                        label="Visualizador 3D Interactivo",
                        height=420,
                        camera_position=(-90, 80, 2.4),
                        clear_color=[0.08, 0.10, 0.16, 1.0]
                    )

                    lib_mesh_stats_display = gr.HTML(
                        value=render_mesh_stats_html(_initial_preview if _has_real_model else None),
                        elem_classes=["mesh-stats-container"]
                    )
                    lib_active_model_path = gr.State(value=_initial_preview if _has_real_model else None)

        # TAB TERCIARIA (Estado del Sistema)
        with gr.Tab("Configuración del Sistema"):
            mode_label = "Kaboomi Party Edition" if MODE == "kambo" else "Estándar / General"
            kambo_sys_info = f"- **Ruta de Juego Godot 4:** `{KABOOMI_ROOT}`\n" if MODE == "kambo" else ""
            auto_vram_key = detect_vram_profile()
            auto_vram_desc = VRAM_PROFILES.get(auto_vram_key, {}).get("desc", auto_vram_key)
            gr.Markdown(
                f"""
                ### Arquitectura y Entorno de Ejecución
                - **Modo Activo:** `{mode_label}`
                - **GPU:** {GPU_SYS_TEXT}
                - **Perfil VRAM Auto-Detectado:** `{auto_vram_key}` ({auto_vram_desc})
                - **Almacenamiento Local:** `D:\\AI_Models` (100% en Disco D:)
                - **Directorio de Salida:** `{OUTPUT_DIR}` (Formatos: `.GLB`, `.STL`, `.3MF`)
                {kambo_sys_info}- **Motor de Inferencia:** Tencent Hunyuan3D-2 Turbo + Rembg U2-Net
                - **Filtro de Geometría:** Eliminación activa de losas e islas parásitas (< 3% volumen)
                """
            )

    # ==========================================================================
    # Conexión de Eventos y Callbacks
    # ==========================================================================

    # 1. Extracción de silueta automática al subir o pegar imagen
    input_img.upload(
        fn=on_image_uploaded,
        inputs=[input_img],
        outputs=[input_img, clean_silhouette_state, status_widget]
    )
    input_img.clear(
        fn=lambda: (None, render_progress_html("Listo para generar. Sube una imagen.", 0)),
        inputs=[],
        outputs=[clean_silhouette_state, status_widget]
    )

    # 2. Generación 3D en streaming con bloqueo anti-doble clic instantáneo
    btn_generate.click(
        fn=lambda: (
            gr.update(interactive=False, value="PROCESANDO MODELO 3D..."),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False)
        ),
        inputs=[],
        outputs=[btn_generate, download_glb_btn, download_stl_btn, download_3mf_btn]
    ).then(
        fn=ui_generate_3d,
        inputs=[
            input_img,
            clean_silhouette_state,
            preset_radio,
            ai_model_dropdown,
            chk_texture,
            txt_model_name,
            vram_dropdown
        ],
        outputs=[viewer_3d, download_glb_btn, download_stl_btn, download_3mf_btn, status_widget]
    ).then(
        fn=on_generation_completed,
        inputs=[],
        outputs=[btn_generate, active_model_path, mesh_stats_display, viewer_shading_mode]
    )

    # 3. Controladores de Visualización Interactiva y Malla (Paso 3)
    viewer_shading_mode.change(
        fn=on_change_shading_mode,
        inputs=[viewer_shading_mode, active_model_path],
        outputs=[viewer_3d]
    )
    viewer_bg_choice.change(
        fn=on_change_background,
        inputs=[viewer_bg_choice],
        outputs=[viewer_3d]
    )
    viewer_cam_choice.change(
        fn=on_camera_preset,
        inputs=[viewer_cam_choice],
        outputs=[viewer_3d]
    )

    # 4. Guardar directamente en el juego o carpeta personalizada
    if MODE == "kambo":
        btn_save_game.click(
            fn=ui_save_to_game,
            inputs=[dest_category, dest_subcategory, txt_model_name],
            outputs=[export_feedback]
        )
    else:
        btn_save_custom.click(
            fn=ui_save_to_custom_dir,
            inputs=[custom_dest_folder, txt_model_name],
            outputs=[export_feedback]
        )

    # 5. Explorador de modelos en biblioteca
    btn_refresh_library.click(
        fn=list_existing_models,
        inputs=[],
        outputs=[dropdown_models]
    )
    
    # Actualización automática al cambiar, ingresar o seleccionar
    dropdown_models.change(
        fn=on_select_library_model,
        inputs=[dropdown_models],
        outputs=[library_viewer, lib_download_glb, lib_download_stl, lib_download_3mf, lib_active_model_path, lib_mesh_stats_display, lib_shading_mode]
    )
    dropdown_models.input(
        fn=on_select_library_model,
        inputs=[dropdown_models],
        outputs=[library_viewer, lib_download_glb, lib_download_stl, lib_download_3mf, lib_active_model_path, lib_mesh_stats_display, lib_shading_mode]
    )
    dropdown_models.select(
        fn=on_select_library_model,
        inputs=[dropdown_models],
        outputs=[library_viewer, lib_download_glb, lib_download_stl, lib_download_3mf, lib_active_model_path, lib_mesh_stats_display, lib_shading_mode]
    )

    # Botón explícito "Visualizar"
    btn_load_model.click(
        fn=on_select_library_model,
        inputs=[dropdown_models],
        outputs=[library_viewer, lib_download_glb, lib_download_stl, lib_download_3mf, lib_active_model_path, lib_mesh_stats_display, lib_shading_mode]
    )

    # Controladores de Visualización Interactiva y Malla (Biblioteca)
    lib_shading_mode.change(
        fn=on_change_shading_mode,
        inputs=[lib_shading_mode, lib_active_model_path],
        outputs=[library_viewer]
    )
    lib_bg_choice.change(
        fn=on_change_background,
        inputs=[lib_bg_choice],
        outputs=[library_viewer]
    )
    lib_cam_choice.change(
        fn=on_camera_preset,
        inputs=[lib_cam_choice],
        outputs=[library_viewer]
    )


# ==============================================================================
# Lanzamiento del Servidor
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    if MODE == "kambo":
        print(" KABOOMI 3D STUDIO (GRADIO WEBUI)")
        print(f" -> Modo: Kaboomi Party Edition")
        print(f" -> Directorio de Salida: {OUTPUT_DIR}")
        print(f" -> Modelos y Caché: D:\\AI_Models (100% Protegido)")
        print(f" -> Proyecto Kaboomi Party: {KABOOMI_ROOT}")
    else:
        print(" 3D AI STUDIO (GRADIO WEBUI)")
        print(f" -> Modo: Estándar / General")
        print(f" -> Directorio de Salida: {OUTPUT_DIR}")
        print(f" -> Modelos y Caché: D:\\AI_Models (100% Protegido)")
    print("="*70)
    
    allowed_list = [str(OUTPUT_DIR.resolve()), str(Path("assets").resolve())]
    kab_dir_obj = get_kaboomi_dir()
    if kab_dir_obj.exists():
        allowed_list.append(str(kab_dir_obj.resolve()))
        allowed_list.append(str((kab_dir_obj / "assets").resolve()))
        allowed_list.append(str((kab_dir_obj / "assets" / "models").resolve()))
        
    demo.launch(
        server_name="127.0.0.1",
        server_port=cmd_args.port,
        share=cmd_args.share,
        inbrowser=False,
        theme=custom_theme,
        css=CUSTOM_CSS,
        allowed_paths=allowed_list
    )
