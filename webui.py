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

import gc
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
# 1. Rutas de Almacenamiento (raíz elegida en setup.bat, guardada en .env)
# ==============================================================================

from studio_env import MODELS_ROOT, OUTPUT_DIR

# Selección de GPU (debe ocurrir antes de importar torch / gradio / rembg)
from gpu_select import select_gpu
select_gpu()

import gradio as gr
from PIL import Image
import rembg
import torch
import trimesh

try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


from img2glb import (
    flush_vram,
    print_real_vram_status,
    unload_sf3d,
    unload_hunyuan,
    unload_all_models,
    park_all_models,
    TEXTURE_SAFE_MAX_FACES,
    apply_taubin_smoothing,
    inpaint_edge_padding,
    preprocess_and_remove_background,
    clean_mesh_floating_artifacts,
    decimate_mesh_to_target_faces,
    decimate_mesh_preserving_texture,
    inpaint_uv_atlas_texture,
    center_and_orient_mesh,
    postprocess_and_validate_glb,
    export_formats_from_mesh,
    convert_glb_to_format,
    Hunyuan3DGenerator,
    SF3DGenerator,
    is_dit_cached,
    VRAM_PROFILES,
    detect_vram_profile,
    DEFAULT_OUTPUT_DIR
)
from thumbnails import get_thumbnail, make_placeholder, warm_thumbnails
from export_profiles import (
    ALL_FORMATS,
    MODEL_FORMATS,
    PRINT_FORMATS,
    PRINTERS,
    CUSTOM_PRINTER,
    SLICERS,
    active_formats,
    detect_slicers,
    export_with_profile,
    load_settings,
    normalize_settings,
    open_in_slicer,
    printer_dims,
    profile_label,
    resolve_slicer,
    save_settings,
)

# ==============================================================================
# 2. Pre-inicialización Inteligente de Rembg (U2-Net) al arrancar el servidor
# ==============================================================================

print("\n" + "="*70)
print(" [PRE-CARGA INTELIGENTE] Inicializando sesión de Rembg U2-Net en CPU (providers=['CPUExecutionProvider'])...")
t_rembg_start = time.time()
try:
    REMBG_SESSION = rembg.new_session("u2net", providers=['CPUExecutionProvider'])
    print(f" -> Rembg U2-Net listo en CPU en {time.time() - t_rembg_start:.2f}s (0 MB VRAM ocupados)")
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
parser.add_argument("--gpu", default=None, help="GPU para la IA: índice de nvidia-smi (0, 1...), 'cpu' o 'ask' para forzar el menú")
cmd_args, _ = parser.parse_known_args()
MODE = cmd_args.mode.lower()

# Rutas específicas para el modo Kaboomi Party (KABOOMI_DIR del .env tiene prioridad;
# las rutas relativas se resuelven desde la carpeta del proyecto, no desde donde se lance)
SCRIPT_DIR = Path(__file__).resolve().parent
POSSIBLE_KABOOMI_ROOTS = [
    *([SCRIPT_DIR / os.environ["KABOOMI_DIR"]] if os.environ.get("KABOOMI_DIR") else []),
    Path.home() / "Proyectos" / "Games" / "kaboomi-party",
    Path.home() / "Proyectos" / "kaboomi-party",
    SCRIPT_DIR.parent / "Games" / "kaboomi-party",
    SCRIPT_DIR.parent / "kaboomi-party"
]

def get_kaboomi_dir() -> Path:
    """Detecta la carpeta raíz del proyecto Kaboomi Party."""
    for p in POSSIBLE_KABOOMI_ROOTS:
        if p.exists() and ((p / "project.godot").exists() or (p / "assets").exists()):
            return p.resolve()
    return POSSIBLE_KABOOMI_ROOTS[0].resolve()

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
LAST_GEN_OK: bool = False
LAST_GEN_SUMMARY: str = ""

# Perfil de exportación (modal de configuración). Cada navegador guarda el suyo en
# localStorage (gr.BrowserState); el .json en disco es el valor inicial para navegadores nuevos.
SETTINGS_FILE = MODELS_ROOT / "studio_settings.json"
INITIAL_SETTINGS: dict = load_settings(SETTINGS_FILE)
EXPORTS_DIR = OUTPUT_DIR / "exports"
PLACEHOLDER_GLB = Path(__file__).parent / "assets" / "placeholder.glb"
EMPTY_GLB = Path(__file__).parent / "assets" / "empty.glb"

def get_initial_preview_model() -> Optional[str]:
    """Retorna la escena vacía para que el visor inicie completamente limpio y sin modelos cargados."""
    if EMPTY_GLB.exists():
        return str(EMPTY_GLB.resolve())
    return None

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    _gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "GeForce ").strip()
    _vram_gb = int(round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)))
    GPU_STATUS_TEXT = f"{_gpu_name} · {_vram_gb} GB"
    GPU_SYS_TEXT = f"NVIDIA {_gpu_name} ({_vram_gb}GB VRAM fp16)"
    print("\n" + "=" * 76)
    print("   [CONFIRMACIÓN DE GPU AL ARRANCAR EL SERVIDOR]")
    print(f"   ✓ GPU ASIGNADA PARA IA: {torch.cuda.get_device_name(0)} (cuda:0)")
    print(f"   ✓ VRAM DEDICADA DETECTADA: {_vram_gb} GB")
    print("   ✓ OTRAS GPUs: OCULTAS PARA CUDA (CUDA_VISIBLE_DEVICES)")
    # Pulso de activación de 1.2s para verificar en el Administrador de Tareas la GPU elegida
    try:
        _a = torch.randn((5120, 5120), device="cuda:0", dtype=torch.float16)
        _b = torch.randn((5120, 5120), device="cuda:0", dtype=torch.float16)
        _t_pulse = time.time()
        while time.time() - _t_pulse < 1.2:
            torch.matmul(_a, _b)
            torch.cuda.synchronize()
        del _a, _b
        torch.cuda.empty_cache()
        print("   ✓ PULSO DE VALIDACIÓN EN VIVO: OK")
    except Exception as _e_pulse:
        print(f"   ! PULSO DE VALIDACIÓN FALLÓ: {_e_pulse}")
    print("=" * 76 + "\n")
else:
    GPU_STATUS_TEXT = "CPU"
    GPU_SYS_TEXT = "CPU (Sin aceleración CUDA)"
    print("\n[AVISO] No se detectó GPU CUDA. Corriendo en CPU.\n")


PRESET_CONFIG_MAP = {
    "Alta Calidad (50,000 caras • Con textura: 40k)": {"faces": 50000, "steps": 5, "octree": 256},
    # octree 384 se limita según el perfil de VRAM (256 en 6-8 GB, 320 en 12 GB, 384 en 16 GB+)
    "Máxima Calidad HD (Malla Completa sin reducir • Con textura: 40k)": {"faces": 0, "steps": 5, "octree": 384},
    "Equilibrado (20,000 caras • Estándar 3D)": {"faces": 20000, "steps": 5, "octree": 256},
    "Móvil Low-Poly (5,000 caras • Juegos Ligeros)": {"faces": 5000, "steps": 4, "octree": 256}
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

# Etiquetas cortas para la interfaz: (texto visible, valor interno). Los valores no cambian.
PRESET_UI_CHOICES = list(zip(["Alta · 50k", "HD · Completa", "Media · 20k", "Low-poly · 5k"], PRESET_CONFIG_MAP))
ENGINE_HUNYUAN = "Hunyuan3D-2 (Secuencial 2 Etapas • Máxima Calidad)"
ENGINE_SF3D = "Stable Fast 3D (SF3D • Rápido 1 Paso)"
ENGINE_UI_CHOICES = [("Hunyuan3D-2", ENGINE_HUNYUAN), ("SF3D · Rápido", ENGINE_SF3D)]
VRAM_UI_CHOICES = list(zip(["Auto", "6 GB", "8 GB", "12 GB", "16 GB+", "CPU"], VRAM_PROFILE_CHOICES))

GENERATE_LABEL = "Generar modelo 3D"
VIEWER_DARK_BG = [0.07, 0.08, 0.10, 1.0]


def render_progress_html(step_text: str, percent: int) -> str:
    """Renderiza la línea de estado compacta: punto de estado, texto, porcentaje y barra fina."""
    if "error" in step_text.lower():
        state = "err"
    elif percent == 100:
        state = "done"
    elif percent > 0:
        state = "run"
    else:
        state = "idle"
    pct_html = f'<span class="st-pct">{percent}%</span>' if percent > 0 else ""
    return f"""
    <div class="st st-{state}">
        <div class="st-row">
            <span class="st-dot"></span>
            <span class="st-text" title="{step_text}">{step_text}</span>
            {pct_html}
        </div>
        <div class="st-track"><div class="st-fill" style="width: {percent}%;"></div></div>
    </div>
    """


# ==============================================================================
# Descarga automática de pesos DiT con detección de fallos (nunca se queda colgada)
# ==============================================================================

DOWNLOAD_STATUS_FILE = MODELS_ROOT / "download_status.json"
DOWNLOAD_LOCK_FILE = MODELS_ROOT / "download_status.lock"
DOWNLOAD_STALL_TIMEOUT_SEC = 300
DOWNLOAD_ERROR = "__ERROR__"
_DOWNLOAD_PROC: Optional[subprocess.Popen] = None


def _download_lock_alive() -> bool:
    """True si otra descarga (p. ej. descargar_modelos.bat) mantiene el bloqueo y su proceso sigue vivo."""
    try:
        import psutil
        pid = int(DOWNLOAD_LOCK_FILE.read_text().strip() or 0)
        return bool(pid) and psutil.pid_exists(pid)
    except Exception:
        return False


def _wait_for_dit_download() -> Generator[Tuple[str, Union[int, str]], None, None]:
    """
    Lanza download_model.py (si no hay otra descarga en curso) y emite (texto, porcentaje)
    mientras avanza. Si la descarga falla, muere o se detiene 5 minutos, emite
    (DOWNLOAD_ERROR, mensaje) y termina.
    """
    global _DOWNLOAD_PROC
    script_dir = Path(__file__).resolve().parent
    proc_running = _DOWNLOAD_PROC is not None and _DOWNLOAD_PROC.poll() is None
    if not proc_running and not _download_lock_alive():
        DOWNLOAD_STATUS_FILE.unlink(missing_ok=True)  # Evita leer el estado de una descarga anterior
        print(" -> [WebUI] Pesos DiT de Hunyuan3D-2 no encontrados. Iniciando descarga en segundo plano...")
        try:
            _DOWNLOAD_PROC = subprocess.Popen([sys.executable, str(script_dir / "download_model.py")], cwd=str(script_dir))
        except Exception as e_dl:
            yield DOWNLOAD_ERROR, f"No se pudo lanzar download_model.py ({e_dl})."
            return

    last_bytes, last_progress_t, last_console_t = -1, time.time(), 0.0
    while not is_dit_cached():
        try:
            sdata = json.loads(DOWNLOAD_STATUS_FILE.read_text(encoding="utf-8"))
        except Exception:
            sdata = {}

        if sdata.get("error"):
            yield DOWNLOAD_ERROR, f"La descarga de pesos falló ({sdata['error']})."
            return
        proc_dead = _DOWNLOAD_PROC is not None and _DOWNLOAD_PROC.poll() is not None
        if proc_dead and not _download_lock_alive() and not is_dit_cached():
            yield DOWNLOAD_ERROR, f"La descarga terminó (código {_DOWNLOAD_PROC.returncode}) sin completar los pesos."
            return

        downloaded = sdata.get("downloaded_bytes", 0)
        if downloaded != last_bytes:
            last_bytes, last_progress_t = downloaded, time.time()
        elif time.time() - last_progress_t > DOWNLOAD_STALL_TIMEOUT_SEC:
            if _DOWNLOAD_PROC is not None and _DOWNLOAD_PROC.poll() is None:
                _DOWNLOAD_PROC.terminate()
            yield DOWNLOAD_ERROR, "La descarga lleva 5 minutos sin avanzar (¿sin conexión?)."
            return

        if sdata:
            pct = sdata.get("percent", 0)
            eta = sdata.get("eta_seconds", 0)
            eta_str = f"{int(eta // 60)}m {int(eta % 60):02d}s" if eta > 0 else "calculando..."
            txt = (f"[2/4] Descargando pesos DiT ({sdata.get('downloaded_mb', 0):.0f}/{sdata.get('total_mb', 4702):.0f} MB"
                   f" • {pct}% • {sdata.get('speed_mb', 0)} MB/s • ETA: {eta_str})")
            bar_pct = int(pct * 0.45 + 5)
        else:
            txt, bar_pct = f"[2/4] Iniciando descarga de pesos Hunyuan3D-2 (~4.9GB) a {MODELS_ROOT}...", 5

        if time.time() - last_console_t >= 5.0:
            print(f" -> {txt}")
            last_console_t = time.time()
        yield txt, bar_pct
        time.sleep(1.5)
    print(" -> [WebUI] Pesos DiT de Hunyuan3D-2 listos en disco.")


# ==============================================================================
# 4. Lógica de Interfaz y Pipeline en Tiempo Real (con yield y streaming granular)
# ==============================================================================

def ui_remove_background(image: Optional[Image.Image]) -> Tuple[Optional[Image.Image], str]:
    """Elimina el fondo usando la sesión pre-inicializada en menos de 1 segundo."""
    if image is None:
        return None, render_progress_html("Sube una imagen para comenzar.", 0)
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
    global CURRENT_GENERATED_GLB

    def step_yield(msg_html: str):
        return gr.skip(), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False), msg_html
    
    if image is None and clean_silhouette is None:
        yield step_yield(render_progress_html("Sube una imagen antes de generar.", 0))
        return

    cfg = PRESET_CONFIG_MAP.get(preset_label, {"faces": 4000, "steps": 5, "octree": 256})
    target_faces = cfg["faces"]
    dit_steps = cfg["steps"]
    octree_res = cfg["octree"]
    total_start = time.time()
    
    print("\n" + "=" * 76)
    print("   [BOTÓN GENERAR PULSADO - INICIANDO PROCESAMIENTO]")
    gpu_name = "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        print(f"   ✓ GPU EN EJECUCIÓN: {gpu_name} (cuda:0)")
        print(f"   ✓ VRAM TOTAL: {total_b / 1024 ** 3:.2f} GB | VRAM LIBRE: {free_b / 1024 ** 2:.0f} MB")
    else:
        print("   ! DISPOSITIVO: CPU")
    print("=" * 76 + "\n")

    # --------------------------------------------------------------------------
    # ETAPA 1/4: Silueta Limpia (15%)
    # --------------------------------------------------------------------------
    yield step_yield(render_progress_html(f"[1/4] {gpu_name}: Extrayendo silueta limpia...", 15))
    
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
    # Cada motor saca a los demás de la VRAM al activarse; los inactivos quedan aparcados en RAM
    print_real_vram_status("WebUI Inicio Inferencia")

    is_sf3d = "sf3d" in str(ai_model_choice).lower() or "stable fast 3d" in str(ai_model_choice).lower()

    if is_sf3d:
        yield step_yield(render_progress_html("[2/4] Motor SF3D: Inicializando Stable Fast 3D en GPU...", 25))
        try:
            sf3d_gen = SF3DGenerator()
        except Exception as e:
            yield step_yield(render_progress_html(f"Error al inicializar Stable Fast 3D: {e}", 0))
            return

        yield step_yield(render_progress_html("[3/4] Inferencia Instantánea: Esculpiendo 3D y horneando PBR en GPU...", 60))
        try:
            raw_mesh = sf3d_gen.generate_mesh(
                clean_img,
                texture_resolution=2048,
                generate_texture=generate_texture,
                foreground_ratio=0.82
            )
        except Exception as e:
            yield step_yield(render_progress_html(f"Error en inferencia SF3D: {e}", 0))
            return

        if target_faces and target_faces < len(raw_mesh.faces):
            yield step_yield(render_progress_html(f"[3/4] Optimización PBR: Decimando a {target_faces:,} caras con textura nativa...", 88))
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
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.90,
                doubleSided=True
            )
            aligned_mesh.visual.material = neutral_mat
            final_mesh = aligned_mesh
        else:
            if hasattr(aligned_mesh, "visual") and hasattr(aligned_mesh.visual, "material") and aligned_mesh.visual.material is not None:
                aligned_mesh.visual.material.metallicFactor = 0.0
                aligned_mesh.visual.material.roughnessFactor = 0.90
                aligned_mesh.visual.material.doubleSided = True
            final_mesh = aligned_mesh
    else:
        # Pipeline Tencent Hunyuan3D-2 Turbo
        if not is_dit_cached():
            for msg, pct in _wait_for_dit_download():
                if msg == DOWNLOAD_ERROR:
                    print(f" -> [WebUI Error] {pct}")
                    yield step_yield(render_progress_html(f"Error: {pct} Ejecuta descargar_modelos.bat o usa el motor SF3D.", 0))
                    return
                yield step_yield(render_progress_html(msg, pct))

        # Determinar perfil VRAM activo
        selected_key = VRAM_CHOICE_TO_KEY.get(vram_profile_choice, "auto")
        active_profile_key = detect_vram_profile() if selected_key == "auto" else selected_key
        profile_info = VRAM_PROFILES.get(active_profile_key, VRAM_PROFILES["low_8gb"])
        print(f" -> [WebUI VRAM] Selección: '{vram_profile_choice}' | Perfil activo: '{active_profile_key}' ({profile_info['desc']})")

        yield step_yield(render_progress_html(f"[2/4] Perfil VRAM: {profile_info['desc']}...", 25))

        target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            generator = Hunyuan3DGenerator(
                model_id="tencent/Hunyuan3D-2",
                shape_subfolder="hunyuan3d-dit-v2-0-turbo",
                paint_subfolder="hunyuan3d-paint-v2-0",
                use_fp16=profile_info.get("use_fp16", True),
                low_vram_mode=None,  # Lo decide el perfil de VRAM
                vram_profile=active_profile_key,
                device=target_device
            )
        except Exception as e:
            yield step_yield(render_progress_html(f"Error al cargar Hunyuan3D-2: {e}", 0))
            return

        # Purga total de VRAM antes de iniciar Etapa 1
        flush_vram()

        # ETAPA 1: Geometría Hunyuan3D-DiT y Decimación Previa
        yield step_yield(render_progress_html(f"[3/4] [Etapa 1/2] Síntesis Geométrica DiT: Inferencia en {gpu_name}...", 30))

        progress_q = queue.Queue()
        gen_result = {"mesh": None, "error": None}

        # El tope de caras solo aplica si se va a texturizar (el texturizado no admite 280k caras).
        # Sin textura, "Máxima Calidad HD" conserva la malla completa.
        texture_cap = TEXTURE_SAFE_MAX_FACES if generate_texture else None
        if texture_cap and (not target_faces or target_faces > texture_cap):
            print(f" -> [WebUI] Con textura la malla se limita a {texture_cap:,} caras (sin textura se conserva completa).")

        def on_shape_progress(step_text, pct):
            progress_q.put((step_text, pct))

        def run_shape_worker():
            try:
                m = generator.generate_shape(
                    clean_img,
                    steps=dit_steps,
                    octree_resolution=octree_res,
                    seed=42,
                    target_faces=target_faces,
                    progress_callback=on_shape_progress,
                    texture_safe_cap=texture_cap
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

        aligned_mesh = center_and_orient_mesh(
            raw_mesh,
            align_floor=False,
            normalize_scale=True,
            target_height=1.2
        )

        # Purga total de VRAM entre Etapa 1 y Etapa 2
        flush_vram()
        print_real_vram_status("WebUI Entre Etapa 1 y 2")

        # ETAPA 2: Texturizado Nativo Multivista (Hunyuan3D-Paint en Low-VRAM)
        if generate_texture:
            yield step_yield(render_progress_html("[4/4] [Etapa 2/2] Texturizado Multivista: Cargando Hunyuan3D-Paint en GPU...", 90))
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
            neutral_mat = trimesh.visual.material.PBRMaterial(
                baseColorFactor=[0.82, 0.84, 0.88, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.90,
                doubleSided=True
            )
            aligned_mesh.visual.material = neutral_mat
            final_mesh = aligned_mesh

        # VRAM vacía al terminar; los modelos quedan aparcados en RAM para la próxima generación
        park_all_models()
        print_real_vram_status("WebUI Fin Generación")

    # Garantizar material PBR mate/limpio en la malla final
    if hasattr(final_mesh, "visual") and hasattr(final_mesh.visual, "material") and final_mesh.visual.material is not None:
        final_mesh.visual.material.metallicFactor = 0.0
        final_mesh.visual.material.roughnessFactor = 0.90
        final_mesh.visual.material.doubleSided = True

    yield step_yield(render_progress_html("[4/4] Guardando modelo .GLB en la biblioteca...", 97))

    # Guardar el .glb maestro en <HY3DGEN_MODELS>\output\ (los formatos de descarga
    # se generan después con el perfil de exportación activo: on_generation_completed)
    global LAST_GEN_SUMMARY
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
            formats=["glb"],
            target_faces=target_faces
        )

        glb_file = exported_map.get("glb")
        CURRENT_GENERATED_GLB = glb_file
        CURRENT_GENERATED_FILES["glb"] = glb_file

        elapsed = time.time() - total_start
        LAST_GEN_SUMMARY = f"Generado en {elapsed:.1f}s · {len(final_mesh.faces):,} caras"
        final_msg = render_progress_html(f"{LAST_GEN_SUMMARY} · Exportando con el perfil activo...", 100)

        # Purga estricta final de VRAM tras exportación completa
        flush_vram()

        yield (str(glb_file), gr.skip(), gr.skip(), gr.skip(), final_msg)

    except Exception as e:
        flush_vram()
        yield step_yield(render_progress_html(f"Error al exportar formatos: {e}", 0))
    finally:
        flush_vram()


def ui_generate_3d(
    image: Optional[Image.Image],
    clean_silhouette: Optional[Image.Image],
    preset_label: str,
    ai_model_choice: str,
    generate_texture: bool,
    model_name: str,
    vram_profile_choice: str = "Auto-detectar (Recomendado)"
) -> Generator[Tuple[Optional[str], str], None, None]:
    """
    Función generadora externa con bloqueo atómico anti-doble clic (mutex).
    Emite (visor 3D, estado); las descargas las prepara on_generation_completed.
    """
    global IS_GENERATING, LAST_GEN_OK
    if IS_GENERATING:
        yield gr.skip(), render_progress_html("Ya hay una generación en progreso. Por favor espera a que finalice.", 0)
        return
    IS_GENERATING = True
    LAST_GEN_OK = False
    previous_glb = CURRENT_GENERATED_GLB
    try:
        for out in _ui_generate_3d_impl(image, clean_silhouette, preset_label, ai_model_choice, generate_texture, model_name, vram_profile_choice):
            yield out[0], out[4]
        LAST_GEN_OK = CURRENT_GENERATED_GLB is not None and CURRENT_GENERATED_GLB != previous_glb
    finally:
        IS_GENERATING = False
        try:
            park_all_models()
        except Exception:
            pass


def ui_save_to_game(category: str, custom_name: str) -> None:
    """Copia directamente el .glb generado a la carpeta de Kaboomi Party (aviso tipo toast)."""
    if CURRENT_GENERATED_GLB is None or not CURRENT_GENERATED_GLB.exists():
        gr.Warning("Primero genera un modelo 3D antes de guardarlo en el juego.")
        return

    if BOMBOS_DIR is None:
        gr.Warning("El modo Kaboomi Party no está activo.")
        return
    
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
        gr.Info(f"Guardado en {dest_file.parent.name}/{safe_name}. Godot 4 lo importará automáticamente.")
    except Exception as e:
        gr.Warning(f"Error al guardar en el juego: {e}")


def ui_save_to_custom_dir(target_dir_str: str, custom_name: str, export_map: Optional[dict]) -> None:
    """Copia los archivos del perfil de exportación activo a una carpeta local (aviso tipo toast)."""
    sources = [Path(p) for p in (export_map or {}).values() if p and Path(p).exists()]
    if not sources:
        gr.Warning("Primero genera un modelo 3D antes de exportarlo.")
        return

    if not target_dir_str or not target_dir_str.strip():
        gr.Warning("Indica una carpeta de destino válida (ej. C:/MiProyecto/Assets).")
        return

    dest_folder = Path(target_dir_str.strip().strip('"').strip("'"))
    try:
        dest_folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        gr.Warning(f"No se pudo crear o acceder a la carpeta destino: {e}")
        return
        
    raw_name = Path((custom_name or "").strip()).stem
    safe_name = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-")).strip()
    # Los exportados se llaman <stem>.glb, <stem>_print.stl, <stem>_obj.zip...: se conserva la cola
    source_stem = Path(sources[0]).parent.name

    copied = []
    try:
        for src in sources:
            tail = src.name[len(source_stem):] if src.name.startswith(source_stem) else src.suffix
            dest_file = dest_folder / f"{safe_name or source_stem}{tail}"
            shutil.copy2(src, dest_file)
            copied.append(dest_file.name)
        gr.Info(f"Copiado a {dest_folder}: {', '.join(copied)}")
    except Exception as e:
        gr.Warning(f"Error al exportar modelo: {e}")


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
            
            # Búsqueda en la carpeta de salidas
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
            metallicFactor=0.0,
            roughnessFactor=0.90,
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
        return '<div class="hud hud-empty">Sin modelo</div>'

    tex_status = "PBR" if stats["has_texture"] else "Sin textura"
    tex_class = "ok" if stats["has_texture"] else "warn"
    size_formatted = f"{stats['size_kb']:.0f} KB" if stats["size_kb"] < 1024 else f"{stats['size_kb']/1024:.1f} MB"

    return f"""
    <div class="hud">
        <span><b>{stats['faces']:,}</b> tris</span>
        <span><b>{stats['vertices']:,}</b> vért</span>
        <span class="{tex_class}">{tex_status}</span>
        <span><b>{size_formatted}</b></span>
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
        return gr.update(clear_color=VIEWER_DARK_BG)


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


def export_for_downloads(glb_path: Optional[str], settings: dict) -> Tuple[dict, str]:
    """Exporta el .glb con el perfil de la sesión a output/exports/<modelo>/. Retorna ({fmt: ruta}, resumen)."""
    if not glb_path or not Path(glb_path).exists():
        return {}, ""
    src = Path(glb_path)
    try:
        files, summary = export_with_profile(src, EXPORTS_DIR / src.stem, src.stem, normalize_settings(settings))
        return {fmt: str(p) for fmt, p in files.items()}, summary
    except Exception as e:
        print(f"[Aviso Exportación] {e}")
        return {}, f"Error al exportar: {e}"


def download_updates(export_map: Optional[dict], settings: dict) -> list:
    """Un update por formato (orden ALL_FORMATS): visible si el perfil lo usa, activo si existe el archivo."""
    formats = active_formats(normalize_settings(settings))
    updates = []
    for fmt in ALL_FORMATS:
        path = (export_map or {}).get(fmt)
        ok = bool(path) and Path(path).exists()
        updates.append(gr.update(visible=fmt in formats, value=path if ok else None, interactive=ok))
    return updates


def slicer_button_update(export_map: Optional[dict], settings: dict):
    """Botón 'Abrir en <slicer>': solo en modo impresión."""
    settings = normalize_settings(settings)
    if settings["mode"] != "impresion":
        return gr.update(visible=False)
    _, name, exe = resolve_slicer(settings)
    has_file = any(f in (export_map or {}) for f in PRINT_FORMATS)
    return gr.update(visible=True, value=f"Abrir en {name}" if exe else "Abrir en slicer", interactive=has_file)


def on_open_slicer(export_map: Optional[dict], settings: dict) -> None:
    """Abre el archivo de impresión (3MF > STL > OBJ) en el slicer configurado."""
    path = next((export_map[f] for f in PRINT_FORMATS if (export_map or {}).get(f)), None)
    if not path or not Path(path).exists():
        gr.Warning("No hay un archivo de impresión exportado todavía.")
        return
    try:
        name = open_in_slicer(path, normalize_settings(settings))
        gr.Info(f"Abriendo {Path(path).name} en {name}...")
    except Exception as e:
        gr.Warning(f"No se pudo abrir el slicer: {e}")


def on_generation_completed(export_map: Optional[dict], settings: dict):
    """Al concluir la generación: exporta con el perfil de la sesión, actualiza métricas y reactiva controles."""
    if not LAST_GEN_OK:
        # Generación fallida o cancelada: se conserva el estado y las descargas anteriores
        return (
            gr.update(interactive=True, value=GENERATE_LABEL),
            gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
            *download_updates(export_map, settings),
            slicer_button_update(export_map, settings)
        )
    glb_path = str(CURRENT_GENERATED_GLB)
    new_map, summary = export_for_downloads(glb_path, settings)
    return (
        gr.update(interactive=True, value=GENERATE_LABEL),
        glb_path,
        render_mesh_stats_html(glb_path),
        "Textura",
        new_map,
        render_progress_html(f"{LAST_GEN_SUMMARY} · {summary}", 100),
        *download_updates(new_map, settings),
        slicer_button_update(new_map, settings)
    )


def on_select_library_model(selection: str, settings: dict):
    """Carga el modelo en el visor y prepara las descargas con el perfil de exportación de la sesión."""
    glb_path_str = load_existing_model_preview(selection)
    if not glb_path_str:
        return (
            gr.skip(),
            None,
            render_mesh_stats_html(None),
            "Textura",
            {},
            *download_updates({}, settings),
            slicer_button_update({}, settings)
        )
    glb_path = str(Path(glb_path_str).resolve())
    export_map, _ = export_for_downloads(glb_path, settings)
    return (
        glb_path,
        glb_path,
        render_mesh_stats_html(glb_path),
        "Textura",
        export_map,
        *download_updates(export_map, settings),
        slicer_button_update(export_map, settings)
    )


# ------------------------------------------------------------------------------
# Modal de configuración de exportación
# ------------------------------------------------------------------------------

MODE_CHOICES = [("Modelos 3D", "modelos"), ("Impresión 3D", "impresion")]
PIVOT_CHOICES = [("Base", "base"), ("Centro", "centro")]
TEXTURE_CHOICES = [("1K", 1024), ("2K", 2048), ("4K", 4096)]
MODEL_FORMAT_CHOICES = [(f.upper(), f) for f in MODEL_FORMATS]
PRINT_FORMAT_CHOICES = [(f.upper(), f) for f in PRINT_FORMATS]


def slicer_choices() -> list:
    installed = detect_slicers()
    choices = [("Automático (según impresora)", "auto")]
    for key, (name, _) in SLICERS.items():
        choices.append((f"{name}" + ("" if key in installed else " (no instalado)"), key))
    choices.append(("Programa predeterminado de Windows", "default"))
    return choices


def render_printer_info(printer: str, slicer: str, bx=None, by=None, bz=None) -> str:
    """Línea informativa: volumen de impresión, tecnología y slicer que se usará."""
    tmp = normalize_settings({"print_printer": printer, "print_slicer": slicer,
                              "print_custom_bed": [bx or 220, by or 220, bz or 250]})
    x, y, z, tech, _ = printer_dims(tmp)
    _, name, exe = resolve_slicer(tmp)
    slicer_txt = f"se abrirá en {name}" if exe else "slicer no detectado: se abrirá con el programa predeterminado"
    return f'<div class="printer-info"><b>{x:g} × {y:g} × {z:g} mm</b> · {tech} · {slicer_txt}</div>'


def on_open_settings(settings: dict):
    """Abre el modal cargando el perfil guardado en el navegador (descarta cambios sin guardar)."""
    s = normalize_settings(settings)
    is_print = s["mode"] == "impresion"
    bed = s["print_custom_bed"]
    return (
        gr.update(visible=True),
        s["mode"],
        gr.update(visible=not is_print),
        gr.update(visible=is_print),
        s["model_formats"],
        s["model_pivot"],
        s["model_height_m"],
        s["texture_max"],
        s["print_printer"],
        gr.update(visible=s["print_printer"] == CUSTOM_PRINTER),
        bed[0], bed[1], bed[2],
        s["print_height_mm"],
        s["print_fit_bed"],
        s["print_formats"],
        s["print_repair"],
        gr.update(choices=slicer_choices(), value=s["print_slicer"]),
        render_printer_info(s["print_printer"], s["print_slicer"], *bed),
    )


def on_save_settings(
    mode, model_formats, model_pivot, model_height_m, texture_max,
    printer, bed_x, bed_y, bed_z, print_height_mm, fit_bed, print_formats, repair, slicer,
    gen_model_path, lib_model_path
):
    """
    Guarda el perfil en el navegador (y en disco como valor inicial para navegadores nuevos),
    cierra el modal y vuelve a exportar los modelos cargados con el nuevo perfil.
    """
    if not model_formats:
        gr.Warning("Modelos 3D necesita al menos un formato: se usará GLB.")
    if not print_formats:
        gr.Warning("Impresión 3D necesita al menos un formato: se usará 3MF.")
    settings = normalize_settings({
        "mode": mode,
        "model_formats": list(model_formats or []),
        "model_pivot": model_pivot,
        "model_height_m": float(model_height_m or 1.0),
        "texture_max": int(texture_max or 2048),
        "print_printer": printer,
        "print_custom_bed": [float(bed_x or 220), float(bed_y or 220), float(bed_z or 250)],
        "print_height_mm": float(print_height_mm or 80.0),
        "print_fit_bed": bool(fit_bed),
        "print_formats": list(print_formats or []),
        "print_repair": bool(repair),
        "print_slicer": slicer or "auto",
    })
    try:
        save_settings(SETTINGS_FILE, settings)
    except Exception as e:
        print(f"[Aviso] No se pudo guardar {SETTINGS_FILE}: {e}")

    gen_map, _ = export_for_downloads(gen_model_path, settings)
    lib_map, _ = export_for_downloads(lib_model_path, settings)
    gr.Info(f"Perfil guardado: {profile_label(settings)}")
    return (
        gr.update(visible=False),
        gr.update(value=profile_label(settings)),
        settings,
        gen_map, *download_updates(gen_map, settings), slicer_button_update(gen_map, settings),
        lib_map, *download_updates(lib_map, settings), slicer_button_update(lib_map, settings),
    )


# ------------------------------------------------------------------------------
# Preferencias de la interfaz guardadas en el navegador (localStorage)
# ------------------------------------------------------------------------------

# Clave y secreto fijos: con valores aleatorios (por defecto en gr.BrowserState) lo guardado
# se perdería en cada reinicio del servidor. Son preferencias locales, no datos sensibles.
BROWSER_STATE_SECRET = "studio3d-local-ui-preferences"
UI_PREFS_STORAGE_KEY = f"studio3d_ui_prefs_{MODE}"
EXPORT_STORAGE_KEY = f"studio3d_export_profile_{MODE}"

# La pestaña activa es estado visual: se guarda en el navegador solo con clics reales del usuario
# (Gradio dispara 'select' del Tab al montar la página, lo que pisaría la preferencia guardada)
TAB_PERSIST_JS = """
() => {
    const KEY = "studio3d_tab_%s";
    const tabButtons = () => [...document.querySelectorAll('.custom-tabs [role="tablist"] button[role="tab"]')];
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    const restore = (attempt) => {
        const buttons = tabButtons();
        if (!buttons.length) {
            if (attempt < 50) setTimeout(() => restore(attempt + 1), 100);
            return;
        }
        const target = buttons.find(b => b.textContent.trim() === saved);
        if (target && !target.classList.contains("selected")) target.click();
    };
    restore(0);
    document.addEventListener("click", (e) => {
        const tab = e.target.closest('.custom-tabs [role="tab"]');
        if (tab && e.isTrusted) {
            try { localStorage.setItem(KEY, tab.textContent.trim()); } catch (err) {}
        }
    }, true);
}
""" % MODE


def remember_pref(key: str):
    """Crea el callback que guarda un valor de la interfaz en las preferencias del navegador."""
    def _store(value, prefs):
        prefs = dict(prefs or {})
        prefs[key] = value
        return prefs
    return _store


def restore_pref(prefs: Optional[dict], key: str, allowed=None, kind=None):
    """Valor guardado si es válido; si no existe o ya no es válido, no toca el componente."""
    if not isinstance(prefs, dict) or key not in prefs:
        return gr.skip()
    value = prefs[key]
    if kind is not None and not isinstance(value, kind):
        return gr.skip()
    if allowed is not None and value not in allowed:
        return gr.skip()
    return value


# ------------------------------------------------------------------------------
# Panel lateral de modelos generados (botón flotante a la derecha)
# ------------------------------------------------------------------------------

THUMBS_DIR = OUTPUT_DIR / "_thumbs"
PLACEHOLDER_THUMB = make_placeholder(THUMBS_DIR / "_placeholder.png")
SWITCH_TO_GENERATE_TAB_JS = f"""
(...args) => {{
    const tab = [...document.querySelectorAll('.custom-tabs [role="tab"]')].find(b => b.textContent.trim() === "Generar");
    if (tab && !tab.classList.contains("selected")) tab.click();
    try {{ localStorage.setItem("studio3d_tab_{MODE}", "Generar"); }} catch (e) {{}}
    return args;
}}
"""


def format_age(timestamp: float) -> str:
    """'hace 5 min', 'hace 3 h', 'hace 2 d' o la fecha si es más antiguo."""
    seconds = max(0, time.time() - timestamp)
    if seconds < 60:
        return "ahora"
    if seconds < 3600:
        return f"hace {int(seconds // 60)} min"
    if seconds < 86400:
        return f"hace {int(seconds // 3600)} h"
    if seconds < 7 * 86400:
        return f"hace {int(seconds // 86400)} d"
    return time.strftime("%d/%m/%Y", time.localtime(timestamp))


def render_drawer_title(count: int) -> str:
    sub = "Sin modelos todavía" if count == 0 else f"{count} modelo{'s' if count != 1 else ''} · clic para verlo"
    return f'<div class="drawer-title">Modelos generados</div><div class="drawer-sub">{sub}</div>'


def drawer_items(render_budget_s: float = 4.0) -> Tuple[list, list]:
    """
    (valores de la galería [(miniatura, pie)], claves de la biblioteca) del más nuevo al más viejo.
    Las miniaturas que falten se renderizan hasta agotar el presupuesto de tiempo; el resto
    muestra un marcador y se completa en segundo plano (warm_thumbnails).
    """
    items, keys = [], []
    started = time.time()
    for key in list_existing_models():
        path = load_existing_model_preview(key)
        if not path:
            continue
        thumb = get_thumbnail(path, THUMBS_DIR, render_missing=time.time() - started < render_budget_s)
        name = Path(key.split("] ", 1)[-1]).stem
        items.append((str(thumb or PLACEHOLDER_THUMB), f"{name} · {format_age(Path(path).stat().st_mtime)}"))
        keys.append(key)
    return items, keys


def on_toggle_drawer(is_open: bool, prefs: Optional[dict]):
    """Abre (recargando la lista) o cierra el panel lateral; recuerda el estado en el navegador."""
    prefs = dict(prefs or {}, drawer_open=not is_open)
    if is_open:
        return False, gr.update(visible=False), gr.update(elem_classes=["drawer-toggle"]), gr.skip(), gr.skip(), gr.skip(), prefs
    items, keys = drawer_items()
    return (True, gr.update(visible=True), gr.update(elem_classes=["drawer-toggle", "is-open"]),
            items, keys, render_drawer_title(len(keys)), prefs)


def on_refresh_drawer():
    items, keys = drawer_items()
    return items, keys, render_drawer_title(len(keys))


def on_drawer_pick(keys: list, settings: dict, evt: gr.SelectData):
    """Carga el modelo elegido en el visor de Generar con sus descargas del perfil activo."""
    global CURRENT_GENERATED_GLB
    key = keys[evt.index] if keys and 0 <= evt.index < len(keys) else None
    result = on_select_library_model(key, settings)
    if result[1]:
        # 'Guardar en el juego' (Kaboomi) usa el modelo activo del generador
        CURRENT_GENERATED_GLB = Path(result[1])
        status = render_progress_html(f"Viendo {Path(result[1]).name}", 100)
    else:
        status = render_progress_html("No se encontró el modelo seleccionado.", 0)
    return (*result, status)



# ==============================================================================
# 5. Estilos CSS Profesionales, Sin Emojis y con Animaciones de Alta Fidelidad
# ==============================================================================

CUSTOM_CSS = """
/* ==============================================================================
   Diseño minimalista de una sola pantalla: todo cabe en 100vh sin scroll
   ============================================================================== */
:root {
    --s-bg: #0a0b0f;
    --s-surface: #111318;
    --s-surface-2: #181b22;
    --s-border: rgba(255, 255, 255, 0.07);
    --s-border-strong: rgba(255, 255, 255, 0.14);
    --s-text: #e6e8ee;
    --s-muted: #8b91a1;
    --s-dim: #5c6273;
    --s-accent: #8b5cf6;
    --s-accent-2: #6366f1;
    --s-accent-soft: rgba(139, 92, 246, 0.14);
    --s-ok: #34d399;
    --s-warn: #fbbf24;
    --s-err: #f87171;
    --s-radius: 12px;
    --s-radius-sm: 8px;
    --s-header-h: 44px;
}

html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: var(--s-bg) !important;
    color: var(--s-text) !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    padding: 8px 14px 10px !important;
    margin: 0 !important;
    background: var(--s-bg) !important;
    position: relative !important;
}
.gradio-container .main.fillable { max-width: min(100%, 2000px) !important; }
.gradio-container .main,
.gradio-container .wrap,
.gradio-container .contain {
    padding: 0 !important;
    gap: 0 !important;
}
footer { display: none !important; }
.gradio-container .html-container { padding: 0 !important; }

/* Etiquetas de los controles: texto pequeño y discreto */
.gradio-container [data-testid="block-info"] {
    color: var(--s-muted) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}

/* ==============================================================================
   Cabecera fina (las pestañas se colocan dentro de ella)
   ============================================================================== */
.studio-header {
    height: var(--s-header-h);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 0 4px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 9px;
    min-width: 0;
}
.brand svg { color: var(--s-accent); flex-shrink: 0; }
.brand-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.01em;
    white-space: nowrap;
}
.brand-tag {
    font-size: 0.72rem;
    color: var(--s-dim);
    white-space: nowrap;
}
.gpu-pill {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--s-muted);
    padding: 5px 11px;
    border: 1px solid var(--s-border);
    border-radius: 999px;
    background: var(--s-surface);
    white-space: nowrap;
}
.gpu-pill .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--s-ok);
    box-shadow: 0 0 8px var(--s-ok);
}

/* Pestañas como control segmentado centrado en la cabecera */
.custom-tabs { position: static !important; padding: 0 !important; border: none !important; background: transparent !important; }
.custom-tabs .tab-wrapper {
    position: absolute !important;
    top: 8px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10;
    width: max-content !important;
    min-width: 300px !important;
    height: 36px !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}
.custom-tabs .tab-container[role="tablist"] {
    gap: 2px !important;
    padding: 3px !important;
    height: auto !important;
    background: var(--s-surface) !important;
    border: 1px solid var(--s-border) !important;
    border-radius: 10px !important;
}
.custom-tabs .tab-container[role="tablist"] button {
    height: 28px !important;
    padding: 0 14px !important;
    border: none !important;
    border-radius: 7px !important;
    background: transparent !important;
    color: var(--s-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}
.custom-tabs .tab-container[role="tablist"] button:hover { color: var(--s-text) !important; }
.custom-tabs .tab-container[role="tablist"] button.selected {
    background: var(--s-surface-2) !important;
    color: #fff !important;
    box-shadow: inset 0 0 0 1px var(--s-border-strong) !important;
}
.custom-tabs .tab-container[role="tablist"] button::after,
.custom-tabs .tab-container[role="tablist"] button.selected::after { display: none !important; }
.custom-tabs .tabitem {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
}

/* ==============================================================================
   Layout: barra lateral fija + visor que ocupa el resto
   ============================================================================== */
.studio-main-row {
    gap: 12px !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
}
.studio-card {
    background: var(--s-surface) !important;
    border: 1px solid var(--s-border) !important;
    border-radius: var(--s-radius) !important;
    padding: 12px !important;
    gap: 10px !important;
    box-shadow: none !important;
}
.studio-sidebar {
    flex: 0 0 340px !important;
    width: 340px !important;
    max-width: 340px !important;
    min-width: 300px !important;
}
.studio-viewport {
    flex: 1 1 auto !important;
    min-width: 0 !important;
}
/* Los bloques no se estiran: el espacio sobrante queda al final de la barra lateral */
.studio-card > * { flex: 0 0 auto !important; }

/* Sin cajas dentro de cajas */
.studio-card .block,
.studio-card .form,
.studio-card .gr-group,
.studio-card .row {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.studio-card .block { padding: 0 !important; }
.studio-card .form { gap: 8px !important; }
.studio-card .row { gap: 8px !important; }

/* Imagen de entrada con fondo de tablero para ver el canal alfa */
.input-image {
    border: 1px dashed var(--s-border-strong) !important;
    border-radius: var(--s-radius-sm) !important;
    overflow: hidden !important;
    background-color: #0e1015 !important;
    background-image:
        linear-gradient(45deg, #15181f 25%, transparent 25%),
        linear-gradient(-45deg, #15181f 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #15181f 75%),
        linear-gradient(-45deg, transparent 75%, #15181f 75%) !important;
    background-size: 16px 16px !important;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0 !important;
}
.input-image:hover { border-color: rgba(139, 92, 246, 0.5) !important; }

/* Presets de calidad en cuadrícula 2x2 */
.preset-chips .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 6px !important;
}
.preset-chips input[type="radio"] { display: none !important; }
.preset-chips label {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 7px 8px !important;
    border: 1px solid var(--s-border) !important;
    border-radius: var(--s-radius-sm) !important;
    background: var(--s-surface-2) !important;
    color: var(--s-muted) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease !important;
}
.preset-chips label:hover { color: var(--s-text) !important; border-color: var(--s-border-strong) !important; }
.preset-chips label.selected,
.preset-chips label:has(input:checked) {
    border-color: var(--s-accent) !important;
    background: var(--s-accent-soft) !important;
    color: #fff !important;
}

/* Entradas y desplegables */
.gradio-container input[type="text"],
.gradio-container textarea,
.gradio-container .wrap-inner {
    background: var(--s-surface-2) !important;
    border-color: var(--s-border) !important;
    border-radius: var(--s-radius-sm) !important;
    color: var(--s-text) !important;
    font-size: 0.82rem !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus,
.gradio-container .wrap-inner:focus-within {
    border-color: var(--s-accent) !important;
    box-shadow: 0 0 0 2px var(--s-accent-soft) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder { color: var(--s-dim) !important; }

/* Casilla de textura */
.tex-check label {
    gap: 8px !important;
    font-size: 0.82rem !important;
    color: var(--s-text) !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.tex-check input[type="checkbox"]:checked { background-color: var(--s-accent) !important; border-color: var(--s-accent) !important; }

/* Botón principal */
.btn-generate {
    min-height: 44px !important;
    border: none !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, var(--s-accent-2), var(--s-accent)) !important;
    color: #fff !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 8px 24px -10px rgba(139, 92, 246, 0.8) !important;
    transition: filter 0.15s ease, transform 0.15s ease !important;
}
.btn-generate:hover { filter: brightness(1.12) !important; transform: translateY(-1px) !important; }
.btn-generate img { filter: invert(1) !important; width: 16px !important; height: 16px !important; }
.btn-generate:disabled,
.btn-generate[disabled] {
    background: var(--s-surface-2) !important;
    color: var(--s-muted) !important;
    box-shadow: inset 0 0 0 1px var(--s-border-strong) !important;
    transform: none !important;
    filter: none !important;
    cursor: progress !important;
}
.btn-generate:disabled img,
.btn-generate[disabled] img { display: none !important; }
.btn-generate:disabled::before,
.btn-generate[disabled]::before {
    content: "";
    width: 14px;
    height: 14px;
    margin-right: 8px;
    border: 2px solid rgba(139, 92, 246, 0.25);
    border-top-color: var(--s-accent);
    border-radius: 50%;
    animation: s-spin 0.75s linear infinite;
}
@keyframes s-spin { to { transform: rotate(360deg); } }

/* Línea de estado + barra de progreso fina */
.st { padding: 2px 2px 0; }
.st-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    font-size: 0.76rem;
    line-height: 1.35;
    color: var(--s-muted);
    min-height: 2.7em;
}
.st-dot {
    width: 7px;
    height: 7px;
    margin-top: 5px;
    border-radius: 50%;
    background: var(--s-dim);
    flex-shrink: 0;
}
.st-text {
    flex: 1;
    min-width: 0;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.st-pct {
    font-weight: 700;
    color: var(--s-text);
    font-variant-numeric: tabular-nums;
}
.st-track {
    height: 3px;
    margin-top: 6px;
    border-radius: 999px;
    background: var(--s-surface-2);
    overflow: hidden;
}
.st-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--s-accent-2), var(--s-accent), #d946ef, var(--s-accent-2));
    background-size: 200% 100%;
    animation: s-shimmer 2s linear infinite;
    transition: width 0.3s ease;
}
@keyframes s-shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
.st-run .st-dot { background: var(--s-accent); box-shadow: 0 0 8px var(--s-accent); animation: s-pulse 1.4s ease-in-out infinite; }
.st-run .st-text { color: var(--s-text); }
.st-done .st-dot { background: var(--s-ok); box-shadow: 0 0 8px var(--s-ok); }
.st-done .st-text { color: var(--s-text); }
.st-done .st-fill { background: var(--s-ok); animation: none; }
.st-err .st-dot { background: var(--s-err); }
.st-err .st-text { color: var(--s-err); }
@keyframes s-pulse { 50% { opacity: 0.35; } }

/* ==============================================================================
   Visor 3D: barra de herramientas, lienzo y estadísticas superpuestas
   ============================================================================== */
.viewer-toolbar {
    gap: 8px !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
}
.viewer-toolbar > * { min-width: 0 !important; }

.segmented .wrap {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 2px !important;
    padding: 3px !important;
    background: var(--s-surface-2) !important;
    border: 1px solid var(--s-border) !important;
    border-radius: 9px !important;
    width: fit-content !important;
}
.segmented input[type="radio"] { display: none !important; }
.segmented label {
    margin: 0 !important;
    padding: 0 12px !important;
    height: 26px !important;
    display: flex !important;
    align-items: center !important;
    border: none !important;
    border-radius: 6px !important;
    background: transparent !important;
    color: var(--s-muted) !important;
    font-size: 0.76rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    cursor: pointer !important;
    box-shadow: none !important;
}
.segmented label:hover { color: var(--s-text) !important; }
.segmented label.selected,
.segmented label:has(input:checked) {
    background: var(--s-surface) !important;
    color: #fff !important;
    box-shadow: inset 0 0 0 1px var(--s-border-strong) !important;
}

.viewer-toolbar .segmented { flex: 1 1 auto !important; }
.viewer-toolbar .mini-dropdown { flex: 0 0 160px !important; width: 160px !important; }
.action-bar .mini-dropdown { flex: 1 1 200px !important; }
.action-bar .btn-dl { flex: 0 0 84px !important; width: 84px !important; }
.action-bar .btn-ghost { flex: 0 0 auto !important; width: auto !important; }
.action-bar .btn-icon { flex: 0 0 36px !important; width: 36px !important; }
.action-bar .compact-input { flex: 1 1 200px !important; }
.tex-check { flex: 0 0 auto !important; width: auto !important; }

.mini-dropdown .wrap-inner {
    min-height: 34px !important;
    padding: 0 8px !important;
    font-size: 0.76rem !important;
}
.mini-dropdown input { font-size: 0.76rem !important; }

.viewer-wrap {
    position: relative !important;
    border: 1px solid var(--s-border) !important;
    border-radius: var(--s-radius-sm) !important;
    overflow: hidden !important;
    background: #121419 !important;
    gap: 0 !important;
}
.viewer-wrap .block { border-radius: 0 !important; }
.viewer-wrap canvas { outline: none !important; }

.hud-block {
    position: absolute !important;
    left: 10px !important;
    bottom: 10px !important;
    z-index: 5 !important;
    width: auto !important;
    pointer-events: none !important;
    background: transparent !important;
}
.hud {
    display: flex;
    gap: 14px;
    padding: 6px 11px;
    border-radius: 8px;
    background: rgba(10, 11, 15, 0.72);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--s-border);
    font-size: 0.72rem;
    color: var(--s-muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}
.hud b { color: var(--s-text); font-weight: 700; }
.hud .ok { color: var(--s-ok); font-weight: 600; }
.hud .warn { color: var(--s-warn); font-weight: 600; }
.hud-empty { color: var(--s-dim); }

/* Barra inferior: descargas + exportar */
.action-bar {
    gap: 8px !important;
    align-items: center !important;
    flex-wrap: nowrap !important;
}
.action-bar > * { min-width: 0 !important; }
.btn-dl,
.btn-ghost {
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 14px !important;
    border-radius: var(--s-radius-sm) !important;
    border: 1px solid var(--s-border) !important;
    background: var(--s-surface-2) !important;
    color: var(--s-text) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    white-space: nowrap !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, background 0.15s ease !important;
}
.btn-dl img,
.btn-ghost img {
    filter: invert(1) !important;
    opacity: 0.75 !important;
    width: 15px !important;
    height: 15px !important;
}
.btn-dl:hover:not(:disabled),
.btn-ghost:hover:not(:disabled) {
    border-color: var(--s-accent) !important;
    background: var(--s-accent-soft) !important;
}
.btn-dl:disabled,
.btn-dl[disabled] {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}
.btn-icon { padding: 0 !important; width: 36px !important; min-width: 36px !important; flex: 0 0 36px !important; }
.btn-icon img { margin: 0 !important; }
.action-spacer { flex: 1 1 auto !important; }
.action-bar .compact-input input {
    height: 36px !important;
    font-size: 0.78rem !important;
}

/* ==============================================================================
   Tuerca de configuración (esquina derecha de la cabecera)
   ============================================================================== */
.studio-header { padding-right: 250px; }
.settings-btn {
    position: absolute !important;
    top: 6px !important;
    right: 0 !important;
    z-index: 10 !important;
    width: 240px !important;
    max-width: 240px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 12px !important;
    justify-content: flex-start !important;
    gap: 8px !important;
    border: 1px solid var(--s-border) !important;
    border-radius: 999px !important;
    background: var(--s-surface) !important;
    color: var(--s-muted) !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, color 0.15s ease !important;
}
.settings-btn img {
    filter: invert(1) !important;
    opacity: 0.8 !important;
    width: 15px !important;
    height: 15px !important;
    transition: transform 0.4s ease !important;
}
.settings-btn:hover { color: #fff !important; border-color: var(--s-accent) !important; }
.settings-btn:hover img { transform: rotate(60deg); }

/* ==============================================================================
   Modal de configuración de exportación
   ============================================================================== */
.settings-modal {
    position: fixed !important;
    inset: 0 !important;
    z-index: 1000 !important;
    margin: 0 !important;
    padding: 16px !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(5, 6, 9, 0.72) !important;
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    border: none !important;
}
.settings-panel {
    flex: 0 0 auto !important;
    width: min(620px, 100%) !important;
    max-height: calc(100vh - 32px) !important;
    overflow-y: auto !important;
    padding: 18px 20px !important;
    gap: 14px !important;
    background: var(--s-surface) !important;
    border: 1px solid var(--s-border-strong) !important;
    border-radius: 14px !important;
    box-shadow: 0 24px 64px rgba(0, 0, 0, 0.6) !important;
    animation: s-pop 0.18s ease-out;
}
@keyframes s-pop { from { opacity: 0; transform: translateY(8px) scale(0.98); } }
.settings-panel > * { flex: 0 0 auto !important; }
.settings-panel .block,
.settings-panel .form,
.settings-panel .row,
.settings-panel .gr-group {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.settings-panel .block { padding: 0 !important; }
.settings-panel .row { gap: 12px !important; }
.modal-head { align-items: flex-start !important; flex-wrap: nowrap !important; }
.modal-head > .block:first-child { flex: 1 1 auto !important; }
.modal-title { font-size: 1rem; font-weight: 700; color: #fff; }
.modal-sub { margin-top: 3px; font-size: 0.78rem; color: var(--s-muted); }
.modal-section { gap: 12px !important; padding: 0 !important; border: none !important; background: transparent !important; }
.modal-hint {
    font-size: 0.74rem;
    line-height: 1.45;
    color: var(--s-dim);
    padding: 8px 10px;
    border-radius: var(--s-radius-sm);
    background: var(--s-surface-2);
}
.printer-info { font-size: 0.76rem; color: var(--s-muted); }
.printer-info b { color: var(--s-text); }
.modal-foot { justify-content: flex-end !important; padding-top: 4px; border-top: 1px solid var(--s-border) !important; }
.modal-foot .btn-ghost { flex: 0 0 auto !important; width: auto !important; height: 38px !important; }
.btn-save { flex: 0 0 auto !important; width: auto !important; min-height: 38px !important; padding: 0 22px !important; }

.mode-switch .wrap { width: 100% !important; }
.mode-switch label { flex: 1 1 0 !important; justify-content: center !important; height: 32px !important; font-size: 0.82rem !important; }

/* Chips de formato (casillas) y radios en línea */
.fmt-chips .wrap { display: flex !important; flex-wrap: wrap !important; gap: 6px !important; }
.fmt-chips input[type="checkbox"] { display: none !important; }
.fmt-chips label {
    margin: 0 !important;
    padding: 6px 14px !important;
    border: 1px solid var(--s-border) !important;
    border-radius: var(--s-radius-sm) !important;
    background: var(--s-surface-2) !important;
    color: var(--s-muted) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    cursor: pointer !important;
    box-shadow: none !important;
}
.fmt-chips label:hover { color: var(--s-text) !important; }
.fmt-chips label.selected,
.fmt-chips label:has(input:checked) {
    border-color: var(--s-accent) !important;
    background: var(--s-accent-soft) !important;
    color: #fff !important;
}
.chips-inline .wrap { display: flex !important; grid-template-columns: none !important; }
.chips-inline label { flex: 1 1 0 !important; padding: 7px 10px !important; }

.gradio-container input[type="number"] {
    background: var(--s-surface-2) !important;
    border-color: var(--s-border) !important;
    border-radius: var(--s-radius-sm) !important;
    color: var(--s-text) !important;
    font-size: 0.82rem !important;
}

/* Botón de slicer resaltado */
.btn-accent { border-color: rgba(139, 92, 246, 0.45) !important; color: #fff !important; }
.action-bar .btn-accent { flex: 0 0 auto !important; width: auto !important; }

/* ==============================================================================
   Botón flotante (borde derecho, a media altura) y panel de modelos generados
   ============================================================================== */
:root { --drawer-w: min(340px, 88vw); }
.drawer-toggle {
    position: fixed !important;
    top: 50% !important;
    right: 0 !important;
    transform: translateY(-50%) !important;
    z-index: 900 !important;
    width: 34px !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: auto !important;
    min-height: 0 !important;
    padding: 14px 0 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 10px !important;
    writing-mode: vertical-rl !important;
    border: 1px solid var(--s-border-strong) !important;
    border-right: none !important;
    border-radius: 10px 0 0 10px !important;
    background: var(--s-surface) !important;
    color: var(--s-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    box-shadow: -8px 0 24px rgba(0, 0, 0, 0.35) !important;
    transition: right 0.22s ease, color 0.15s ease, border-color 0.15s ease !important;
}
.drawer-toggle img {
    writing-mode: horizontal-tb !important;
    margin: 0 !important;
    width: 15px !important;
    height: 15px !important;
    filter: invert(1) !important;
    opacity: 0.8 !important;
}
.drawer-toggle:hover { color: #fff !important; border-color: var(--s-accent) !important; }
.drawer-toggle.is-open {
    right: var(--drawer-w) !important;
    color: #fff !important;
    border-color: rgba(139, 92, 246, 0.6) !important;
}

.models-drawer {
    position: fixed !important;
    top: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    width: var(--drawer-w) !important;
    z-index: 890 !important;
    margin: 0 !important;
    padding: 12px !important;
    gap: 10px !important;
    background: var(--s-surface) !important;
    border: none !important;
    border-left: 1px solid var(--s-border-strong) !important;
    border-radius: 0 !important;
    box-shadow: -24px 0 64px rgba(0, 0, 0, 0.5) !important;
    animation: s-slide-in 0.22s ease-out;
}
@keyframes s-slide-in { from { transform: translateX(100%); } }
.models-drawer > * { flex: 0 0 auto !important; }
.models-drawer .block,
.models-drawer .row { background: transparent !important; border: none !important; box-shadow: none !important; }
.drawer-head { align-items: center !important; flex-wrap: nowrap !important; gap: 6px !important; }
.drawer-head > .block:first-child { flex: 1 1 auto !important; min-width: 0 !important; }
.drawer-head .btn-icon { flex: 0 0 34px !important; width: 34px !important; height: 34px !important; min-height: 34px !important; }
.drawer-title { font-size: 0.92rem; font-weight: 700; color: #fff; }
.drawer-sub { margin-top: 2px; font-size: 0.74rem; color: var(--s-muted); }

.drawer-gallery { padding: 0 !important; }
.drawer-gallery .grid-wrap { padding: 0 !important; }
.drawer-gallery .grid-container { gap: 8px !important; }
.drawer-gallery .thumbnail-item {
    border: 1px solid var(--s-border) !important;
    border-radius: var(--s-radius-sm) !important;
    background: radial-gradient(circle at 50% 40%, #1c1f29, #121419) !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease, transform 0.15s ease !important;
}
.drawer-gallery .thumbnail-item:hover { border-color: rgba(139, 92, 246, 0.6) !important; transform: translateY(-1px); }
.drawer-gallery .thumbnail-item.selected { border-color: var(--s-accent) !important; box-shadow: 0 0 0 1px var(--s-accent) !important; }
.drawer-gallery .caption-label {
    font-size: 0.68rem !important;
    color: var(--s-text) !important;
    background: rgba(10, 11, 15, 0.78) !important;
    border-radius: 6px !important;
}

/* Pestaña Sistema */
.sys-card {
    max-width: 760px !important;
    margin: 12px auto 0 !important;
    background: var(--s-surface) !important;
    border: 1px solid var(--s-border) !important;
    border-radius: var(--s-radius) !important;
    padding: 18px 22px !important;
}
.sys-card h3 { margin-top: 0 !important; font-size: 0.95rem !important; }
.sys-card li { font-size: 0.84rem !important; color: var(--s-muted) !important; }
.sys-card strong { color: var(--s-text) !important; }

/* Toasts (gr.Info / gr.Warning) */
.toast-body { background: var(--s-surface) !important; border-color: var(--s-border-strong) !important; }

/* ==============================================================================
   Pantallas estrechas: apilar columnas y permitir scroll
   ============================================================================== */
@media (max-width: 900px) {
    .custom-tabs .tab-wrapper { position: static !important; transform: none !important; margin: 0 0 8px !important; }
    .studio-main-row { flex-wrap: wrap !important; }
    .studio-sidebar { flex: 1 1 100% !important; width: 100% !important; max-width: 100% !important; }
    .brand-tag { display: none; }
    .gpu-pill { display: none; }
    .studio-header { padding-right: 178px; }
    .settings-btn { width: 170px !important; max-width: 170px !important; }
    .action-bar { flex-wrap: wrap !important; }
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "Segoe UI", "sans-serif"]
).set(
    body_background_fill="#0a0b0f",
    body_background_fill_dark="#0a0b0f",
    background_fill_primary="#111318",
    background_fill_primary_dark="#111318",
    background_fill_secondary="#181b22",
    background_fill_secondary_dark="#181b22",
    block_background_fill="#111318",
    block_background_fill_dark="#111318",
    block_border_color="rgba(255, 255, 255, 0.07)",
    block_border_color_dark="rgba(255, 255, 255, 0.07)",
    block_radius="12px",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_label_text_color="#8b91a1",
    block_label_text_color_dark="#8b91a1",
    block_label_text_size="*text_xs",
    block_title_text_color="#e6e8ee",
    block_title_text_color_dark="#e6e8ee",
    border_color_primary="rgba(255, 255, 255, 0.07)",
    border_color_primary_dark="rgba(255, 255, 255, 0.07)",
    input_background_fill="#181b22",
    input_background_fill_dark="#181b22",
    input_border_color="rgba(255, 255, 255, 0.07)",
    input_border_color_dark="rgba(255, 255, 255, 0.07)",
    input_border_color_focus="#8b5cf6",
    input_border_color_focus_dark="#8b5cf6",
    body_text_color="#e6e8ee",
    body_text_color_dark="#e6e8ee",
    body_text_color_subdued="#8b91a1",
    body_text_color_subdued_dark="#8b91a1",
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_dark="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    layout_gap="10px"
)

# ==============================================================================
# 6. Construcción del Dashboard con Gradio Blocks
# ==============================================================================

page_title = "Kaboomi 3D Studio" if MODE == "kambo" else "3D AI Studio"
header_title = page_title
badge_tag = "Hunyuan3D-2 Turbo · Godot 4" if MODE == "kambo" else "Hunyuan3D-2 Turbo"

# Alturas calculadas para que la vista completa quepa en 100vh sin scroll
VIEWER_HEIGHT = "max(320px, calc(100vh - 192px))"
INPUT_IMAGE_HEIGHT = "max(150px, calc(100vh - 450px))"

SHADING_CHOICES = ["Textura", "Arcilla", "Malla", "Puntos"]
BG_CHOICES = ["Fondo: Oscuro", "Fondo: CAD", "Fondo: Claro"]
CAM_CHOICES = ["Cámara: 3D", "Cámara: Frontal", "Cámara: Lateral", "Cámara: Superior", "Cámara: Centrar"]


def build_viewer_toolbar():
    """Fila superior del visor: modo de sombreado, fondo y cámara."""
    with gr.Row(elem_classes=["viewer-toolbar"]):
        shading = gr.Radio(
            choices=SHADING_CHOICES,
            value="Textura",
            show_label=False,
            container=False,
            scale=1,
            min_width=280,
            elem_classes=["segmented"]
        )
        bg = gr.Dropdown(
            choices=BG_CHOICES,
            value="Fondo: Oscuro",
            show_label=False,
            container=False,
            scale=0,
            min_width=150,
            elem_classes=["mini-dropdown"]
        )
        cam = gr.Dropdown(
            choices=CAM_CHOICES,
            value="Cámara: 3D",
            show_label=False,
            container=False,
            scale=0,
            min_width=160,
            elem_classes=["mini-dropdown"]
        )
    return shading, bg, cam


FORMAT_ICONS = {"glb": "cube", "obj": "palette", "ply": "sparkles", "stl": "printer", "3mf": "layers"}


def build_download_buttons() -> list:
    """Un botón compacto por formato (orden ALL_FORMATS); solo se ven los del perfil activo."""
    formats = active_formats(INITIAL_SETTINGS)
    return [
        gr.DownloadButton(
            fmt.upper(),
            icon=f"assets/icons/{FORMAT_ICONS[fmt]}.svg",
            interactive=False,
            visible=fmt in formats,
            scale=0,
            min_width=84,
            elem_classes=["btn-dl"]
        )
        for fmt in ALL_FORMATS
    ]


def build_slicer_button():
    return gr.Button(
        "Abrir en slicer",
        icon="assets/icons/printer.svg",
        interactive=False,
        visible=INITIAL_SETTINGS["mode"] == "impresion",
        scale=0,
        min_width=150,
        elem_classes=["btn-ghost", "btn-accent"]
    )


with gr.Blocks(title=page_title) as demo:

    # Cabecera fina: marca a la izquierda, GPU a la derecha (las pestañas se centran encima con CSS)
    gr.HTML(
        f"""
        <div class="studio-header">
            <div class="brand">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m21.12 6.4-6.05-3.5a2 2 0 0 0-2.14 0L6.88 6.4A2 2 0 0 0 6 8.13v7.74a2 2 0 0 0 1 1.73l6.05 3.5a2 2 0 0 0 2.14 0l6.05-3.5a2 2 0 0 0 1-1.73V8.13a2 2 0 0 0-.88-1.73Z"/>
                    <path d="m12 22 9-5"/>
                    <path d="M12 12V2"/>
                    <path d="M3 8.5 12 13l9-4.5"/>
                </svg>
                <span class="brand-name">{header_title}</span>
                <span class="brand-tag">{badge_tag}</span>
            </div>
            <div class="gpu-pill"><span class="dot"></span>{GPU_STATUS_TEXT}</div>
        </div>
        """,
        padding=False
    )

    # Preferencias guardadas en el navegador (localStorage): sobreviven a recargas y reinicios
    ui_prefs = gr.BrowserState(
        default_value={},
        storage_key=UI_PREFS_STORAGE_KEY,
        secret=BROWSER_STATE_SECRET
    )
    export_settings = gr.BrowserState(
        default_value=INITIAL_SETTINGS,
        storage_key=EXPORT_STORAGE_KEY,
        secret=BROWSER_STATE_SECRET
    )

    # Tuerca de configuración (se coloca en la esquina derecha de la cabecera con CSS)
    btn_settings = gr.Button(
        profile_label(INITIAL_SETTINGS),
        icon="assets/icons/settings.svg",
        elem_classes=["settings-btn"]
    )

    # Modal de configuración de exportación (estilo Meshy)
    with gr.Column(visible=False, elem_classes=["settings-modal"]) as settings_modal:
        with gr.Column(elem_classes=["settings-panel"]):
            with gr.Row(elem_classes=["modal-head"]):
                gr.HTML(
                    """
                    <div class="modal-title">Configuración de exportación</div>
                    <div class="modal-sub">Elige para qué es el modelo. Se aplica a las descargas, a Copiar y al slicer.</div>
                    """,
                    padding=False
                )
                btn_close_settings = gr.Button(
                    "",
                    icon="assets/icons/x.svg",
                    scale=0,
                    min_width=36,
                    elem_classes=["btn-ghost", "btn-icon"]
                )

            set_mode = gr.Radio(
                choices=MODE_CHOICES,
                value=INITIAL_SETTINGS["mode"],
                show_label=False,
                container=False,
                elem_classes=["segmented", "mode-switch"]
            )

            with gr.Column(visible=INITIAL_SETTINGS["mode"] != "impresion", elem_classes=["modal-section"]) as model_group:
                set_model_formats = gr.CheckboxGroup(
                    choices=MODEL_FORMAT_CHOICES,
                    value=INITIAL_SETTINGS["model_formats"],
                    label="Formatos",
                    elem_classes=["fmt-chips"]
                )
                with gr.Row():
                    set_pivot = gr.Radio(
                        choices=PIVOT_CHOICES,
                        value=INITIAL_SETTINGS["model_pivot"],
                        label="Origen",
                        min_width=140,
                        elem_classes=["preset-chips", "chips-inline"]
                    )
                    set_height_m = gr.Number(
                        value=INITIAL_SETTINGS["model_height_m"],
                        label="Altura (m)",
                        minimum=0.01,
                        step=0.1,
                        min_width=100
                    )
                    set_texture = gr.Radio(
                        choices=TEXTURE_CHOICES,
                        value=INITIAL_SETTINGS["texture_max"],
                        label="Textura máx.",
                        min_width=170,
                        elem_classes=["preset-chips", "chips-inline"]
                    )
                gr.HTML(
                    '<div class="modal-hint">Y-up, listo para Godot, Unity, Unreal, Blender y web. '
                    'El OBJ con textura se descarga como .zip (obj + mtl + png).</div>',
                    padding=False
                )

            with gr.Column(visible=INITIAL_SETTINGS["mode"] == "impresion", elem_classes=["modal-section"]) as print_group:
                with gr.Row():
                    set_printer = gr.Dropdown(
                        choices=list(PRINTERS),
                        value=INITIAL_SETTINGS["print_printer"],
                        label="Impresora",
                        min_width=180
                    )
                    set_slicer = gr.Dropdown(
                        choices=slicer_choices(),
                        value=INITIAL_SETTINGS["print_slicer"],
                        label="Slicer",
                        min_width=180
                    )
                printer_info = gr.HTML(
                    value=render_printer_info(INITIAL_SETTINGS["print_printer"], INITIAL_SETTINGS["print_slicer"], *INITIAL_SETTINGS["print_custom_bed"]),
                    padding=False
                )
                with gr.Row(visible=INITIAL_SETTINGS["print_printer"] == CUSTOM_PRINTER) as custom_bed_row:
                    set_bed_x = gr.Number(value=INITIAL_SETTINGS["print_custom_bed"][0], label="Cama X (mm)", minimum=10, min_width=90)
                    set_bed_y = gr.Number(value=INITIAL_SETTINGS["print_custom_bed"][1], label="Cama Y (mm)", minimum=10, min_width=90)
                    set_bed_z = gr.Number(value=INITIAL_SETTINGS["print_custom_bed"][2], label="Altura Z (mm)", minimum=10, min_width=90)
                with gr.Row():
                    set_height_mm = gr.Number(
                        value=INITIAL_SETTINGS["print_height_mm"],
                        label="Altura del modelo (mm)",
                        minimum=1,
                        step=5,
                        min_width=140
                    )
                    set_print_formats = gr.CheckboxGroup(
                        choices=PRINT_FORMAT_CHOICES,
                        value=INITIAL_SETTINGS["print_formats"],
                        label="Formatos",
                        min_width=200,
                        elem_classes=["fmt-chips"]
                    )
                with gr.Row():
                    set_fit_bed = gr.Checkbox(
                        value=INITIAL_SETTINGS["print_fit_bed"],
                        label="Ajustar al volumen de impresión",
                        container=False,
                        elem_classes=["tex-check"]
                    )
                    set_repair = gr.Checkbox(
                        value=INITIAL_SETTINGS["print_repair"],
                        label="Reparar malla (normales y huecos)",
                        container=False,
                        elem_classes=["tex-check"]
                    )
                gr.HTML(
                    '<div class="modal-hint">Solo geometría, en milímetros y Z-up, apoyada en la cama y centrada '
                    'en el volumen de la impresora. El 3MF lleva las unidades incluidas.</div>',
                    padding=False
                )

            with gr.Row(elem_classes=["modal-foot"]):
                btn_cancel_settings = gr.Button("Cancelar", scale=0, min_width=100, elem_classes=["btn-ghost"])
                btn_save_settings = gr.Button("Guardar", scale=0, min_width=120, elem_classes=["btn-generate", "btn-save"])

    # Botón flotante (borde derecho, a media altura) y panel lateral de modelos generados
    drawer_open = gr.State(False)
    drawer_keys = gr.State([])
    btn_drawer = gr.Button(
        "Modelos",
        icon="assets/icons/layers.svg",
        elem_classes=["drawer-toggle"]
    )
    with gr.Column(visible=False, elem_classes=["models-drawer"]) as models_drawer:
        with gr.Row(elem_classes=["drawer-head"]):
            drawer_title = gr.HTML(render_drawer_title(0), padding=False)
            btn_drawer_refresh = gr.Button(
                "",
                icon="assets/icons/refresh.svg",
                scale=0,
                min_width=34,
                elem_classes=["btn-ghost", "btn-icon"]
            )
            btn_drawer_close = gr.Button(
                "",
                icon="assets/icons/x.svg",
                scale=0,
                min_width=34,
                elem_classes=["btn-ghost", "btn-icon"]
            )
        drawer_gallery = gr.Gallery(
            show_label=False,
            columns=2,
            height="calc(100vh - 84px)",
            allow_preview=False,
            object_fit="contain",
            buttons=[],
            interactive=False,
            elem_classes=["drawer-gallery"]
        )

    _initial_preview = get_initial_preview_model()
    _has_real_model = bool(_initial_preview and not _initial_preview.endswith("placeholder.glb") and not _initial_preview.endswith("empty.glb"))

    with gr.Tabs(elem_classes=["custom-tabs"]):
        # ======================================================================
        # TAB 1: Generador
        # ======================================================================
        with gr.Tab("Generar"):
            with gr.Row(elem_classes=["studio-main-row"], equal_height=True):

                # Barra lateral: entrada y ajustes
                with gr.Column(scale=0, min_width=300, elem_classes=["studio-card", "studio-sidebar"]):
                    input_img = gr.Image(
                        show_label=False,
                        height=INPUT_IMAGE_HEIGHT,
                        type="pil",
                        sources=["upload", "clipboard"],
                        elem_classes=["input-image"]
                    )
                    clean_silhouette_state = gr.State(value=None)

                    preset_radio = gr.Radio(
                        choices=PRESET_UI_CHOICES,
                        value="Alta Calidad (50,000 caras • Con textura: 40k)",
                        label="Calidad",
                        elem_classes=["preset-chips"]
                    )

                    with gr.Row():
                        ai_model_dropdown = gr.Dropdown(
                            choices=ENGINE_UI_CHOICES,
                            value=ENGINE_SF3D if MODE == "kambo" else ENGINE_HUNYUAN,
                            label="Motor",
                            min_width=120,
                            interactive=True
                        )
                        vram_dropdown = gr.Dropdown(
                            choices=VRAM_UI_CHOICES,
                            value="Auto-detectar (Recomendado)",
                            label="VRAM",
                            min_width=90,
                            interactive=True
                        )

                    with gr.Row(equal_height=True):
                        chk_texture = gr.Checkbox(
                            value=False,
                            label="Textura PBR",
                            container=False,
                            scale=0,
                            min_width=120,
                            elem_classes=["tex-check"]
                        )
                        txt_model_name = gr.Textbox(
                            placeholder="Nombre (opcional)",
                            show_label=False,
                            container=False,
                            lines=1,
                            scale=1,
                            min_width=100
                        )

                    btn_generate = gr.Button(
                        GENERATE_LABEL,
                        icon="assets/icons/wand.svg",
                        elem_classes=["btn-generate"]
                    )

                    status_widget = gr.HTML(
                        value=render_progress_html("Sube una imagen y pulsa Generar.", 0),
                        padding=False
                    )

                # Visor 3D y exportación
                with gr.Column(scale=1, elem_classes=["studio-card", "studio-viewport"]):
                    viewer_shading_mode, viewer_bg_choice, viewer_cam_choice = build_viewer_toolbar()

                    with gr.Group(elem_classes=["viewer-wrap"]):
                        viewer_3d = gr.Model3D(
                            value=_initial_preview,
                            show_label=False,
                            height=VIEWER_HEIGHT,
                            camera_position=(-90, 80, 2.4),
                            clear_color=VIEWER_DARK_BG
                        )
                        mesh_stats_display = gr.HTML(
                            value=render_mesh_stats_html(_initial_preview if _has_real_model else None),
                            padding=False,
                            elem_classes=["hud-block"]
                        )
                    active_model_path = gr.State(value=_initial_preview if _has_real_model else None)
                    gen_export_state = gr.State(value={})

                    with gr.Row(elem_classes=["action-bar"]):
                        gen_dl_buttons = build_download_buttons()
                        if MODE == "kambo":
                            dest_category = gr.Dropdown(
                                choices=["/bombos/", "/bozis/", "/cosmetics/"],
                                value="/bombos/",
                                show_label=False,
                                container=False,
                                scale=1,
                                min_width=140,
                                elem_classes=["mini-dropdown"]
                            )
                            btn_save_game = gr.Button(
                                "Guardar en el juego",
                                icon="assets/icons/gamepad.svg",
                                scale=0,
                                min_width=170,
                                elem_classes=["btn-ghost"]
                            )
                        else:
                            custom_dest_folder = gr.Textbox(
                                placeholder="Copiar exportados a carpeta (ej. C:/Proyectos/Godot/assets)",
                                show_label=False,
                                container=False,
                                lines=1,
                                scale=1,
                                min_width=160,
                                elem_classes=["compact-input"]
                            )
                            btn_save_custom = gr.Button(
                                "Copiar",
                                icon="assets/icons/download.svg",
                                scale=0,
                                min_width=96,
                                elem_classes=["btn-ghost"]
                            )
                        gen_slicer_btn = build_slicer_button()

        # ======================================================================
        # TAB 2: Biblioteca
        # ======================================================================
        with gr.Tab("Biblioteca"):
            with gr.Column(elem_classes=["studio-card"]):
                lib_shading_mode, lib_bg_choice, lib_cam_choice = build_viewer_toolbar()

                with gr.Group(elem_classes=["viewer-wrap"]):
                    library_viewer = gr.Model3D(
                        value=_initial_preview,
                        show_label=False,
                        height=VIEWER_HEIGHT,
                        camera_position=(-90, 80, 2.4),
                        clear_color=VIEWER_DARK_BG
                    )
                    lib_mesh_stats_display = gr.HTML(
                        value=render_mesh_stats_html(_initial_preview if _has_real_model else None),
                        padding=False,
                        elem_classes=["hud-block"]
                    )
                lib_active_model_path = gr.State(value=_initial_preview if _has_real_model else None)
                lib_export_state = gr.State(value={})

                with gr.Row(elem_classes=["action-bar"]):
                    dropdown_models = gr.Dropdown(
                        choices=list_existing_models(),
                        show_label=False,
                        container=False,
                        scale=1,
                        min_width=200,
                        allow_custom_value=True,
                        elem_classes=["mini-dropdown"]
                    )
                    btn_refresh_library = gr.Button(
                        "",
                        icon="assets/icons/refresh.svg",
                        scale=0,
                        min_width=36,
                        elem_classes=["btn-ghost", "btn-icon"]
                    )
                    lib_dl_buttons = build_download_buttons()
                    lib_slicer_btn = build_slicer_button()

        # ======================================================================
        # TAB 3: Sistema
        # ======================================================================
        with gr.Tab("Sistema"):
            mode_label = "Kaboomi Party Edition" if MODE == "kambo" else "Estándar / General"
            kambo_sys_info = f"- **Ruta de Juego Godot 4:** `{KABOOMI_ROOT}`\n" if MODE == "kambo" else ""
            auto_vram_key = detect_vram_profile()
            auto_vram_desc = VRAM_PROFILES.get(auto_vram_key, {}).get("desc", auto_vram_key)
            gr.Markdown(
                f"""
### Entorno de ejecución
- **Modo:** `{mode_label}`
- **GPU:** {GPU_SYS_TEXT}
- **Perfil VRAM auto-detectado:** `{auto_vram_key}` ({auto_vram_desc})
- **Modelos y caché:** `{MODELS_ROOT}`
- **Salida:** `{OUTPUT_DIR}` (GLB maestro) y `{EXPORTS_DIR}` (exportados por perfil)
- **Configuración de exportación:** `{SETTINGS_FILE}`
{kambo_sys_info}- **Motores:** Tencent Hunyuan3D-2 Turbo / Stable Fast 3D + Rembg U2-Net
- **Limpieza de geometría:** eliminación de losas e islas parásitas (< 3% del volumen)
                """,
                elem_classes=["sys-card"]
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
        fn=lambda: (None, render_progress_html("Sube una imagen y pulsa Generar.", 0)),
        inputs=[],
        outputs=[clean_silhouette_state, status_widget]
    )

    # 2. Generación 3D en streaming con bloqueo anti-doble clic instantáneo
    btn_generate.click(
        fn=lambda: (
            gr.update(interactive=False, value="Generando..."),
            *[gr.update(interactive=False) for _ in range(len(gen_dl_buttons) + 1)]
        ),
        inputs=[],
        outputs=[btn_generate, *gen_dl_buttons, gen_slicer_btn]
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
        outputs=[viewer_3d, status_widget]
    ).then(
        fn=on_generation_completed,
        inputs=[gen_export_state, export_settings],
        outputs=[btn_generate, active_model_path, mesh_stats_display, viewer_shading_mode,
                 gen_export_state, status_widget, *gen_dl_buttons, gen_slicer_btn]
    ).then(
        # El modelo nuevo aparece en el panel lateral si está abierto
        fn=lambda is_open: on_refresh_drawer() if is_open else (gr.skip(), gr.skip(), gr.skip()),
        inputs=[drawer_open],
        outputs=[drawer_gallery, drawer_keys, drawer_title],
        show_progress="hidden"
    )

    # 3. Controles del visor (sombreado, fondo, cámara)
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

    # 4. Guardar directamente en el juego, en una carpeta o abrir en el slicer
    if MODE == "kambo":
        btn_save_game.click(
            fn=ui_save_to_game,
            inputs=[dest_category, txt_model_name],
            outputs=[]
        )
    else:
        btn_save_custom.click(
            fn=ui_save_to_custom_dir,
            inputs=[custom_dest_folder, txt_model_name, gen_export_state],
            outputs=[]
        )
    gen_slicer_btn.click(fn=on_open_slicer, inputs=[gen_export_state, export_settings], outputs=[])
    lib_slicer_btn.click(fn=on_open_slicer, inputs=[lib_export_state, export_settings], outputs=[])

    # 5. Explorador de modelos en biblioteca
    btn_refresh_library.click(
        fn=lambda: gr.update(choices=list_existing_models()),
        inputs=[],
        outputs=[dropdown_models]
    )
    dropdown_models.change(
        fn=on_select_library_model,
        inputs=[dropdown_models, export_settings],
        outputs=[library_viewer, lib_active_model_path, lib_mesh_stats_display, lib_shading_mode,
                 lib_export_state, *lib_dl_buttons, lib_slicer_btn]
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

    # 6. Modal de configuración de exportación
    btn_settings.click(
        fn=on_open_settings,
        inputs=[export_settings],
        outputs=[settings_modal, set_mode, model_group, print_group,
                 set_model_formats, set_pivot, set_height_m, set_texture,
                 set_printer, custom_bed_row, set_bed_x, set_bed_y, set_bed_z,
                 set_height_mm, set_fit_bed, set_print_formats, set_repair, set_slicer, printer_info]
    )
    for _btn in (btn_close_settings, btn_cancel_settings):
        _btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[settings_modal])

    set_mode.change(
        fn=lambda m: (gr.update(visible=m != "impresion"), gr.update(visible=m == "impresion")),
        inputs=[set_mode],
        outputs=[model_group, print_group]
    )
    for _comp in (set_printer, set_slicer, set_bed_x, set_bed_y, set_bed_z):
        _comp.change(
            fn=lambda p, s, x, y, z: (render_printer_info(p, s, x, y, z), gr.update(visible=p == CUSTOM_PRINTER)),
            inputs=[set_printer, set_slicer, set_bed_x, set_bed_y, set_bed_z],
            outputs=[printer_info, custom_bed_row]
        )

    btn_save_settings.click(
        fn=on_save_settings,
        inputs=[set_mode, set_model_formats, set_pivot, set_height_m, set_texture,
                set_printer, set_bed_x, set_bed_y, set_bed_z, set_height_mm, set_fit_bed,
                set_print_formats, set_repair, set_slicer,
                active_model_path, lib_active_model_path],
        outputs=[settings_modal, btn_settings, export_settings,
                 gen_export_state, *gen_dl_buttons, gen_slicer_btn,
                 lib_export_state, *lib_dl_buttons, lib_slicer_btn]
    )

    # 7. Preferencias de la interfaz en el navegador: cada cambio se guarda en localStorage
    #    y al abrir la página se restauran (solo valores que sigan siendo válidos)
    persisted_controls = {
        "preset": (preset_radio, list(PRESET_CONFIG_MAP), str),
        "engine": (ai_model_dropdown, [ENGINE_HUNYUAN, ENGINE_SF3D], str),
        "vram": (vram_dropdown, VRAM_PROFILE_CHOICES, str),
        "texture": (chk_texture, None, bool),
        "model_name": (txt_model_name, None, str),
        "shading": (viewer_shading_mode, SHADING_CHOICES, str),
        "bg": (viewer_bg_choice, BG_CHOICES, str),
        "cam": (viewer_cam_choice, CAM_CHOICES, str),
        "lib_shading": (lib_shading_mode, SHADING_CHOICES, str),
        "lib_bg": (lib_bg_choice, BG_CHOICES, str),
        "lib_cam": (lib_cam_choice, CAM_CHOICES, str),
        "lib_model": (dropdown_models, None, str),
    }
    if MODE == "kambo":
        persisted_controls["dest_category"] = (dest_category, ["/bombos/", "/bozis/", "/cosmetics/"], str)
    else:
        persisted_controls["dest_folder"] = (custom_dest_folder, None, str)

    for _key, (_comp, _allowed, _kind) in persisted_controls.items():
        _comp.change(
            fn=remember_pref(_key),
            inputs=[_comp, ui_prefs],
            outputs=[ui_prefs],
            show_progress="hidden",
            trigger_mode="always_last"
        )

    def restore_session(prefs, settings):
        """Al abrir la página: restaura la interfaz y el perfil de exportación guardados en el navegador."""
        settings = normalize_settings(settings)
        values = []
        for key, (_comp, allowed, kind) in persisted_controls.items():
            if key == "lib_model":
                allowed = [m for m in list_existing_models() if not m.startswith("(")]
            values.append(restore_pref(prefs, key, allowed, kind))
        if isinstance(prefs, dict) and prefs.get("drawer_open") is True:
            items, keys = drawer_items()
            drawer = (True, gr.update(visible=True), gr.update(elem_classes=["drawer-toggle", "is-open"]),
                      items, keys, render_drawer_title(len(keys)))
        else:
            drawer = (gr.skip(),) * 6
        return (
            *values,
            gr.update(value=profile_label(settings)),
            *download_updates({}, settings), slicer_button_update({}, settings),
            *download_updates({}, settings), slicer_button_update({}, settings),
            *drawer,
        )

    demo.load(
        fn=restore_session,
        inputs=[ui_prefs, export_settings],
        outputs=[*[c for c, _, _ in persisted_controls.values()], btn_settings,
                 *gen_dl_buttons, gen_slicer_btn, *lib_dl_buttons, lib_slicer_btn,
                 drawer_open, models_drawer, btn_drawer, drawer_gallery, drawer_keys, drawer_title],
        show_progress="hidden"
    )

    # 8. Panel lateral de modelos generados
    for _btn, _forced_open in ((btn_drawer, None), (btn_drawer_close, True)):
        _btn.click(
            fn=(lambda is_open, prefs, forced=_forced_open:
                on_toggle_drawer(is_open if forced is None else forced, prefs)),
            inputs=[drawer_open, ui_prefs],
            outputs=[drawer_open, models_drawer, btn_drawer, drawer_gallery, drawer_keys, drawer_title, ui_prefs],
            show_progress="hidden"
        )
    btn_drawer_refresh.click(
        fn=on_refresh_drawer,
        inputs=[],
        outputs=[drawer_gallery, drawer_keys, drawer_title]
    )
    drawer_gallery.select(
        fn=on_drawer_pick,
        inputs=[drawer_keys, export_settings],
        outputs=[viewer_3d, active_model_path, mesh_stats_display, viewer_shading_mode,
                 gen_export_state, *gen_dl_buttons, gen_slicer_btn, status_widget],
        js=SWITCH_TO_GENERATE_TAB_JS
    )


# ==============================================================================
# Lanzamiento del Servidor
# ==============================================================================

if __name__ == "__main__":
    # Miniaturas del panel lateral: se generan en segundo plano las que falten
    warm_thumbnails([p for p in map(load_existing_model_preview, list_existing_models()) if p], THUMBS_DIR)
    print("\n" + "="*70)
    if MODE == "kambo":
        print(" KABOOMI 3D STUDIO (GRADIO WEBUI)")
        print(" -> Modo: Kaboomi Party Edition")
        print(f" -> Directorio de Salida: {OUTPUT_DIR}")
        print(" -> Modelos y Caché: D:\\AI_Models (100% Protegido)")
        print(f" -> Proyecto Kaboomi Party: {KABOOMI_ROOT}")
    else:
        print(" 3D AI STUDIO (GRADIO WEBUI)")
        print(" -> Modo: Estándar / General")
        print(f" -> Directorio de Salida: {OUTPUT_DIR}")
        print(" -> Modelos y Caché: D:\\AI_Models (100% Protegido)")
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
        js=TAB_PERSIST_JS,
        allowed_paths=allowed_list
    )
