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
import subprocess
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple, Generator

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


from img2glb import (
    preprocess_and_remove_background,
    decimate_mesh_to_target_faces,
    center_and_orient_mesh,
    postprocess_and_validate_glb,
    Hunyuan3DGenerator,
    is_dit_cached,
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
# 3. Detección de Rutas de Kaboomi Party (Godot 4)
# ==============================================================================

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

KABOOMI_ROOT = get_kaboomi_dir()
MODELS_DIR = KABOOMI_ROOT / "assets" / "models"
BOMBOS_DIR = MODELS_DIR / "bombos"
BOZIS_DIR = MODELS_DIR / "bozis"
COSMETICS_DIR = MODELS_DIR / "cosmetics"

for folder in [BOMBOS_DIR, BOZIS_DIR, COSMETICS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

CURRENT_GENERATED_GLB: Optional[Path] = None
IS_GENERATING: bool = False

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
            session=REMBG_SESSION
        )
        elapsed = time.time() - start_t
        msg = render_progress_html(f"Silueta extraída con éxito en {elapsed:.2f}s (Canal alfa listo)", 0)
        return clean_img, msg
    except Exception as e:
        return None, render_progress_html(f"Error al procesar la imagen: {e}", 0)


def _ui_generate_3d_impl(
    image: Optional[Image.Image],
    clean_silhouette: Optional[Image.Image],
    preset_label: str,
    generate_texture: bool,
    model_name: str
) -> Generator[Tuple[Optional[str], any, str], None, None]:
    """
    Función generadora interna en streaming que emite actualizaciones en vivo etapa por etapa:
      - Etapa 1: Extracción alfa (15%)
      - Etapa 2: GPU VRAM init (25%)
      - Etapa 3: Síntesis DiT granular (30% -> 65%), Volume Decoding (66% -> 84%), Decimación (88%)
      - Etapa 4: Texturizado UV (94%), Empaquetado GLB (97%), Finalizado (100%)
    """
    global CURRENT_GENERATED_GLB
    
    if image is None and clean_silhouette is None:
        yield None, gr.update(interactive=False), render_progress_html("Sube una imagen en el Paso 1 antes de generar.", 0)
        return

    cfg = PRESET_CONFIG_MAP.get(preset_label, {"faces": 4000, "steps": 5, "octree": 256})
    target_faces = cfg["faces"]
    dit_steps = cfg["steps"]
    octree_res = cfg["octree"]
    total_start = time.time()
    
    # --------------------------------------------------------------------------
    # ETAPA 1/4: Silueta Limpia (15%)
    # --------------------------------------------------------------------------
    yield None, gr.update(interactive=False), render_progress_html("[1/4] Extracción Alfa: Procesando silueta con Rembg U2-Net...", 15)
    
    try:
        if clean_silhouette is not None:
            clean_img = clean_silhouette
            print(" -> Usando silueta pre-calculada de la interfaz.")
        else:
            clean_img = preprocess_and_remove_background(
                image,
                force_rembg=True,
                session=REMBG_SESSION
            )
    except Exception as e:
        yield None, gr.update(interactive=False), render_progress_html(f"Error en Rembg: {e}", 0)
        return

    # --------------------------------------------------------------------------
    # ETAPA 2/4: Hunyuan3D-2 GPU / VRAM (25%)
    # --------------------------------------------------------------------------
    cached = is_dit_cached()
    if not cached:
        status_file = Path("D:/AI_Models/download_status.json")
        try:
            subprocess.Popen([sys.executable, "download_model.py"], cwd=str(Path("D:/3d-generator").resolve()))
        except Exception:
            pass

        while not is_dit_cached():
            if status_file.exists():
                try:
                    with open(status_file, "r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    if sdata.get("is_complete"):
                        break
                    pct = sdata.get("percent", 0)
                    dl_mb = sdata.get("downloaded_mb", 0)
                    tot_mb = sdata.get("total_mb", 4702)
                    spd = sdata.get("speed_mb", 0)
                    eta = sdata.get("eta_seconds", 0)
                    eta_str = f"{int(eta//60)}m {int(eta%60):02d}s" if eta > 0 else "calculando..."
                    txt = f"[2/4] Descargando pesos DiT ({dl_mb:.0f}/{tot_mb:.0f} MB • {pct}% • {spd} MB/s • ETA: {eta_str})"
                    yield None, gr.update(interactive=False), render_progress_html(txt, int(pct * 0.45 + 5))
                except Exception:
                    pass
            else:
                yield None, gr.update(interactive=False), render_progress_html("[2/4] Iniciando descarga de pesos Hunyuan3D-2 (~4.9GB) a Disco D:...", 25)
            
            time.sleep(1.5)

    yield None, gr.update(interactive=False), render_progress_html("[2/4] Aceleración GPU: Cargando Hunyuan3D-2 DiT en VRAM (fp16)...", 25)

    try:
        generator = Hunyuan3DGenerator(
            model_id="tencent/Hunyuan3D-2",
            shape_subfolder="hunyuan3d-dit-v2-0-turbo",
            paint_subfolder="hunyuan3d-paint-v2-0",
            use_fp16=True,
            low_vram_mode=False
        )
    except Exception as e:
        yield None, gr.update(interactive=False), render_progress_html(f"Error al cargar Hunyuan3D-2: {e}", 0)
        return

    # --------------------------------------------------------------------------
    # ETAPA 3/4: Geometría 3D y Decimación (Streaming granular 30% -> 88%)
    # --------------------------------------------------------------------------
    yield None, gr.update(interactive=False), render_progress_html("[3/4] Síntesis Geométrica: Iniciando inferencia en GPU...", 30)

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
            yield None, gr.update(interactive=False), render_progress_html(msg, pct)
        except queue.Empty:
            pass

    worker_thread.join()

    if gen_result["error"] is not None:
        yield None, gr.update(interactive=False), render_progress_html(f"Error en geometría: {gen_result['error']}", 0)
        return

    raw_mesh = gen_result["mesh"]

    yield None, gr.update(interactive=False), render_progress_html(f"[3/4] Optimización de Malla: Decimando a {target_faces:,} caras...", 88)
    decimated_mesh = decimate_mesh_to_target_faces(raw_mesh, target_faces=target_faces)
    aligned_mesh = center_and_orient_mesh(
        decimated_mesh,
        align_floor=False,
        normalize_scale=True,
        target_height=1.2
    )

    # --------------------------------------------------------------------------
    # ETAPA 4/4: Texturizado y Empaquetado GLB (94% -> 100%)
    # --------------------------------------------------------------------------
    if generate_texture:
        yield None, gr.update(interactive=False), render_progress_html("[4/4] Texturizado PBR: Horneando mapas de color y proyección UV...", 94)
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
        yield None, gr.update(interactive=False), render_progress_html("[4/4] Empaquetado: Ensamblando malla base (sin textura)...", 94)
        import trimesh
        neutral_mat = trimesh.visual.material.PBRMaterial(
            baseColorFactor=[0.84, 0.86, 0.90, 1.0],
            metallicFactor=0.08,
            roughnessFactor=0.55,
            doubleSided=True
        )
        aligned_mesh.visual.material = neutral_mat
        final_mesh = aligned_mesh

    yield None, gr.update(interactive=False), render_progress_html("[4/4] Exportación glTF: Validando y guardando archivo .GLB con normales PBR...", 97)

    # Guardar archivo .glb en D:\AI_Models\output\
    timestamp = int(time.time())
    raw_name = (model_name or "").strip()
    if raw_name.lower().endswith(".glb"):
        raw_name = raw_name[:-4]
    safe_base = "".join(c for c in raw_name if c.isalnum() or c in ("_", "-")).strip()
    if not safe_base:
        safe_base = f"modelo_{timestamp}"
    safe_base += ".glb"

    out_file = OUTPUT_DIR / safe_base
    if out_file.exists():
        out_file = OUTPUT_DIR / f"{out_file.stem}_{timestamp}.glb"

    try:
        final_mesh.export(str(out_file), file_type="glb", include_normals=True)
        postprocess_and_validate_glb(out_file, asset_name=out_file.stem, target_faces=target_faces)
        
        CURRENT_GENERATED_GLB = out_file
        elapsed = time.time() - total_start
        file_size_mb = out_file.stat().st_size / (1024 * 1024)
        num_faces = len(final_mesh.faces)

        final_msg = render_progress_html(
            f"Modelo generado en {elapsed:.1f}s | {num_faces:,} caras | {file_size_mb:.2f} MB",
            100
        )
        
        yield str(out_file), gr.update(value=str(out_file), interactive=True), final_msg

    except Exception as e:
        yield None, gr.update(interactive=False), render_progress_html(f"Error al exportar GLB: {e}", 0)


def ui_generate_3d(
    image: Optional[Image.Image],
    clean_silhouette: Optional[Image.Image],
    preset_label: str,
    generate_texture: bool,
    model_name: str
) -> Generator[Tuple[Optional[str], any, str], None, None]:
    """
    Función generadora externa con bloqueo atómico anti-doble clic (mutex).
    """
    global IS_GENERATING
    if IS_GENERATING:
        yield None, gr.update(interactive=False), render_progress_html("Ya hay una generación en progreso. Por favor espera a que finalice.", 0)
        return
    IS_GENERATING = True
    try:
        for out in _ui_generate_3d_impl(image, clean_silhouette, preset_label, generate_texture, model_name):
            yield out
    finally:
        IS_GENERATING = False


def ui_save_to_game(category: str, subcategory: str, custom_name: str) -> str:
    """Copia directamente el .glb generado a la carpeta de Kaboomi Party."""
    global CURRENT_GENERATED_GLB
    if CURRENT_GENERATED_GLB is None or not CURRENT_GENERATED_GLB.exists():
        return "Primero genera un modelo 3D antes de guardarlo en el juego."
    
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


def list_existing_models():
    """Escanea los modelos disponibles en Kaboomi Party."""
    categories = {"bombos": BOMBOS_DIR, "bozis": BOZIS_DIR, "cosmetics": COSMETICS_DIR}
    choices = []
    for cat_name, folder in categories.items():
        if folder.exists():
            for f in folder.glob("*.glb"):
                choices.append(f"[{cat_name.upper()}] {f.name}")
    return choices or ["(No hay modelos exportados aún)"]


def load_existing_model_preview(selection: str) -> Optional[str]:
    """Carga un modelo existente de Godot en el visor."""
    if not selection or "(" in selection:
        return None
    try:
        cat_part, name_part = selection.split("] ", 1)
        cat = cat_part.replace("[", "").lower()
        target = MODELS_DIR / cat / name_part
        if target.exists():
            return str(target)
    except Exception:
        pass
    return None


# ==============================================================================
# 5. Estilos CSS Profesionales, Sin Emojis y con Animaciones de Alta Fidelidad
# ==============================================================================

CUSTOM_CSS = """
/* Reset y base oscura premium */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
    background-color: #090c14 !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased !important;
}

.gradio-container {
    max-width: 1760px !important;
    padding: 10px 20px !important;
    margin: 0 auto !important;
    background-color: #090c14 !important;
}

/* Cabecera Profesional de Grado Estudio */
.custom-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    margin-bottom: 12px;
    background: linear-gradient(135deg, rgba(18, 24, 38, 0.75), rgba(15, 20, 32, 0.75));
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}
.header-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.header-logo-icon {
    stroke: #a855f7;
    filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.5));
}
.header-title-row {
    display: flex;
    align-items: center;
    gap: 10px;
}
.custom-header-title {
    font-size: 1.32rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.4px;
    margin: 0;
}
.header-badge-tag {
    background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(99, 102, 241, 0.15));
    border: 1px solid rgba(168, 85, 247, 0.4);
    color: #c084fc;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.custom-header-sub1 {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 2px 0 0 0;
}

.header-meta {
    display: flex;
    align-items: center;
    gap: 10px;
}
.system-status-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    padding: 5px 12px;
    border-radius: 20px;
}
.status-pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
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
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.76rem;
    font-weight: 600;
    color: #cbd5e1;
}

/* Pestañas compactas superiores */
.custom-tabs > .tab-nav {
    background: transparent !important;
    border-bottom: none !important;
    gap: 8px !important;
    margin-bottom: 12px !important;
}
.custom-tabs > .tab-nav > button {
    background: #121726 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.84rem !important;
    padding: 6px 14px !important;
    font-weight: 600 !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.custom-tabs > .tab-nav > button:hover {
    background: #1c2438 !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
}
.custom-tabs > .tab-nav > button.selected {
    background: #1e263d !important;
    border-color: #8b5cf6 !important;
    color: #ffffff !important;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.25) !important;
}

/* Fila principal de 3 Columnas */
.wizard-row {
    display: flex !important;
    gap: 16px !important;
}

/* Tarjetas individuales de cada paso */
.wizard-card {
    background: #101524 !important;
    border: 1.5px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 14px 16px !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
    transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s ease !important;
}
.wizard-card:hover {
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.55) !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
}

/* Cabeceras de Tarjeta con Badge */
.step-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.step-badge {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(168, 85, 247, 0.18));
    border: 1px solid rgba(139, 92, 246, 0.4);
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

/* Botón 'Extraer Silueta' de Columna 1 */
.btn-silhouette {
    background: #182033 !important;
    border: 1px solid #2a3754 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    height: 36px !important;
    margin: 6px 0 !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.btn-silhouette:hover {
    background: #202b44 !important;
    border-color: #3b4d75 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Contenedor de Opciones de Geometría en Columna 2 */
.geometry-box {
    background: #131929 !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    margin-bottom: 8px !important;
}
.geometry-box-title {
    font-size: 0.86rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    margin-bottom: 4px !important;
}

/* Opciones de Radio como Tarjetas con Borde Amarillo al Seleccionar */
.preset-radio-list .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}
.preset-radio-list label {
    background: #171f33 !important;
    border: 1.5px solid #25314a !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    color: #f1f5f9 !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.preset-radio-list label:hover {
    border-color: #3b4d75 !important;
    background: #1c263e !important;
    transform: translateX(2px) !important;
}
.preset-radio-list label:has(input:checked),
.preset-radio-list label.selected {
    border-color: #eab308 !important;
    background: rgba(234, 179, 8, 0.09) !important;
    box-shadow: 0 0 12px rgba(234, 179, 8, 0.25) !important;
}

/* Botón Gigante Morado con Borde Amarillo Brillante */
.btn-generate-glow {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    border: 1.5px solid #eab308 !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.45), 0 0 10px rgba(234, 179, 8, 0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.5px !important;
    height: 48px !important;
    margin: 6px 0 !important;
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

/* Estado Deshabilitado Anti-Doble Clic */
.btn-generate-glow:disabled,
.btn-generate-glow[disabled],
.btn-busy {
    opacity: 0.65 !important;
    cursor: not-allowed !important;
    pointer-events: none !important;
    transform: none !important;
    box-shadow: none !important;
    background: #251e36 !important;
    border: 1.5px dashed #64748b !important;
    color: #94a3b8 !important;
}

/* Caja de Estado y Barra de Progreso Dinámica */
.live-status-container {
    background: #0c101a;
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
.status-icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
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
    background: #182033;
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

/* Botón Azul de Descarga en Columna 3 */
.btn-download-blue {
    background: linear-gradient(135deg, #0284c7, #0369a1) !important;
    border: 1px solid #38bdf8 !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    height: 42px !important;
    margin: 8px 0 6px 0 !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.btn-download-blue:hover:not(:disabled) {
    background: #0284c7 !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Tarjeta 'Guardar directo en el Juego' en Columna 3 */
.game-save-card {
    background: #131929 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 10px !important;
    padding: 10px 12px !important;
    margin-top: 6px !important;
}
.game-save-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.86rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 6px;
}
.btn-save-game {
    background: #1b2336 !important;
    border: 1px solid #2a3754 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    height: 36px !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
.btn-save-game:hover {
    background: #222d45 !important;
    border-color: #3b4d75 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Fondo de tablero de ajedrez para previsualizar transparencia */
.checkerboard-image {
    background-image: linear-gradient(45deg, #151a29 25%, transparent 25%), linear-gradient(-45deg, #151a29 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #151a29 75%), linear-gradient(-45deg, transparent 75%, #151a29 75%) !important;
    background-size: 16px 16px !important;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px !important;
    background-color: #0f1320 !important;
}
"""

custom_theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="blue",
    neutral_hue="slate"
)

# ==============================================================================
# 6. Construcción del Dashboard con Gradio Blocks
# ==============================================================================

with gr.Blocks(title="Kaboomi 3D Studio") as demo:
    
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
                        <span class="custom-header-title">Kaboomi 3D Studio</span>
                        <span class="header-badge-tag">Hunyuan3D-2 Turbo</span>
                    </div>
                    <p class="custom-header-sub1">Generador de Modelos 3D (.glb) Optimizados para Godot 4</p>
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
            with gr.Row(elem_classes=["wizard-row"]):
                
                # ==============================================================
                # COLUMNA 1: 1. Imagen y Silueta
                # ==============================================================
                with gr.Column(scale=1, elem_classes=["wizard-card"]):
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
                        height=230,
                        type="pil",
                        sources=["upload", "clipboard"]
                    )
                    
                    btn_rembg = gr.Button(
                        "Extraer Silueta Alfa",
                        icon="assets/icons/scissors.svg",
                        elem_classes=["btn-silhouette"]
                    )
                    
                    preview_rembg = gr.Image(
                        label="Silueta Limpia (Canal Alfa)",
                        height=195,
                        interactive=False,
                        type="pil",
                        elem_classes=["checkerboard-image"]
                    )

                # ==============================================================
                # COLUMNA 2: 2. Configuración y Generación
                # ==============================================================
                with gr.Column(scale=1, elem_classes=["wizard-card"]):
                    gr.HTML(
                        """
                        <div class="step-card-header">
                            <span class="step-badge">PASO 2</span>
                            <span class="step-title">CONFIGURACIÓN Y GENERACIÓN</span>
                        </div>
                        """
                    )
                    
                    with gr.Group(elem_classes=["geometry-box"]):
                        gr.HTML('<div class="geometry-box-title">Opciones de Topología y Malla</div><div style="font-size:0.78rem;color:#94a3b8;margin-bottom:6px;">Presets de Rendimiento:</div>')
                        
                        preset_radio = gr.Radio(
                            choices=list(PRESET_CONFIG_MAP.keys()),
                            value="Móvil Rápido (4,000 caras • Balanceado)",
                            label="",
                            show_label=False,
                            elem_classes=["preset-radio-list"]
                        )
                        
                        chk_texture = gr.Checkbox(
                            value=True,
                            label="Generar y Proyectar Textura de Color"
                        )

                    btn_generate = gr.Button(
                        "GENERAR MODELO 3D (.GLB)",
                        icon="assets/icons/wand.svg",
                        elem_classes=["btn-generate-glow"]
                    )
                    
                    status_widget = gr.HTML(
                        value=render_progress_html("Listo para generar. Sube una imagen y presiona generar.", 0)
                    )

                # ==============================================================
                # COLUMNA 3: 3. Visor 3D y Descarga
                # ==============================================================
                with gr.Column(scale=1, elem_classes=["wizard-card"]):
                    gr.HTML(
                        """
                        <div class="step-card-header">
                            <span class="step-badge">PASO 3</span>
                            <span class="step-title">VISOR 3D Y EXPORTACIÓN</span>
                        </div>
                        """
                    )
                    
                    viewer_3d = gr.Model3D(
                        label="Visor 3D Interactivo",
                        height=340,
                        clear_color=[0.10, 0.12, 0.18, 1.0]
                    )
                    
                    download_btn = gr.DownloadButton(
                        "DESCARGAR ARCHIVO .GLB",
                        icon="assets/icons/download.svg",
                        elem_classes=["btn-download-blue"],
                        interactive=False
                    )
                    
                    with gr.Group(elem_classes=["game-save-card"]):
                        gr.HTML(
                            """
                            <div class="game-save-header">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <line x1="6" x2="10" y1="12" y2="12"/><line x1="8" x2="8" y1="10" y2="14"/><line x1="15" x2="15.01" y1="13" y2="13"/><line x1="18" x2="18.01" y1="11" y2="11"/><rect width="20" height="12" x="2" y="6" rx="6"/>
                                </svg>
                                <span>Integración Directa con Godot 4</span>
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

        # TAB SECUNDARIA (Modelos en Godot)
        with gr.Tab("Biblioteca de Modelos"):
            gr.Markdown("### Biblioteca de Modelos `.glb` en Kaboomi Party")
            with gr.Row():
                with gr.Column(scale=1):
                    btn_refresh_library = gr.Button("Actualizar Lista", icon="assets/icons/refresh.svg")
                    dropdown_models = gr.Dropdown(
                        choices=list_existing_models(),
                        label="Selecciona un modelo existente"
                    )
                with gr.Column(scale=1):
                    library_viewer = gr.Model3D(
                        label="Visualizador",
                        height=360,
                        clear_color=[0.10, 0.12, 0.18, 1.0]
                    )

        # TAB TERCIARIA (Estado del Sistema)
        with gr.Tab("Configuración del Sistema"):
            gr.Markdown(
                f"""
                ### Arquitectura y Entorno de Ejecución
                - **GPU:** {GPU_SYS_TEXT}
                - **Almacenamiento Local:** `D:\\AI_Models` (100% en Disco D:)
                - **Directorio de Salida (.GLB):** `{OUTPUT_DIR}`
                - **Ruta de Juego Godot 4:** `{KABOOMI_ROOT}`
                - **Motor de Inferencia:** Tencent Hunyuan3D-2 Turbo + Rembg U2-Net
                """
            )

    # ==========================================================================
    # Conexión de Eventos y Callbacks
    # ==========================================================================

    # 1. Extracción de silueta (automática al subir o manual por botón)
    input_img.upload(
        fn=ui_remove_background,
        inputs=[input_img],
        outputs=[preview_rembg, status_widget]
    )
    
    btn_rembg.click(
        fn=ui_remove_background,
        inputs=[input_img],
        outputs=[preview_rembg, status_widget]
    )

    # 2. Generación 3D en streaming con bloqueo anti-doble clic instantáneo
    btn_generate.click(
        fn=lambda: (
            gr.update(interactive=False, value="PROCESANDO MODELO 3D...", icon="assets/icons/loader.svg"),
            gr.update(interactive=False)
        ),
        inputs=[],
        outputs=[btn_generate, download_btn]
    ).then(
        fn=ui_generate_3d,
        inputs=[
            input_img,
            preview_rembg,
            preset_radio,
            chk_texture,
            txt_model_name
        ],
        outputs=[viewer_3d, download_btn, status_widget]
    ).then(
        fn=lambda: gr.update(interactive=True, value="GENERAR MODELO 3D (.GLB)", icon="assets/icons/wand.svg"),
        inputs=[],
        outputs=[btn_generate]
    )

    # 3. Guardar directamente en el juego Kaboomi Party
    btn_save_game.click(
        fn=ui_save_to_game,
        inputs=[dest_category, dest_subcategory, txt_model_name],
        outputs=[export_feedback]
    )

    # 4. Explorador de modelos en biblioteca
    btn_refresh_library.click(
        fn=list_existing_models,
        inputs=[],
        outputs=[dropdown_models]
    )
    dropdown_models.change(
        fn=load_existing_model_preview,
        inputs=[dropdown_models],
        outputs=[library_viewer]
    )


# ==============================================================================
# Lanzamiento del Servidor
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" KABOOMI 3D STUDIO (GRADIO WEBUI)")
    print(f" -> Directorio de Salida: {OUTPUT_DIR}")
    print(f" -> Modelos y Caché: D:\\AI_Models (100% Protegido)")
    print(f" -> Proyecto Kaboomi Party: {KABOOMI_ROOT}")
    print("="*70)
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=custom_theme,
        css=CUSTOM_CSS,
        allowed_paths=[str(OUTPUT_DIR), str(MODELS_DIR), str(Path("assets").resolve())]
    )
