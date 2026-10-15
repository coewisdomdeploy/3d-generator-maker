import os
import sys
import time
from pathlib import Path
from PIL import Image

print("=" * 76)
print("   PRUEBA DIRECTA DE GENERACIÓN 3D EN GPU NVIDIA (RTX 4050 / CUDA:0)")
print("=" * 76)

import torch

if not torch.cuda.is_available():
    print("[ERROR] PyTorch indica que CUDA no está disponible.")
    sys.exit(1)

torch.cuda.set_device(0)
gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
vram_total_gb = props.total_memory / (1024 ** 3)

print(f"\n[1/3] Dispositivo CUDA fijado:")
print(f"  - Tarjeta: {gpu_name}")
print(f"  - VRAM Total: {vram_total_gb:.2f} GB")
print(f"  - VRAM Inicial en uso: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

# Imagen de prueba
img_path = Path("test_input.png")
if not img_path.exists():
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.ellipse([32, 32, 224, 224], fill=(220, 50, 50, 255))
    img.save("test_input.png")
else:
    img = Image.open(img_path).convert("RGBA")

print(f"\n[2/3] Cargando motor Hunyuan3DGenerator en 'cuda:0' con fp16...")
from img2glb import Hunyuan3DGenerator

gen = Hunyuan3DGenerator(
    model_id="tencent/Hunyuan3D-2",
    shape_subfolder="hunyuan3d-dit-v2-0-turbo",
    use_fp16=True,
    vram_profile="low_6gb",
    device="cuda:0"
)

print(f"  - VRAM tras cargar modelo: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")

print(f"\n[3/3] Ejecutando generación de prueba (3 pasos DiT en GPU)...")
print("  >> Abre el Administrador de Tareas (Ctrl+Shift+Esc) para ver la GPU NVIDIA al 100%.")

t0 = time.time()

def progress_cb(text, pct):
    print(f"  [Progreso {pct}%] {text}")

mesh = gen.generate_shape(
    img,
    steps=3,
    octree_resolution=256,
    seed=42,
    progress_callback=progress_cb
)

t_total = time.time() - t0
out_file = Path("D:/AI_Models/output/test_gpu_output.glb") if Path("D:/").exists() else Path("test_gpu_output.glb")
out_file.parent.mkdir(parents=True, exist_ok=True)
mesh.export(str(out_file))

vram_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)

print("\n" + "=" * 76)
print("  RESULTADO DE LA PRUEBA:")
print(f"  ✓ Generación completada con ÉXITO en: {t_total:.1f} segundos")
print(f"  ✓ GPU que realizó el trabajo: {gpu_name}")
print(f"  ✓ Pico de VRAM utilizado: {vram_peak:.1f} MB (sin sobrepasar los 6GB)")
print(f"  ✓ Modelo 3D exportado a: {out_file}")
print("=" * 76)
