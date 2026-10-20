import os
import sys
import time
from PIL import Image
import torch

os.environ["HY3DGEN_MODELS"] = "D:/AI_Models"
from webui import _ui_generate_3d_impl

img = Image.open("D:/AI_Models/temp/boy_dino_input.png")

print("=" * 70)
print("  PROBANDO FLUJO EXACTO DE WEBUI.PY CON DINO BOY")
print("=" * 70)

gen = _ui_generate_3d_impl(
    image=img,
    clean_silhouette=None,
    preset_label="Alta Calidad (50,000 caras • Con textura: 40k)",
    ai_model_choice="Hunyuan3D-2 Turbo (Geometría HD • Sólidos 3D)",
    generate_texture=False,
    model_name="test_boy_dino",
    vram_profile_choice="Auto-detectar (Recomendado)"
)

for step in gen:
    pass

print("=" * 70)
print("  FLUJO WEBUI TERMINADO EXITOSAMENTE!")
print("=" * 70)
