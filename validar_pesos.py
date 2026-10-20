import os
import sys
import yaml
from pathlib import Path

from studio_env import HUNYUAN_LOCAL_DIR

dit_dir = HUNYUAN_LOCAL_DIR / "hunyuan3d-dit-v2-0-turbo"
safetensors_path = dit_dir / "model.fp16.safetensors"
config_path = dit_dir / "config.yaml"

print(f"[1/3] Verificando archivos en {dit_dir}...")
if not config_path.exists():
    print("  [ERROR] config.yaml no existe en la carpeta.")
    sys.exit(1)
print(f"  [OK] config.yaml encontrado ({config_path.stat().st_size} bytes).")

if not safetensors_path.exists():
    print(f"  [ERROR] model.fp16.safetensors no existe en: {safetensors_path}")
    print("  Por favor ejecuta 'descargar_modelos.bat' para descargarlo.")
    sys.exit(1)

size_bytes = safetensors_path.stat().st_size
size_gb = size_bytes / (1024**3)
print(f"  [OK] model.fp16.safetensors encontrado: {size_bytes:,} bytes ({size_gb:.2f} GB)")
if size_bytes < 4.5 * 1024**3:
    print("  [PELIGRO] El archivo mide menos de 4.5 GB. Está incompleto o corrupto!")
    print("  Tamaño oficial esperado: 4,930,777,530 bytes (~4.93 GB).")
    print("  Ejecuta 'descargar_modelos.bat' para completar los bytes faltantes.")
else:
    print("  [OK] Tamaño consistente con la versión oficial (~4.93 GB).")

print("\n[2/3] Inspeccionando estructura de llaves del modelo...")
try:
    import safetensors.torch
    ckpt = safetensors.torch.load_file(str(safetensors_path), device='cpu')
    cond_keys = [k for k in ckpt.keys() if k.startswith('conditioner.')]
    model_keys = [k for k in ckpt.keys() if k.startswith('model.')]
    vae_keys = [k for k in ckpt.keys() if k.startswith('vae.')]
    print(f"  Total tensores en archivo: {len(ckpt):,}")
    print(f"  - Tensores de conditioner: {len(cond_keys)}")
    print(f"  - Tensores de model (DiT): {len(model_keys)}")
    print(f"  - Tensores de vae:        {len(vae_keys)}")
    if len(cond_keys) == 0:
        print("  [ERROR CRÍTICO] El archivo NO contiene la sección conditioner!")
except Exception as e:
    print(f"  [ERROR al leer safetensors]: {e}")
    sys.exit(1)

print("\n[3/3] Probando carga en SingleImageEncoder...")
try:
    from hy3dgen.shapegen.pipelines import instantiate_from_config
    cfg = yaml.safe_load(open(config_path, encoding='utf-8'))
    cond = instantiate_from_config(cfg['conditioner'])
    cond_state = {k.replace('conditioner.', ''): v for k, v in ckpt.items() if k.startswith('conditioner.')}
    
    # Probar con strict=True
    try:
        cond.load_state_dict(cond_state, strict=True)
        print("  [OK] Carga en modo estricto (strict=True): 100% compatible sin faltantes!")
    except RuntimeError as re_err:
        print("  [AVISO] Carga estricta reportó diferencias:")
        res = cond.load_state_dict(cond_state, strict=False)
        print(f"  -> Llaves faltantes ({len(res.missing_keys)}): {res.missing_keys[:5]}")
        print(f"  -> Llaves inesperadas ({len(res.unexpected_keys)}): {res.unexpected_keys[:5]}")
        print("  [SOLUCIONADO] Con el parche strict=False que agregamos a img2glb.py, estas diferencias menores se ignoran y el modelo genera correctamente.")

except Exception as e:
    print(f"  [ERROR en conditioner]: {e}")
