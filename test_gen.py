import os
import sys
import time
import traceback
from PIL import Image

os.environ['HY3DGEN_MODELS'] = 'D:\\AI_Models'

print("=== INICIANDO TEST DE GENERACIÓN ===")
t_start = time.time()

try:
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    print(f"[{time.time()-t_start:.1f}s] Importación de hy3dgen exitosa.")
    
    print(f"[{time.time()-t_start:.1f}s] Cargando Hunyuan3DDiTFlowMatchingPipeline...")
    t_load = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2',
        subfolder='hunyuan3d-dit-v2-0-turbo',
        torch_dtype='fp16'
    )
    print(f"[{time.time()-t_start:.1f}s] Modelo cargado en {time.time()-t_load:.2f}s. Moviendo a CUDA...")
    pipe.to('cuda')
    print(f"[{time.time()-t_start:.1f}s] En GPU CUDA.")
    
    img = Image.open('test_input.png')
    print(f"[{time.time()-t_start:.1f}s] Iniciando inferencia (num_inference_steps=5)...")
    t_infer = time.time()
    outputs = pipe(
        image=img,
        num_inference_steps=5,
        octree_resolution=256,
        output_type='trimesh'
    )
    mesh = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    print(f"[{time.time()-t_start:.1f}s] Inferencia completada en {time.time()-t_infer:.2f}s!")
    print(f"Vertices: {len(mesh.vertices):,}, Caras: {len(mesh.faces):,}")
    print(f"Bounds: {mesh.bounds}")
    
    # Exportar GLB de prueba
    out_glb = "D:/AI_Models/output/test_model.glb"
    mesh.export(out_glb, file_type="glb")
    print(f"[{time.time()-t_start:.1f}s] Exportado a {out_glb} con éxito!")
    print("=== TEST COMPLETADO CON ÉXITO TOTAL ===")

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
