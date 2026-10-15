import os
import sys
import time
import subprocess

print("=" * 76)
print("       VERIFICADOR Y DIAGNÓSTICO EN VIVO DE GPU (NVIDIA vs INTEL)")
print("=" * 76)

# 1. Adaptadores gráficos según Windows
print("\n[1/4] Adaptadores gráficos detectados por Windows:")
try:
    ps_cmd = 'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'
    res = subprocess.run(['powershell', '-NoProfile', '-Command', ps_cmd], capture_output=True, text=True, check=True)
    gpus = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    for idx, name in enumerate(gpus):
        print(f"  - Adaptador Windows [{idx}]: {name}")
except Exception as e:
    print(f"  [AVISO] No se pudo consultar WMI: {e}")

# 2. Driver NVIDIA (nvidia-smi)
print("\n[2/4] Estado del Driver NVIDIA (nvidia-smi):")
try:
    res = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,utilization.gpu', '--format=csv,noheader'],
        capture_output=True, text=True, check=True
    )
    for line in res.stdout.strip().splitlines():
        if line.strip():
            parts = [p.strip() for p in line.split(',')]
            print(f"  - NVIDIA GPU #{parts[0]}: {parts[1]} | Driver: {parts[2]} | VRAM: {parts[3]} | Uso GPU: {parts[4]}")
except Exception:
    print("  - nvidia-smi no disponible directamente en el PATH (común en algunas laptops).")

# 3. Diagnóstico PyTorch CUDA
print("\n[3/4] Detección de aceleración CUDA en PyTorch:")
try:
    import torch
    print(f"  - PyTorch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"  - torch.cuda.is_available(): {cuda_ok}")

    if not cuda_ok:
        print("  [ERROR CRÍTICO] PyTorch reporta que CUDA NO está disponible.")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"  - Total GPUs NVIDIA registradas en PyTorch: {device_count}")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"    -> PyTorch 'cuda:{i}': {gpu_name} ({vram_gb:.2f} GB VRAM)")

    env_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "Sin restricción")
    print(f"  - Variable CUDA_VISIBLE_DEVICES: {env_cuda}")

except Exception as e:
    print(f"  [ERROR al inicializar PyTorch CUDA]: {e}")
    sys.exit(1)

# Mapeo explicativo
print("\n" + "-" * 76)
print("  ¿POR QUÉ EL ADMINISTRADOR DE TAREAS Y PYTORCH USAN NÚMEROS DISTINTOS?")
print("-" * 76)
print("  1. En el Administrador de Tareas de Windows (Rendimiento):")
print("     • GPU 0 = Intel(R) UHD Graphics (la pantalla de la laptop va conectada a ella)")
print("     • GPU 1 = NVIDIA GeForce RTX 4050 (tarjeta dedicada)")
print("  2. En PyTorch / CUDA:")
print("     • ¡CUDA NO RECONOCE A INTEL! Solo cuenta tarjetas NVIDIA.")
print("     • Por eso, en PyTorch:")
print("       'cuda:0' = NVIDIA GeForce RTX 4050 (Tu GPU dedicada de alto rendimiento).")
print("       'cuda:1' = NO existe (fallaría porque solo tienes 1 tarjeta NVIDIA).")
print("  3. ¿Por qué se ve porcentaje en la GPU Intel?")
print("     • Google Chrome / Edge usan la GPU 0 (Intel) para dibujar la ventana y el visor 3D.")
print("     • Si el modelo falló antes de generar (ej. en la descarga o en SingleImageEncoder),")
print("       la GPU NVIDIA nunca llegó a recibir órdenes, quedándose en 0%.")
print("-" * 76)

# 4. Prueba de estrés en vivo
print("\n[4/4] TEST EN VIVO: COMPROBANDO QUE CUDA:0 ES TU TARJETA NVIDIA")
print("  >> Abre el Administrador de Tareas (Ctrl + Shift + Esc) -> Pestaña Rendimiento.")
print("  >> Mantén a la vista 'GPU 1 (NVIDIA)' y 'GPU 0 (Intel)'.")
print("  >> Ejecutando cálculo intensivo de matrices en 'cuda:0' durante 6 segundos...")

try:
    device = torch.device("cuda:0")
    a = torch.randn((6144, 6144), device=device, dtype=torch.float16)
    b = torch.randn((6144, 6144), device=device, dtype=torch.float16)

    start_time = time.time()
    iterations = 0
    while time.time() - start_time < 6.0:
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        iterations += 1

    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

    print(f"\n  [RESULTADO TEST]:")
    print(f"  ✓ Dispositivo ejecutado: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ Multiplicaciones completadas: {iterations} operaciones tensoriales")
    print(f"  ✓ Memoria VRAM activa en la tarjeta: {allocated_mb:.1f} MB (Reservada: {reserved_mb:.1f} MB)")
    print(f"  ✓ Conclusión: PyTorch está enviando TODO a tu tarjeta NVIDIA.")

except Exception as e:
    print(f"  [ERROR]: {e}")
    sys.exit(1)

print("\n" + "=" * 76)
print("  FIN DEL DIAGNOSTICO")
print("=" * 76)
