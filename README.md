# 🎮 Kaboomi 3D Generator (Tencent Hunyuan3D-2)

Sistema de generación local ilimitada de modelos 3D en formato `.glb` a partir de imágenes 2D, optimizado para GPU NVIDIA (16GB VRAM) y diseñado específicamente para el pipeline de **Godot 4** en **Kaboomi Party**.

---

## ⚡ Características Principales

- **Arquitectura de IA de Última Generación:** Utiliza `Tencent/Hunyuan3D-2` con la variante `Hunyuan3D-DiT-v2-0-Turbo` (generación de geometría en 5 pasos de inferencia mediante destilación de flujo) y `Hunyuan3D-Paint-v2-0` para el horneado de texturas multi-vista.
- **Optimización para 16GB VRAM en fp16:** Carga y ejecución secuencial con liberación de memoria CUDA (`empty_cache`) entre las fases de forma y textura para caber holgadamente sin desbordamiento de memoria (OOM).
- **Almacenamiento Seguro 100% en Disco `D:\`:**
  - Pesos de Hugging Face (`HF_HOME = D:\AI_Models\huggingface`)
  - Modelos de eliminación de fondo (`U2NET_HOME = D:\AI_Models\u2net`)
  - Modelos de PyTorch (`TORCH_HOME = D:\AI_Models\torch`)
  - Entorno virtual (`.venv` enlazado a `D:\AI_Models\venv_3dgen`)
  - Cache de descargas (`PIP_CACHE_DIR = D:\AI_Models\pip_cache`)
  *El disco `C:\` permanece completamente protegido.*
- **Decimación Automática a 4000 Triángulos:** Algoritmo *Quadric Edge Collapse* (QEM) en cascada (`PyMeshLab` -> `pyfqmr` -> `Trimesh`) para garantizar rendimiento de 60 FPS en dispositivos móviles en Godot 4.
- **Centrado Exacto en el Origen `(0, 0, 0)`:** Opción de centrado geométrico puro o pivote en la base (pies en `Y=0.0`) para que los personajes se apoyen perfectamente sobre el suelo en Godot.
- **Validación con `pygltflib`:** Estándar glTF 2.0 con metadatos embebidos e inspección de materiales PBR.
- **Exportación con 1 Clic a Kaboomi Party:** Integración directa con `kaboomi-party/assets/models/bombos/`, `bozis/` y `cosmetics/`.

---

## 📁 Estructura del Proyecto

```text
Proyectos/
├── 3d-generator/
│   ├── .env                   # Variables que configuran todas las rutas hacia D:\
│   ├── requirements.txt       # Dependencias PyTorch CUDA, hy3dgen, rembg, trimesh, etc.
│   ├── setup.bat              # Script de instalación automática para Windows
│   ├── setup.sh               # Script de instalación para Linux / WSL
│   ├── img2glb.py             # CLI de consola para generación por lotes o scripts
│   ├── webui.py               # Interfaz interactiva en Gradio (navegador web)
│   ├── run_webui.bat          # Lanzador directo de la interfaz web
│   ├── run_cli.bat            # Lanzador directo de comandos por consola
│   └── README.md              # Documentación técnica
└── Games/
    └── kaboomi-party/
        └── assets/
            └── models/
                ├── bombos/    # Modelos jugables de Bombos
                ├── bozis/     # Modelos de Bozis
                └── cosmetics/ # Accesorios y sombreros
```

---

## 🚀 Instalación Rápida (Windows)

1. Haz doble clic en **`setup.bat`** (o ejecútalo en la consola dentro de `3d-generator/`):
   ```cmd
   setup.bat
   ```
2. El instalador recorre 9 pasos (se puede volver a ejecutar: omite lo que ya está bien):
   1. **Python 3.10–3.12 de 64 bits** (ofrece instalar 3.11 con `winget` si no hay).
   2. **Entorno virtual** en `<carpeta de modelos>\venv_3dgen`, enlazado como `.venv`.
   3. **PyTorch**: CUDA 12.4 si hay tarjeta NVIDIA, o solo CPU. La tarjeta concreta se elige al abrir la app.
   4. **Dependencias**: con Python 3.11 instala el entorno exacto verificado (`requirements-lock.txt`); si no, `requirements.txt`. La versión CUDA de PyTorch queda fijada para que ninguna dependencia la cambie por la de CPU.
   5. **Motores**: Hunyuan3D-2 (`hy3dgen`) y/o Stable Fast 3D (repositorio + token de Hugging Face).
   6. **Pesos de los modelos**: muestra qué falta y ofrece descargarlo ya (geometría ~4.9 GB, textura ~9 GB, SF3D ~3.8 GB), para que el primer uso no se quede esperando.
   7. **Textura Hunyuan3D-Paint** (opcional): ofrece instalar Visual Studio Build Tools y CUDA Toolkit 12.4 con `winget` (sin tocar el driver) y compilar `custom_rasterizer`. También se puede hacer después con `instalar_textura_hunyuan.bat`.
   8. **Verificación de la GPU** con PyTorch.
   9. **Configuración**: guarda la carpeta de modelos elegida en `.env` (la usan la app, la CLI y los lanzadores) y muestra el estado final de cada componente.

   Estado de los pesos en cualquier momento (no usa la GPU): `.venv\Scripts\python.exe preparar_modelos.py status`

---

## 🖥️ Selección de GPU (equipos con 2 o más tarjetas)

Al arrancar `webui.bat`, `webui-kambo.bat` o `run_cli.bat`, `gpu_select.py` enumera las tarjetas con `nvidia-smi` y, si el equipo tiene más de un adaptador, muestra un menú:

```text
   [0] NVIDIA GeForce RTX 4060 (8 GB VRAM)  <- predeterminada
   [1] NVIDIA GeForce RTX 3090 (24 GB VRAM)
   [C] CPU (sin aceleración, muy lento)
    -  Intel(R) UHD Graphics: no compatible con CUDA (solo pantalla / navegador)
   Elige GPU [Enter = 0, automático en 15s]:
```

- La tarjeta elegida se convierte en `cuda:0` dentro del proceso (`CUDA_VISIBLE_DEVICES` por UUID + `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
- La última elección se recuerda en `.gpu_choice` y es la opción predeterminada la próxima vez.
- Las GPUs Intel/AMD aparecen solo como información: CUDA no puede usarlas.
- Para saltar el menú: `webui.bat --gpu 1`, `run_cli.bat ... --gpu cpu` o `set STUDIO_GPU=0`. Para forzarlo: `--gpu ask`.
- Probar el selector por separado: `.venv\Scripts\python.exe gpu_select.py --gpu ask`

---

## 💻 Uso por Consola (CLI: `img2glb.py`)

Comando oficial según especificación:

```bash
python img2glb.py --input foto.png --output ../Games/kaboomi-party/assets/models/bombos/mi_modelo.glb --faces 4000
```
*(O utilizando el lanzador rápido: `run_cli.bat --input foto.png --output ../Games/kaboomi-party/assets/models/bombos/mi_modelo.glb --faces 4000`)*

### Opciones Disponibles:

| Parámetro | Valor por Defecto | Descripción |
| :--- | :--- | :--- |
| `--input`, `-i` | *(Requerido)* | Ruta a la imagen (PNG, JPG, WEBP). Si no es transparente, `rembg` quita el fondo. |
| `--output`, `-o` | *(Requerido)* | Ruta de destino del modelo (`.glb`, `.stl` o `.3mf`). Crea carpetas automáticamente. |
| `--faces`, `-f` | `4000` | Número de triángulos objetivo (ideal móvil: 3000-5000). |
| `--steps`, `-s` | `5` | Pasos de inferencia DiT (5 pasos en Turbo = ~15 segundos). |
| `--seed` | `42` | Semilla de generación (-1 para aleatorio). |
| `--align-bottom` | `False` | Si se activa, coloca los pies en `Y=0.0` en lugar de centrar en el centro geométrico. |
| `--formats` | `None` | Formatos a exportar separados por comas (ej: `glb,stl,3mf`). |
| `--export-stl` | `False` | Exporta adicionalmente el modelo en formato `.stl` para impresión 3D / CAD. |
| `--export-3mf` | `False` | Exporta adicionalmente el modelo en formato `.3mf` para slicers de manufactura (Bambu, Prusa, Orca). |

---

## 🌐 Interfaz Web Interactiva (WebUI)

Dispones de dos versiones según tu flujo de trabajo:

### 1. Versión General / Estudio 3D (`webui.bat`)
Para cualquier proyecto (Godot, Unity, Unreal, Blender, Impresión 3D):
```cmd
webui.bat
```
*(O `python webui.py` / `run_webui.bat`)*
- Interfaz neutral y profesional (3D AI Studio).
- Biblioteca de modelos que lista y visualiza todos los `.glb` generados en la carpeta de salidas.
- Permite exportar y copiar con 1 clic a cualquier carpeta o proyecto local que especifiques.

### 2. Edición Kaboomi Party (`webui-kambo.bat`)
Específica con la integración y biblioteca de Kaboomi Party:
```cmd
webui-kambo.bat
```
*(O `python webui_kambo.py` / `run_webui_kambo.bat`)*
- Detección automática del proyecto `kaboomi-party`.
- Guardado con 1 clic en `bombos/`, `bozis/` o `cosmetics/`.
- Biblioteca integrada con los modelos existentes del juego.

### 🎨 Flujo de Trabajo en 3 Pasos (Ambas Versiones):
1. **Paso 1:** Arrastra una imagen y pulsa *"Extraer Silueta Alfa"* si deseas previsualizar la silueta.
2. **Paso 2:** Ajusta los triángulos (presets de 2K, 4K u 8K caras) y pulsa **"GENERAR MODELO 3D (GLB / STL / 3MF)"**.
3. **Paso 3:** Inspecciona el modelo en el visor 3D interactivo con rotación de 360° y:
   - **Descargar .GLB:** Con texturas PBR completas y normales (Godot, Unity, Web).
   - **Descargar .STL:** Geometría triangular sólida optimizada para cualquier impresora 3D.
   - **Descargar .3MF:** Formato moderno de manufactura aditiva con metadatos para Bambu Studio, OrcaSlicer y PrusaSlicer.
   - **Exportar directo a tu proyecto o juego.**

---

## 🎮 Integración en Godot 4 (Kaboomi Party)

1. Al enviar el modelo con 1 clic desde la WebUI o el CLI, el archivo `.glb` se copia inmediatamente a la carpeta correspondiente dentro de `kaboomi-party/assets/models/`.
2. Al regresar a **Godot Engine**, el motor detectará el nuevo archivo `.glb` y lo importará automáticamente.
3. **Recomendaciones de importación en Godot:**
   - En el panel de **Importación** del `.glb` en Godot:
     - `Mesh Save to File`: Habilitado si se desea instanciar múltiples veces.
     - `Lightmapping`: Disabled (para móviles).
     - Añadir un `CollisionShape3D` con forma de cápsula o cilindro para física de personajes.

---

## 🛠️ Diagnóstico y Resolución de Problemas

- **¿Dónde están los archivos descargados?**
  Todos los pesos y cachés residen en `D:\AI_Models\`. Si necesitas liberar espacio en el futuro, puedes limpiar esa carpeta sin afectar tu sistema operativo.
- **Memoria: un solo modelo en la GPU a la vez, con validación antes de cada carga.**
  Flujo con textura: 1) se genera la geometría, 2) el modelo de forma sale de la VRAM y se comprueba que la VRAM quedó libre, 3) se valida que haya VRAM (~4.8 GB) y RAM (~7.5 GB + margen) libres para Hunyuan3D-Paint; si la RAM no alcanza, se libera también la copia en RAM del modelo de forma, 4) solo entonces se carga Paint. Si aun así no cabe, no se carga (se usa la textura de contingencia) para no trabar el equipo; con forma o SF3D se muestra un error claro.
  Opciones en `.env` (ver `.env.example`): `KEEP_MODELS_IN_RAM`, `RAM_SAFETY_MARGIN_GB`, `RAM_PARK_MIN_FREE_GB`, `SHAPE_LOAD_MODE`.
- **Mínima RAM al cargar.** hy3dgen y SF3D cargan sus pesos pasando por la RAM (pico de ~14 GB y ~8 GB). El generador los carga directamente a la GPU: la geometría usa ~0 GB de RAM y se sirve en modo `full` (todo en GPU, si hay ~6 GB de VRAM libre) o `stream` (por partes desde el disco, ~3 GB de VRAM). Misma calidad en ambos modos. Diagnóstico sin usar la GPU: `.venv\Scripts\python.exe preparar_modelos.py memoria`.
- **Perfiles de VRAM:**
  | Perfil | Pipeline de forma | Octree máx. | Textura |
  | :--- | :--- | :--- | :--- |
  | 6 GB / 8 GB | Un componente en GPU a la vez (condicionador → DiT → VAE) | 256 | 1024 |
  | 12 GB | Completo en GPU | 320 | 1024 |
  | 16 GB+ | Completo en GPU | 384 | 2048 |
- **Calidad y caras:** "Máxima Calidad HD" conserva la malla completa **sin textura**. Con textura, cualquier preset se limita a 40.000 caras (el mismo tope que usa Hunyuan3D-2 antes de pintar).
- **Textura Hunyuan3D-Paint en GPUs de 6 GB, sin compilar nada:**
  - `custom_rasterizer` se sustituye por `torch_rasterizer.py` (PyTorch en GPU, reproduce el kernel original: mismo z-buffer y baricéntricas con corrección de perspectiva).
  - Delight y multivista se cargan directo a la GPU de uno en uno (RAM de pico ~3 GB).
  - El multivista elige su ubicación según la VRAM libre y, si se queda sin VRAM, reintenta con la siguiente:

    | Ubicación | Qué hay en la GPU durante los 30 pasos | VRAM aprox. |
    | :--- | :--- | :--- |
    | `gpu` | UNet (2 copias), text encoder y VAE | ~7 GB |
    | `lean` | UNet principal; la copia de referencia (`unet_dual`, se usa 1 vez) y el text encoder suben solo al usarse | ~5 GB |
    | `bloques` | solo el bloque del UNet en curso (≤ 0.56 GB), leído del archivo con mmap | ~3 GB |

    Misma calidad en las tres. Los pesos transmitidos se apoyan en la caché de disco de Windows, sin reservar RAM.
  - Aceleradores nativos opcionales (`instalar_textura_hunyuan.bat`): `mesh_processor` en C++ (solo Visual Studio Build Tools) y `custom_rasterizer` en CUDA (además CUDA Toolkit 12.4).
  - Recomendado en Windows: Panel de control de NVIDIA → Administrar configuración 3D → *Política de respaldo de memoria del sistema CUDA* = **Preferir sin respaldo**. Así, si falta VRAM, la app cambia de ubicación en vez de desbordar a la RAM compartida (muy lento).
- **La descarga automática de pesos falla:** la interfaz muestra el error (no se queda colgada) si la descarga falla, termina sin completar o pasa 5 minutos sin avanzar. Reintenta con `descargar_modelos.bat`.
