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
2. El script realizará automáticamente:
   - Detección de Python 3.11 en `D:\Programas\Python311` o en el sistema.
   - Creación del entorno virtual en `D:\AI_Models\venv_3dgen`.
   - Instalación de PyTorch con soporte **CUDA 12.4**.
   - Instalación de `requirements.txt` y `hy3dgen`.
   - Verificación del estado de la GPU NVIDIA.

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
- **Memoria VRAM (GPU 16GB):**
  El pipeline opera en `fp16` con descarga secuencial de modelos. El pico máximo de consumo de VRAM no supera los 9-11 GB, dejando suficiente margen para que el sistema operativo y Godot permanezcan abiertos simultáneamente.
