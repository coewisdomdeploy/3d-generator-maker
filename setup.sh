#!/usr/bin/env bash
# ==============================================================================
# 3D GENERATOR - SETUP (Tencent Hunyuan3D-2 para GPU 16GB VRAM)
# Optimizador y Exportador de Modelos .GLB para Kaboomi Party (Linux/WSL)
# ==============================================================================

set -e

echo "=============================================================================="
echo "      3D GENERATOR - SETUP (Tencent Hunyuan3D-2 en Linux/WSL)"
echo "=============================================================================="

# 1. Comprobación de Python
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "[ERROR] Python 3 no está instalado. Ejecuta: sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    exit 1
fi

echo "[1/6] Python detectado: $($PY_CMD --version)"

# 2. Configuración de Cache Hugging Face
if [ -z "$HF_HOME" ]; then
    export HF_HOME="$HOME/.cache/huggingface"
fi
mkdir -p "$HF_HOME"

cat <<EOF > .env
HF_HOME=$HF_HOME
KABOOMI_DIR=../Games/kaboomi-party
LOW_VRAM=True
DIT_SUBFOLDER=hunyuan3d-dit-v2-0-turbo
PAINT_SUBFOLDER=hunyuan3d-paint-v2-0
EOF
echo "[2/6] Configuración de cache guardada en .env ($HF_HOME)"

# 3. Creación y activación de entorno virtual
echo "[3/6] Configurando entorno virtual (.venv)..."
if [ ! -d ".venv" ]; then
    $PY_CMD -m venv .venv
fi
source .venv/bin/activate

# 4. Actualización de pip y PyTorch con CUDA
echo "[4/6] Instalando PyTorch con aceleración CUDA 12.4..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Instalación de requerimientos
echo "[5/6] Instalando dependencias de Hunyuan3D-2..."
pip install -r requirements.txt

# Instalar hy3dgen si es necesario
if ! python -c "import hy3dgen" &>/dev/null; then
    echo "Instalando hy3dgen desde Tencent-Hunyuan/Hunyuan3D-2..."
    pip install git+https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
fi

# 6. Verificación de GPU
echo "[6/6] Verificando aceleración por GPU..."
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()} | Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Crear scripts ejecutables auxiliares
cat <<'EOF' > run_webui.sh
#!/usr/bin/env bash
source .venv/bin/activate
python webui.py "$@"
EOF
chmod +x run_webui.sh

cat <<'EOF' > run_cli.sh
#!/usr/bin/env bash
source .venv/bin/activate
python img2glb.py "$@"
EOF
chmod +x run_cli.sh

echo "=============================================================================="
echo " ¡INSTALACIÓN COMPLETADA! Inicia con ./run_webui.sh o ./run_cli.sh"
echo "=============================================================================="
