cd /workspace

wget --quiet --show-progress https://github.com/Run-Pod/runpodctl/releases/download/v1.10.0/runpodctl-linux-amd -O runpodctl && chmod +x runpodctl && sudo cp runpodctl /usr/bin/runpodctl

git clone https://github.com/comfyanonymous/ComfyUI

cd /workspace/ComfyUI

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

pip uninstall torch torchvision torchaudio --yes

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install xformers==0.0.22

cd /workspace/ComfyUI/custom_nodes

git clone https://github.com/ltdrdata/ComfyUI-Manager

cd /workspace/ComfyUI

wget -c https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors -P ./models/vae/

mv /workspace/auto-models/sd_xl_base_1.0.safetensors /workspace/ComfyUI/models/checkpoints/

mv /workspace/auto-models/sd_xl_refiner_1.0.safetensors /workspace/ComfyUI/models/checkpoints/

wget -c https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors -P ./models/vae/

cd /workspace

rm -r auto-models

python /workspace/auto_download_models.py