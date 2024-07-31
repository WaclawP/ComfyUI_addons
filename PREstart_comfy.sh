#!/bin/bash

apt update
apt install psmisc
fuser -k 3002/tcp


# Aktywacja wirtualnego środowiska
source /workspace/ComfyUI/venv/bin/activate

# Przejście do katalogu projektu
cd /workspace/ComfyUI

pip install torch torchvision torchaudio

# Uruchomienie skryptu Pythona
python main.py --listen 0.0.0.0 --port 3002
