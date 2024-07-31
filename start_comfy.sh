#!/bin/bash

# Aktywacja wirtualnego środowiska
source /workspace/ComfyUI/venv/bin/activate

# Przejście do katalogu projektu
cd /workspace/ComfyUI

# Uruchomienie skryptu Pythona
python main.py --listen 0.0.0.0 --port 3002
