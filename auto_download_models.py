import shutil
import os
import subprocess

subprocess.run(['git', 'lfs', 'install'])

os.chdir("/workspace/ComfyUI/custom_nodes")

custom_extensions = [
    'https://github.com/ltdrdata/ComfyUI-Manager',
    'https://github.com/ltdrdata/ComfyUI-Impact-Pack.git',
    'https://github.com/ltdrdata/ComfyUI-Inspire-Pack.git',
    'https://github.com/comfyanonymous/ComfyUI_experiments.git',
    'https://github.com/Stability-AI/stability-ComfyUI-nodes.git',
    'https://github.com/Fannovel16/comfyui_controlnet_aux.git',
    'https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git',
    'https://github.com/jags111/efficiency-nodes-comfyui.git',
    'https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes.git',
    'https://github.com/WASasquatch/was-node-suite-comfyui.git',
    'https://github.com/WASasquatch/ComfyUI_Preset_Merger.git',
    'https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92.git',
    'https://github.com/LEv145/images-grid-comfy-plugin.git',
    'https://github.com/BadCafeCode/masquerade-nodes-comfyui.git',
    'https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git',
    'https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git',
    'https://github.com/RockOfFire/ComfyUI_Comfyroll_CustomNodes.git',
    'https://github.com/bmad4ever/ComfyUI-Bmad-DirtyUndoRedo.git',
    'https://github.com/FizzleDorf/ComfyUI_FizzNodes.git',
    'https://github.com/cubiq/ComfyUI_IPAdapter_plus.git',
    'https://github.com/evanspearman/ComfyMath.git',
    'https://github.com/civitai/civitai_comfy_nodes.git',
    'https://github.com/ArtVentureX/comfyui-animatediff.git',
    'https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git',
    'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git',
    'https://github.com/Gourieff/comfyui-reactor-node.git',
    'https://github.com/badjeff/comfyui_lora_tag_loader.git',
    'https://github.com/shingo1228/ComfyUI-SDXL-EmptyLatentImage.git',
    'https://github.com/marhensa/sdxl-recommended-res-calc.git',
    'https://github.com/skfoo/ComfyUI-Coziness.git',
    'https://github.com/picturesonpictures/comfy_PoP.git',
    'https://github.com/kinfolk0117/ComfyUI_GradientDeepShrink.git',
    'https://github.com/kijai/ComfyUI-SVD.git',
    'https://github.com/palant/image-resize-comfyui.git',
    'https://github.com/shadowcz007/comfyui-mixlab-nodes.git',
    'https://github.com/ceruleandeep/ComfyUI-LLaVA-Captioner.git',
    "https://github.com/Acly/comfyui-inpaint-nodes.git",
    "https://github.com/sipherxyz/comfyui-art-venture.git",
    "https://github.com/rgthree/rgthree-comfy.git",
    "https://github.com/storyicon/comfyui_segment_anything.git",
    "https://github.com/Nuked88/ComfyUI-N-Nodes.git",
    "https://github.com/gokayfem/ComfyUI_VLM_nodes.git",
    "https://github.com/chflame163/ComfyUI_LayerStyle.git",
    "https://github.com/digitaljohn/comfyui-propost.git",
    "https://github.com/cubiq/ComfyUI_essentials.git",
]

for repo_url in custom_extensions:
    subprocess.run(['git', 'clone', repo_url])


os.chdir("/workspace/ComfyUI")

os.chdir("/workspace/ComfyUI/models/checkpoints")

model_mapping = {
    'https://civitai.com/api/download/models/240840?type=Model&format=SafeTensor&size=full&fp=fp16': 'SDXL_JuggernautXL.safetensors',
    'https://civitai.com/api/download/models/348913?type=Model&format=SafeTensor&size=full&fp=fp16': 'SDXL_JuggernautXL_v9.safetensors',
    'https://civitai.com/api/download/models/344487?type=Model&format=SafeTensor&size=pruned&fp=fp16': 'SDXL_RealVisXL_V4.safetensors',
    'https://civitai.com/api/download/models/143906?type=Model&format=SafeTensor&size=pruned&fp=fp16': 'SD15_epiCRealism.safetensors',
    'https://civitai.com/api/download/models/130090?type=Model&format=SafeTensor&size=full&fp=fp16': 'SD15_RealisticVisionV51-inpainting.safetensors',
    "https://civitai.com/api/download/models/255474?type=Model&format=SafeTensor&size=pruned&fp=fp16": 'SDXL_TurboVisionXL.safetensors',
    "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors": "SD15_Realistic_Vision_V5.1.safetensors",
    "https://civitai.com/api/download/models/266533?type=Model&format=SafeTensor&size=pruned&fp=fp16": 'SDXL_RVXL2_turbo.safetensors',
    "https://civitai.com/api/download/models/141050?type=Model&format=SafeTensor&size=full&fp=fp16": 'architectureV70.safetensors',
}



for url, destination_filename in model_mapping.items():
    subprocess.run(['wget', '-c', url, '-q'])
    print(f"File downloaded successfully from {url}")
    shutil.move(url.split('/')[-1], destination_filename)
    print(f"File moved successfully to {destination_filename}")

print("Model download and rename done!")

os.chdir("/workspace/ComfyUI/models/loras")


loras_mapping = {
    "https://civitai.com/api/download/models/16576?type=Model&format=SafeTensor&size=full&fp=fp16": 'SD15_epi_noiseoffset.safetensors',
    "https://civitai.com/api/download/models/38113?type=Model&format=SafeTensor": 'SD15_Arhictecture_Illustrate.safetensors',
    "https://civitai.com/api/download/models/39140?type=Model&format=SafeTensor": 'SD15_Architecture_Sketch.safetensors',
    "https://civitai.com/api/download/models/159829?type=Model&format=SafeTensor": 'SD15_LAN_Architecture Design Input.safetensors',
    "https://civitai.com/api/download/models/40665?type=Model&format=SafeTensor": 'SD15_Architecture_Marker Sketch.safetensors',
    "https://civitai.com/api/download/models/141133?type=Model&format=SafeTensor": 'SDXL_Detail Enhancer.safetensors',
    "https://civitai.com/api/download/models/180569?type=Model&format=SafeTensor": 'SDXL_Film Photography Style.safetensors',
    "https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true": 'lcm-lora-sdxl.safetensors',
    "https://civitai.com/api/download/models/345252?type=Model&format=SafeTensor": 'SDXL_FormFinder-XL.safetensors',
    
}

for url, destination_filename in loras_mapping.items():
    subprocess.run(['wget', '-c', url, '-q'])
    print(f"File downloaded successfully from {url}")

    shutil.move(url.split('/')[-1], destination_filename)
    print(f"File moved successfully to {destination_filename}")

print("LoRA download and rename done!")


os.chdir("/workspace/ComfyUI/models/controlnet")

controlnet_models = [
    "https://huggingface.co/lllyasviel/ControlNet-v1-1",
    "https://huggingface.co/stabilityai/control-lora",
]
for repo_url in controlnet_models:
    subprocess.run(['git', 'clone', repo_url])

import subprocess

# SDXL_Canny_1.0_d.safetensors
url_canny = "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true"
subprocess.run(["wget", "-O", "SDXL_Canny_1.0_d.safetensors", url_canny])

# SDXL_Depth_1.0_d.safetensors
url_depth = "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors?download=true"
subprocess.run(["wget", "-O", "SDXL_Depth_1.0_d.safetensors", url_depth])

# SDXL_SoftEdge_sargezt.safetensors
url_softedge = "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/sargezt_xl_softedge.safetensors?download=true"
subprocess.run(["wget", "-O", "SDXL_SoftEdge_sargezt.safetensors", url_softedge])

# SDXL_OpenPose_thibaud.safetensors
url_openpose = "https://huggingface.co/lllyasviel/sd_control_collection/resolve/main/thibaud_xl_openpose.safetensors?download=true"
subprocess.run(["wget", "-O", "SDXL_OpenPose_thibaud.safetensors", url_openpose])


print("ControlNet 1.5 models downloaded!")
print("ControlNet SDXL models downloaded!")
print("T2I models downloaded!")
print ("Controlnet download done!")

os.mkdir("/workspace/ComfyUI/models/inpaint")
os.chdir("/workspace/ComfyUI/models/inpaint")

controlnet_models = [
    "https://huggingface.co/lllyasviel/fooocus_inpaint",
]
for repo_url in controlnet_models:
    subprocess.run(['git', 'clone', repo_url])

print("inpainting models downloaded!")

os.chdir("/workspace/ComfyUI/models/upscale_models")

upscale_models = [
    'https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth?download=true',
    'https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true',
    'https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x8.pth?download=true',
    "https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth?download=true",
]

for url in upscale_models:
    subprocess.run(['wget', '-c', url, '-q'])
    print(f"File downloaded successfully from {url}")
    
os.rename("RealESRGAN_x2.pth?download=true", "RealESRGAN_x2.pth")
print(f"RealESRGAN_x2 renamed")
os.rename("RealESRGAN_x4.pth?download=true", "RealESRGAN_x4.pth")
print(f"RealESRGAN_x4 renamed")
os.rename("RealESRGAN_x8.pth?download=true", "RealESRGAN_x8.pth")
print(f"RealESRGAN_x8 renamed")
os.rename("4x_NMKD-Siax_200k.pth?download=true", "4x_NMKD-Siax_200k.pth")
print(f"4x_NMKD-Siax_200k.pth renamed")
    
print ("upscale models download done!")

os.mkdir("/workspace/ComfyUI/models/ipadapter")
os.chdir("/workspace/ComfyUI/models/ipadapter")

ip_adapter = [
    "https://huggingface.co/h94/IP-Adapter",
    "https://huggingface.co/ostris/ip-composition-adapter",
]

for repo_url in ip_adapter:
    subprocess.run(['git', 'clone', repo_url])

print ("ip adapter models download done!")



os.chdir("/workspace/ComfyUI/models/clip_vision")

clip_vision = [
    'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors?download=true',
    "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.safetensors?download=true",
]

for url in clip_vision:
    subprocess.run(['wget', '-c', url, '-q'])
    print(f"File downloaded successfully from {url}")
    
os.rename("model.safetensors?download=true", "clip_vision_model.safetensors")
print(f"clip vision model renamed")
os.rename("open_clip_pytorch_model.safetensors?download=true", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors")
print(f"clip vision model renamed")


clip_vision_2 = [
    "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.safetensors?download=true",
]

for url in clip_vision_2:
    subprocess.run(['wget', '-c', url, '-q'])
    print(f"File downloaded successfully from {url}")
    
os.rename("open_clip_pytorch_model.safetensors?download=true", "CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors")
print(f"CLIP-ViT-bigG-14-laion2B-39B-b160k renamed")
print ("clip vision models download done!")


os.chdir("/workspace/ComfyUI/custom_nodes/ComfyUI-LLaVA-Captioner")
subprocess.run(['python', 'install.py'])

os.chdir("/workspace/ComfyUI/custom_nodes/ComfyUI-LLaVA-Captioner/models")
llava = [
    'https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K.gguf',
    "https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-mmproj-Q4_0.gguf",
]

for url in llava:
    subprocess.run(['wget', '-c', url, '-q'])
    print(f"File downloaded successfully from {url}")

    
print ("LlaVA models download done!")

print ("Installation is done, enjoy!")