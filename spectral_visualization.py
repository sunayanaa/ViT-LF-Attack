#filename spectral_visualization.py
# Generates Figure 2: Visual decomposition of adversarial perturbation into Wavelet bands
!pip install torchattacks
import torch
import timm
import torchattacks
import pywt
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'vit_tiny_patch16_224'
SAVE_NAME = "fig2_spectral_bands.png"

# Robust Image Source (The "Cat" from COCO)
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

def get_data(model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print(f"⬇️  Downloading test image...")
    try:
        r = requests.get(IMG_URL, headers=headers, timeout=10)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        # Get label
        with torch.no_grad(): 
            label = model(img_t).argmax(1)
        return img_t, label
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def plot_spectral_decomposition(clean, adv, save_path):
    # 1. Compute Perturbation
    perturbation = (adv - clean).squeeze(0).cpu().detach().numpy()
    
    # We visualize the mean perturbation across RGB channels for clarity
    delta_gray = np.mean(perturbation, axis=0) 
    
    # 2. Wavelet Decomposition
    coeffs = pywt.dwt2(delta_gray, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # 3. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Row 1: The Attack Context
    # Clean Image
    clean_np = clean.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(clean_np)
    axes[0, 0].set_title("Clean Image")
    axes[0, 0].axis('off')
    
    # Total Perturbation (The Noise)
    # Scale for visibility (x50)
    axes[0, 1].imshow(delta_gray, cmap='seismic', vmin=-0.1, vmax=0.1)
    axes[0, 1].set_title("Total Perturbation (x50 Gain)")
    axes[0, 1].axis('off')

    # LL Band (The "Shape" of the noise)
    axes[0, 2].imshow(LL, cmap='seismic', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title("Low-Frequency (LL)\n[Structure/Shape]")
    axes[0, 2].axis('off')
    
    # Row 2: The "Inert" High Frequencies
    # LH
    axes[1, 0].imshow(LH, cmap='seismic', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title("Horizontal (LH)\n[Texture/Noise]")
    axes[1, 0].axis('off')
    
    # HL
    axes[1, 1].imshow(HL, cmap='seismic', vmin=-0.1, vmax=0.1)
    axes[1, 1].set_title("Vertical (HL)\n[Texture/Noise]")
    axes[1, 1].axis('off')
    
    # HH
    axes[1, 2].imshow(HH, cmap='seismic', vmin=-0.1, vmax=0.1)
    axes[1, 2].set_title("Diagonal (HH)\n[Texture/Noise]")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to {save_path}")
    plt.show()

def run_vis():
    print(f"🚀 Generating Spectral Visualization...")
    model = timm.create_model(MODEL_NAME, pretrained=True).to(DEVICE).eval()
    
    img, label = get_data(model)
    if img is None: return

    # Generate Attack
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
    adv_img = atk(img, label)
    
    plot_spectral_decomposition(img, adv_img, SAVE_NAME)

if __name__ == "__main__":
    run_vis()