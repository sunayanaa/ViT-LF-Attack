#filename spectral_energy_analysis.py
#This script generates the data for Table II: Perturbation Energy Distribution by Wavelet Band
#It is essential for reproducing Section ("The Energy Distribution Paradox")
!pip install torchattacks
#Title Analysis 5b: Granular Energy Distribution (LL, LH, HL, HH)
import torch
import timm
import torchattacks
import pywt
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Setup Data ---
img_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg", # Cat
    "http://images.cocodataset.org/val2017/000000037777.jpg", # Dog
    "http://images.cocodataset.org/val2017/000000078315.jpg", # Bird
    "http://images.cocodataset.org/val2017/000000252219.jpg", # Plane
]

def get_data_and_labels(model):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    images, labels = [], []
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print("⬇️  Downloading images...")
    for url in img_urls:
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad(): pred = model(img_t).argmax(1).item()
                images.append(img_t.squeeze(0))
                labels.append(torch.tensor(pred))
        except: continue
            
    if not images: raise RuntimeError("No images downloaded.")
    return torch.stack(images).to(DEVICE), torch.stack(labels).to(DEVICE)

# --- 2. Calculate Granular Energy ---
def calculate_band_energy(clean_tensor, adv_tensor):
    perturbation = (adv_tensor - clean_tensor).cpu().detach().numpy()
    
    # Store sums for each band
    energies = {'LL': 0.0, 'LH': 0.0, 'HL': 0.0, 'HH': 0.0}
    
    for i in range(perturbation.shape[0]):
        for c in range(3):
            delta = perturbation[i, c, :, :]
            coeffs = pywt.dwt2(delta, 'haar')
            LL, (LH, HL, HH) = coeffs
            
            energies['LL'] += np.sum(LL ** 2)
            energies['LH'] += np.sum(LH ** 2)
            energies['HL'] += np.sum(HL ** 2)
            energies['HH'] += np.sum(HH ** 2)
            
    total = sum(energies.values())
    return {k: (v / total * 100) for k, v in energies.items()}

# --- 3. Run ---
def run_analysis():
    print(f"\nAnalyzing Granular Energy (ViT-Tiny)...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE).eval()
    images, labels = get_data_and_labels(model)
    
    atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
    adv_images = atk(images, labels)
    
    results = calculate_band_energy(images, adv_images)
    
    print("\n" + "="*40)
    print(f"GRANULAR ENERGY DISTRIBUTION:")
    print("="*40)
    print(f"Approximation (LL): {results['LL']:.1f}%")
    print(f"Horizontal (LH):    {results['LH']:.1f}%")
    print(f"Vertical (HL):      {results['HL']:.1f}%")
    print(f"Diagonal (HH):      {results['HH']:.1f}%")
    print("="*40)

run_analysis()
