#filename analysis_2_latent_lowpass.py
#This script implements the "Nuclear" latent ablation mechanism.
#Title The "Nuclear Option" (CLS Cleaning + Low-Pass Only)
import torch
import torch.nn as nn
import timm
import torchattacks
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

# 1. Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(DEVICE)
model.eval()

# 2. Wavelet Helpers (Aggressive)
def haar_dwt_2d(x):
    # Standard Haar Decomposition
    x00 = x[..., 0::2, 0::2]; x01 = x[..., 0::2, 1::2]
    x10 = x[..., 1::2, 0::2]; x11 = x[..., 1::2, 1::2]
    LL = (x00 + x01 + x10 + x11) / 4.0
    return LL # We strictly return ONLY the Low-Frequency approximation

def haar_idwt_2d_LL_only(LL):
    # "Blur" reconstruction: repeat the LL pixels 2x2
    # This effectively deletes 100% of high-frequency details
    return torch.repeat_interleave(torch.repeat_interleave(LL, 2, dim=-1), 2, dim=-2)

# 3. The "Nuclear" Hook
def nuclear_hook(module, input, output):
    # A. Clean CLS Token (The "Driver")
    # PGD often inflates the norm of the CLS token. We shrink it towards the mean.
    cls_token = output[:, 0:1, :]
    cls_mean = cls_token.mean(dim=-1, keepdim=True)
    # Shrinkage factor 0.9: Pull outliers 10% closer to the mean
    cls_clean = (cls_token - cls_mean) * 0.9 + cls_mean 

    # B. Clean Spatial Patches (The "Windows")
    patches = output[:, 1:, :]
    B, N, D = patches.shape
    Size = int(N**0.5)
    
    # Reshape to 2D
    x = patches.transpose(1, 2).view(B, D, Size, Size)
    
    # --- NUCLEAR FILTER ---
    # Keep ONLY LL band. Kill ALL Details.
    LL = haar_dwt_2d(x)
    x_clean = haar_idwt_2d_LL_only(LL)
    # ----------------------
    
    patches_clean = x_clean.view(B, D, N).transpose(1, 2)
    
    # Reassemble
    return torch.cat([cls_clean, patches_clean], dim=1)

# 4. Generate Attack Data
print("⏳ Generating Attack Data (PGD-10)...")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# Using 32 images for statistical significance
dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
loader = DataLoader(Subset(dataset, range(32)), batch_size=32, shuffle=False)
images, _ = next(iter(loader))
images = images.to(DEVICE)

# Baseline
with torch.no_grad():
    clean_preds = model(images).argmax(1)

# Attack
atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
adv_images = atk(images, clean_preds)

with torch.no_grad():
    adv_preds = model(adv_images).argmax(1)

broken_mask = (clean_preds != adv_preds)
num_broken = broken_mask.sum().item()
print(f"Attack Broken: {num_broken}/32 images")

if num_broken == 0:
    print("Attack failed! We cannot test defense.")
else:
    # 5. Run Nuclear Test across Layers
    LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    print("\nNUCLEAR TEST RESULTS (Keep LL Only + Clean CLS)")
    print(f"{'Layer':<6} | {'Recovery Rate':<15}")
    print("-" * 30)

    for layer_idx in LAYERS:
        # Register Hook
        handle = model.blocks[layer_idx].register_forward_hook(nuclear_hook)
        
        try:
            with torch.no_grad():
                def_preds = model(adv_images).argmax(1)
            
            recovered = (def_preds[broken_mask] == clean_preds[broken_mask]).sum().item()
            rate = 100. * recovered / num_broken
            
            marker = "^" if rate > 0 else " "
            print(f"Blk {layer_idx:<2} | {rate:.1f}% {marker}")
            
        finally:
            handle.remove()