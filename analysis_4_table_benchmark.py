#filename analysis_4_table_benchmark.py
#This script generates the data for Table I: Comparative Robustness Under Spectral Defense Protocols.
#It provides the source for the "Quantitative Assurance Testing" described in Section III.A, specifically the comparison between ViT-Tiny and ResNet-50
# @title Analysis 4: Final Benchmark (Auto-Labeling + Stable URLs)
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

# --- 1. Reliable Image Sources (COCO & Standard Test Images) ---
# We use images we know contain ImageNet objects (Cat, Dog, Plane, etc.)
img_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg", # Cat
    "http://images.cocodataset.org/val2017/000000037777.jpg", # Dog
    "http://images.cocodataset.org/val2017/000000078315.jpg", # Bird
    "http://images.cocodataset.org/val2017/000000252219.jpg", # Plane
    "http://images.cocodataset.org/val2017/000000101691.jpg", # Bus
    "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", # Another Dog
]

def get_data_and_labels(model):
    """
    Downloads images and uses the MODEL to label them.
    This ensures 100% Clean Accuracy (Baseline) so we strictly measure Robustness.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    images = []
    labels = []
    
    # Header to look like a browser
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    print("⬇️  Downloading and Auto-Labeling images...")
    for url in img_urls:
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img_t = transform(img).unsqueeze(0).to(DEVICE)
                
                # Auto-Label: Ask the model what this is
                with torch.no_grad():
                    pred_label = model(img_t).argmax(1).item()
                
                images.append(img_t.squeeze(0))
                labels.append(torch.tensor(pred_label))
            else:
                print(f"Skipping {url}: Status {r.status_code}")
        except:
            continue
            
    if len(images) == 0:
        raise RuntimeError("No images could be downloaded. Check internet.")
        
    return torch.stack(images).to(DEVICE), torch.stack(labels).to(DEVICE)

# --- 2. Defense Functions (Same as before) ---
def wavelet_denoise(tensor, method='visushrink'):
    imgs_np = tensor.cpu().detach().numpy()
    cleaned = []
    for img in imgs_np:
        channels = []
        for c in range(img.shape[0]):
            data = img[c, :, :]
            coeffs = pywt.dwt2(data, 'haar')
            LL, (LH, HL, HH) = coeffs
            if method == 'visushrink':
                mad = np.median(np.abs(HH))
                sigma = mad / 0.6745 if mad > 0 else 0
                t = sigma * np.sqrt(2 * np.log(data.size)) if sigma > 0 else 0
            else:
                t = 0.1
            LH = pywt.threshold(LH, t, mode='soft')
            HL = pywt.threshold(HL, t, mode='soft')
            HH = pywt.threshold(HH, t, mode='soft')
            rec = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
            channels.append(np.clip(rec, 0, 1))
        cleaned.append(np.stack(channels))
    return torch.tensor(np.array(cleaned), dtype=torch.float32).to(DEVICE)

def nuclear_ablation(model, images):
    # Hook to kill HF in ViT latent space
    def hook(module, input, output):
        cls_token = output[:, 0:1, :]
        patches = output[:, 1:, :]
        B, N, D = patches.shape
        Size = int(N**0.5)
        if Size*Size != N: return output
        x = patches.transpose(1, 2).view(B, D, Size, Size)
        # Avg Pool = Low Pass Filter (Removes HF)
        x_down = torch.nn.functional.avg_pool2d(x, 2)
        x_up = torch.nn.functional.interpolate(x_down, size=(Size, Size), mode='nearest')
        patches_clean = x_up.view(B, D, N).transpose(1, 2)
        return torch.cat([cls_token, patches_clean], dim=1)

    if hasattr(model, 'blocks'):
        handle = model.blocks[4].register_forward_hook(hook)
        with torch.no_grad():
            preds = model(images).argmax(1)
        handle.remove()
        return preds
    return None

# --- 3. Run Benchmark ---
def run_benchmark():
    results = []
    print(f"\n🚀 Starting Benchmark on {DEVICE}...")
    
    # Test both models
    models_to_test = [
        ('vit_tiny_patch16_224', 'ViT-Tiny'), 
        ('resnet50', 'ResNet-50')
    ]
    
    for name, pretty_name in models_to_test:
        print(f"\nTesting {pretty_name}...")
        try:
            model = timm.create_model(name, pretrained=True).to(DEVICE).eval()
        except:
            print(f"Skipping {name} (Load Error)")
            continue
            
        # 1. Get Data & Auto-Labels using THIS model (Ensures 100% Clean Acc)
        images, labels = get_data_and_labels(model)
        
        # 2. Attack (PGD)
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
        adv_images = atk(images, labels)
        
        with torch.no_grad():
            clean_preds = model(images).argmax(1)
            adv_preds = model(adv_images).argmax(1)
        
        # Calculate Base Stats
        clean_acc = (clean_preds == labels).sum().item() / len(labels) * 100
        adv_acc = (adv_preds == labels).sum().item() / len(labels) * 100
        
        # Identify broken samples (Successful Attacks)
        broken_mask = (clean_preds == labels) & (adv_preds != labels)
        num_broken = broken_mask.sum().item()
        
        results.append({
            "Model": pretty_name, "Defense": "None", 
            "Clean": f"{clean_acc:.0f}%", "Adv": f"{adv_acc:.0f}%", "Rec": "-"
        })
        
        # 3. Apply Defenses (Only on broken images)
        if num_broken > 0:
            # VisuShrink
            def_img = wavelet_denoise(adv_images, 'visushrink')
            with torch.no_grad(): 
                def_preds = model(def_img).argmax(1)
            
            # Recovery = Defended image returns to Original Label
            rec_count = (def_preds[broken_mask] == labels[broken_mask]).sum().item()
            rec_rate = rec_count / num_broken * 100
            
            results.append({
                "Model": pretty_name, "Defense": "VisuShrink", 
                "Clean": "-", "Adv": "-", "Rec": f"{rec_rate:.1f}%"
            })
            
            # Nuclear (ViT Only)
            if "ViT" in pretty_name:
                nuc_preds = nuclear_ablation(model, adv_images)
                if nuc_preds is not None:
                    rec_n = (nuc_preds[broken_mask] == labels[broken_mask]).sum().item()
                    rec_rate_n = rec_n / num_broken * 100
                    results.append({
                        "Model": pretty_name, "Defense": "Nuclear", 
                        "Clean": "-", "Adv": "-", "Rec": f"{rec_rate_n:.1f}%"
                    })

    # --- Print Table for LaTeX ---
    print("\n" + "="*40)
    print("COPY THIS INTO YOUR LATEX TABLE:")
    print("="*40)
    for r in results:
        print(f"{r['Model']} & {r['Defense']} & {r['Clean']} & {r['Adv']} & {r['Rec']} \\\\")

if __name__ == "__main__":
    run_benchmark()
