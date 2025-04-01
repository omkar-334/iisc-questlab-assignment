import numpy as np
import torch
import torch.nn as nn
from dataset import load_datasets
from model import DnCNN
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DnCNN().to(device)

model_path = "model_2053.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


train_d, test_d = load_datasets("CBSD68-dataset/CBSD68/noisy35", "CBSD68-dataset/CBSD68/original_png")


ssim_metric = StructuralSimilarityIndexMeasure().to("cuda" if torch.cuda.is_available() else "cpu")


def calculate_metrics(img1, img2):
    """Calculates PSNR and SSIM between two images.

    Args:
        img1 (torch.Tensor or np.ndarray): Noisy/Denoised image.
        img2 (torch.Tensor or np.ndarray): Clean image.

    Returns:
        dict: Dictionary with PSNR and SSIM values.
    """

    if isinstance(img1, np.ndarray):
        img1 = torch.tensor(img1).unsqueeze(0)
    if isinstance(img2, np.ndarray):
        img2 = torch.tensor(img2).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img1, img2 = img1.to(device), img2.to(device)

    psnr = peak_signal_noise_ratio(img2.cpu().numpy(), img1.cpu().numpy(), data_range=1.0)

    ssim = ssim_metric(img1, img2).item()

    return {"PSNR": psnr, "SSIM": ssim}


noisy_img, clean_img = train_d[0]
noisy_img, clean_img = noisy_img.to(device), clean_img.to(device)


noisy_img = noisy_img.unsqueeze(0)
clean_img = clean_img.unsqueeze(0)


with torch.no_grad():
    denoised_img = model(noisy_img)


denoised_img = denoised_img.squeeze(0)
noisy_img = noisy_img.squeeze(0)
clean_img = clean_img.squeeze(0)


noisy_np = noisy_img.cpu().numpy()
clean_np = clean_img.cpu().numpy()
denoised_np = denoised_img.cpu().numpy()


metrics_noisy = calculate_metrics(noisy_np, clean_np)
metrics_denoised = calculate_metrics(denoised_np, clean_np)

print(f"PSNR (Noisy vs Clean): {metrics_noisy['PSNR']:.2f}, SSIM: {metrics_noisy['SSIM']:.4f}")
print(f"PSNR (Denoised vs Clean): {metrics_denoised['PSNR']:.2f}, SSIM: {metrics_denoised['SSIM']:.4f}")
