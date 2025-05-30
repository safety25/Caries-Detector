import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Checkpoint dosyanızın yolu
ckpt_path = "/content/drive/MyDrive/Colab Notebooks/dental_images/UAMT10-epoch=188-val_mean_iou=0.6908-val_mean_dice=0.7931-val_mean_spe=0.9973-val_mean_sen=0.7798-val_mean_pre=0.8413.ckpt"

# Modeli yükleyin (modeliniz tek kanallı eğitildiği için in_channels=1 olmalı)
model = CariesSSLNet.load_from_checkpoint(ckpt_path)
model.eval()
model.cuda()

# Test görüntüsünün yolu (orijinal renkli görüntüyü kullanıyoruz, overlay için)
test_img_path = "/content/drive/MyDrive/Colab Notebooks/dental_images/val/images/155_patch_r1_c2.png"

# Orijinal görüntüyü BGR formatında okuyun ve yeniden boyutlandırın
orig_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
orig_img = cv2.resize(orig_img, (384, 384))

# Modelin eğitimi sırasında tek kanallı (grayscale) kullanıldıysa, test görüntüsünü grayscale'e çevirin
gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
gray_img = gray_img.astype(np.float32) / 255.0
img_tensor = torch.tensor(gray_img).unsqueeze(0).unsqueeze(0).cuda()  # (1, 1, 384, 384)

with torch.no_grad():
    output = model(img_tensor)
if isinstance(output, tuple):
    output = output[0]
prob_mask = torch.sigmoid(output)
binary_mask = (prob_mask > 0.5).squeeze().cpu().numpy().astype(np.uint8)  # (384,384)

# Overlay yapmak için:
alpha = 0.5  # Şeffaflık oranı

# Yeşil renk overlay oluşturmak için, orijinal görüntü ile aynı boyutta yeşil bir görüntü oluşturun.
green_overlay = np.full_like(orig_img, (0, 255, 0))  # BGR formatında yeşil

# Maske uygulanacak alanı boolean hale getirin
mask_bool = binary_mask.astype(bool)

# Yeni overlay görüntüsünü oluşturun:
overlay_img = orig_img.copy()
# Maske uygulanan piksellerde, orijinal ve yeşil overlay görüntülerini blend edin.
overlay_img[mask_bool] = cv2.addWeighted(orig_img[mask_bool], 1 - alpha, green_overlay[mask_bool], alpha, 0)

# Görüntüyü gösterin ve kaydedin
plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
plt.title("Superposition : masque vert")
plt.axis("off")
plt.show()

cv2.imwrite("overlay_pred_mask.png", overlay_img)
