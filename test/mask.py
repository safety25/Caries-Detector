import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Checkpoint dosyanızın yolu (örnek)
ckpt_path = "/content/drive/MyDrive/Colab Notebooks/dental_images/UAMT10-epoch=188-val_mean_iou=0.6908-val_mean_dice=0.7931-val_mean_spe=0.9973-val_mean_sen=0.7798-val_mean_pre=0.8413.ckpt"

model = CariesSSLNet.load_from_checkpoint(ckpt_path)
model.eval()
model.cuda()  

# Test görüntüsünün yolu (tek kanallı, grayscale görüntü)
test_img_path = "/content/drive/MyDrive/Colab Notebooks/dental_images/val/images/155_patch_r1_c2.png"

# Görüntüyü grayscale olarak oku
img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)  # Bu şekilde tek kanal elde edilir

# Yeniden boyutlandır (model eğitiminde kullanılan boyuta getirin)
img = cv2.resize(img, (384, 384))
img = img.astype(np.float32) / 255.0

# PyTorch tensöre dönüştür: (1, 1, 384, 384)
img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)
img_tensor = img_tensor.cuda()

with torch.no_grad():
    output = model(img_tensor)  # Modelin çıktısı tuple ise, ana maskeyi seçin
if isinstance(output, tuple):
    output = output[0]

# Çıktı raw logits olabilir; sigmoid uygulayın
prob_mask = torch.sigmoid(output)
prob_mask = prob_mask.squeeze().cpu().numpy()  # (384,384) boyutunda

# Eşik uygulayarak binary maske oluşturun
binary_mask = (prob_mask > 0.5).astype(np.uint8)

# Sonucu görselleştirin
plt.imshow(binary_mask, cmap='gray')
plt.title("Masque de segmentation")
plt.show()

# İsterseniz dosyaya kaydedin
cv2.imwrite("pred_mask.png", binary_mask * 255)
