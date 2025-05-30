import os
import shutil
import random
from PIL import Image
import numpy as np

# Ayarlar
dataset_path = "/content/drive/MyDrive/Colab Notebooks/dental_images"
patch_size = 384
positive_threshold = 0.01  # %2 eşik (maskede pozitif piksellerin oranı)

# Orijinal görüntü ve maske klasörleri
image_path = os.path.join(dataset_path, "images")
mask_path = os.path.join(dataset_path, "labels_clean")

# Yeni train/val klasörleri
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")

train_image_path = os.path.join(train_path, "images")
train_mask_path = os.path.join(train_path, "labels_clean")
val_image_path = os.path.join(val_path, "images")
val_mask_path = os.path.join(val_path, "labels_clean")

# Klasörleri oluştur
os.makedirs(train_image_path, exist_ok=True)
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(val_image_path, exist_ok=True)
os.makedirs(val_mask_path, exist_ok=True)

# Tüm görüntü ve maskeleri listele (593 tane işaretlenmiş veri varsayılıyor)
image_files = sorted(os.listdir(image_path))
mask_files = sorted(os.listdir(mask_path))

# Rastgele shuffle yapıp %80 - %20 oranında bölme
combined = list(zip(image_files, mask_files))
random.shuffle(combined)
split_idx = int(0.8 * len(combined))
train_files = combined[:split_idx]
val_files = combined[split_idx:]

print(f"Toplam {len(image_files)} görüntü vardı.")
print(f"{len(train_files)} tanesi train için kullanılacak.")
print(f"{len(val_files)} tanesi validation için kullanılacak.")

# --- TRAIN VERİSİNİ İŞLEME (PATCHLEME) ---
def extract_valid_patches(img, mask, patch_size, pos_thresh):
    """
    img ve mask, PIL Image formatında.
    Non-overlapping patch'lere bölerek, mask'teki pozitif oranı kontrol eder.
    Eğer patch'teki pozitif oran pos_thresh'dan büyükse, patch'i kaydeder.
    """
    img_width, img_height = img.size  # (width, height)
    n_cols = img_width // patch_size
    n_rows = img_height // patch_size
    valid_patches = []

    for r in range(n_rows):
        for c in range(n_cols):
            left = c * patch_size
            upper = r * patch_size
            right = left + patch_size
            lower = upper + patch_size

            img_patch = img.crop((left, upper, right, lower))
            mask_patch = mask.crop((left, upper, right, lower))

            # Maske patch'ini numpy array'ye çevir, 0/255 değerlerini normalize et
            mask_np = np.array(mask_patch)
            if mask_np.max() > 1:
                mask_np = mask_np / 255.0
            ratio = np.sum(mask_np > 0) / (patch_size * patch_size)
            if ratio >= pos_thresh:
                valid_patches.append((img_patch, mask_patch, r, c))
    return valid_patches

train_patch_count = 0
# Her train görüntüsü için patch çıkarma ve geçerli olanları kaydetme
for img_file, mask_file in train_files:
    img_full_path = os.path.join(image_path, img_file)
    mask_full_path = os.path.join(mask_path, mask_file)

    try:
        # X‑ray görüntüsü olduğu için grayscale (L) formatında açıyoruz.
        img = Image.open(img_full_path).convert("L")
        mask = Image.open(mask_full_path).convert("L")
    except Exception as e:
        print(f"Error opening {img_file} veya {mask_file}: {e}")
        continue

    valid_patches = extract_valid_patches(img, mask, patch_size, positive_threshold)

    for patch, mask_patch, r, c in valid_patches:
        new_img_name = f"{os.path.splitext(img_file)[0]}_patch_r{r}_c{c}.png"
        new_mask_name = f"{os.path.splitext(mask_file)[0]}_patch_r{r}_c{c}.png"

        patch.save(os.path.join(train_image_path, new_img_name))
        mask_patch.save(os.path.join(train_mask_path, new_mask_name))
        train_patch_count += 1

print(f"Train için toplam {train_patch_count} adet patch kaydedildi.")

# --- VALIDATION VERİSİNİ İŞLEME (PATCH ÇIKARMA) ---
val_patch_count = 0
for img_file, mask_file in val_files:
    img_full_path = os.path.join(image_path, img_file)
    mask_full_path = os.path.join(mask_path, mask_file)

    try:
        img = Image.open(img_full_path).convert("L")
        mask = Image.open(mask_full_path).convert("L")
    except Exception as e:
        print(f"Error opening {img_file} veya {mask_file}: {e}")
        continue

    # Eğitimde kullanılan extract_valid_patches fonksiyonunu validation için de kullanıyoruz.
    valid_patches = extract_valid_patches(img, mask, patch_size, positive_threshold)

    for patch, mask_patch, r, c in valid_patches:
        new_img_name = f"{os.path.splitext(img_file)[0]}_patch_r{r}_c{c}.png"
        new_mask_name = f"{os.path.splitext(mask_file)[0]}_patch_r{r}_c{c}.png"

        patch.save(os.path.join(val_image_path, new_img_name))
        mask_patch.save(os.path.join(val_mask_path, new_mask_name))
        val_patch_count += 1

print(f"Validation için toplam {val_patch_count} adet patch kaydedildi.")
