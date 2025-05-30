import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import albumentations as A
# from evaluate.utils import get_data_test_overlap, rgb2gray, paint_border_overlap, extract_ordered_overlap, recompone_overlap, metric_calculate


class TrainDataset(Dataset):
    def __init__(self, image_list, label_list, ul_image_list = None, transize = 384):
        self.transize = transize
        self.data_list = []
        for image_path, label_path in zip(image_list, label_list):
            self.data_list.append([image_path, label_path])
        if ul_image_list is not None:
            for ul_image_path in ul_image_list:
                self.data_list.append([ul_image_path, None])
        self.img_transform = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5),
        ])
        self.both_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(45),
        ])
        self.resize_transform = T.Resize((self.transize, self.transize))
        self.nomalize_transform = T.ToTensor()
        print("data set num:", len(self.data_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        [image_path, label_path] = self.data_list[index]
        image = Image.open(image_path).convert("L")
        if label_path is not None:
            label = Image.open(label_path).convert("L")
        else:
            label = Image.fromarray(np.zeros((self.transize, self.transize)))
        seed = random.randint(0, 10000)
        torch.random.manual_seed(seed)
        image = self.both_transform(image)
        torch.random.manual_seed(seed)
        label = self.both_transform(label)
        image = self.img_transform(image)
        image = self.resize_transform(image)
        label = self.resize_transform(label)
        image = self.nomalize_transform(image)
        label = self.nomalize_transform(label)
        image = torch.tensor(np.array(image), dtype=torch.float32)
        label = torch.tensor(np.array(label), dtype=torch.float32)
        return image, label



class ValDataset(Dataset):
    def __init__(self, image_list, mask_list, transform=None, resize=(384,384)):
        """
        Args:
            image_list (list): Validation görüntü dosya yolları.
            mask_list (list): İlgili maske dosya yolları.
            transform (callable, optional): Uygulanacak dönüşümler.
            resize (tuple): (width, height) boyutuna yeniden boyutlandırma.
        """
        self.image_list = image_list
        self.mask_list = mask_list
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = self.mask_list[idx]
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        image = image.resize(self.resize, Image.BILINEAR)
        mask = mask.resize(self.resize, Image.NEAREST)
        image = np.array(image)
        mask = np.array(mask)
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.transpose(2, 0, 1)
        else:
            image = image[..., 0:1].transpose(2, 0, 1)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.transpose(2, 0, 1)
        else:
            mask = mask[..., 0:1].transpose(2, 0, 1)
        image = torch.tensor(image)
        mask = torch.tensor(mask)
        return image, mask

