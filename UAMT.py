#UAMT
import random
import os
import numpy as np
import torch
import argparse
import albumentations as A
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pytorch_lightning import LightningModule, Trainer, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch, torchvision
from torchvision import transforms as T
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
import random
from PIL import Image
#from util.utils import mean_metric, DiceLoss, mse_loss, sigmoid_rampup, get_current_consistency_weight, sigmoid_mse_loss
#from evaluate.utils import recompone_overlap, metric_calculate
# from colab.dataset import TrainDataset, ValDataset, UnlabeledDataset
from medpy import metric

gpu_list = [0]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

parser = argparse.ArgumentParser(description='CariesNet')
parser.add_argument('--sigma', '-s', type=float, default=5, required=False)
args = parser.parse_args(args=[])

# Model tanımı: Unet bazlı
class Net(smp.Unet):
    def __init__(self, in_c: int = 1, out_c: int = 1):
        super().__init__(in_channels=in_c, classes=out_c)
    def forward(self, x):
        return super().forward(x)

# LightningModule: Yarı denetimli (SSL) eğitim için
class CariesSSLNet(LightningModule):
    def __init__(self, lr: float = 0.001, l_batch_size: int = 8, theta: float = 0.99, SSL: bool = True):
        super().__init__()
        self.semi_train = SSL
        self.learning_rate = lr
        self.p = theta
        self.max_epoch = 200
        self.l_batch_size = l_batch_size
        self.glob_step = 0

        # Teacher ve Student modelleri
        self.model_tea = Net()
        self.model_stu = Net()
        for para in self.model_stu.parameters():
            para.detach_()

        self.dice_loss = DiceLoss()
        self.bce_loss = F.binary_cross_entropy_with_logits
        self.mse_loss = sigmoid_mse_loss

        self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})

    def forward(self, l_x):
        return self.model_tea(l_x)

    def training_step(self, batch, batch_idx):
        self.glob_step += 1
        x, y = batch
        # İlk l_batch_size örnek labeled, geri kalanı unlabeled
        sup_data   = x[:self.l_batch_size]
        sup_label  = y[:self.l_batch_size]
        ul_data    = x[self.l_batch_size:]

        # Öğretmen modelinden tüm batch için çıktı al
        preds = self.model_tea(x)  # shape: [batch,1,H,W]

        # Labeled kısmın prediksiyonunu ayıkla
        pred_sup = preds[:self.l_batch_size]

        #   Consistency Loss (SSL)
        consistency_loss = 0.0
        if self.semi_train and ul_data.size(0) > 0:
            # Unlabeled veriye gürültü ekle
            noise = torch.clamp(torch.randn_like(ul_data) * 0.1, -0.2, 0.2)
            ul_noisy = ul_data + noise
            with torch.no_grad():
                ul_pred = self.model_stu(ul_noisy)
        else:
            consistency_loss = torch.tensor(0.0, device=x.device)

        #   Supervised Loss & Metrikler

        # Etiketli veri için kayıp hesaplaması
        acc, iou, dice, spe, sen = mean_metric(pred_sup, sup_label)
        self.log('train_mean_acc', acc, on_step=False, on_epoch=True)
        self.log('train_mean_iou', iou, on_step=False, on_epoch=True)
        self.log('train_mean_dice', dice, on_step=False, on_epoch=True)
        self.log('train_mean_spe', spe, on_step=False, on_epoch=True)
        self.log('train_mean_sen', sen, on_step=False, on_epoch=True)

        bce_loss = self.bce_loss(pred_sup, sup_label)
        dice_loss = self.dice_loss(pred_sup, sup_label)
        seg_loss = 0.5 * (bce_loss + dice_loss)

        consistency_weight = get_current_consistency_weight(self.current_epoch, 200)
        self.log('train_consistency_loss', consistency_loss, on_step=False, on_epoch=True)
        self.log('train_bce_loss', bce_loss, on_step=False, on_epoch=True)
        self.log('train_dice_loss', dice_loss, on_step=False, on_epoch=True)
        self.log('train_seg_loss', seg_loss, on_step=False, on_epoch=True)

        return seg_loss + consistency_weight * consistency_loss

    def validation_step(self, batch, batch_idx):
        if self.current_epoch > 0:
            self.eval()
            imgs, gt = batch
            # imgs = imgs.permute(1, 0, 2, 3)
            with torch.no_grad():
                outputs = self(imgs)
            pred = torch.sigmoid(outputs)
            # Eğer validation görüntüleriniz tam görüntü ise, reassemble yapmanıza gerek yok:
            pred_imgs = pred.squeeze().cpu().numpy()
            gt_imgs = gt.squeeze().cpu().numpy()

            # Metrikleri hesaplayın (örneğin, dice, iou vs.)
            dice_val = metric.binary.dc(pred_imgs > 0.5, gt_imgs > 0.5)
            jc = metric.binary.jc(pred_imgs > 0.5, gt_imgs > 0.5)
            sen = metric.binary.sensitivity(pred_imgs > 0.5, gt_imgs > 0.5)
            pre = metric.binary.precision(pred_imgs > 0.5, gt_imgs > 0.5)
            spe = metric.binary.specificity(pred_imgs > 0.5, gt_imgs > 0.5)
            self.eval_dict["iou"].append(jc)
            self.eval_dict["dice"].append(dice_val)
            self.eval_dict["pre"].append(pre)
            self.eval_dict["sen"].append(sen)
            self.eval_dict["spe"].append(spe)

    def on_validation_epoch_end(self):
        if self.current_epoch > 0 and len(self.eval_dict["iou"]) > 0:
            mean_iou = sum(self.eval_dict["iou"]) / len(self.eval_dict["iou"])
            mean_dice = sum(self.eval_dict["dice"]) / len(self.eval_dict["dice"])
            mean_spe = sum(self.eval_dict["spe"]) / len(self.eval_dict["spe"])
            mean_pre = sum(self.eval_dict["pre"]) / len(self.eval_dict["pre"])
            mean_sen = sum(self.eval_dict["sen"]) / len(self.eval_dict["sen"])
            self.log('val_mean_iou', mean_iou)
            self.log('val_mean_dice', mean_dice)
            self.log('val_mean_spe', mean_spe)
            self.log('val_mean_pre', mean_pre)
            self.log('val_mean_sen', mean_sen)
            self.eval_dict = dict({"acc": [], "iou": [], "dice": [], "pre": [], "spe": [], "sen": []})
        else:
            self.log('val_mean_iou', 0)
            self.log('val_mean_dice', 0)
            self.log('val_mean_spe', 0)
            self.log('val_mean_pre', 0)
            self.log('val_mean_sen', 0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        poly_lr = lambda epoch: (1 - float(epoch) / self.max_epoch) ** 0.9
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lr)
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx, unused: int = 0):
        alpha = min(1 - 1 / (self.glob_step + 1), self.p)
        for para1, para2 in zip(self.model_stu.parameters(), self.model_tea.parameters()):
            para1.data.copy_(alpha * para1.data + (1 - alpha) * para2.data)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_process(model, train_loaders, val_loader, max_epochs):
    tb_logger = pl_loggers.TensorBoardLogger('./Cariouslog/UAMT')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mean_dice',
        filename='UAMT10-{epoch:02d}-{val_mean_iou:.4f}-{val_mean_dice:.4f}-{val_mean_spe:.4f}-{val_mean_sen:.4f}-{val_mean_pre:.4f}',
        save_top_k=5,
        mode='max',
        save_weights_only=True
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=tb_logger,
        accelerator="gpu",
        devices=[0],
        precision=16,
        check_val_every_n_epoch=1,
        benchmark=True,
        callbacks=[lr_monitor, checkpoint_callback]
    )
    trainer.fit(model, train_loaders, val_loader)

def main():
    SSL_flag = True
    learning_rate = 1e-3
    theta = 0.99
    labeled_ratio = {"0.1": 265, "0.2": 530, "0.5": 1325}
    labeled_rate = "0.2"
    if SSL_flag:
        batch_size, l_batch_size = 8, 4
    else:
        batch_size, l_batch_size = 4, 4

    dataset_path = "/content/drive/MyDrive/Colab Notebooks/dental_images"

# Train, Validation ve Unlabeled yolları
    train_image_path = os.path.join(dataset_path, "train/images")
    train_mask_path = os.path.join(dataset_path, "train/labels_clean")
    val_image_path = os.path.join(dataset_path, "val/images")
    val_mask_path = os.path.join(dataset_path, "val/labels_clean")
    unlabeled_image_path = os.path.join(dataset_path, "unlabeled_images")


    # Liste oluşturuluyor
    train_image_list = sorted([os.path.join(train_image_path, file) for file in os.listdir(train_image_path)])
    train_label_list = sorted([os.path.join(train_mask_path, file) for file in os.listdir(train_mask_path)])
    val_image_list = sorted([os.path.join(val_image_path, file) for file in os.listdir(val_image_path)])
    val_label_list = sorted([os.path.join(val_mask_path, file) for file in os.listdir(val_mask_path)])
    ul_image_list = sorted([os.path.join(unlabeled_image_path, file) for file in os.listdir(unlabeled_image_path)])


    # TrainDataset: Hem etiketli hem etiketsiz verileri kabul eden yapı.
    #esize_size = (384, 384)
    train_data = TrainDataset(train_image_list, train_label_list, ul_image_list)
    val_data = ValDataset(val_image_list, val_label_list)

    from PIL import Image
    img = Image.open(train_image_list[0]).convert('L')
    print("Görüntü modu:", img.mode)  # Beklenen: 'L'
    print("NumPy array shape:", np.array(img).shape)  # Beklenen: (H, W)


    sample_img, sample_mask = train_data[0]
    print("TrainDataset örnek görüntü şekli:", sample_img.shape)  # Beklenen: (1, 384, 384)
    print("TrainDataset örnek maske şekli:", sample_mask.shape)    # Beklenen: (1, 384, 384)

    sample_img_val, sample_mask_val = val_data[0]
    print("ValDataset örnek görüntü şekli:", sample_img_val.shape)  # Beklenen: (1, 384, 384)
    print("ValDataset örnek maske şekli:", sample_mask_val.shape)    # Beklenen: (1, 384, 384)

    # Labeled verilerin indeksi listenin başında olduğu varsayılıyor.
    all_idxs = list(range(len(train_image_list) + len(ul_image_list)))
    labeled_len = labeled_ratio[labeled_rate]
    labeled_idxs = all_idxs[:labeled_len]
    unlabeled_idxs = list(set(all_idxs) - set(labeled_idxs))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, l_batch_size)

    train_loader = DataLoader(train_data, batch_sampler=batch_sampler, num_workers=10, pin_memory=True)
    # Validation için batch_size 1 kullanılıyor.
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=10, pin_memory=True)

    model = CariesSSLNet(learning_rate, l_batch_size, theta, SSL_flag)
    max_epoch = 200
    train_process(model, train_loader, val_loader, max_epoch)


if __name__ == '__main__':
    seed_everything()
    main()
