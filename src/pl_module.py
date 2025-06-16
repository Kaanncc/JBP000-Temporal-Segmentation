import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from src.datasets import CoronaryAngiographyDataset
from src.models import make_cnn_model, make_transformer_model
from src.metrics import (
    dice_coefficient, iou_score,
    accuracy_score, precision_score, recall_score,
    temporal_stability, unsupervised_temporal_consistency, stability_rate
)
from src.losses import TverskyLoss 
from random import shuffle

class AngioDataModule(pl.LightningDataModule):
    def __init__(self,
                 img_dir: str,
                 msk_dir: str,
                 seq_len: int = 1,
                 batch_size: int = 1,
                 num_workers: int = 4):
        super().__init__()
        self.img_dir     = img_dir
        self.msk_dir     = msk_dir
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if self.seq_len == 1:
            full_ds = CoronaryAngiographyDataset(
                self.img_dir, self.msk_dir,
                sequence_length=1,
                transform=None
            )
            if not hasattr(self, 'train_idx'): 
                indices = list(range(len(full_ds)))
                shuffle(indices)
                n = len(indices)
                n_train = int(0.7 * n)
                n_val   = int(0.15 * n)
                self.train_idx = indices[:n_train]
                self.val_idx   = indices[n_train:n_train+n_val]
                self.test_idx  = indices[n_train+n_val:]

            self.train_ds = Subset(full_ds, self.train_idx)
            self.val_ds   = Subset(full_ds, self.val_idx)
            self.test_ds  = Subset(full_ds, self.test_idx)
            return

    
        if not hasattr(self, 'base_ds_seq'):
            self.base_ds_seq = CoronaryAngiographyDataset(
                self.img_dir, self.msk_dir,
                sequence_length=self.seq_len,
                step=1,
                transform=None
            )
            all_video_ids = list(self.base_ds_seq.sequences.keys())
    
            shuffle(all_video_ids)
            n = len(all_video_ids)
            n_train = int(0.7 * n)
            n_val   = int(0.15 * n)
            self.train_ids = all_video_ids[:n_train]
            self.val_ids   = all_video_ids[n_train:n_train+n_val]
            self.test_ids  = all_video_ids[n_train+n_val:]

        def filter_ds(video_ids, base_ds):
            keep = [i for i,(vid,_) in enumerate(base_ds.index_map) if vid in video_ids]
            return Subset(base_ds, keep)

        current_base_ds = CoronaryAngiographyDataset(
            self.img_dir, self.msk_dir,
            sequence_length=self.seq_len,
            step=1,
            transform=None
        )

        self.train_ds = filter_ds(self.train_ids, current_base_ds)
        self.val_ds   = filter_ds(self.val_ids, current_base_ds)
        self.test_ds  = filter_ds(self.test_ids, current_base_ds)


    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )


class SegmentationLitModel(pl.LightningModule):
    def __init__(self,
                 arch: str = "resnet50_unet",
                 seq_len: int = 1,
                 lr: float = 1e-4, 
                 tversky_alpha: float = 0.7, 
                 tversky_beta: float = 0.3, 
            
                 ):
        super().__init__()
        self.save_hyperparameters("arch", "seq_len", "lr", "tversky_alpha", "tversky_beta")

        if arch.endswith("_unet"):
            backbone = arch.split("_")[0]
            self.net = make_cnn_model(
                encoder_name=backbone,
                encoder_weights="imagenet",
                in_channels=3, 
                out_classes=1
            )
        else:
            self.net = make_transformer_model(
                pretrained_backbone=arch,
                in_channels=3,
                out_classes=1
            )
        
        self.loss_fn = TverskyLoss(
            alpha=self.hparams.tversky_alpha,
            beta=self.hparams.tversky_beta
        )
  

    def forward(self, x):
        if not x.ndim == 5:
             raise ValueError(f"Expected 5D input (B, T, C, H, W), but got {x.ndim}D")

        B, T, C, H, W = x.shape
        original_device = x.device 

        if T == 1:
            x = x.squeeze(1) 
            output = self.net(x)

            if hasattr(output, 'logits'):
                logits = output.logits 
            else:
                logits = output

            if logits.shape[-2:] != (H, W):
                 logits = torch.nn.functional.interpolate(
                     logits, size=(H, W), mode='bilinear', align_corners=False
                 )

            return logits.unsqueeze(1)

        elif T > 1:
            x = x.view(B * T, C, H, W)
            output = self.net(x) 

            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output

            _, _, logit_H, logit_W = logits.shape
            if (logit_H, logit_W) != (H, W):
                logits = torch.nn.functional.interpolate(
                    logits, size=(H, W), mode='bilinear', align_corners=False
                )

            C_out = logits.shape[1]
            return logits.view(B, T, C_out, H, W)

        else:
            raise ValueError(f"Invalid input time dimension T: {T}")


    def compute_metrics(self, logits, masks):
        with torch.no_grad(): 
            preds = torch.sigmoid(logits)
            flat_p = (preds > 0.5).int()
            flat_t = masks.int() 

            has_pred = flat_p.sum() > 0
            has_target = flat_t.sum() > 0

            dice = dice_coefficient(flat_p, flat_t)
            iou  = iou_score(flat_p, flat_t)
            acc  = accuracy_score(flat_p, flat_t)
            prec = precision_score(flat_p, flat_t) if has_pred or has_target else torch.tensor(1.0, device=logits.device)
            rec  = recall_score(flat_p, flat_t) if has_target else torch.tensor(1.0, device=logits.device)

        return dict(dice=dice, iou=iou,
                    accuracy=acc,
                    precision=prec,
                    recall=rec)

    def step(self, batch, stage):
        x, y = batch['image'], batch['mask'] 
        logits = self(x) 

        loss = self.loss_fn(logits, y.float())

        metrics = self.compute_metrics(logits, y)

        self.log(f"{stage}_loss", loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))
        for key, val in metrics.items():
            self.log(f"{stage}_{key}", val, on_step=(stage=='train'), on_epoch=True, prog_bar=True, logger=True, batch_size=x.size(0))

        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, "train")
        return loss 

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val") 

    def test_step(self, batch, batch_idx):
        _, logits, y = self.step(batch, "test") 

        if logits.ndim == 5 and logits.shape[1] > 1:  
            preds_binary = (torch.sigmoid(logits) > 0.5).int().squeeze(0).squeeze(1) 

            if preds_binary.shape[0] >= 2: 
                ts  = temporal_stability(preds_binary)
                self.log("test_temp_stability", ts, batch_size=1) 

            if preds_binary.shape[0] >= 3: 
                sr  = stability_rate(preds_binary)
                self.log("test_stability_rate", sr, batch_size=1)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',      
                factor=0.1,      
                patience=5,     
            ),
            "interval": "epoch", 
            "monitor": "val_loss" 
        }
    
        return [opt], [scheduler]
