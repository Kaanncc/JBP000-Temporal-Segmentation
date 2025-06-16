import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2 

from src.datasets import CoronaryAngiographyDataset

class CoronaryDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Coronary Angiography dataset.

    Args:
        data_dir (str): Path to the base directory ('CADICA_prepared').
        batch_size (int): Batch size for DataLoader. Defaults to 4.
        num_workers (int): Number of workers for DataLoader. Defaults to 2.
        img_size (int): Target image size (height and width). Defaults to 512.
    """
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 2, img_size: int = 512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.save_hyperparameters()

        self.train_transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),  
            A.Rotate(limit=10, p=0.7), 
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2(), 
            A.Lambda(mask=lambda mask, **kwargs: (mask / 255.0 > 0.5).float()) 
        ])

        self.val_test_transform = A.Compose([
            A.Resize(height=self.img_size, width=self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
            ToTensorV2(), 
            A.Lambda(mask=lambda mask, **kwargs: (mask / 255.0 > 0.5).float()) 
        ])


    def setup(self, stage: str | None = None):
        """Load data. Set variables: self.train_dataset, self.val_dataset, self.test_dataset."""
        if stage == 'fit' or stage is None:
            train_img_dir = os.path.join(self.data_dir, 'train', 'A')
            train_mask_dir = os.path.join(self.data_dir, 'train', 'B')
            self.train_dataset = CoronaryAngiographyDataset(train_img_dir, train_mask_dir, transform=self.train_transform)

            val_img_dir = os.path.join(self.data_dir, 'val', 'A')
            val_mask_dir = os.path.join(self.data_dir, 'val', 'B')
            self.val_dataset = CoronaryAngiographyDataset(val_img_dir, val_mask_dir, transform=self.val_test_transform)

        if stage == 'test' or stage is None:
            test_img_dir = os.path.join(self.data_dir, 'test', 'A')
            test_mask_dir = os.path.join(self.data_dir, 'test', 'B')
            self.test_dataset = CoronaryAngiographyDataset(test_img_dir, test_mask_dir, transform=self.val_test_transform)

        if stage == 'predict' or stage is None:
             predict_img_dir = os.path.join(self.data_dir, 'test', 'A') 
             predict_mask_dir = os.path.join(self.data_dir, 'test', 'B')
             self.predict_dataset = CoronaryAngiographyDataset(predict_img_dir, predict_mask_dir, transform=self.val_test_transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
