import os
import re
from collections import defaultdict
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CoronaryAngiographyDataset(Dataset):
    """
    PyTorch Dataset for Coronary Angiography that supports:
      - sequence_length=1: single-frame samples (for CNNs)
      - sequence_length>1: sliding windows of frames (for Transformers or temporal metrics)

    Expects filenames like:
      p<patient>_v<video>_<frame_number>.png  (e.g. p42_v10_00046.png)
      or with an optional 'frame' token before the number.
    """
    def __init__(self,
                 image_dir: str,
                 mask_dir: str,
                 sequence_length: int = 1,
                 step: int = 1,
                 transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.seq_len   = sequence_length
        self.step      = step

        all_files = sorted([
            fn for fn in os.listdir(self.image_dir)
            if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        pattern = re.compile(
            r"^p(?P<p>\d+)_v(?P<v>\d+)(?:_frame)?[_\-]?(?P<f>\d+)\.(?:png|jpg|jpeg)$",
            re.IGNORECASE
        )
        groups = defaultdict(list)
        for fn in all_files:
            m = pattern.match(fn)
            if m:
                vid = f"p{m.group('p')}_v{m.group('v')}"
                frame_idx = int(m.group('f'))
                groups[vid].append((frame_idx, fn))

        if self.seq_len > 1 and groups:
            self.sequences = {}
            for vid, frames in groups.items():
                frames.sort(key=lambda x: x[0])
                self.sequences[vid] = [fn for _, fn in frames]
        else:
            self.sequences = {'all': all_files}

        self.index_map = []
        for vid, fn_list in self.sequences.items():
            n = len(fn_list)
            if self.seq_len == 1:
                for fn in fn_list:
                    self.index_map.append((vid, fn))
            else:
                for start in range(0, n - self.seq_len + 1, self.step):
                    self.index_map.append((vid, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        vid, sub = self.index_map[idx]
        if self.seq_len == 1:
            filenames = [sub]
        else:
            filenames = self.sequences[vid][sub:sub + self.seq_len]

        images, masks = [], []
        for fn in filenames:
            img = np.array(
                Image.open(os.path.join(self.image_dir, fn)).convert("RGB")
            )
            msk = np.array(
                Image.open(os.path.join(self.mask_dir, fn)).convert("L")
            )
            if self.transform:
                aug = self.transform(image=img, mask=msk)
                img, msk = aug['image'], aug['mask']

            img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)
            msk_t = torch.from_numpy(msk).unsqueeze(0).float().div(255.0)
            msk_t = (msk_t > 0.5).float()

            images.append(img_t)
            masks.append(msk_t)

        images = torch.stack(images, dim=0)  
        masks  = torch.stack(masks,  dim=0)  

        return {'image': images, 'mask': masks}
