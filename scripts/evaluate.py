
import torch.optim.lr_scheduler as _sch
if not hasattr(_sch, "LRScheduler"):
    _sch.LRScheduler = _sch._LRScheduler
import os
import torch
from pytorch_lightning import Trainer
from src.pl_module import AngioDataModule, SegmentationLitModel
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import argparse

DEFAULT_CHECKPOINT_PATH = "../checkpoints_debug/resnet50_unet_seq1_bs4_lr0.0001_tv0.7-0.3/initial_tversky_run/epoch=09-val_loss=0.2099-val_dice=0.7475.ckpt"

DEFAULT_IMG_DIR = "../CADICA_prepared/test/A"
DEFAULT_MSK_DIR = "../CADICA_prepared/test/B"

SEQ_LEN_EVAL = 10  
BATCH_SIZE_EVAL = 1 
GPUS_EVAL = 1
MAX_WINDOWS_VIS = 5  
OVERLAY_DIR = "overlays_output" 

ARCH = "resnet50_unet"
TRAIN_SEQ_LEN = 1 
TRAIN_LR = 1e-4
TRAIN_TVERSKY_ALPHA = 0.7
TRAIN_TVERSKY_BETA = 0.3

def get_device(gpus: int):
    """Gets the appropriate torch device."""
    if gpus > 0 and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        if gpus > 0:
            print("Warning: CUDA requested but not available, using CPU.")
        return torch.device("cpu")

def save_frame_grid(batch, preds, output_dir, batch_index, seq_len):
    """Saves a 2x2 grid visualization for each frame in a sequence."""
    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()

    B, T, _, H, W = preds.shape
    if T != seq_len:
        print(f"Warning: Expected sequence length {seq_len} but got {T} frames in batch {batch_index} for visualization.")

    for t in range(T):
        img = to_pil(batch['image'][0, t].cpu()) 

        gt_mask = batch['mask'][0, t, 0].cpu().numpy()
        gt_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0)) 
        dg = ImageDraw.Draw(gt_rgba)
        ys, xs = np.where(gt_mask > 0.5) 
        for y, x in zip(ys, xs):
            dg.point((x, y), fill=(255, 0, 0, 150)) 

        pr_mask = preds[0, t, 0].cpu().numpy() 
        pr_rgba = Image.new("RGBA", (W, H), (0, 0, 0, 0)) 
        dp = ImageDraw.Draw(pr_rgba)
        ys, xs = np.where(pr_mask > 0.5) 
        for y, x in zip(ys, xs):
            dp.point((x, y), fill=(0, 0, 255, 150)) 

        overlay_img = Image.alpha_composite(img.convert("RGBA"), pr_rgba)
        overlay_img = Image.alpha_composite(overlay_img, gt_rgba) 

        canvas = Image.new("RGB", (W * 2, H * 2), "white") 
        canvas.paste(img, (0, 0))

        gt_vis = Image.new("RGB", (W, H), "black")
        gt_vis.paste(gt_rgba, mask=gt_rgba) 
        canvas.paste(gt_vis, (W, 0))

        pr_vis = Image.new("RGB", (W, H), "black")
        pr_vis.paste(pr_rgba, mask=pr_rgba)
        canvas.paste(pr_vis, (0, H))

        canvas.paste(overlay_img.convert("RGB"), (W, H))

        fname = f"seq{batch_index:03d}_frame{t:03d}.png"
        canvas.save(os.path.join(output_dir, fname))

def main(args):
    print("--- Evaluation Script Started ---")
    print(f"Using checkpoint: {args.checkpoint_path}")
    print(f"Evaluating sequences of length: {SEQ_LEN_EVAL}")

    device = get_device(GPUS_EVAL)
    print(f"Using device: {device}")

    print("Setting up DataModule...")
    dm = AngioDataModule(
        img_dir=args.img_dir,
        msk_dir=args.msk_dir,
        seq_len=SEQ_LEN_EVAL, 
        batch_size=BATCH_SIZE_EVAL, 
        num_workers=args.num_workers 
    )

    print("Loading model from checkpoint...")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    model = SegmentationLitModel.load_from_checkpoint(
        args.checkpoint_path,
        arch=ARCH,
        seq_len=TRAIN_SEQ_LEN, 
        lr=TRAIN_LR,
        tversky_alpha=TRAIN_TVERSKY_ALPHA,
        tversky_beta=TRAIN_TVERSKY_BETA,
        map_location=device
    )
    model.eval() 
    model.to(device)
    print("Model loaded successfully.")

    print("Calculating test metrics (spatial and temporal)...")
    trainer = Trainer(
        accelerator="gpu" if device.type == "cuda" else device.type,
        devices=1 if device.type != "cpu" else None, 
        logger=False 
    )
    test_results = trainer.test(model, datamodule=dm, verbose=True)
    print("Test Metrics Results:")
    print(test_results)


    print(f"\nGenerating visualization overlays for {MAX_WINDOWS_VIS} sequences...")
    test_loader = dm.test_dataloader() 

    model.to(device) 

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= MAX_WINDOWS_VIS:
                break

            images = batch['image'].to(device)
            masks = batch['mask'].to(device) 

            logits = model(images) 
            preds = (torch.sigmoid(logits) > 0.5).int() 

            save_frame_grid(batch, preds, args.overlay_dir, batch_index=i, seq_len=SEQ_LEN_EVAL)
            print(f"  Saved overlays for sequence {i+1}/{MAX_WINDOWS_VIS}")

    print(f"\n✅ Visualization overlays saved to '{args.overlay_dir}/'")
    print("--- Evaluation Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Model and Generate Overlays")
    parser.add_argument('--checkpoint_path', type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help='Path to the model checkpoint (.ckpt) file')
    parser.add_argument('--img_dir', type=str, default=DEFAULT_IMG_DIR,
                        help='Directory containing input test images')
    parser.add_argument('--msk_dir', type=str, default=DEFAULT_MSK_DIR,
                        help='Directory containing ground truth test masks')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for the test dataloader')
    parser.add_argument('--overlay_dir', type=str, default=OVERLAY_DIR,
                        help='Directory to save the visualization overlays')

    args = parser.parse_args()
    main(args)
