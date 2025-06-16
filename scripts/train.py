
import torch.optim.lr_scheduler as _sch
if not hasattr(_sch, "LRScheduler"):
    _sch.LRScheduler = _sch._LRScheduler
import os
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse 
import sys
from src.pl_module import AngioDataModule, SegmentationLitModel

def main(args):
    pl.seed_everything(args.seed, workers=True)

    IMG_DIR       = args.img_dir
    MSK_DIR       = args.msk_dir
    BATCH_SIZE    = args.batch_size
    LR            = args.lr
    MAX_EPOCHS    = args.max_epochs
    GPUS          = args.gpus
    ARCH          = args.arch
    TRAIN_SEQ_LEN = 1  
    TEST_SEQ_LEN  = args.test_seq_len  
    TVERSKY_ALPHA = args.tversky_alpha
    TVERSKY_BETA  = args.tversky_beta

    print(f"--- Training Configuration ---")
    print(f"Architecture:   {ARCH}")
    print(f"Image Dir:      {IMG_DIR}")
    print(f"Mask Dir:       {MSK_DIR}")
    print(f"Train Seq Len:  {TRAIN_SEQ_LEN}")
    print(f"Test Seq Len:   {TEST_SEQ_LEN}")
    print(f"Batch Size:     {BATCH_SIZE}")
    print(f"Learning Rate:  {LR}")
    print(f"Tversky Alpha:  {TVERSKY_ALPHA}")
    print(f"Tversky Beta:   {TVERSKY_BETA}")
    print(f"Max Epochs:     {MAX_EPOCHS}")
    print(f"GPUs:           {GPUS}")
    print(f"Num Workers:    {args.num_workers}")
    print(f"Seed:           {args.seed}")
    print(f"----------------------------")


    dm_fit = AngioDataModule(
        img_dir     = IMG_DIR,
        msk_dir     = MSK_DIR,
        seq_len     = TRAIN_SEQ_LEN, 
        batch_size  = BATCH_SIZE,
        num_workers = args.num_workers,
    )

    model = SegmentationLitModel(
        arch    = ARCH,
        seq_len = TRAIN_SEQ_LEN, 
        lr      = LR,
        tversky_alpha = TVERSKY_ALPHA,
        tversky_beta  = TVERSKY_BETA
    )

    sanitized_arch = ARCH.replace('/', '_')  

    log_name = f"{sanitized_arch}_seq{TRAIN_SEQ_LEN}_bs{BATCH_SIZE}_lr{LR}_tv{TVERSKY_ALPHA}-{TVERSKY_BETA}"
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=log_name,
        version=args.exp_name
    )
    progbar_cb = TQDMProgressBar(refresh_rate=max(1, args.progress_bar_refresh_rate)) 
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.checkpoint_dir, log_name, args.exp_name),
        filename="{epoch:02d}-{val_loss:.4f}-{val_dice:.4f}", 
        monitor="val_dice", 
        mode="max",         
        save_top_k=args.save_top_k,       
        save_last=True     
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        logger               = tb_logger,
        accelerator          = "gpu" if GPUS > 0 else "cpu",
        devices              = GPUS if GPUS > 0 else None,
        max_epochs           = MAX_EPOCHS,
        limit_train_batches  = args.limit_train_batches if args.limit_train_batches else 1.0,
        limit_val_batches    = args.limit_val_batches if args.limit_val_batches else 1.0,
        callbacks            = [progbar_cb, ckpt_cb, lr_monitor],
        deterministic        = args.deterministic,
        num_sanity_val_steps = args.num_sanity_val_steps, 
        log_every_n_steps    = args.log_every_n_steps,   
    )

    print("--- Starting Training ---")
    trainer.fit(model, datamodule=dm_fit) 

    best_model_path = ckpt_cb.best_model_path
    last_model_path = ckpt_cb.last_model_path  

    model_to_test = None
    ckpt_path_for_test = None

    if args.max_epochs > 0:
        print(f"--- Best model saved at: {best_model_path} ---")
        if best_model_path:
            ckpt_path_for_test = best_model_path
            print(f"--- Loading best model from checkpoint for testing: {ckpt_path_for_test} ---")
            model_to_test = SegmentationLitModel.load_from_checkpoint(
                ckpt_path_for_test,
            )
        elif last_model_path:
            print(f"!!! No best model path found, testing with last model: {last_model_path} !!!")
            ckpt_path_for_test = last_model_path
            model_to_test = SegmentationLitModel.load_from_checkpoint(
                ckpt_path_for_test,
            )
        else:
            print("!!! No checkpoint found after training. Cannot test. !!!")

    elif args.max_epochs == 0:
        print("--- Max epochs is 0: Evaluating baseline model (untrained state) ---")
        model_to_test = model 
        ckpt_path_for_test = None  
        model_to_test.eval()

    else:  
        print("!!! Invalid max_epochs value. Cannot determine model for testing. !!!")

    if args.run_test and model_to_test is not None:
        print(f"\n--- Setting up Test DataModule with seq_len={TEST_SEQ_LEN} ---")
        dm_test = AngioDataModule(
            img_dir=IMG_DIR,
            msk_dir=MSK_DIR,
            seq_len=TEST_SEQ_LEN,  
            batch_size=1,  
            num_workers=args.num_workers,
        )

        if ckpt_path_for_test:
            print(f"--- Starting Testing using checkpoint: {ckpt_path_for_test} ---")
        else:
            print(f"--- Starting Testing using initial model state (no checkpoint loaded) ---")

        trainer.test(model=model_to_test, datamodule=dm_test, ckpt_path=None)

    elif args.run_test:
        print("\n--- Skipping Test Phase: No valid model state available for testing ---")
    else:
        print("\n--- Skipping Test Phase: --run_test not specified ---")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Coronary Angiography Segmentation Model")

    # Data Args
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--msk_dir', type=str, required=True, help='Directory containing mask images')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')

    # Model Args
    parser.add_argument('--arch', type=str, default='resnet50_unet', help='Model architecture (e.g., resnet50_unet, nvidia/mit-b2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--tversky_alpha', type=float, default=0.7, help='Alpha parameter for Tversky loss (weights FP)')
    parser.add_argument('--tversky_beta', type=float, default=0.3, help='Beta parameter for Tversky loss (weights FN)')

    # Training Args
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (0 for CPU)')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='Fraction (0.0-1.0) or number (>1) of training batches per epoch')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='Fraction (0.0-1.0) or number (>1) of validation batches per epoch')
    parser.add_argument('--num_sanity_val_steps', type=int, default=2, help='Number of validation batches to check before training starts')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true', help='Force deterministic operations (may slow down training)')
    parser.add_argument('--progress_bar_refresh_rate', type=int, default=10, help='How often to refresh the progress bar (steps)')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='How often to log metrics to logger (steps)')

    # Testing Args
    parser.add_argument('--run_test', action='store_true', help='Run testing phase after training')
    parser.add_argument('--test_seq_len', type=int, default=10, help='Sequence length for temporal evaluation during testing')

    # Logging & Checkpointing Args
    parser.add_argument('--log_dir', type=str, default='tb_logs', help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for model checkpoints')
    parser.add_argument('--exp_name', type=str, default='run_01', help='Experiment name (subfolder within log/checkpoint dirs)')
    parser.add_argument('--save_top_k', type=int, default=2, help='How many best checkpoints to save')


    args = parser.parse_args()

    if args.limit_train_batches <= 0: args.limit_train_batches = 1.0
    if args.limit_val_batches <= 0: args.limit_val_batches = 1.0
    if args.gpus > 0 and not torch.cuda.is_available():
        print("WARNING: GPUs requested but CUDA not available! Using CPU.")
        args.gpus = 0
    if not os.path.isdir(args.img_dir): raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.isdir(args.msk_dir): raise FileNotFoundError(f"Mask directory not found: {args.msk_dir}")

    main(args)
