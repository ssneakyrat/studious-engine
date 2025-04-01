import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf, DictConfig
import argparse

from dataset import SVSDataModule # Assuming dataset.py contains SVSDataModule
from model import FFTLightSVS     # Assuming model.py contains FFTLightSVS

def train(config: DictConfig):
    """Main training loop."""
    pl.seed_everything(config.seed, workers=True) # Set seed for reproducibility

    # --- Data Module ---
    data_module = SVSDataModule(config)
    # data_module.prepare_data() # Called by Trainer internally
    data_module.setup('fit')   # Setup explicitly for 'fit' stage

    # --- Model ---
    # Ensure vocab size is updated after datamodule setup
    config.model.vocab_size = len(data_module.phoneme_to_id)
    print(f"Model vocab size set to: {config.model.vocab_size}")
    model = FFTLightSVS(config)

    # --- Logger ---
    logger = TensorBoardLogger("logs", name="FFTLightSVS_PoC")

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename='fftlight-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,          # Save top 3 models based on validation loss
        monitor='val_loss',    # Monitor validation loss
        mode='min',            # Mode should be 'min' for loss
        save_last=True         # Save the last checkpoint as well
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=config.training.grad_clip_val,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1, # Use 1 GPU
        log_every_n_steps=config.validation.log_every_n_steps,
        val_check_interval=config.validation.val_check_interval,
        # Enable Automatic Mixed Precision (AMP) for potential speedup/VRAM savings
        precision=16 if torch.cuda.is_available() else 32 # Use 16-bit precision on GPU
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to the configuration file.')
    args = parser.parse_args()

    # Load YAML configuration
    try:
        conf = OmegaConf.load(args.config)
        print("Configuration loaded successfully:")
        print(OmegaConf.to_yaml(conf))
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading or parsing configuration file: {e}")
        exit(1)

    train(conf)
    print("Training finished.")