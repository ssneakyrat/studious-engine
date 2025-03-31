import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import load_config
from dataset import MelDataset
from model import ConditionalVAE
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(config_path="config/default.yaml"):
    """Loads data, initializes model, and runs training loop."""

    # 1. Load Configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    # print(config) # Optional: Print config

    # 2. Setup Logging and Checkpointing
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    logger_tb = TensorBoardLogger(log_dir, name="ConditionalVAE")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger_tb.log_dir, "checkpoints"),
        filename='cvae-{epoch:02d}-{val/loss:.2f}',
        save_top_k=3,
        verbose=True,
        monitor='val/loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.001,
        patience=10, # Stop if val_loss doesn't improve for 10 epochs
        verbose=True,
        mode='min'
    )

    # 3. Load Data
    try:
        dataset = MelDataset(config['data']['h5_path'], config)
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}")
        return

    if len(dataset) == 0:
        logger.error("Dataset is empty. Please check the HDF5 file and preprocessing.")
        return

    # Split dataset into train and validation
    total_size = len(dataset)
    val_size = max(1, int(0.1 * total_size)) # Ensure at least 1 validation sample
    train_size = total_size - val_size

    if train_size <= 0:
        logger.warning(f"Dataset size ({total_size}) is very small. Using all data for training and validation.")
        train_dataset = dataset
        val_dataset = dataset # Use the same data if split is not possible
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        logger.info(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True # Improves GPU transfer speed if possible
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # 4. Initialize Model
    model = ConditionalVAE(config)

    # 5. Initialize Trainer
    trainer = pl.Trainer(
        logger=logger_tb,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=config['training']['epochs'],
        val_check_interval=config['training']['val_check_interval'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        log_every_n_steps=10,
        detect_anomaly=True, # Useful for debugging NaN/Inf issues
        gradient_clip_val=0.5
    )

    # 6. Start Training
    logger.info("Starting training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        logger.info("Training finished.")
        logger.info(f"Best model checkpoint saved at: {checkpoint_callback.best_model_path}")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == '__main__':
    # --- Argument Parsing (Optional but Recommended) ---
    # import argparse
    # parser = argparse.ArgumentParser(description="Train Conditional VAE Model")
    # parser.add_argument('--config', type=str, default="config/default.yaml",
    #                     help="Path to the configuration file.")
    # args = parser.parse_args()
    # train(config_path=args.config)
    # --- Or run with default ---
    train()