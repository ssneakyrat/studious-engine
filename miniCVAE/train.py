import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging, GradientAccumulationScheduler

from utils import load_config
from dataset import MelDataset
from model import ConditionalVAE
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_callbacks(config):
    """Setup and return callbacks for training."""
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Checkpoint callback with improved settings
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename='cvae-{epoch:03d}-{val/loss:.4f}',
        save_top_k=5,  # Save more checkpoints for better monitoring
        verbose=True,
        monitor='val/loss',
        mode='min',
        save_last=True  # Always save latest model as well
    )
    
    # Early stopping with more patience
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        min_delta=0.001,
        patience=20,  # Increased patience for more training opportunity
        verbose=True,
        mode='min'
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Gradient accumulation scheduler - helps with small batch sizes
    # Start with 1 (no accumulation) and increase if needed based on batch size
    if config['training'].get('batch_size', 16) < 8:
        # For very small batches, accumulate more gradients
        grad_accum_schedule = {
            0: 1,  # Start with no accumulation
            10: 2,  # After 10 epochs, accumulate 2 batches
            20: 4   # After 20 epochs, accumulate 4 batches
        }
        grad_accum = GradientAccumulationScheduler(scheduling=grad_accum_schedule)
        return [checkpoint_callback, early_stop_callback, lr_monitor, grad_accum]
    
    # For reasonably sized batches, use Stochastic Weight Averaging
    # This helps generalization and can smooth out optimization
    swa = StochasticWeightAveraging(
        swa_epoch_start=0.75,  # Start in the last quarter of training
        swa_lrs=1e-5,  # SWA learning rate (lower than main lr)
        annealing_epochs=10
    )
    
    return [checkpoint_callback, early_stop_callback, lr_monitor, swa]

def train(config_path="config/default.yaml", resume_from=None):
    """Loads data, initializes model, and runs training loop with improved settings."""

    # 1. Load Configuration
    config = load_config(config_path)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Check for important config options and adjust as needed
    if config['model']['latent_dim'] > 64:
        logger.info(f"Reducing latent dimension from {config['model']['latent_dim']} to 64 for better stability")
        config['model']['latent_dim'] = 64
    
    if config['training']['annealing_epochs'] < 100:
        logger.info(f"Increasing annealing_epochs from {config['training']['annealing_epochs']} to 100")
        config['training']['annealing_epochs'] = 100
    
    if config['model']['mask_probability'] < 0.3:
        logger.info(f"Increasing mask_probability from {config['model']['mask_probability']} to 0.3")
        config['model']['mask_probability'] = 0.3

    # 2. Setup Logging
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    logger_tb = TensorBoardLogger(log_dir, name="ConditionalVAE")

    # Get callbacks including checkpointing, early stopping, and more
    callbacks = get_callbacks(config)

    # 3. Load Data with improved error handling and validation
    try:
        # Create dataset
        dataset = MelDataset(config['data']['h5_path'], config, augment=True)
        
        if len(dataset) == 0:
            logger.error("Dataset is empty. Please check the HDF5 file and preprocessing.")
            return
            
        # Check dataset consistency - this will print in MelDataset
        # Analyze a few samples to detect issues early
        logger.info("Validating sample data...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            # Check for NaN or extreme values
            if torch.isnan(sample).any():
                logger.error(f"NaN values detected in sample {i}. Check preprocessing.")
                return
                
            if torch.isinf(sample).any():
                logger.error(f"Infinite values detected in sample {i}. Check preprocessing.")
                return
        
        # Split dataset with stratification if possible (helps with imbalanced data)
        total_size = len(dataset)
        val_size = max(min(int(0.1 * total_size), 100), 1)  # 10% up to 100 samples, minimum 1
        train_size = total_size - val_size
        
        if train_size <= 0:
            logger.warning(f"Dataset size ({total_size}) is very small. Using all data for training and validation.")
            train_dataset = dataset
            val_dataset = dataset
        else:
            # Use random split with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                dataset, 
                [train_size, val_size],
                generator=generator
            )
            logger.info(f"Dataset split: Train={len(train_dataset)}, Validation={len(val_dataset)}")
        
        # Create data loaders with improved settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=True,
            drop_last=True,  # Avoid issues with batch norm on small final batch
            persistent_workers=config['training']['num_workers'] > 0  # Keep workers alive between epochs
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True,
            persistent_workers=config['training']['num_workers'] > 0
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize dataset: {e}", exc_info=True)
        return

    # 4. Initialize Model
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        model = ConditionalVAE.load_from_checkpoint(resume_from)
        # Make sure we're using the current config
        model.config = config
        model.hparams.update(config)
    else:
        model = ConditionalVAE(config)

    # 5. Initialize Trainer with better settings
    trainer = pl.Trainer(
        logger=logger_tb,
        callbacks=callbacks,
        max_epochs=config['training']['epochs'],
        val_check_interval=config['training']['val_check_interval'],
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        log_every_n_steps=10,
        detect_anomaly=True,
        gradient_clip_val=1.0,  # Prevent exploding gradients
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision for speedup on GPU
        accumulate_grad_batches=1,  # Use the scheduler instead for better control
        benchmark=True,  # Can speed up training if input sizes don't change
    )

    # 6. Start Training with error handling
    logger.info("Starting training...")
    try:
        trainer.fit(model, train_loader, val_loader)
        logger.info("Training finished.")
        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"Best model checkpoint saved at: {best_model_path}")
        else:
            logger.warning("No best model checkpoint was saved.")
    except KeyboardInterrupt:
        logger.info("Training interrupted. Last checkpoint is available.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        
    # Return the trainer object for potential further use
    return trainer

if __name__ == '__main__':
    # Add command line arguments for more flexibility
    parser = argparse.ArgumentParser(description="Train Conditional VAE Model")
    parser.add_argument('--config', type=str, default="config/default.yaml",
                        help="Path to the configuration file.")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume training from.")
    args = parser.parse_args()
    
    # Run training with parsed arguments
    train(config_path=args.config, resume_from=args.resume)