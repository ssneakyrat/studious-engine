import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import yaml
import argparse
from data import SVSDataModule
from model import FFTLightSVS
import os
from pathlib import Path

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Initialize DataModule ---
    data_module = SVSDataModule(config_path=args.config)
    
    # --- Setup initial preparation ---
    # This will ensure the phone map is created before model initialization
    data_module.prepare_data()
    data_module.setup('fit')
    
    # --- Initialize Model ---
    # Use the same updated config from the data module to avoid conflicts
    model = FFTLightSVS(config_path=args.config)
    
    # Make sure model's config reflects latest phoneme_vocab_size from data_module
    model.hparams.data['phoneme_vocab_size'] = data_module.config['data']['phoneme_vocab_size']
    
    # Print phoneme vocabulary size
    print(f"Training with phoneme vocabulary size: {model.hparams.data['phoneme_vocab_size']}")

    # --- Setup Logging ---
    log_dir = "logs/"
    experiment_name = Path(args.config).stem # Use config name as experiment name
    logger = TensorBoardLogger(save_dir=log_dir, name=experiment_name)

    # --- Setup Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3, # Save top 3 models based on validation loss
        save_last=True # Also save the last checkpoint
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # --- Initialize Trainer ---
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=config['training']['max_epochs'],
        gradient_clip_val=config['training']['grad_clip_val'],
        accelerator="auto", # Automatically selects GPU if available
        devices="auto",
        # Use '16-mixed' for Automatic Mixed Precision (AMP) - helps with VRAM on 3060
        precision="16-mixed",
        val_check_interval=config['training']['val_check_interval'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        # Enable deterministic mode for reproducibility if needed, might slow down
        # deterministic=True,
    )

    # --- Start Training ---
    print("Starting training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from)
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FFT-Light SVS Model")
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume training from.')
    args = parser.parse_args()
    main(args)