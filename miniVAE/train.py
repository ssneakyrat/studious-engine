import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import yaml
import os
from pathlib import Path

from model import SVSModelLightning
from data import SVSDataModule
from validate_module import SVSValidationModule

def main(args):
    # Ensure config and logs directories exist
    Path("config").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.h5_file:
        h5_file = config['dataset']['processed_file']
    else:
        h5_file=args.h5_file

    # Create data module
    data_module = SVSDataModule(
        h5_file,
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
        train_split=config['dataset']['train_split']
    )
    
    # Create model
    model = SVSModelLightning(config_path=args.config)
    
    # Set up logging and callbacks
    logger = TensorBoardLogger("logs", name="svs_model")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="logs/checkpoints",
        filename="svs_model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=config['training']['early_stop_patience'],
        mode="min"
    )
    
    # Validate model if specified
    if args.validate:
        print("Running model validation before training...")
        validator = SVSValidationModule(args.h5_file, args.config)
        validation_success = validator.run_all_validations()
        
        if not validation_success and not args.force:
            print("Validation failed. Use --force to train anyway.")
            return
    
    # Train model
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=config['training']['grad_clip'],
        log_every_n_steps=config['logging']['log_interval'],
        check_val_every_n_epoch=config['logging']['eval_interval']
    )
    
    trainer.fit(model, data_module)
    
    print(f"Training complete. Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVS Model")
    parser.add_argument("--h5_file", default="svs_dataset.h5", help="Path to dataset file")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    parser.add_argument("--validate", action="store_true", help="Run validation before training")
    parser.add_argument("--force", action="store_true", help="Force train even if validation fails")
    
    args = parser.parse_args()
    
    main(args)