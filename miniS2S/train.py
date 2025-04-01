import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import SingingVoiceDataModule
from model import SingingVoiceSynthesisModel

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a singing voice synthesis model")
    
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--name", type=str, default="singing_voice_synthesis",
                        help="Name of the experiment for logging")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (fast dev run)")
    
    return parser.parse_args()

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Store config path for data module
    config["config_path"] = args.config
    
    # Initialize data module
    data_module = SingingVoiceDataModule(config)
    
    # Prepare data (this will run preprocessing if the dataset doesn't exist)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    
    # Initialize model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = SingingVoiceSynthesisModel.load_from_checkpoint(args.checkpoint)
    else:
        print("Initializing new model")
        model = SingingVoiceSynthesisModel(config)
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir=config["logging"]["log_dir"],
        name=args.name,
        default_hp_metric=False
    )
    
    # Setup callbacks
    callbacks = [
        # Model checkpoint callback
        ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor=config["logging"]["monitor"],
            mode=config["logging"]["mode"],
            save_top_k=config["logging"]["save_top_k"],
            save_last=config["logging"]["save_last"],
            verbose=True
        ),
        # Early stopping callback
        EarlyStopping(
            monitor=config["logging"]["monitor"],
            mode=config["logging"]["mode"],
            patience=config["training"]["early_stopping"]["patience"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
            verbose=True
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        accelerator="auto",
        devices=args.gpus if args.gpus > 0 else None,
        fast_dev_run=args.debug,
        gradient_clip_val=config["training"]["grad_clip_thresh"],
        deterministic=True,
        precision=16 if torch.cuda.is_available() else 32,  # Use FP16 if available
        accumulate_grad_batches=1,  # Adjust if needed
        check_val_every_n_epoch=1,
        num_sanity_val_steps=2,
        benchmark=True
    )
    
    # Train model
    print("Starting training")
    trainer.fit(model, data_module)
    
    # Test model
    if not args.debug:
        print("Starting testing")
        trainer.test(model, data_module)
    
    # Save final model
    model_path = os.path.join(logger.log_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

if __name__ == "__main__":
    main()
