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
    parser.add_argument("--fast_train", action="store_true",
                        help="Enable fast training mode with reduced model size and dataset")
    parser.add_argument("--precision", type=str, default="32", choices=["16", "32", "bf16"],
                        help="Floating point precision to use for training")
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="Limit training to a fraction of batches (for testing)")
    # Keep limit_val_batches separate from limit_train_batches to avoid the error
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Limit validation to a fraction of batches (for testing)")
    
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
    
    # Apply fast training mode if specified
    if args.fast_train:
        print("Applying fast training configuration...")
        
        # Reduce model size
        config["model"]["embedding_dim"] = 128  # Instead of 256
        config["model"]["encoder"]["lstm_hidden_size"] = 64  # Instead of 128
        config["model"]["decoder"]["lstm_dim"] = 128  # Instead of 256
        
        # Reduce dataset size for quick iteration
        max_mel_length = 500  # Reduce from 1000 to 500 for faster processing
        
        # Speed up training
        config["training"]["batch_size"] = 16  # Increase batch size if memory allows
        #config["training"]["max_epochs"] = 3  # Fewer epochs for testing
        config["logging"]["log_every_n_steps"] = 10  # More frequent logging
        
        # Simplify model to reduce computation
        config["model"]["encoder"]["lstm_layers"] = 1  # Reduce from 2
        config["model"]["decoder"]["lstm_layers"] = 1  # Reduce from 2
        
        # Reduce validation frequency
        val_check_interval = 0.5  # Check validation every half epoch
    else:
        max_mel_length = 1000  # Default value
        val_check_interval = 1.0  # Default: check validation every epoch
    
    # Initialize data module with potentially modified config
    data_module = SingingVoiceDataModule(config)
    data_module.max_mel_length = max_mel_length  # Apply max_mel_length modification
    
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
    
    # Save modified config if fast_train is enabled
    if args.fast_train:
        fast_train_config_path = os.path.join(logger.log_dir, "fast_train_config.yaml")
        os.makedirs(os.path.dirname(fast_train_config_path), exist_ok=True)
        with open(fast_train_config_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Saved fast training configuration to {fast_train_config_path}")
    
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
    
    # Setup trainer with precision based on args
    precision = args.precision
    if precision == "16" and not torch.cuda.is_available():
        print("Warning: FP16 precision requested but CUDA is not available. Falling back to FP32.")
        precision = "32"
    
    # Convert precision to proper format for PyTorch Lightning
    if precision == "16":
        precision_value = 16
    elif precision == "bf16":
        precision_value = "bf16"
    else:
        precision_value = 32
    
    # Ensure validation batches aren't limited below 1.0 if dataset is small
    limit_val_batches = args.limit_val_batches
    
    # Print dataset sizes for debugging
    print(f"Training dataset size: {len(data_module.train_dataset)} samples")
    print(f"Validation dataset size: {len(data_module.val_dataset)} samples")
    
    # Calculate approximate number of batches
    train_batches = len(data_module.train_dataset) // config["training"]["batch_size"] + 1
    val_batches = len(data_module.val_dataset) // config["training"]["batch_size"] + 1
    
    print(f"Estimated training batches: {train_batches}")
    print(f"Estimated validation batches: {val_batches}")
    
    # Warn if limit_val_batches might cause issues
    if val_batches <= 1 and limit_val_batches < 1.0:
        print(f"Warning: Validation has only {val_batches} batch(es). Setting limit_val_batches=1.0 to avoid errors.")
        limit_val_batches = 1.0
    
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
        precision=precision_value,
        accumulate_grad_batches=1,  # Adjust if needed
        check_val_every_n_epoch=1,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=1,  # Reduced from 2 to 1 for faster startup
        benchmark=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=limit_val_batches
    )
    
    # Train model
    print(f"Starting training with {config['training']['max_epochs']} epochs")
    if args.fast_train:
        print("Fast training mode enabled: reduced model size and faster iteration")
    
    print(f"Training with limit_train_batches={args.limit_train_batches}, limit_val_batches={limit_val_batches}")
    
    trainer.fit(model, data_module)
    
    # Test model
    if not args.debug and not args.fast_train:
        print("Starting testing")
        trainer.test(model, data_module)
    
    # Save final model
    model_path = os.path.join(logger.log_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")

if __name__ == "__main__":
    main()