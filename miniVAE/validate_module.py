import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import yaml  # Add yaml import
from pathlib import Path
import random
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchaudio
import parselmouth

# Import model and data modules
from preprocessor import SVSPreprocessor
from model import SVSModelLightning
from data import SVSDataset, SVSDataModule

class SVSValidationModule:
    """Module for validating SVS model components."""
    
    def __init__(self, h5_file="svs_dataset.h5", config_path="config/default.yaml"):
        self.h5_file = h5_file
        self.config_path = config_path
        self.logger = TensorBoardLogger("logs", name="validation")
        
        # Make sure logs directory exists
        Path("logs").mkdir(exist_ok=True)
    
    def validate_dataset_loading(self):
        """Test dataset loading and visualization."""
        print("Validating dataset loading...")
        
        # Check if file exists
        if not os.path.exists(self.h5_file):
            print(f"Error: {self.h5_file} not found.")
            return False
        
        # Try loading the dataset
        try:
            with h5py.File(self.h5_file, 'r') as h5:
                n_samples = len(h5['data'])
                n_phonemes = h5['metadata']['n_phonemes'][()]
                print(f"Dataset has {n_samples} samples and {n_phonemes} phonemes.")
                
                # Select a random sample
                sample_keys = list(h5['data'].keys())
                random_key = random.choice(sample_keys)
                
                sample = h5['data'][random_key]
                phonemes = sample['phonemes'][:]
                midi = sample['midi'][:]
                f0 = sample['f0'][:]
                mel_spec = sample['mel_spectrogram'][:]
                
                print(f"Sample {random_key} has {len(phonemes)} frames.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
        
        # Try loading with DataLoader
        try:
            dataset = SVSDataset(self.h5_file)
            loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=SVSDataModule.collate_fn)
            
            batch = next(iter(loader))
            print(f"Successfully loaded batch: {batch['ids']}")
            
            # Plot sample
            self._plot_dataset_sample(batch, random_key)
            dataset.close()
        except Exception as e:
            print(f"Error with DataLoader: {e}")
            return False
        
        return True
    
    def _plot_dataset_sample(self, batch, sample_id):
        """Plot a dataset sample for visualization."""
        idx = 0
        
        # Extract sample data
        phonemes = batch['phonemes'][idx].numpy()
        midi = batch['midi'][idx].numpy()
        f0 = batch['f0'][idx].numpy()
        mel_spec = batch['mel_spectrogram'][idx].numpy()
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Plot phonemes
        axs[0].plot(phonemes, 'bo-', markersize=2, alpha=0.5)
        axs[0].set_ylabel('Phoneme ID')
        axs[0].set_title(f'Sample Visualization - {batch["ids"][idx]}')
        
        # Plot MIDI notes
        axs[1].plot(midi, 'ro-', markersize=2, alpha=0.5)
        axs[1].set_ylabel('MIDI Note')
        
        # Plot F0
        axs[2].plot(f0, 'go-', markersize=2, alpha=0.5)
        axs[2].set_ylabel('Normalized F0')
        
        # Plot mel spectrogram
        im = axs[3].imshow(mel_spec.T, aspect='auto', origin='lower')
        axs[3].set_ylabel('Mel bins')
        axs[3].set_xlabel('Frames')
        fig.colorbar(im, ax=axs[3])
        
        plt.tight_layout()
        plt.savefig("logs/dataset_sample.png")
        self.logger.experiment.add_figure("dataset_sample", fig, 0)
        plt.close(fig)
    
    def validate_encoder(self):
        """Test the encoder component."""
        print("Validating encoder...")
        
        # Create mini model for encoder testing
        model = SVSModelLightning(self.config_path)
        model.setup()
        
        # Load a mini-batch
        dataset = SVSDataset(self.h5_file)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, 
                            collate_fn=SVSDataModule.collate_fn)
        
        batch = next(iter(loader))
        
        # Test encoder forward pass
        try:
            z_mean, z_logvar = model.encoder(
                batch['phonemes'],
                batch['midi'],
                batch['f0'],
                batch['lengths']
            )
            
            # Sample from latent space
            z = model.reparameterize(z_mean, z_logvar)
            
            print(f"Encoder forward pass successful.")
            print(f"Latent shape: {z.shape}")
            
            # Visualize latent space
            self._plot_latent_space(z_mean, z_logvar, z)
            dataset.close()
            return True
        except Exception as e:
            print(f"Encoder error: {e}")
            dataset.close()
            return False
    
    def _plot_latent_space(self, z_mean, z_logvar, z):
        """Visualize latent space."""
        z_mean_np = z_mean.detach().cpu().numpy()
        z_logvar_np = z_logvar.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        
        # Plot means
        axs[0].bar(range(z_mean_np.shape[1]), z_mean_np[0])
        axs[0].set_title("Latent Mean (z_mean)")
        axs[0].set_ylabel("Value")
        
        # Plot log variances
        axs[1].bar(range(z_logvar_np.shape[1]), z_logvar_np[0])
        axs[1].set_title("Latent Log Variance (z_logvar)")
        axs[1].set_ylabel("Value")
        
        # Plot sampled z
        axs[2].bar(range(z_np.shape[1]), z_np[0])
        axs[2].set_title("Sampled Latent Vector (z)")
        axs[2].set_xlabel("Latent Dimension")
        axs[2].set_ylabel("Value")
        
        plt.tight_layout()
        plt.savefig("logs/latent_space.png")
        self.logger.experiment.add_figure("latent_space", fig, 0)
        plt.close(fig)
    
    def validate_decoder(self):
        """Test the decoder component."""
        print("Validating decoder...")
        
        # Create mini model for decoder testing
        model = SVSModelLightning(self.config_path)
        model.setup()
        
        # Load a mini-batch
        dataset = SVSDataset(self.h5_file)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                            collate_fn=SVSDataModule.collate_fn)
        
        batch = next(iter(loader))
        
        # Test decoder forward pass
        try:
            # Get latent vector from encoder
            z_mean, z_logvar = model.encoder(
                batch['phonemes'],
                batch['midi'],
                batch['f0'],
                batch['lengths']
            )
            
            z = model.reparameterize(z_mean, z_logvar)
            
            # Decoder forward pass
            decoder_out, postnet_out = model.decoder(
                z, 
                batch['f0'], 
                batch['lengths']
            )
            
            print(f"Decoder forward pass successful.")
            print(f"Output shapes: {decoder_out.shape}, {postnet_out.shape}")
            
            # Visualize decoder output
            self._plot_decoder_output(batch['mel_spectrogram'][0], decoder_out[0], postnet_out[0])
            dataset.close()
            return True
        except Exception as e:
            print(f"Decoder error: {e}")
            dataset.close()
            return False
    
    def _plot_decoder_output(self, gt_mel, pre_mel, post_mel):
        """Visualize decoder output."""
        # Convert to numpy
        gt_mel_np = gt_mel.detach().cpu().numpy()
        pre_mel_np = pre_mel.detach().cpu().numpy()
        post_mel_np = post_mel.detach().cpu().numpy()
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot ground truth
        im1 = axs[0].imshow(gt_mel_np.T, aspect='auto', origin='lower')
        axs[0].set_title("Ground Truth Mel Spectrogram")
        axs[0].set_ylabel("Mel bins")
        fig.colorbar(im1, ax=axs[0])
        
        # Plot decoder output
        im2 = axs[1].imshow(pre_mel_np.T, aspect='auto', origin='lower')
        axs[1].set_title("Decoder Output (Before Postnet)")
        axs[1].set_ylabel("Mel bins")
        fig.colorbar(im2, ax=axs[1])
        
        # Plot postnet output
        im3 = axs[2].imshow(post_mel_np.T, aspect='auto', origin='lower')
        axs[2].set_title("Decoder Output (After Postnet)")
        axs[2].set_ylabel("Mel bins")
        axs[2].set_xlabel("Frames")
        fig.colorbar(im3, ax=axs[2])
        
        plt.tight_layout()
        plt.savefig("logs/decoder_output.png")
        self.logger.experiment.add_figure("decoder_output", fig, 0)
        plt.close(fig)
    
    def validate_full_model(self):
        """Test the full model forward pass."""
        print("Validating full model...")
        
        # Create model
        model = SVSModelLightning(self.config_path)
        model.setup()
        
        # Load a mini-batch
        dataset = SVSDataset(self.h5_file)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0,
                            collate_fn=SVSDataModule.collate_fn)
        
        batch = next(iter(loader))
        
        # Test full model forward pass
        try:
            # Forward pass
            decoder_out, postnet_out, z_mean, z_logvar = model(
                batch['phonemes'],
                batch['midi'],
                batch['f0'],
                batch['lengths']
            )
            
            # Calculate loss
            mask = batch['mask'].unsqueeze(-1)
            mel_gt = batch['mel_spectrogram']
            
            decoder_loss = F.l1_loss(decoder_out * mask, mel_gt * mask)
            postnet_loss = F.l1_loss(postnet_out * mask, mel_gt * mask)
            kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            
            total_loss = decoder_loss + postnet_loss + 0.01 * kl_loss
            
            print(f"Full model forward pass successful.")
            print(f"Loss values: Decoder={decoder_loss.item():.4f}, Postnet={postnet_loss.item():.4f}, KL={kl_loss.item():.4f}")
            print(f"Total Loss: {total_loss.item():.4f}")
            
            dataset.close()
            return True
        except Exception as e:
            print(f"Full model error: {e}")
            dataset.close()
            return False
    
    def run_all_validations(self):
        """Run all validation tests."""
        print("Running all validation tests...")
        
        results = {}
        
        print("\n=== Dataset Loading Validation ===")
        results['dataset'] = self.validate_dataset_loading()
        
        print("\n=== Encoder Validation ===")
        results['encoder'] = self.validate_encoder()
        
        print("\n=== Decoder Validation ===")
        results['decoder'] = self.validate_decoder()
        
        print("\n=== Full Model Validation ===")
        results['full_model'] = self.validate_full_model()
        
        print("\n=== Validation Summary ===")
        for test, result in results.items():
            status = "PASSED" if result else "FAILED"
            print(f"{test.ljust(15)}: {status}")
        
        return all(results.values())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SVS model components.")
    # Default to None, will construct path from config if not provided
    parser.add_argument("--h5_file", default=None, help="Path to dataset file (defaults to config dataset_dir + svs_dataset.h5)")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    parser.add_argument("--test", choices=["dataset", "encoder", "decoder", "full", "all"],
                        default="all", help="Which test to run")

    args = parser.parse_args()

    h5_file_path = args.h5_file
    config_path = args.config

    # If h5_file is not provided, construct it from config
    if h5_file_path is None:
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            datasets_dir = config_data.get('dataset', {}).get('datasets_dir')
            if datasets_dir:
                # Assuming the standard filename is svs_dataset.h5 based on previous default
                h5_file_path = os.path.join(datasets_dir, "processed", "svs_dataset.h5")
                print(f"Using H5 file path from config: {h5_file_path}")
            else:
                print(f"Warning: --h5_file not provided and 'dataset.datasets_dir' not found in {config_path}. Using default 'svs_dataset.h5'.")
                h5_file_path = "svs_dataset.h5" # Fallback to original default filename if config key is missing
        except FileNotFoundError:
            print(f"Warning: Config file '{config_path}' not found. Using default 'svs_dataset.h5'.")
            h5_file_path = "svs_dataset.h5" # Fallback if config file doesn't exist
        except Exception as e:
            print(f"Warning: Error reading config file '{config_path}': {e}. Using default 'svs_dataset.h5'.")
            h5_file_path = "svs_dataset.h5" # Fallback on any other config reading error

    # Instantiate validator with the determined h5_file_path and config_path
    validator = SVSValidationModule(h5_file_path, config_path)
    
    if args.test == "dataset":
        validator.validate_dataset_loading()
    elif args.test == "encoder":
        validator.validate_encoder()
    elif args.test == "decoder":
        validator.validate_decoder()
    elif args.test == "full":
        validator.validate_full_model()
    else:
        validator.run_all_validations()