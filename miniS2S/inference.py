import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Union

from dataset import SingingVoiceDataModule
from model import SingingVoiceSynthesisModel
from audio_utils import griffin_lim, save_mel_spectrogram


class InferenceManager:
    """
    Manager class for model inference and audio generation.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 config_path: str = "config/default.yaml",
                 output_dir: str = "outputs",
                 device: Optional[str] = None):
        """
        Initialize the inference manager.
        
        Args:
            checkpoint_path: Path to the model checkpoint
            config_path: Path to the configuration file
            output_dir: Directory to save outputs
            device: Device to use for inference ('cpu', 'cuda', or None for auto-detection)
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        print(f"Loading model from checkpoint: {checkpoint_path}")
        self.model = SingingVoiceSynthesisModel.load_from_checkpoint(
            checkpoint_path, config=self.config
        ).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize data module
        self.data_module = SingingVoiceDataModule(self.config)
        self.data_module.prepare_data()
        self.data_module.setup(stage="test")
        
        # Get phoneme vocabulary
        self.phoneme_to_id, self.id_to_phoneme, self.num_phonemes = (
            self.data_module.get_phone_id_mappings()
        )
        
        print(f"Model loaded successfully. Ready for inference.")
    
    def phonemes_to_ids(self, phonemes: List[str]) -> torch.Tensor:
        """
        Convert phoneme sequence to phoneme IDs.
        
        Args:
            phonemes: List of phonemes
            
        Returns:
            Tensor of phoneme IDs
        """
        # Map phonemes to IDs
        phoneme_ids = [self.phoneme_to_id.get(p, 0) for p in phonemes]
        
        return torch.tensor(phoneme_ids, dtype=torch.long, device=self.device)
    
    def create_musical_features(self, 
                                seq_length: int, 
                                pitch_values: Optional[List[int]] = None,
                                duration_values: Optional[List[float]] = None,
                                velocity_values: Optional[List[float]] = None,
                                phrasing_values: Optional[List[float]] = None) -> torch.Tensor:
        """
        Create musical features tensor with provided values or defaults.
        
        Args:
            seq_length: Length of the sequence
            pitch_values: List of pitch values (MIDI note numbers)
            duration_values: List of duration values
            velocity_values: List of velocity values
            phrasing_values: List of phrasing values
            
        Returns:
            Tensor of musical features [seq_length, 4]
        """
        # Initialize features with zeros
        features = torch.zeros((seq_length, 4), dtype=torch.float, device=self.device)
        
        # Fill in pitch values (normalized to 0-1 range)
        if pitch_values is not None:
            for i, pitch in enumerate(pitch_values):
                if i < seq_length:
                    features[i, 0] = pitch / 127.0  # Normalize by max MIDI value
        
        # Fill in duration values
        if duration_values is not None:
            for i, duration in enumerate(duration_values):
                if i < seq_length:
                    features[i, 1] = duration
        
        # Fill in velocity values (normalized to 0-1 range)
        if velocity_values is not None:
            for i, velocity in enumerate(velocity_values):
                if i < seq_length:
                    features[i, 2] = velocity
        
        # Fill in phrasing values
        if phrasing_values is not None:
            for i, phrasing in enumerate(phrasing_values):
                if i < seq_length:
                    features[i, 3] = phrasing
        
        return features
    
    def generate_from_sample(self, 
                             sample_idx: Optional[int] = None,
                             save_results: bool = True) -> Dict:
        """
        Generate audio from a test dataset sample.
        
        Args:
            sample_idx: Index of the sample to use (None for random)
            save_results: Whether to save the results to disk
            
        Returns:
            Dictionary with generation results
        """
        # Get test dataloader
        test_loader = self.data_module.test_dataloader()
        
        # Get sample
        if sample_idx is None:
            # Random sample
            batch = next(iter(test_loader))
            sample_idx = 0
        else:
            # Specific sample
            for i, batch in enumerate(test_loader):
                if i == sample_idx // test_loader.batch_size:
                    sample_idx = sample_idx % test_loader.batch_size
                    break
        
        # Extract the specific sample from the batch
        sample = {
            'id': batch['ids'][sample_idx],
            'phoneme_ids': batch['phoneme_ids'][sample_idx].unsqueeze(0).to(self.device),
            'musical_features': batch['musical_features'][sample_idx].unsqueeze(0).to(self.device),
            'mel': batch['mel'][sample_idx].unsqueeze(0).to(self.device),
            'mel_length': batch['mel_lengths'][sample_idx].unsqueeze(0).to(self.device)
        }
        
        print(f"Generating from sample: {sample['id']}")
        
        # Generate with the model
        with torch.no_grad():
            outputs = self.model(
                phoneme_ids=sample['phoneme_ids'],
                musical_features=sample['musical_features'],
                mel_lengths=sample['mel_length'],
                teacher_forcing_ratio=0.0  # No teacher forcing for inference
            )
        
        # Extract numpy arrays for visualization and audio conversion
        generated_mel = outputs['mel_outputs_postnet'][0].cpu().numpy()
        target_mel = sample['mel'][0].cpu().numpy()
        attention = outputs['alignments'][0].cpu().numpy()
        
        # Generate audio using Griffin-Lim
        print("Generating audio with Griffin-Lim...")
        
        # Audio configuration
        audio_config = {
            'sample_rate': self.config['audio']['sample_rate'],
            'n_fft': self.config['audio']['n_fft'],
            'hop_length': self.config['audio']['hop_length'],
            'win_length': self.config['audio']['win_length'],
            'n_mels': self.config['audio']['n_mels'],
            'fmin': self.config['audio']['fmin'],
            'fmax': self.config['audio']['fmax']
        }
        
        # Generate audio
        generated_audio = griffin_lim(
            generated_mel.T,  # Transpose to [n_mels, time]
            **audio_config
        )
        
        # Generate target audio (for comparison)
        target_audio = griffin_lim(
            target_mel.T,  # Transpose to [n_mels, time]
            **audio_config
        )
        
        # Save results
        if save_results:
            # Create output directory for this sample
            sample_dir = os.path.join(self.output_dir, f"sample_{sample['id']}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save audio files
            sf.write(
                os.path.join(sample_dir, "generated.wav"),
                generated_audio,
                audio_config['sample_rate']
            )
            sf.write(
                os.path.join(sample_dir, "target.wav"),
                target_audio,
                audio_config['sample_rate']
            )
            
            # Save spectrograms
            save_mel_spectrogram(
                generated_mel.T,
                os.path.join(sample_dir, "generated_mel.png"),
                title="Generated Mel Spectrogram"
            )
            save_mel_spectrogram(
                target_mel.T,
                os.path.join(sample_dir, "target_mel.png"),
                title="Target Mel Spectrogram"
            )
            
            # Save attention plot
            plt.figure(figsize=(10, 10))
            plt.imshow(attention, aspect='auto', origin='lower')
            plt.title("Attention Alignment")
            plt.xlabel("Encoder Steps")
            plt.ylabel("Decoder Steps")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, "attention.png"))
            plt.close()
            
            print(f"Results saved to {sample_dir}")
        
        return {
            'id': sample['id'],
            'generated_mel': generated_mel,
            'target_mel': target_mel,
            'attention': attention,
            'generated_audio': generated_audio,
            'target_audio': target_audio
        }
    
    def generate_from_phonemes(self, 
                               phonemes: List[str],
                               musical_features: Optional[torch.Tensor] = None,
                               output_name: str = "custom",
                               save_results: bool = True) -> Dict:
        """
        Generate audio from a custom phoneme sequence.
        
        Args:
            phonemes: List of phonemes
            musical_features: Optional tensor of musical features
            output_name: Name for the output files
            save_results: Whether to save the results to disk
            
        Returns:
            Dictionary with generation results
        """
        print(f"Generating from custom phoneme sequence: {' '.join(phonemes)}")
        
        # Convert phonemes to IDs
        phoneme_ids = self.phonemes_to_ids(phonemes)
        phoneme_ids = phoneme_ids.unsqueeze(0)  # Add batch dimension
        
        # Create musical features if not provided
        seq_length = len(phonemes)
        if musical_features is None:
            musical_features = self.create_musical_features(seq_length)
            musical_features = musical_features.unsqueeze(0)  # Add batch dimension
        
        # Sequence length
        mel_length = torch.tensor([seq_length], device=self.device)
        
        # Generate with the model
        with torch.no_grad():
            outputs = self.model(
                phoneme_ids=phoneme_ids,
                musical_features=musical_features,
                mel_lengths=mel_length,
                teacher_forcing_ratio=0.0  # No teacher forcing for inference
            )
        
        # Extract numpy arrays for visualization and audio conversion
        generated_mel = outputs['mel_outputs_postnet'][0].cpu().numpy()
        attention = outputs['alignments'][0].cpu().numpy()
        
        # Generate audio using Griffin-Lim
        print("Generating audio with Griffin-Lim...")
        
        # Audio configuration
        audio_config = {
            'sample_rate': self.config['audio']['sample_rate'],
            'n_fft': self.config['audio']['n_fft'],
            'hop_length': self.config['audio']['hop_length'],
            'win_length': self.config['audio']['win_length'],
            'n_mels': self.config['audio']['n_mels'],
            'fmin': self.config['audio']['fmin'],
            'fmax': self.config['audio']['fmax']
        }
        
        # Generate audio
        generated_audio = griffin_lim(
            generated_mel.T,  # Transpose to [n_mels, time]
            **audio_config
        )
        
        # Save results
        if save_results:
            # Create output directory for this sample
            sample_dir = os.path.join(self.output_dir, f"custom_{output_name}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save audio file
            sf.write(
                os.path.join(sample_dir, "generated.wav"),
                generated_audio,
                audio_config['sample_rate']
            )
            
            # Save spectrogram
            save_mel_spectrogram(
                generated_mel.T,
                os.path.join(sample_dir, "generated_mel.png"),
                title="Generated Mel Spectrogram"
            )
            
            # Save attention plot
            plt.figure(figsize=(10, 10))
            plt.imshow(attention, aspect='auto', origin='lower')
            plt.title("Attention Alignment")
            plt.xlabel("Encoder Steps")
            plt.ylabel("Decoder Steps")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, "attention.png"))
            plt.close()
            
            # Save phoneme sequence
            with open(os.path.join(sample_dir, "phonemes.txt"), "w") as f:
                f.write(' '.join(phonemes))
            
            print(f"Results saved to {sample_dir}")
        
        return {
            'id': output_name,
            'generated_mel': generated_mel,
            'attention': attention,
            'generated_audio': generated_audio
        }
    
    def batch_generate(self, num_samples: int = 5) -> None:
        """
        Generate audio for multiple samples from the test set.
        
        Args:
            num_samples: Number of samples to generate
        """
        print(f"Generating audio for {num_samples} samples...")
        
        for i in range(num_samples):
            print(f"\nGenerating sample {i+1}/{num_samples}")
            self.generate_from_sample(sample_idx=i, save_results=True)
        
        print(f"\nBatch generation complete. Results saved to {self.output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference for singing voice synthesis")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save outputs")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"],
                        help="Device to use for inference ('cpu', 'cuda', or None for auto-detection)")
    parser.add_argument("--mode", type=str, default="sample",
                        choices=["sample", "batch", "custom"],
                        help="Inference mode: single sample, batch, or custom phonemes")
    parser.add_argument("--sample_idx", type=int, default=None,
                        help="Index of the sample to use (None for random)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to generate in batch mode")
    parser.add_argument("--phonemes", type=str, default=None,
                        help="Custom phoneme sequence (space-separated)")
    parser.add_argument("--output_name", type=str, default="custom",
                        help="Name for custom output files")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create inference manager
    inference_manager = InferenceManager(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Run inference based on mode
    if args.mode == "sample":
        inference_manager.generate_from_sample(
            sample_idx=args.sample_idx,
            save_results=True
        )
    elif args.mode == "batch":
        inference_manager.batch_generate(
            num_samples=args.num_samples
        )
    elif args.mode == "custom":
        if args.phonemes is None:
            print("Error: Custom mode requires phoneme sequence.")
            return
        
        # Parse phoneme sequence
        phonemes = args.phonemes.split()
        
        inference_manager.generate_from_phonemes(
            phonemes=phonemes,
            output_name=args.output_name,
            save_results=True
        )


if __name__ == "__main__":
    main()
