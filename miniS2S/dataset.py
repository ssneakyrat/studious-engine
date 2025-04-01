import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import yaml
import random
from typing import Dict, List, Optional, Tuple, Union

class SingingVoiceDataset(Dataset):
    """Dataset for singing voice synthesis using preprocessed HDF5 data."""
    
    def __init__(self, 
                 h5_path: str,
                 sample_ids: Optional[List[str]] = None,
                 max_mel_length: int = 1000,
                 random_crop: bool = True):
        """
        Args:
            h5_path: Path to the H5 file containing the preprocessed dataset
            sample_ids: List of sample IDs to use (if None, use all)
            max_mel_length: Maximum length of mel spectrograms (for memory efficiency)
            random_crop: Whether to randomly crop mel spectrograms during training
        """
        super().__init__()
        self.h5_path = h5_path
        self.max_mel_length = max_mel_length
        self.random_crop = random_crop
        
        # Initialize phoneme vocabulary attributes
        self.phoneme_to_id = None
        self.id_to_phoneme = None
        self.num_phonemes = None
        
        # Open H5 file and get metadata
        with h5py.File(h5_path, 'r') as f:
            # Get sample IDs from file list or use provided ones
            if sample_ids is None:
                if 'metadata' in f and 'file_list' in f['metadata']:
                    file_list = f['metadata']['file_list'][:]
                    self.sample_ids = [s.decode('utf-8') if isinstance(s, bytes) else s for s in file_list]
                else:
                    # Fallback: use all group names as sample IDs except 'metadata'
                    self.sample_ids = [k for k in f.keys() if k != 'metadata']
            else:
                self.sample_ids = sample_ids
                
            # Store audio configuration from metadata
            if 'metadata' in f:
                self.config = dict(f['metadata'].attrs)
            else:
                # Default configuration if metadata is not available
                self.config = {
                    'sample_rate': 22050,
                    'hop_length': 256,
                    'n_mels': 80
                }
        
        print(f"Loaded {len(self.sample_ids)} samples from {h5_path}")
    
    def set_phoneme_vocabulary(self, phoneme_to_id, id_to_phoneme, num_phonemes):
        """Set the phoneme vocabulary for the dataset."""
        self.phoneme_to_id = phoneme_to_id
        self.id_to_phoneme = id_to_phoneme
        self.num_phonemes = num_phonemes
    
    def _get_phoneme_id(self, phoneme):
        """
        Get the ID for a phoneme.
        
        Args:
            phoneme: Phoneme string
        
        Returns:
            ID for the phoneme or 0 for unknown
        """
        if self.phoneme_to_id is None:
            raise ValueError("Phoneme vocabulary not set. Call set_phoneme_vocabulary() first.")
        return self.phoneme_to_id.get(phoneme, 0)
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample_id = self.sample_ids[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            if sample_id not in f:
                raise KeyError(f"Sample ID {sample_id} not found in HDF5 file")
            
            sample_group = f[sample_id]
            
            # Get mel spectrogram
            mel_spec = sample_group['features']['mel_spectrogram'][:]  # Shape: [n_mels, time]
            mel_spec = torch.FloatTensor(mel_spec)  # Convert to torch tensor
            
            # Get phoneme data
            phones_bytes = sample_group['phonemes']['phones'][:]
            phones = [p.decode('utf-8') for p in phones_bytes]
            start_frames = sample_group['phonemes']['start_frames'][:]
            end_frames = sample_group['phonemes']['end_frames'][:]
            durations = sample_group['phonemes']['durations'][:]
            
            # Get musical features (F0 and others)
            f0_values = sample_group['features']['f0_values'][:]
            f0_values = torch.FloatTensor(f0_values)  # Convert to torch tensor
            
            # Get MIDI data if available
            has_midi = (
                'midi' in sample_group and 
                'notes' in sample_group['midi'] and 
                sample_group['midi']['notes'].shape[0] > 0
            )
            
            # Create a frame-level phoneme sequence
            # Each mel frame will have its corresponding phoneme ID
            n_frames = mel_spec.shape[1]
            phoneme_ids = torch.zeros(n_frames, dtype=torch.long)
            
            # Create frame-level musical features
            # Initialize with zeros - 4 features: F0, duration, velocity, phrasing
            musical_features = torch.zeros((n_frames, 4), dtype=torch.float)
            
            # Fill in F0 values
            musical_features[:, 0] = f0_values[:n_frames]  # F0 (first column)
            
            # Map each frame to its corresponding phoneme ID using the global vocabulary
            for i in range(len(phones)):
                start_frame = int(start_frames[i])
                end_frame = int(end_frames[i])
                p_id = self._get_phoneme_id(phones[i])  # Use global mapping
                
                # Ensure end_frame is within bounds
                end_frame = min(end_frame, n_frames)
                
                if start_frame < end_frame:
                    # Assign phoneme ID to all frames in this phoneme's duration
                    phoneme_ids[start_frame:end_frame] = p_id
                    
                    # Add duration as a feature (column 1)
                    frame_duration = durations[i] / (end_frame - start_frame) if end_frame > start_frame else 0
                    musical_features[start_frame:end_frame, 1] = frame_duration
            
            # Add MIDI-based features if available (velocity in column 2)
            if has_midi and 'notes' in sample_group['midi']:
                midi_notes = sample_group['midi']['notes'][:]
                midi_start_frames = sample_group['midi']['start_frames'][:]
                midi_end_frames = sample_group['midi']['end_frames'][:]
                
                # We'll use MIDI velocity as a proxy (set to 1.0 for all notes)
                # In a real dataset, this might come from the MIDI file
                for i in range(len(midi_notes)):
                    start_frame = int(midi_start_frames[i])
                    end_frame = int(midi_end_frames[i])
                    
                    # Ensure end_frame is within bounds
                    end_frame = min(end_frame, n_frames)
                    
                    if start_frame < end_frame:
                        # Set velocity to 1.0 (assuming max velocity)
                        musical_features[start_frame:end_frame, 2] = 1.0
            
            # Phrasing markers (column 3) - for demonstration, we'll mark phrase boundaries
            # In a real implementation, this would come from musical phrase analysis
            # Here we'll just mark the beginning of each phoneme that follows a pause/silence
            for i in range(1, len(phones)):
                if phones[i-1] in ['pau', 'sil', 'sp'] and start_frames[i] < n_frames:
                    # Mark the beginning of a phrase after a pause
                    phrase_start = int(start_frames[i])
                    phrase_length = min(10, n_frames - phrase_start)  # Mark first 10 frames of phrase
                    if phrase_length > 0:
                        # Decaying importance through the phrase start
                        values = torch.linspace(1.0, 0.0, phrase_length)
                        musical_features[phrase_start:phrase_start+phrase_length, 3] = values
            
            # Normalize F0 values to be in the range [0, 1]
            # Replace NaN values with 0 first
            musical_features[:, 0] = torch.nan_to_num(musical_features[:, 0], nan=0.0)
            if torch.max(musical_features[:, 0]) > 0:
                musical_features[:, 0] = musical_features[:, 0] / 800.0  # Normalize by max expected F0
            
            # Handle sequences that are too long
            if n_frames > self.max_mel_length:
                if self.random_crop:
                    # Randomly crop the sequence to max_mel_length
                    max_start = n_frames - self.max_mel_length
                    start_idx = random.randint(0, max_start)
                    end_idx = start_idx + self.max_mel_length
                else:
                    # Just take the first max_mel_length frames
                    start_idx = 0
                    end_idx = self.max_mel_length
                    
                mel_spec = mel_spec[:, start_idx:end_idx]
                phoneme_ids = phoneme_ids[start_idx:end_idx]
                musical_features = musical_features[start_idx:end_idx]
                n_frames = self.max_mel_length
            
            # Create target sequence for teacher forcing (shifted mel frames)
            # Last frame is repeated as a simple solution for handling the final prediction
            target_mel = torch.cat([mel_spec[:, 1:], mel_spec[:, -1:]], dim=1)
            
            # Create stop tokens: 0 for all frames except the last one
            stop_tokens = torch.zeros(n_frames, dtype=torch.float)
            stop_tokens[-1] = 1.0  # Mark the last frame as stop
            
            # Prepare attention mask for encoder-decoder alignment
            # In training, we might use the actual alignment for teacher forcing
            # Here we just create a simple mask based on sequence length
            attention_mask = torch.ones(n_frames, dtype=torch.bool)
            
            # Create a sample dictionary with all necessary data
            sample = {
                'id': sample_id,
                'mel': mel_spec.T,  # [time, n_mels] for batch processing
                'target_mel': target_mel.T,  # [time, n_mels]
                'phoneme_ids': phoneme_ids,  # [time]
                'musical_features': musical_features,  # [time, 4]
                'stop_tokens': stop_tokens,  # [time]
                'attention_mask': attention_mask,  # [time]
                'mel_length': n_frames,
            }
            
            return sample

    def get_phoneme_vocabulary(self):
        """Extract the phoneme vocabulary from the dataset."""
        phoneme_set = set()
        
        with h5py.File(self.h5_path, 'r') as f:
            for sample_id in self.sample_ids:
                if sample_id in f and 'phonemes' in f[sample_id] and 'phones' in f[sample_id]['phonemes']:
                    phones_bytes = f[sample_id]['phonemes']['phones'][:]
                    phones = [p.decode('utf-8') for p in phones_bytes]
                    phoneme_set.update(phones)
        
        # Sort for deterministic ordering
        phoneme_list = sorted(list(phoneme_set))
        # Create mapping
        phoneme_to_id = {p: i+1 for i, p in enumerate(phoneme_list)}  # Start from 1, 0 for padding
        id_to_phoneme = {i+1: p for i, p in enumerate(phoneme_list)}
        id_to_phoneme[0] = '<PAD>'  # Add padding token
        
        return phoneme_to_id, id_to_phoneme, len(phoneme_to_id) + 1  # +1 for padding

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for variable length sequences."""
        # Find max length in batch
        max_mel_length = max(item['mel_length'] for item in batch)
        
        # Initialize tensors with proper padding
        batch_size = len(batch)
        n_mels = batch[0]['mel'].shape[1]
        
        # Prepare padded tensors
        padded_mel = torch.zeros((batch_size, max_mel_length, n_mels))
        padded_target_mel = torch.zeros((batch_size, max_mel_length, n_mels))
        padded_phoneme_ids = torch.zeros((batch_size, max_mel_length), dtype=torch.long)
        padded_musical_features = torch.zeros((batch_size, max_mel_length, 4))
        padded_stop_tokens = torch.zeros((batch_size, max_mel_length))
        attention_mask = torch.zeros((batch_size, max_mel_length), dtype=torch.bool)
        mel_lengths = torch.zeros(batch_size, dtype=torch.long)
        
        # Fill padded tensors with data
        sample_ids = []
        for i, item in enumerate(batch):
            mel_length = item['mel_length']
            mel_lengths[i] = mel_length
            
            # Add data to padded tensors
            padded_mel[i, :mel_length] = item['mel']
            padded_target_mel[i, :mel_length] = item['target_mel']
            padded_phoneme_ids[i, :mel_length] = item['phoneme_ids']
            padded_musical_features[i, :mel_length] = item['musical_features']
            padded_stop_tokens[i, :mel_length] = item['stop_tokens']
            attention_mask[i, :mel_length] = item['attention_mask']
            sample_ids.append(item['id'])
            
        # Create final batch dictionary
        collated_batch = {
            'ids': sample_ids,
            'mel': padded_mel,
            'target_mel': padded_target_mel,
            'phoneme_ids': padded_phoneme_ids,
            'musical_features': padded_musical_features,
            'stop_tokens': padded_stop_tokens,
            'attention_mask': attention_mask,
            'mel_lengths': mel_lengths,
        }
        
        return collated_batch


class SingingVoiceDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for singing voice synthesis."""
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary for the data module
        """
        super().__init__()
        self.config = config
        self.h5_path = config['data']['data_processed']#os.path.join(config['data']['data_raw'], 'binary', 'dataset.h5')
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['dataloader']['num_workers']
        self.pin_memory = config['dataloader']['pin_memory']
        self.train_val_split = config['preprocessing']['train_val_split']
        self.max_mel_length = 1000  # Can be parameterized from config
        
        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Placeholders for phoneme vocabulary
        self.phoneme_to_id = None
        self.id_to_phoneme = None
        self.num_phonemes = None
    
    def prepare_data(self):
        """Check if the dataset exists, if not run the preprocessing script."""
        h5_path = Path(self.h5_path)
        if not h5_path.exists():
            print(f"Dataset not found at {h5_path}. Please run the preprocessing script first.")
            # Importing here to avoid circular imports
            from preprocess import process_files, validate_dataset
            
            # Run preprocessing
            print("Running preprocessing...")
            process_files(
                config_path=self.config['config_path'] if 'config_path' in self.config else "config/default.yaml",
                min_phonemes=self.config['preprocessing']['min_phonemes']
            )
            
            # Validate the dataset
            validate_dataset(h5_path)
    
    def setup(self, stage=None):
        """Set up train, validation, and test datasets."""
        # Create a full dataset first to get the sample IDs
        full_dataset = SingingVoiceDataset(
            h5_path=self.h5_path,
            max_mel_length=self.max_mel_length,
            random_crop=True
        )
        
        # Get phoneme vocabulary
        self.phoneme_to_id, self.id_to_phoneme, self.num_phonemes = full_dataset.get_phoneme_vocabulary()
        
        # Split dataset into train and validation
        all_sample_ids = full_dataset.sample_ids
        random.shuffle(all_sample_ids)  # Shuffle for random split
        
        split_idx = int(len(all_sample_ids) * self.train_val_split)
        train_ids = all_sample_ids[:split_idx]
        val_ids = all_sample_ids[split_idx:]
        
        # Create separate datasets for train and validation
        if stage == 'fit' or stage is None:
            self.train_dataset = SingingVoiceDataset(
                h5_path=self.h5_path,
                sample_ids=train_ids,
                max_mel_length=self.max_mel_length,
                random_crop=True  # Use random crop for training
            )
            # Set the phoneme vocabulary for the training dataset
            self.train_dataset.set_phoneme_vocabulary(
                self.phoneme_to_id, self.id_to_phoneme, self.num_phonemes
            )
            
            self.val_dataset = SingingVoiceDataset(
                h5_path=self.h5_path,
                sample_ids=val_ids,
                max_mel_length=self.max_mel_length,
                random_crop=False  # No random crop for validation
            )
            # Set the phoneme vocabulary for the validation dataset
            self.val_dataset.set_phoneme_vocabulary(
                self.phoneme_to_id, self.id_to_phoneme, self.num_phonemes
            )
            
            print(f"Training set: {len(self.train_dataset)} samples")
            print(f"Validation set: {len(self.val_dataset)} samples")
        
        if stage == 'test' or stage is None:
            # For testing, use validation set if no specific test set is provided
            self.test_dataset = SingingVoiceDataset(
                h5_path=self.h5_path,
                sample_ids=val_ids[:min(10, len(val_ids))],  # Use a subset for testing
                max_mel_length=self.max_mel_length,
                random_crop=False
            )
            # Set the phoneme vocabulary for the test dataset
            self.test_dataset.set_phoneme_vocabulary(
                self.phoneme_to_id, self.id_to_phoneme, self.num_phonemes
            )
            
            print(f"Test set: {len(self.test_dataset)} samples")
        
        # Store vocabulary size in config for model initialization
        self.config['model']['num_phonemes'] = self.num_phonemes
    
    def train_dataloader(self):
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SingingVoiceDataset.collate_fn,
            prefetch_factor=self.config['dataloader']['prefetch_factor'] if 'prefetch_factor' in self.config['dataloader'] else 2,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SingingVoiceDataset.collate_fn,
            prefetch_factor=self.config['dataloader']['prefetch_factor'] if 'prefetch_factor' in self.config['dataloader'] else 2,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=SingingVoiceDataset.collate_fn,
            prefetch_factor=self.config['dataloader']['prefetch_factor'] if 'prefetch_factor' in self.config['dataloader'] else 2
        )
    
    def get_phone_id_mappings(self):
        """Return phoneme to ID and ID to phoneme mappings."""
        return self.phoneme_to_id, self.id_to_phoneme, self.num_phonemes


def test_dataset():
    """Test the dataset class with a small subset of data."""
    # Load configuration
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create a test dataset
    h5_path = os.path.join(config['data']['data_raw'], 'binary', 'dataset.h5')
    if not os.path.exists(h5_path):
        print(f"Dataset not found at {h5_path}. Please run the preprocessing script first.")
        return
    
    dataset = SingingVoiceDataset(h5_path=h5_path, max_mel_length=1000, random_crop=False)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Get a random sample
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    
    print(f"Sample ID: {sample['id']}")
    print(f"Mel spectrogram shape: {sample['mel'].shape}")
    print(f"Phoneme IDs shape: {sample['phoneme_ids'].shape}")
    print(f"Musical features shape: {sample['musical_features'].shape}")
    print(f"Stop tokens shape: {sample['stop_tokens'].shape}")
    
    # Test phoneme vocabulary extraction
    phoneme_to_id, id_to_phoneme, num_phonemes = dataset.get_phoneme_vocabulary()
    print(f"Number of unique phonemes: {num_phonemes}")
    print(f"Sample phonemes: {list(phoneme_to_id.items())[:10]}")
    
    # Test collate function with a mini-batch
    indices = [random.randint(0, len(dataset) - 1) for _ in range(3)]
    mini_batch = [dataset[i] for i in indices]
    collated_batch = SingingVoiceDataset.collate_fn(mini_batch)
    
    print(f"Collated batch mel shape: {collated_batch['mel'].shape}")
    print(f"Collated batch phoneme IDs shape: {collated_batch['phoneme_ids'].shape}")
    print(f"Collated batch mel lengths: {collated_batch['mel_lengths']}")
    
    # Test data module
    data_module = SingingVoiceDataModule(config)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    print(f"Training dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    
    # Test dataloader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    
    print(f"Batch from dataloader:")
    print(f"  Mel: {batch['mel'].shape}")
    print(f"  Phoneme IDs: {batch['phoneme_ids'].shape}")
    print(f"  Musical features: {batch['musical_features'].shape}")
    print(f"  Mel lengths: {batch['mel_lengths']}")
    
    print("Dataset test completed successfully")


if __name__ == "__main__":
    test_dataset()
