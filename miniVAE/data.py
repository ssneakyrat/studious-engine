import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Optional
import random

class SVSDataset(Dataset):
    """Singing Voice Synthesis Dataset."""
    
    def __init__(self, h5_file, ids=None, max_len=None):
        """
        Args:
            h5_file: Path to the h5py file containing processed data
            ids: List of sample IDs to use (for train/val split)
            max_len: Maximum sequence length (for padding/truncating)
        """
        self.h5_file = h5_file
        self.h5 = None  # Will be opened in __getitem__ to support multiprocessing
        
        # Open temporarily to get metadata and sample IDs
        with h5py.File(h5_file, 'r') as h5:
            self.metadata = {
                'n_phonemes': h5['metadata']['n_phonemes'][()],
                'midi_min': h5['metadata']['midi_min'][()],
                'midi_max': h5['metadata']['midi_max'][()],
                'n_mels': h5['metadata']['n_mels'][()]
            }
            
            # Get dataset IDs
            if ids is None:
                self.ids = list(h5['data'].keys())
            else:
                self.ids = ids
        
        self.max_len = max_len
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_file, 'r')
        
        sample_id = self.ids[idx]
        sample_data = self.h5['data'][sample_id]
        
        # Get data
        phonemes = sample_data['phonemes'][:]
        midi = sample_data['midi'][:]
        f0 = sample_data['f0'][:]
        mel_spec = sample_data['mel_spectrogram'][:]
        
        # Handle sequence length
        seq_len = len(phonemes)
        if self.max_len and seq_len > self.max_len:
            # Random crop for training
            start = random.randint(0, seq_len - self.max_len)
            end = start + self.max_len
            
            phonemes = phonemes[start:end]
            midi = midi[start:end]
            f0 = f0[start:end]
            mel_spec = mel_spec[start:end]
            seq_len = self.max_len
        
        # Normalize features
        f0_norm = self._normalize_f0(f0)
        mel_norm = self._normalize_mel(mel_spec)
        
        # Convert to tensors
        phonemes_tensor = torch.tensor(phonemes, dtype=torch.long)
        midi_tensor = torch.tensor(midi, dtype=torch.long)
        f0_tensor = torch.tensor(f0_norm, dtype=torch.float)
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float)
        
        return {
            'id': sample_id,
            'phonemes': phonemes_tensor,
            'midi': midi_tensor,
            'f0': f0_tensor,
            'mel_spectrogram': mel_tensor,
            'length': seq_len
        }
    
    def _normalize_f0(self, f0):
        """Normalize F0 values to [0, 1] range."""
        # Log scale and normalize
        f0_log = np.log(np.maximum(f0, 1e-5))
        return (f0_log - np.log(10)) / (np.log(800) - np.log(10))
    
    def _normalize_mel(self, mel_spec):
        """Normalize mel spectrogram."""
        # Assuming mel_spec is already in dB scale
        return (mel_spec + 80) / 80  # Typical range for dB mel: [-80, 0]
    
    def close(self):
        if self.h5 is not None:
            self.h5.close()
            self.h5 = None

class SVSDataModule(pl.LightningDataModule):
    """Data module for Singing Voice Synthesis."""
    
    def __init__(self, h5_file, batch_size=16, num_workers=4, 
                 train_split=0.9, max_len=None):
        super().__init__()
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.max_len = max_len
        self.train_ids = None
        self.val_ids = None
        
    def setup(self, stage=None):
        # Open h5 file to get all sample IDs
        with h5py.File(self.h5_file, 'r') as h5:
            all_ids = list(h5['data'].keys())
        
        # Shuffle and split
        random.shuffle(all_ids)
        train_size = int(len(all_ids) * self.train_split)
        
        self.train_ids = all_ids[:train_size]
        self.val_ids = all_ids[train_size:]
    
    def train_dataloader(self):
        train_dataset = SVSDataset(self.h5_file, self.train_ids, self.max_len)
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=SVSDataModule.collate_fn
        )
    
    def val_dataloader(self):
        val_dataset = SVSDataset(self.h5_file, self.val_ids, self.max_len)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=SVSDataModule.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for variable length sequences."""
        # Sort by length for packing
        batch.sort(key=lambda x: x['length'], reverse=True)
        
        # Get batch info
        ids = [sample['id'] for sample in batch]
        lengths = torch.tensor([sample['length'] for sample in batch])
        max_len = lengths[0].item()
        
        # Prepare padded tensors
        phonemes = torch.zeros(len(batch), max_len, dtype=torch.long)
        midi = torch.zeros(len(batch), max_len, dtype=torch.long)
        f0 = torch.zeros(len(batch), max_len, dtype=torch.float)
        mel_specs = torch.zeros(len(batch), max_len, batch[0]['mel_spectrogram'].size(1), dtype=torch.float)
        
        # Create mask (1 for real data, 0 for padding)
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        
        # Fill in tensors
        for i, sample in enumerate(batch):
            seq_len = sample['length']
            phonemes[i, :seq_len] = sample['phonemes']
            midi[i, :seq_len] = sample['midi']
            f0[i, :seq_len] = sample['f0']
            mel_specs[i, :seq_len, :] = sample['mel_spectrogram']
            mask[i, :seq_len] = 1
        
        return {
            'ids': ids,
            'phonemes': phonemes,
            'midi': midi,
            'f0': f0,
            'mel_spectrogram': mel_specs,
            'lengths': lengths,
            'mask': mask
        }