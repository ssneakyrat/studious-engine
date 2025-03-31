import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import calculate_target_frames, pad_or_truncate  # Import utility functions
import logging
import random

logger = logging.getLogger(__name__)

class AudioDataAugmentation:
    """
    Provides data augmentation techniques for mel spectrograms.
    """
    def __init__(self, prob=0.5, time_mask_param=70, freq_mask_param=20, noise_scale=0.005):
        self.prob = prob
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.noise_scale = noise_scale
    
    def add_noise(self, mel_spectrogram):
        """Add small gaussian noise to the mel spectrogram"""
        noise = torch.randn_like(mel_spectrogram) * self.noise_scale * torch.abs(mel_spectrogram.mean())
        return mel_spectrogram + noise
    
    def time_mask(self, mel_spectrogram):
        """Apply time masking to mel spectrogram"""
        batch_size, channels, n_mels, time_steps = mel_spectrogram.shape
        
        # Don't mask too wide for very short spectrograms
        mask_param = min(self.time_mask_param, time_steps // 4)
        if mask_param <= 0:
            return mel_spectrogram
            
        result = mel_spectrogram.clone()
        for b in range(batch_size):
            # Apply 1-3 masks
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                t = random.randint(0, mask_param)
                t_zero = random.randint(0, time_steps - t)
                # Mask t consecutive time steps
                result[b, :, :, t_zero:t_zero+t] = mel_spectrogram.min()
        return result
        
    def freq_mask(self, mel_spectrogram):
        """Apply frequency masking to mel spectrogram"""
        batch_size, channels, n_mels, time_steps = mel_spectrogram.shape
        
        # Don't mask too wide for very small mel bins
        mask_param = min(self.freq_mask_param, n_mels // 4)
        if mask_param <= 0:
            return mel_spectrogram
            
        result = mel_spectrogram.clone()
        for b in range(batch_size):
            # Apply 1-2 masks
            num_masks = random.randint(1, 2)
            for _ in range(num_masks):
                f = random.randint(0, mask_param)
                f_zero = random.randint(0, n_mels - f)
                # Mask f consecutive mel bins
                result[b, :, f_zero:f_zero+f, :] = mel_spectrogram.min()
        return result
    
    def time_stretch(self, mel_spectrogram, padding_value):
        """Time stretching approximation on mel spectrogram"""
        batch_size, channels, n_mels, time_steps = mel_spectrogram.shape
        
        if time_steps < 10:  # Too short to stretch meaningfully
            return mel_spectrogram
            
        result = torch.zeros_like(mel_spectrogram)
        for b in range(batch_size):
            # Random stretch factor between 0.8 and 1.2
            stretch_factor = 0.8 + 0.4 * random.random()
            
            # Calculate new time steps
            new_time_steps = int(time_steps * stretch_factor)
            if new_time_steps < 2:  # Too short after stretching
                result[b] = mel_spectrogram[b]
                continue
                
            # Use interpolate for time stretching approximation
            stretched = torch.nn.functional.interpolate(
                mel_spectrogram[b].unsqueeze(0),
                size=(n_mels, new_time_steps),
                mode='bilinear',
                align_corners=False
            )[0]
            
            # Pad or truncate to original size
            if new_time_steps > time_steps:
                result[b] = stretched[:, :, :time_steps]
            else:
                # Pad with padding_value
                pad_size = time_steps - new_time_steps
                padded = torch.nn.functional.pad(
                    stretched, (0, pad_size), mode='constant', value=padding_value
                )
                result[b] = padded
                
        return result
    
    def apply(self, mel_spectrogram, padding_value):
        """Apply random augmentations based on probability"""
        if random.random() < self.prob:
            mel_spectrogram = self.add_noise(mel_spectrogram)
        
        if random.random() < self.prob:
            mel_spectrogram = self.time_mask(mel_spectrogram)
            
        if random.random() < self.prob:
            mel_spectrogram = self.freq_mask(mel_spectrogram)
            
        if random.random() < self.prob:
            mel_spectrogram = self.time_stretch(mel_spectrogram, padding_value)
            
        return mel_spectrogram

class MelDataset(Dataset):
    """
    PyTorch Dataset for loading Mel Spectrograms from an HDF5 file.
    Includes optional data augmentation.
    """
    def __init__(self, h5_path, config, augment=True):
        super().__init__()
        self.h5_path = h5_path
        self.config = config
        self.padding_value = config['audio']['padding_value']
        self.target_frames = calculate_target_frames(config)
        self.augment = augment
        
        # Initialize augmentation if enabled
        if self.augment:
            print("Data augmentation enabled for MelDataset")
            self.augmentation = AudioDataAugmentation(
                prob=0.5,  # 50% chance to apply each augmentation
                time_mask_param=min(70, self.target_frames // 10),  # Max mask size as 10% of frames
                freq_mask_param=min(20, config['audio']['n_mels'] // 8),  # Max mask size as 1/8 of freq bins
                noise_scale=0.005  # Small noise for stability
            )
        
        # Additional statistics tracking - helps detect data inconsistencies
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.sum_value = 0
        self.count = 0
        self.stats_samples = 0

        self.file_handle = None # Initialize file handle
        try:
            # Open HDF5 file in read mode
            self.file_handle = h5py.File(self.h5_path, 'r')
            if 'metadata' not in self.file_handle or 'file_list' not in self.file_handle['metadata']:
                 raise ValueError("HDF5 file must contain 'metadata' group with 'file_list' dataset.")

            # Load the list of sample IDs (decode from bytes if needed)
            file_list_ds = self.file_handle['metadata']['file_list']
            if h5py.check_string_dtype(file_list_ds.dtype):
                 self.sample_ids = [s.decode('utf-8') for s in file_list_ds[:]]
            else:
                 # Assuming it's already strings or another handleable type
                 self.sample_ids = list(file_list_ds[:])

            if not self.sample_ids:
                logger.warning(f"No sample IDs found in metadata/file_list in {self.h5_path}.")

        except FileNotFoundError:
            logger.error(f"HDF5 dataset file not found at: {self.h5_path}")
            raise
        except Exception as e:
            logger.error(f"Error opening or reading metadata from HDF5 file {self.h5_path}: {e}")
            if self.file_handle:
                self.file_handle.close() # Close if opened before error
            raise

        print(f"Dataset initialized. Found {len(self.sample_ids)} samples in {self.h5_path}.")

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        if not self.file_handle:
            # Re-open if closed (e.g., after serialization for multiprocessing)
            # Note: This might be needed if num_workers > 0
            self.file_handle = h5py.File(self.h5_path, 'r')

        sample_id = self.sample_ids[idx]
        try:
            # Access the sample group and load the mel spectrogram
            sample_group = self.file_handle[sample_id]
            mel_spectrogram_db = sample_group['features']['mel_spectrogram'][:]  # Load data into memory

            # Convert to PyTorch tensor
            mel_tensor = torch.from_numpy(mel_spectrogram_db).float()  # Shape: (N_MELS, T_original)

            # Pad or truncate to the target length
            processed_mel = pad_or_truncate(mel_tensor, self.target_frames, self.padding_value)

            # Add channel dimension -> (1, N_MELS, T_target)
            processed_mel = processed_mel.unsqueeze(0)

            # Apply data normalization - helps stabilize training
            # Normalize within a reasonable dB range
            # Assuming mel specs typically range from -80 to 0 dB
            # Set floor at padding_value if it's reasonable, otherwise use -80dB
            floor_db = self.padding_value if self.padding_value > -100 else -80
            processed_mel = torch.clamp(processed_mel, min=floor_db)
            
            # Scale to [0, 1] range for network stability
            # Adjust scaling based on audio characteristics
            processed_mel = (processed_mel - floor_db) / (0 - floor_db)
            
            # Track statistics for the first 100 samples to detect data issues
            if self.stats_samples < 100:
                curr_min = processed_mel.min().item()
                curr_max = processed_mel.max().item()
                curr_mean = processed_mel.mean().item()
                
                self.min_value = min(self.min_value, curr_min)
                self.max_value = max(self.max_value, curr_max)
                self.sum_value += curr_mean
                self.count += 1
                self.stats_samples += 1
                
                if self.stats_samples == 100:
                    print(f"=== Mel spectrogram statistics (first 100 samples) ===")
                    print(f"Min value: {self.min_value:.4f}")
                    print(f"Max value: {self.max_value:.4f}")
                    print(f"Mean value: {self.sum_value/self.count:.4f}")
                    print(f"================================================")

            # Apply data augmentation for training if enabled
            if self.augment:
                processed_mel = self.augmentation.apply(processed_mel.unsqueeze(0), self.padding_value).squeeze(0)

            return processed_mel

        except KeyError as e:
            logger.error(f"KeyError accessing data for sample_id '{sample_id}' in {self.h5_path}: {e}")
            # Return a dummy tensor with normalized padding value
            floor_db = self.padding_value if self.padding_value > -100 else -80
            padding_normalized = (self.padding_value - floor_db) / (0 - floor_db)
            return torch.full((1, self.config['audio']['n_mels'], self.target_frames), 
                             padding_normalized, dtype=torch.float)
        except Exception as e:
            logger.error(f"Error loading sample_id '{sample_id}' from {self.h5_path}: {e}")
            floor_db = self.padding_value if self.padding_value > -100 else -80
            padding_normalized = (self.padding_value - floor_db) / (0 - floor_db)
            return torch.full((1, self.config['audio']['n_mels'], self.target_frames), 
                             padding_normalized, dtype=torch.float)

    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset object is destroyed
        if self.file_handle:
            try:
                self.file_handle.close()
                self.file_handle = None
            except Exception as e:
                # Can happen if file is already closed elsewhere
                logger.debug(f"Exception closing HDF5 file in dataset destructor: {e}")