import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import calculate_target_frames, pad_or_truncate # Import utility functions
import logging

logger = logging.getLogger(__name__)

class MelDataset(Dataset):
    """
    PyTorch Dataset for loading Mel Spectrograms from an HDF5 file created
    by preprocess.py. Handles padding/truncating to a fixed length.
    """
    def __init__(self, h5_path, config):
        super().__init__()
        self.h5_path = h5_path
        self.config = config
        self.padding_value = config['audio']['padding_value']
        self.target_frames = calculate_target_frames(config)

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
            mel_spectrogram_db = sample_group['features']['mel_spectrogram'][:] # Load data into memory

            # Convert to PyTorch tensor
            mel_tensor = torch.from_numpy(mel_spectrogram_db).float() # Shape: (N_MELS, T_original)

            # Pad or truncate to the target length
            processed_mel = pad_or_truncate(mel_tensor, self.target_frames, self.padding_value)

            # Add channel dimension -> (1, N_MELS, T_target)
            processed_mel = processed_mel.unsqueeze(0)

            return processed_mel

        except KeyError as e:
            logger.error(f"KeyError accessing data for sample_id '{sample_id}' in {self.h5_path}: {e}")
            # Return a dummy tensor or raise an error, depending on desired behavior
            # Returning dummy allows training to continue but might skew results
            return torch.full((1, self.config['audio']['n_mels'], self.target_frames), self.padding_value, dtype=torch.float)
        except Exception as e:
            logger.error(f"Error loading sample_id '{sample_id}' from {self.h5_path}: {e}")
            return torch.full((1, self.config['audio']['n_mels'], self.target_frames), self.padding_value, dtype=torch.float)

    def __del__(self):
        # Ensure the HDF5 file is closed when the dataset object is destroyed
        if self.file_handle:
            try:
                self.file_handle.close()
                self.file_handle = None
            except Exception as e:
                # Can happen if file is already closed elsewhere
                logger.debug(f"Exception closing HDF5 file in dataset destructor: {e}")


# Example Usage (optional, for testing dataset)
# if __name__ == '__main__':
#     from utils import load_config
#     import os

#     # Create dummy config and HDF5 if they don't exist
#     DUMMY_CONFIG_PATH = "config/dummy_test.yaml"
#     DUMMY_H5_PATH = "data_raw/binary/dummy_dataset.h5"
#     if not os.path.exists(DUMMY_CONFIG_PATH):
#         os.makedirs(os.path.dirname(DUMMY_CONFIG_PATH), exist_ok=True)
#         dummy_cfg_data = {
#             'data': {'h5_path': DUMMY_H5_PATH},
#             'audio': {
#                 'sample_rate': 22050, 'n_fft': 1024, 'hop_length': 256,
#                 'n_mels': 80, 'target_duration_sec': 5, 'padding_value': -100.0
#             }
#         }
#         with open(DUMMY_CONFIG_PATH, 'w') as f: yaml.dump(dummy_cfg_data, f)

#     if not os.path.exists(DUMMY_H5_PATH):
#          os.makedirs(os.path.dirname(DUMMY_H5_PATH), exist_ok=True)
#          with h5py.File(DUMMY_H5_PATH, 'w') as f:
#             md = f.create_group('metadata')
#             md.create_dataset('file_list', data=['sample1', 'sample2'], dtype=h5py.string_dtype(encoding='utf-8'))
#             s1 = f.create_group('sample1')
#             feat1 = s1.create_group('features')
#             feat1.create_dataset('mel_spectrogram', data=np.random.rand(80, 500) * -80) # Shorter
#             s2 = f.create_group('sample2')
#             feat2 = s2.create_group('features')
#             feat2.create_dataset('mel_spectrogram', data=np.random.rand(80, 1000) * -80) # Longer

#     print("Testing MelDataset...")
#     try:
#         config = load_config(DUMMY_CONFIG_PATH)
#         dataset = MelDataset(config['data']['h5_path'], config)
#         print(f"Dataset length: {len(dataset)}")

#         dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#         batch = next(iter(dataloader))

#         print(f"Batch shape: {batch.shape}")
#         print(f"Expected shape: (Batch Size, 1, N_MELS, Target Frames)")
#         expected_frames = calculate_target_frames(config)
#         print(f"Expected frames: {expected_frames}")
#         print(f"Actual frames in batch: {batch.shape[-1]}")
#         assert batch.shape == (2, 1, config['audio']['n_mels'], expected_frames)
#         print("Dataset and DataLoader test passed!")

#     except Exception as e:
#         print(f"Dataset test failed: {e}")
#     finally:
#         # Clean up dummy files
#         # if os.path.exists(DUMMY_CONFIG_PATH): os.remove(DUMMY_CONFIG_PATH)
#         # if os.path.exists(DUMMY_H5_PATH): os.remove(DUMMY_H5_PATH)
#         pass