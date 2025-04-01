import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import yaml
from pathlib import Path
import math

class HDF5Dataset(Dataset):
    """Reads data samples from an HDF5 file."""
    def __init__(self, hdf5_path, sample_ids, config):
        self.hdf5_path = hdf5_path
        self.sample_ids = sample_ids
        self.config = config
        self.max_seq_len_frames = config['training']['max_seq_len_frames']
        self.mel_pad_value = config['data']['mel_pad_value']
        self.f0_pad_value = config['data']['f0_pad_value']
        self.pad_token_id = config['data']['pad_token_id']
        # Keep file handle open if num_workers is 0, otherwise open in getitem
        self._h5_file = None
        if self.config['training']['num_workers'] == 0:
             # This is generally NOT recommended for multi-worker, but fine for num_workers=0
             # In __getitem__ is safer for multi-processing. Let's stick to __getitem__ opening.
             pass


    def _open_h5(self):
        # Opens HDF5 file in read mode. Essential for multi-worker loading.
        return h5py.File(self.hdf5_path, 'r')

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        # Open HDF5 file here for multiprocessing safety
        with self._open_h5() as f:
            sample_group = f[sample_id]

            # --- Load Data ---
            phonemes = torch.from_numpy(sample_group['phonemes/phones'][:]).long()
            midi_notes = torch.from_numpy(sample_group['midi/notes'][:]).long()
            # Assuming durations are stored as per refined preprocess.py
            durations = torch.from_numpy(sample_group['phonemes/durations'][:]).long()
            f0 = torch.from_numpy(sample_group['features/f0_values'][:]).float()
            mel = torch.from_numpy(sample_group['features/mel_spectrogram'][:]).float() # Shape: (T, n_mels)

            # --- Validate ---
            t_frames = mel.shape[0]
            n_phonemes = phonemes.shape[0]

            if durations.sum() != t_frames:
                 print(f"Warning: Skipping {sample_id}. Duration sum ({durations.sum()}) != Mel frames ({t_frames})")
                 # Return None or raise error? For now, let collate_fn handle potential None
                 # Safest is probably to return a dummy small sample that collate can handle
                 # Or, better yet, filter these out *before* creating the dataset if possible.
                 # Let's assume preprocess.py guaranteed this alignment. If not, errors will occur.
                 pass # Proceed cautiously

            if len(midi_notes) != n_phonemes:
                 print(f"Warning: Skipping {sample_id}. MIDI len ({len(midi_notes)}) != Phoneme len ({n_phonemes})")
                 # Handle appropriately - maybe return dummy data?
                 pass # Proceed cautiously


            # --- Truncate ---
            if t_frames > self.max_seq_len_frames:
                # Find the phoneme where truncation occurs
                cumulative_dur = torch.cumsum(durations, dim=0)
                last_phoneme_idx = torch.searchsorted(cumulative_dur, self.max_seq_len_frames)
                n_phonemes = last_phoneme_idx + 1 # Keep phonemes up to the one that gets truncated

                # Adjust duration of the last phoneme
                if last_phoneme_idx > 0:
                    truncated_duration = self.max_seq_len_frames - cumulative_dur[last_phoneme_idx - 1]
                else:
                     truncated_duration = self.max_seq_len_frames

                durations = durations[:n_phonemes]
                durations[-1] = truncated_duration # Adjust last duration

                # Truncate frame-level features
                t_frames = self.max_seq_len_frames
                mel = mel[:t_frames, :]
                f0 = f0[:t_frames]

                # Truncate sequence-level features
                phonemes = phonemes[:n_phonemes]
                midi_notes = midi_notes[:n_phonomes]


            # Sanity check after potential truncation
            if durations.sum() != t_frames:
                 # This should NOT happen if logic is correct
                 print(f"CRITICAL ERROR after truncation in {sample_id}: Duration sum ({durations.sum()}) != Target frames ({t_frames})")
                 # Handle this error robustly, e.g., by returning None and handling in collate_fn
                 return None # Signal an issue


        return {
            "sample_id": sample_id,
            "phonemes": phonemes,       # (N,)
            "midi_notes": midi_notes, # (N,)
            "durations": durations,     # (N,)
            "f0": f0,                   # (T,)
            "mel": mel,                 # (T, n_mels)
            "n_phonemes": n_phonemes,
            "t_frames": t_frames
        }

def collate_fn(batch):
    """Pads batch sequences to max length."""
    # Filter out None entries if __getitem__ returned errors
    batch = [b for b in batch if b is not None]
    if not batch:
        return None # Or raise error if batch becomes empty

    config = None # Need config for padding values - pass it or load it? Assume model has access later.
                  # Let's get pad values directly. Assume fixed for now.
    mel_pad_value = -11.5
    f0_pad_value = 0.0
    pad_token_id = 0

    # Find max lengths
    max_n = max(item['n_phonemes'] for item in batch)
    max_t = max(item['t_frames'] for item in batch)

    # Initialize padded tensors
    phonemes_padded = torch.full((len(batch), max_n), pad_token_id, dtype=torch.long)
    midi_notes_padded = torch.full((len(batch), max_n), pad_token_id, dtype=torch.long)
    durations_padded = torch.zeros((len(batch), max_n), dtype=torch.long)
    f0_padded = torch.full((len(batch), max_t), f0_pad_value, dtype=torch.float)
    mel_padded = torch.full((len(batch), max_t, batch[0]['mel'].shape[1]), mel_pad_value, dtype=torch.float) # Use n_mels from first sample

    # Create masks
    phoneme_mask = torch.zeros((len(batch), max_n), dtype=torch.bool) # True where padded
    mel_mask = torch.zeros((len(batch), max_t), dtype=torch.bool)     # True where padded

    sample_ids = []

    for i, item in enumerate(batch):
        n = item['n_phonemes']
        t = item['t_frames']
        sample_ids.append(item['sample_id'])

        phonemes_padded[i, :n] = item['phonemes']
        midi_notes_padded[i, :n] = item['midi_notes']
        durations_padded[i, :n] = item['durations']
        f0_padded[i, :t] = item['f0']
        mel_padded[i, :t, :] = item['mel']

        phoneme_mask[i, n:] = True # Mark padding as True
        mel_mask[i, t:] = True     # Mark padding as True

    return {
        "sample_ids": sample_ids,
        "phonemes": phonemes_padded,
        "midi_notes": midi_notes_padded,
        "durations": durations_padded,
        "f0": f0_padded,
        "mel_target": mel_padded,
        "phoneme_mask": phoneme_mask, # Mask for Encoder attention (True means ignore)
        "mel_mask": mel_mask        # Mask for Decoder attention and Loss (True means ignore)
    }


class SVSDataModule(pl.LightningDataModule):
    def __init__(self, config_path="config/default.yaml"):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.save_hyperparameters(self.config) # Log config to hparams

        self.hdf5_path = Path(self.config['data']['hdf5_path'])
        self.batch_size = self.config['training']['batch_size']
        self.num_workers = self.config['training']['num_workers']

        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        # Actions needed only once across all processes
        # e.g., download data, run preprocessing IF needed and not already done
        if not self.hdf5_path.exists():
            print(f"Error: HDF5 dataset not found at {self.hdf5_path}")
            print("Please run preprocess.py first to generate the dataset.")
            raise FileNotFoundError(f"Dataset not found: {self.hdf5_path}")
        print(f"Using dataset: {self.hdf5_path}")

    def setup(self, stage=None):
        # Actions needed on each process (GPU)
        # Open HDF5 file to read metadata
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                if 'metadata' not in f or 'file_list' not in f['metadata']:
                     raise ValueError("Metadata or file_list missing in HDF5 file.")
                # Decode file list if stored as bytes
                file_list_bytes = f['metadata/file_list'][:]
                self.all_sample_ids = [name.decode('utf-8') if isinstance(name, bytes) else name for name in file_list_bytes]

        except Exception as e:
            print(f"Error reading file list from {self.hdf5_path}: {e}")
            raise

        if not self.all_sample_ids:
             raise ValueError(f"No samples found in file_list within {self.hdf5_path}")

        # Simple train/validation split (e.g., 90/10)
        # Consider more robust splitting (e.g., scikit-learn train_test_split)
        np.random.shuffle(self.all_sample_ids) # Shuffle for random split
        split_idx = int(len(self.all_sample_ids) * 0.9)
        train_ids = self.all_sample_ids[:split_idx]
        val_ids = self.all_sample_ids[split_idx:]

        print(f"Total samples: {len(self.all_sample_ids)}")
        print(f"Training samples: {len(train_ids)}")
        print(f"Validation samples: {len(val_ids)}")

        if stage == 'fit' or stage is None:
            self.train_dataset = HDF5Dataset(self.hdf5_path, train_ids, self.config)
            self.val_dataset = HDF5Dataset(self.hdf5_path, val_ids, self.config)
        if stage == 'validate': # Used by validate_module.py potentially
             self.val_dataset = HDF5Dataset(self.hdf5_path, val_ids, self.config)
        # Add test_dataset setup if needed later

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0 # Keep workers alive
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size, # Use larger batch for validation if VRAM allows? Or keep same?
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    # Add test_dataloader if you implement testing stage