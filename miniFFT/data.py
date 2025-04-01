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
    def __init__(self, hdf5_path, sample_ids, config, phone_map=None):
        self.hdf5_path = hdf5_path
        self.sample_ids = sample_ids
        self.config = config
        self.phone_map = phone_map  # Added phone_map parameter
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

        try:
            # Open HDF5 file here for multiprocessing safety
            with self._open_h5() as f:
                sample_group = f[sample_id]

                # --- Load Data ---
                if self.phone_map:
                    # Load phonemes as strings and map to IDs using phone_map
                    phones_bytes = sample_group['phonemes/phones'][:]
                    phonemes = []
                    for phone_bytes in phones_bytes:
                        try:
                            phone = phone_bytes.decode('utf-8')
                            phonemes.append(self.phone_map.get(phone, self.pad_token_id))
                        except UnicodeDecodeError:
                            phonemes.append(self.pad_token_id)
                    phonemes = torch.tensor(phonemes, dtype=torch.long)
                else:
                    # Fall back to direct conversion (which might not work correctly)
                    try:
                        phonemes = torch.from_numpy(sample_group['phonemes/phones'][:]).long()
                    except Exception as e:
                        print(f"Error converting phonemes in sample {sample_id}: {e}")
                        phones_bytes = sample_group['phonemes/phones'][:]
                        phonemes = torch.tensor([self.pad_token_id + 1 for _ in range(len(phones_bytes))], dtype=torch.long)
                
                midi_notes = torch.from_numpy(sample_group['midi/notes'][:]).long()
                # Assuming durations are stored as per refined preprocess.py
                durations = torch.from_numpy(sample_group['phonemes/durations'][:]).long()
                f0 = torch.from_numpy(sample_group['features/f0_values'][:]).float()
                mel = torch.from_numpy(sample_group['features/mel_spectrogram'][:]).float() # Shape: (T, n_mels)

                # --- Validate ---
                t_frames = mel.shape[0]
                n_phonemes = phonemes.shape[0]

                if durations.sum() != t_frames:
                    print(f"Warning: Mismatch in sample {sample_id}. Duration sum ({durations.sum()}) != Mel frames ({t_frames})")
                    # Let's attempt to fix the durations to match the number of frames
                    if n_phonemes > 0:
                        total_frames_diff = int(t_frames - durations.sum())
                        # Distribute the difference across phonemes proportionally
                        if total_frames_diff != 0:
                            durations_float = durations.float()
                            durations_weight = durations_float / durations_float.sum()
                            frames_to_add = durations_weight * total_frames_diff
                            # Apply corrections, ensuring integers
                            durations = torch.maximum(torch.ones_like(durations), 
                                                    (durations_float + frames_to_add).round().long())
                            # Final adjustment to exactly match total
                            remaining_diff = t_frames - durations.sum()
                            if remaining_diff != 0 and len(durations) > 0:
                                durations[-1] += remaining_diff
                    
                    # Validate again after correction
                    if durations.sum() != t_frames:
                        print(f"Failed to correct durations for {sample_id}. Skipping sample.")
                        return None

                if len(midi_notes) != n_phonemes:
                    print(f"Warning: MIDI notes length mismatch in sample {sample_id}. MIDI len ({len(midi_notes)}) != Phoneme len ({n_phonemes})")
                    # Try to fix by padding or truncating
                    if len(midi_notes) < n_phonemes:
                        # Pad with zeros
                        padding = torch.zeros(n_phonemes - len(midi_notes), dtype=torch.long)
                        midi_notes = torch.cat([midi_notes, padding])
                    else:
                        # Truncate
                        midi_notes = midi_notes[:n_phonemes]

                # --- Truncate ---
                if t_frames > self.max_seq_len_frames:
                    # Find the phoneme where truncation occurs
                    cumulative_dur = torch.cumsum(durations, dim=0)
                    last_phoneme_idx = torch.searchsorted(cumulative_dur, self.max_seq_len_frames)
                    
                    # Ensure valid index and at least one phoneme
                    if last_phoneme_idx >= len(durations):
                        last_phoneme_idx = len(durations) - 1
                    if last_phoneme_idx < 0:
                        last_phoneme_idx = 0
                        
                    n_phonemes = last_phoneme_idx + 1  # Keep phonemes up to the one that gets truncated

                    # Adjust duration of the last phoneme
                    if last_phoneme_idx > 0:
                        prev_sum = cumulative_dur[last_phoneme_idx - 1]
                        truncated_duration = self.max_seq_len_frames - prev_sum
                    else:
                        truncated_duration = self.max_seq_len_frames

                    durations = durations[:n_phonemes]
                    
                    # Make sure we have at least one phoneme
                    if len(durations) == 0:
                        durations = torch.ones(1, dtype=torch.long) * self.max_seq_len_frames
                        phonemes = torch.tensor([self.pad_token_id], dtype=torch.long)
                        midi_notes = torch.tensor([self.pad_token_id], dtype=torch.long)
                        n_phonemes = 1
                    else:
                        durations[-1] = truncated_duration  # Adjust last duration

                    # Truncate frame-level features
                    t_frames = self.max_seq_len_frames
                    mel = mel[:t_frames, :]
                    f0 = f0[:t_frames]

                    # Truncate sequence-level features
                    phonemes = phonemes[:n_phonemes]
                    midi_notes = midi_notes[:n_phonemes]

                # Sanity check after potential truncation
                if durations.sum() != t_frames:
                    print(f"CRITICAL ERROR after truncation in {sample_id}: Duration sum ({durations.sum()}) != Target frames ({t_frames})")
                    # Try to force fix the durations
                    if len(durations) > 0:
                        durations[-1] = durations[-1] + (t_frames - durations.sum())
                        # If still not matching, force create a valid sample
                        if durations.sum() != t_frames:
                            print(f"Creating fallback sample for {sample_id} with minimum phonemes and forced durations")
                            n_phonemes = 1
                            phonemes = torch.tensor([self.pad_token_id], dtype=torch.long)
                            midi_notes = torch.tensor([self.pad_token_id], dtype=torch.long)
                            durations = torch.tensor([t_frames], dtype=torch.long)
                    else:
                        # If no durations at all, create a dummy one
                        print(f"Creating fallback sample for {sample_id} with minimum phonemes")
                        n_phonemes = 1
                        phonemes = torch.tensor([self.pad_token_id], dtype=torch.long)
                        midi_notes = torch.tensor([self.pad_token_id], dtype=torch.long)
                        durations = torch.tensor([t_frames], dtype=torch.long)

            return {
                "sample_id": sample_id,
                "phonemes": phonemes,      # (N,)
                "midi_notes": midi_notes,  # (N,)
                "durations": durations,    # (N,)
                "f0": f0,                  # (T,)
                "mel": mel,                # (T, n_mels)
                "n_phonemes": n_phonemes,
                "t_frames": t_frames
            }
        
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            return None

def collate_fn(batch):
    """Pads batch sequences to max length."""
    # Filter out None entries if __getitem__ returned errors
    batch = [b for b in batch if b is not None]
    if not batch:
        print("WARNING: Batch became empty after filtering out None samples. Creating a dummy batch for forward continuation.")
        # Return a minimal dummy batch instead of None to prevent errors downstream
        # Use small dimensions to minimize computation waste
        n_mels = 80  # Default n_mels value from config
        dummy_batch = {
            "sample_ids": ["dummy_sample"],
            "phonemes": torch.full((1, 1), 0, dtype=torch.long),
            "midi_notes": torch.full((1, 1), 0, dtype=torch.long),
            "durations": torch.ones((1, 1), dtype=torch.long),
            "f0": torch.zeros((1, 1), dtype=torch.float),
            "mel_target": torch.full((1, 1, n_mels), -11.5, dtype=torch.float),
            "phoneme_mask": torch.zeros((1, 1), dtype=torch.bool),
            "mel_mask": torch.zeros((1, 1), dtype=torch.bool)
        }
        return dummy_batch

    # Get n_mels from first sample for dimensionality
    n_mels = batch[0]['mel'].shape[1]
    
    # Use consistent pad values
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
    mel_padded = torch.full((len(batch), max_t, n_mels), mel_pad_value, dtype=torch.float)

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
        
        # DO NOT save all hyperparameters to avoid conflicts with the model
        # Instead, store config directly as an attribute
        # self.save_hyperparameters(self.config)  # REMOVE THIS LINE

        self.hdf5_path = Path(self.config['data']['hdf5_path'])
        self.batch_size = self.config['training']['batch_size']
        self.num_workers = self.config['training']['num_workers']

        self.train_dataset = None
        self.val_dataset = None
        self.phone_map = None

    def prepare_data(self):
        # Actions needed only once across all processes
        # e.g., download data, run preprocessing IF needed and not already done
        if not self.hdf5_path.exists():
            print(f"Error: HDF5 dataset not found at {self.hdf5_path}")
            print("Please run preprocess.py first to generate the dataset.")
            raise FileNotFoundError(f"Dataset not found: {self.hdf5_path}")
        print(f"Using dataset: {self.hdf5_path}")

    def _build_phone_map(self):
        """Builds a mapping from phoneme strings to integer IDs."""
        phone_set = set(['<PAD>'])  # Start with PAD token
        print("Building phoneme vocabulary...")
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                for sample_id in self.all_sample_ids:
                    if sample_id in f:
                        sample_group = f[sample_id]
                        if 'phonemes' in sample_group and 'phones' in sample_group['phonemes']:
                            phones_bytes = sample_group['phonemes']['phones'][:]
                            for phone_bytes in phones_bytes:
                                try:
                                    phone = phone_bytes.decode('utf-8')
                                    phone_set.add(phone)
                                except UnicodeDecodeError:
                                    print(f"Warning: Could not decode phoneme in sample {sample_id}")
        except Exception as e:
            print(f"Error building phoneme vocabulary: {e}")
            return None

        # Create mapping from phonemes to IDs (reserve 0 for padding)
        phone_map = {phone: i for i, phone in enumerate(sorted(phone_set))}
        
        print(f"Built phoneme vocabulary with {len(phone_map)} unique phones (including <PAD>)")
        
        # Save phone map for later reference
        phone_map_path = Path(self.hdf5_path).parent / "phone_map.yaml"
        with open(phone_map_path, 'w') as f:
            yaml.dump(phone_map, f)
        print(f"Saved phone map to {phone_map_path}")
        
        # Also save inverted map for easy ID to phoneme lookup
        inv_phone_map = {str(id): phone for phone, id in phone_map.items()}  # Convert keys to strings for YAML
        inv_phone_map_path = Path(self.hdf5_path).parent / "inv_phone_map.yaml"
        with open(inv_phone_map_path, 'w') as f:
            yaml.dump(inv_phone_map, f)
        print(f"Saved inverse phone map to {inv_phone_map_path}")
        
        return phone_map

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

        # Build phoneme vocabulary
        if self.phone_map is None:
            self.phone_map = self._build_phone_map()
            # Update config with actual vocabulary size for model initialization
            if self.phone_map:
                self.config['data']['phoneme_vocab_size'] = len(self.phone_map)
                print(f"Updated phoneme_vocab_size to {len(self.phone_map)}")
                # Instead of saving hyperparameters, just update the config directly

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
            self.train_dataset = HDF5Dataset(self.hdf5_path, train_ids, self.config, phone_map=self.phone_map)
            self.val_dataset = HDF5Dataset(self.hdf5_path, val_ids, self.config, phone_map=self.phone_map)
        if stage == 'validate': # Used by validate_module.py potentially
             self.val_dataset = HDF5Dataset(self.hdf5_path, val_ids, self.config, phone_map=self.phone_map)
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