import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import json
import os
from omegaconf import DictConfig, OmegaConf # Use OmegaConf for config handling

# --- Phoneme Map Handling ---
def build_phoneme_map(hdf5_path, sil_phonemes, unk_token='<UNK>', pad_token='<PAD>'):
    """Scans HDF5 file to build a phoneme map."""
    phonemes = set()
    print(f"Building phoneme map from: {hdf5_path}")
    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'metadata' not in f or 'file_list' not in f['metadata']:
                 raise ValueError("HDF5 file missing 'metadata/file_list'. Was preprocessing run correctly?")
            sample_ids = [s.decode('utf-8') for s in f['metadata']['file_list'][:]]
            print(f"Found {len(sample_ids)} samples listed.")
            for i, sample_id in enumerate(sample_ids):
                if i % 200 == 0: # Print progress
                    print(f"  Scanning sample {i}/{len(sample_ids)}...", end='\r')
                if sample_id in f and 'phonemes/phones' in f[sample_id]:
                    sample_phonemes = [p.decode('utf-8') for p in f[sample_id]['phonemes/phones'][:]]
                    phonemes.update(sample_phonemes)
                else:
                    print(f"\nWarning: Sample {sample_id} or its phonemes not found in HDF5.")
            print(f"\nFound {len(phonemes)} unique phonemes in dataset.")

    except FileNotFoundError:
        raise FileNotFoundError(f"HDF5 file not found at {hdf5_path}. Run preprocess.py first.")
    except Exception as e:
        raise RuntimeError(f"Error reading HDF5 file {hdf5_path}: {e}")

    # Create map, ensuring PAD is 0 and UNK exists
    phoneme_list = sorted(list(phonemes))
    phoneme_to_id = {pad_token: 0}
    if unk_token not in phoneme_list:
        phoneme_to_id[unk_token] = 1
        current_id = 2
    else:
        phoneme_to_id[unk_token] = 1
        phoneme_list.remove(unk_token) # Ensure it gets id 1
        current_id = 2

    for p in phoneme_list:
        if p != pad_token: # Should not happen if PAD is not in dataset phonemes
             if p not in phoneme_to_id: # Avoid overwriting PAD/UNK if they were in data
                phoneme_to_id[p] = current_id
                current_id += 1

    print(f"Generated phoneme map with {len(phoneme_to_id)} entries (including PAD, UNK).")
    # Add silence phonemes if they weren't in the dataset but specified in config
    for sil in sil_phonemes:
        if sil not in phoneme_to_id:
            print(f"Warning: Specified silence phoneme '{sil}' not found in dataset phonemes. Adding to map.")
            phoneme_to_id[sil] = current_id
            current_id += 1

    return phoneme_to_id

def load_phoneme_map(map_path):
    """Loads phoneme map from JSON file."""
    print(f"Loading phoneme map from: {map_path}")
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            phoneme_to_id = json.load(f)
        if '<PAD>' not in phoneme_to_id or phoneme_to_id['<PAD>'] != 0:
            raise ValueError("Phoneme map must contain '<PAD>' with ID 0.")
        if '<UNK>' not in phoneme_to_id:
             raise ValueError("Phoneme map must contain '<UNK>'.")
        print(f"Loaded phoneme map with {len(phoneme_to_id)} entries.")
        return phoneme_to_id
    except FileNotFoundError:
        raise FileNotFoundError(f"Phoneme map file not found at {map_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding JSON from {map_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading phoneme map: {e}")

def save_phoneme_map(phoneme_map, map_path):
    """Saves phoneme map to JSON file."""
    print(f"Saving phoneme map to: {map_path}")
    try:
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        with open(map_path, 'w', encoding='utf-8') as f:
            json.dump(phoneme_map, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"Error saving phoneme map to {map_path}: {e}")


# --- Dataset ---
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, sample_ids, phoneme_to_id, pad_phoneme_id, unk_phoneme_id):
        self.hdf5_path = hdf5_path
        self.sample_ids = sample_ids
        self.phoneme_to_id = phoneme_to_id
        self.id_to_phoneme = {v: k for k, v in phoneme_to_id.items()} # For debugging/logging
        self.pad_phoneme_id = pad_phoneme_id
        self.unk_phoneme_id = unk_phoneme_id
        # Keep file open if not using multiple workers? Might cause issues.
        # Best practice for multiprocessing is often to open in __getitem__
        # self.h5_file = h5py.File(self.hdf5_path, 'r') # Keep open?

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        try:
            # Open file here for better multiprocessing safety
            with h5py.File(self.hdf5_path, 'r') as f:
                sample_group = f[sample_id]

                mel = torch.from_numpy(sample_group['features/mel_spectrogram'][:]).float()
                f0 = torch.from_numpy(sample_group['features/f0_values'][:]).float()

                # Ensure F0 is 1D (T,) or 2D (T, 1) -> make it (T,)
                if f0.ndim > 1:
                    f0 = f0.squeeze()
                # Handle potential NaN in F0 (replace with 0 or interpolate?)
                f0 = torch.nan_to_num(f0, nan=0.0)


                phoneme_strs = [p.decode('utf-8') for p in sample_group['phonemes/phones'][:]]
                phonemes = torch.tensor([self.phoneme_to_id.get(p, self.unk_phoneme_id) for p in phoneme_strs], dtype=torch.long)

                midi_notes = torch.from_numpy(sample_group['midi/notes'][:]).long()
                # Clamp MIDI notes to valid range (e.g., 0-127) if necessary
                midi_notes = torch.clamp(midi_notes, 0, 127)

                durations = torch.from_numpy(sample_group['phonemes/durations'][:]).long()

                # --- Data Consistency Checks (Optional but Recommended) ---
                T_mel = mel.shape[0]
                T_f0 = f0.shape[0]
                N_phonemes = phonemes.shape[0]
                N_midi = midi_notes.shape[0]
                N_durations = durations.shape[0]
                T_dur_sum = torch.sum(durations).item()

                if not (T_mel == T_f0):
                    print(f"Warning: Alignment mismatch in {sample_id}: Mel({T_mel}) != F0({T_f0}). Skipping sample.")
                    # Return None or raise error? For now, let's try returning None and handle in collate
                    # Or better: Handle this during preprocessing validation! This check here is redundant if preprocess is good.
                    # For robustness now, let's just assert, assuming preprocessing handled it.
                    assert T_mel == T_f0, f"Mel/F0 length mismatch in {sample_id}: {T_mel} vs {T_f0}"


                if not (N_phonemes == N_midi == N_durations):
                    print(f"Warning: Phoneme/MIDI/Duration count mismatch in {sample_id}: P({N_phonemes}), M({N_midi}), D({N_durations}).")
                    assert N_phonemes == N_midi == N_durations, f"Input sequence length mismatch in {sample_id}"

                if T_dur_sum != T_mel:
                     # This SHOULD NOT happen if preprocess.py validation is correct.
                     # If it does, the data is corrupted or preprocess logic failed.
                    print(f"CRITICAL WARNING: Duration sum mismatch in {sample_id}: sum(D)={T_dur_sum}, Mel(T)={T_mel}. Check preprocessing!")
                    # Attempt a fix? Dangerous. Best to fix preprocessing.
                    # Truncate/pad mel/f0? Adjust last duration? Let's assert for now.
                    assert T_dur_sum == T_mel, f"Sum of durations != Mel length in {sample_id}"
                # ----------------------------------------------------------

                return {
                    "sample_id": sample_id,
                    "mel": mel,         # (T, n_mels)
                    "f0": f0,           # (T,)
                    "phonemes": phonemes, # (N,)
                    "midi": midi_notes, # (N,)
                    "durations": durations, # (N,)
                    "mel_len": T_mel,   # Scalar T
                    "phone_len": N_phonemes # Scalar N
                }
        except KeyError as e:
            print(f"Error loading data for {sample_id}: Missing key {e}. Check HDF5 structure.")
            # Return None might cause issues in dataloader, maybe raise?
            raise KeyError(f"Missing data for {sample_id}: {e}") from e
        except Exception as e:
            print(f"Unexpected error loading {sample_id}: {e}")
            raise e # Re-raise other unexpected errors

    # def __del__(self):
    #     # Close the file if it was kept open
    #     if hasattr(self, 'h5_file') and self.h5_file:
    #         self.h5_file.close()


# --- Collation ---
def collate_fn(batch, pad_phoneme_id=0, pad_mel_value=0.0):
    """Pads sequences in a batch and creates masks."""
    # Filter out None items if __getitem__ returned None for bad samples
    batch = [b for b in batch if b is not None]
    if not batch:
        return None # Return None if the whole batch was bad

    # Find max lengths
    max_mel_len = max(item['mel_len'] for item in batch)
    max_phone_len = max(item['phone_len'] for item in batch)

    # Prepare lists for padding
    sample_ids = [item['sample_id'] for item in batch]
    mels = []
    f0s = []
    phonemes = []
    midis = []
    durations = []
    mel_lens = []
    phone_lens = []

    for item in batch:
        # Pad Mel: (T, n_mels) -> (max_T, n_mels)
        mel_pad_len = max_mel_len - item['mel_len']
        mels.append(torch.nn.functional.pad(item['mel'], (0, 0, 0, mel_pad_len), value=pad_mel_value))

        # Pad F0: (T,) -> (max_T,)
        f0_pad_len = max_mel_len - item['mel_len']
        f0s.append(torch.nn.functional.pad(item['f0'], (0, f0_pad_len), value=0.0)) # Pad F0 with 0

        # Pad Phonemes: (N,) -> (max_N,)
        phone_pad_len = max_phone_len - item['phone_len']
        phonemes.append(torch.nn.functional.pad(item['phonemes'], (0, phone_pad_len), value=pad_phoneme_id))

        # Pad MIDI: (N,) -> (max_N,)
        midi_pad_len = max_phone_len - item['phone_len']
        midis.append(torch.nn.functional.pad(item['midi'], (0, midi_pad_len), value=0)) # Pad MIDI with 0

        # Pad Durations: (N,) -> (max_N,)
        duration_pad_len = max_phone_len - item['phone_len']
        durations.append(torch.nn.functional.pad(item['durations'], (0, duration_pad_len), value=0)) # Pad durations with 0

        mel_lens.append(item['mel_len'])
        phone_lens.append(item['phone_len'])

    # Stack into batch tensors
    batch_dict = {
        "sample_ids": sample_ids,
        "mels": torch.stack(mels),             # (B, max_T, n_mels)
        "f0": torch.stack(f0s),               # (B, max_T)
        "phonemes": torch.stack(phonemes),    # (B, max_N)
        "midi": torch.stack(midis),           # (B, max_N)
        "durations": torch.stack(durations),  # (B, max_N)
        "mel_lens": torch.tensor(mel_lens, dtype=torch.long),       # (B,) - Original lengths
        "phone_lens": torch.tensor(phone_lens, dtype=torch.long)    # (B,) - Original lengths
    }

    # --- Create Masks ---
    # Input mask (phonemes/midi/durations): True for non-padded elements
    batch_dict["input_mask"] = (batch_dict["phonemes"] != pad_phoneme_id) # (B, max_N)

    # Output mask (mel/f0): True for non-padded frames
    # Create range tensor (0, 1, ..., max_T-1) and compare with lengths
    mel_len_indices = torch.arange(max_mel_len).unsqueeze(0).expand(len(batch), -1) # (B, max_T)
    batch_dict["output_mask"] = (mel_len_indices < batch_dict["mel_lens"].unsqueeze(1)) # (B, max_T)

    return batch_dict


# --- Lightning DataModule ---
class SVSDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.hdf5_path = config.data.hdf5_path
        self.batch_size = config.training.batch_size
        self.num_workers = config.training.num_workers
        self.train_val_split = config.data.train_val_split
        self.pad_phoneme_id = config.preprocessing.pad_phoneme_id
        self.unk_phoneme_id = config.preprocessing.unk_phoneme_id

        self.train_ids = None
        self.val_ids = None
        self.dataset_train = None
        self.dataset_val = None
        self.phoneme_to_id = None
        self.id_to_phoneme = None
        self._prepare_phoneme_map()
        # Update vocab size in config *after* map is built
        self.config.model.vocab_size = len(self.phoneme_to_id)

    def _prepare_phoneme_map(self):
        map_path = self.config.preprocessing.phoneme_map_path
        if map_path and os.path.exists(map_path):
            self.phoneme_to_id = load_phoneme_map(map_path)
        else:
            self.phoneme_to_id = build_phoneme_map(
                self.hdf5_path,
                self.config.preprocessing.sil_phonemes,
                unk_token='<UNK>',
                pad_token='<PAD>'
            )
            # Ensure PAD and UNK have correct IDs after building
            assert self.phoneme_to_id.get('<PAD>') == self.pad_phoneme_id, "PAD ID mismatch after build"
            assert '<UNK>' in self.phoneme_to_id, "UNK token missing after build"
            self.unk_phoneme_id = self.phoneme_to_id['<UNK>'] # Update UNK ID from map

            if map_path: # Save if path was specified but file didn't exist
                 try:
                     # Try saving map next to hdf5 if path is null/relative
                     if not os.path.isabs(map_path) and not os.path.dirname(map_path):
                         save_path = os.path.join(os.path.dirname(self.hdf5_path), "phoneme_map.json")
                     else:
                         save_path = map_path
                     save_phoneme_map(self.phoneme_to_id, save_path)
                 except Exception as e:
                     print(f"Warning: Could not save generated phoneme map: {e}")

        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}


    def prepare_data(self):
        # Called only on 1 GPU. Good place to download/preprocess data
        # In our case, preprocessing is separate (preprocess.py).
        # We just need to ensure the HDF5 file exists.
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at {self.hdf5_path}. Run preprocess.py")
        # Phoneme map is handled in __init__/load/build

    def setup(self, stage=None):
        # Called on each GPU separately - stage is 'fit', 'validate', 'test', 'predict'
        # Read sample IDs from HDF5
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                 if 'metadata' not in f or 'file_list' not in f['metadata']:
                     raise ValueError("HDF5 file missing 'metadata/file_list'.")
                 all_sample_ids = [s.decode('utf-8') for s in f['metadata']['file_list'][:]]
        except Exception as e:
             raise RuntimeError(f"Error reading sample IDs from {self.hdf5_path}: {e}")

        if not all_sample_ids:
            raise ValueError(f"No sample IDs found in metadata of {self.hdf5_path}. Is the dataset empty?")

        # Split train/val
        num_samples = len(all_sample_ids)
        num_train = int(num_samples * self.train_val_split)
        num_val = num_samples - num_train

        # Ensure reproducibility of split
        np.random.seed(self.config.seed) # Use the global seed
        shuffled_indices = np.random.permutation(num_samples)
        train_indices = shuffled_indices[:num_train]
        val_indices = shuffled_indices[num_train:]

        self.train_ids = [all_sample_ids[i] for i in train_indices]
        self.val_ids = [all_sample_ids[i] for i in val_indices]

        print(f"Total samples: {num_samples}, Train: {len(self.train_ids)}, Val: {len(self.val_ids)}")

        # Create datasets
        common_args = {
            'hdf5_path': self.hdf5_path,
            'phoneme_to_id': self.phoneme_to_id,
            'pad_phoneme_id': self.pad_phoneme_id,
            'unk_phoneme_id': self.unk_phoneme_id
        }
        if stage == 'fit' or stage is None:
            self.dataset_train = HDF5Dataset(sample_ids=self.train_ids, **common_args)
            self.dataset_val = HDF5Dataset(sample_ids=self.val_ids, **common_args)
        if stage == 'validate' or stage == 'test': # Use val set for testing too in this PoC
             self.dataset_val = HDF5Dataset(sample_ids=self.val_ids, **common_args)
        # Add stage == 'predict' if needed

    def _collate_wrapper(self, batch):
        """Wrapper for collate_fn that can be properly pickled."""
        return collate_fn(batch, self.pad_phoneme_id)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,  # Use class method instead of lambda
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_wrapper,  # Use class method instead of lambda
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False
        )

    # def test_dataloader(self):
    #     # If you have a separate test set
    #     # return DataLoader(self.dataset_test, ...)
    #     return self.val_dataloader() # Use val set for testing in this PoC