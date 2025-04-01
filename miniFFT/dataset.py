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

                # Transpose the mel spectrogram from (n_mels, T) to (T, n_mels)
                mel = torch.from_numpy(sample_group['features/mel_spectrogram'][:]).float().transpose(0, 1)
                f0 = torch.from_numpy(sample_group['features/f0_values'][:]).float()

                # Ensure F0 is 1D (T,) or 2D (T, 1) -> make it (T,)
                if f0.ndim > 1:
                    f0 = f0.squeeze()
                # Handle potential NaN in F0 (replace with 0)
                f0 = torch.nan_to_num(f0, nan=0.0)

                # Get phoneme data
                phoneme_strs = [p.decode('utf-8') for p in sample_group['phonemes/phones'][:]]
                phonemes = torch.tensor([self.phoneme_to_id.get(p, self.unk_phoneme_id) for p in phoneme_strs], dtype=torch.long)

                midi_notes = torch.from_numpy(sample_group['midi/notes'][:]).long()
                # Clamp MIDI notes to valid range (0-127)
                midi_notes = torch.clamp(midi_notes, 0, 127)

                durations = torch.from_numpy(sample_group['phonemes/durations'][:]).long()

                # --- Alignment Validation & Correction ---
                T_mel = mel.shape[0]  # Target length (frames)
                T_f0 = f0.shape[0]
                N_phonemes = phonemes.shape[0]
                N_midi = midi_notes.shape[0]
                N_durations = durations.shape[0]
                T_dur_sum = torch.sum(durations).item()

                # 1. Fix F0 length to match mel length
                if T_f0 != T_mel:
                    print(f"Fixing F0 length mismatch in {sample_id}: F0({T_f0}) -> Mel({T_mel})")
                    if T_f0 < T_mel:
                        # Pad F0
                        pad_len = T_mel - T_f0
                        f0 = torch.nn.functional.pad(f0, (0, pad_len), value=0.0)
                    else:
                        # Truncate F0
                        f0 = f0[:T_mel]
                    T_f0 = T_mel  # Update T_f0 to new length

                # 2. Ensure phoneme, MIDI, and duration arrays have consistent lengths
                if not (N_phonemes == N_midi == N_durations):
                    print(f"Warning: Input sequence length mismatch in {sample_id}: P({N_phonemes}), M({N_midi}), D({N_durations})")
                    # Find the minimum length to truncate to
                    min_len = min(N_phonemes, N_midi, N_durations)
                    phonemes = phonemes[:min_len]
                    midi_notes = midi_notes[:min_len]
                    durations = durations[:min_len]
                    N_phonemes = N_midi = N_durations = min_len
                    print(f"Truncated all to minimum length: {min_len}")

                # 3. Fix duration sum to match mel length
                if T_dur_sum != T_mel:
                    print(f"Fixing duration sum mismatch in {sample_id}: sum(D)={T_dur_sum}, Mel(T)={T_mel}")
                    
                    # Calculate difference
                    diff = T_mel - T_dur_sum
                    
                    if diff > 0:  # Need to add frames
                        # Find non-zero durations to adjust (prefer vowels or longer phonemes)
                        non_zero_indices = torch.nonzero(durations > 0).squeeze(-1)
                        if len(non_zero_indices) > 0:
                            # Distribute additional frames proportionally to existing duration
                            total_dur = float(durations[non_zero_indices].sum())
                            remaining = diff
                            
                            # First pass - distribute proportionally with floor
                            for idx in non_zero_indices:
                                proportion = durations[idx].item() / total_dur
                                add_frames = min(remaining, int(diff * proportion))
                                durations[idx] += add_frames
                                remaining -= add_frames
                            
                            # Second pass - add remaining frames one by one to largest durations
                            if remaining > 0:
                                sorted_indices = non_zero_indices[torch.argsort(durations[non_zero_indices], descending=True)]
                                for i in range(min(remaining, len(sorted_indices))):
                                    durations[sorted_indices[i % len(sorted_indices)]] += 1
                                    remaining -= 1
                        else:
                            # If no non-zero durations, add to the first phoneme
                            durations[0] += diff
                            
                    elif diff < 0:  # Need to remove frames
                        # Remove frames from longest durations first
                        remaining = -diff
                        while remaining > 0 and torch.any(durations > 1):  # Keep at least duration 1
                            max_idx = torch.argmax(durations)
                            remove = min(durations[max_idx] - 1, remaining)  # Keep at least 1 frame
                            durations[max_idx] -= remove
                            remaining -= remove
                        
                        # If we still need to remove and all durations are 1, then we need to remove phonemes
                        if remaining > 0:
                            print(f"  Warning: Need to remove {remaining} more frames but all durations are 1.")
                            # We could potentially remove entire phonemes, but that's risky
                    
                    # Verify the correction worked
                    T_dur_sum_new = torch.sum(durations).item()
                    
                    # If we still have a mismatch, apply a final force-fix
                    if T_dur_sum_new != T_mel:
                        print(f"  Warning: First pass duration adjustment failed. Applying forced correction.")
                        diff = T_mel - T_dur_sum_new
                        
                        # Find a suitable index for adjustment (prefer longer durations)
                        adjust_idx = 0
                        if len(durations) > 0:
                            non_zero_indices = torch.nonzero(durations > 0).squeeze(-1)
                            if len(non_zero_indices) > 0:
                                # Use the longest duration if removing frames, or distribute if adding
                                if diff < 0:  # Need to remove frames
                                    adjust_idx = torch.argmax(durations).item()
                                    # Make sure we don't go below 1
                                    removal = min(-diff, durations[adjust_idx].item() - 1)
                                    if removal > 0:
                                        durations[adjust_idx] -= removal
                                        diff += removal
                                
                                # If we still have frames to adjust, distribute among all non-zero durations
                                if diff != 0:
                                    for idx in non_zero_indices:
                                        if diff > 0:  # Add remaining frames
                                            durations[idx] += 1
                                            diff -= 1
                                            if diff == 0:
                                                break
                                        elif diff < 0 and durations[idx] > 1:  # Remove frames (keep at least 1)
                                            durations[idx] -= 1
                                            diff += 1
                                            if diff == 0:
                                                break
                        
                        # As a last resort, adjust the first/last phoneme
                        if diff != 0:
                            if len(durations) > 0:
                                # If adding frames, add to first phoneme
                                if diff > 0:
                                    durations[0] += diff
                                # If removing, try to remove from any phoneme with duration > 1
                                else:
                                    # Find any phoneme with duration > 1
                                    for i in range(len(durations)):
                                        if durations[i] > 1:
                                            remove = min(durations[i] - 1, -diff)
                                            durations[i] -= remove
                                            diff += remove
                                            if diff == 0:
                                                break
                                    
                                    # If we still couldn't fix it, we'll have to accept the mismatch
                                    if diff < 0:
                                        print(f"  Critical: Could not completely align durations. Mismatch: {diff}")
                        
                        # Final check
                        T_dur_sum_final = torch.sum(durations).item()
                        print(f"  Final duration sum: {T_dur_sum_final} (target: {T_mel})")
                        
                        # We'll allow it to continue without assertion,
                        # but log if there's still a mismatch after our best effort
                        if T_dur_sum_final != T_mel:
                            print(f"  Critical: Duration sum ({T_dur_sum_final}) still doesn't match Mel length ({T_mel}) for {sample_id}")
                            # Don't assert - allow the data to be returned with this warning

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
            raise KeyError(f"Missing data for {sample_id}: {e}") from e
        except Exception as e:
            print(f"Unexpected error loading {sample_id}: {e}")
            raise e # Re-raise other unexpected errors


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