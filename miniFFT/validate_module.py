import torch
import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import TensorBoardLogger # To log figures
import random

# Import necessary components from your project files
from dataset import SVSDataModule, HDF5Dataset, collate_fn, build_phoneme_map, load_phoneme_map
from model import FFTLightSVS, LengthRegulator # Assuming model parts are accessible

# --- Helper Function for TensorBoard Logging ---
# Note: This requires TensorBoard to be installed (`pip install tensorboard`)
# Create a dummy logger instance for validation plots
val_logger = TensorBoardLogger("logs", name="ValidationTests")
os.makedirs(val_logger.log_dir, exist_ok=True)

def log_figure_to_tb(tag, fig, step=0):
    """Logs a matplotlib figure to TensorBoard."""
    val_logger.experiment.add_figure(tag, fig, global_step=step)
    plt.close(fig) # Close figure to free memory
    print(f"Logged figure '{tag}' to TensorBoard logs: {val_logger.log_dir}")

# --- Test Cases ---

def test_dataset_loading(config: DictConfig):
    """Tests HDF5Dataset loading, collation, and visualizes one sample."""
    print("\n--- Testing Dataset Loading ---")
    try:
        data_module = SVSDataModule(config)
        data_module.prepare_data() # Ensure file exists
        data_module.setup('fit')   # Setup datasets and splits

        # Get a small batch from validation loader
        val_loader = data_module.val_dataloader()
        batch = next(iter(val_loader))

        if batch is None:
             print("Failed to retrieve a batch. Check dataset/collate function.")
             return

        print("Successfully loaded and collated one batch.")
        print("Batch Keys:", batch.keys())
        print("Sample IDs:", batch['sample_ids'][:4]) # Print first few sample IDs
        print("Mel shape:", batch['mels'].shape)
        print("F0 shape:", batch['f0'].shape)
        print("Phonemes shape:", batch['phonemes'].shape)
        print("MIDI shape:", batch['midi'].shape)
        print("Durations shape:", batch['durations'].shape)
        print("Mel Lens shape:", batch['mel_lens'].shape)
        print("Phone Lens shape:", batch['phone_lens'].shape)
        print("Input Mask shape:", batch['input_mask'].shape)
        print("Output Mask shape:", batch['output_mask'].shape)


        # --- Visualize one sample from the batch ---
        idx_to_vis = 0
        sample_id = batch['sample_ids'][idx_to_vis]
        mel_len = batch['mel_lens'][idx_to_vis].item()
        phone_len = batch['phone_lens'][idx_to_vis].item()

        mel = batch['mels'][idx_to_vis][:mel_len].cpu().numpy()
        f0 = batch['f0'][idx_to_vis][:mel_len].cpu().numpy()
        phoneme_ids = batch['phonemes'][idx_to_vis][:phone_len].cpu().numpy()
        midi = batch['midi'][idx_to_vis][:phone_len].cpu().numpy()
        durations = batch['durations'][idx_to_vis][:phone_len].cpu().numpy()

        # Convert phoneme IDs back to symbols
        id_to_phoneme = data_module.id_to_phoneme
        phoneme_syms = [id_to_phoneme.get(pid, '?') for pid in phoneme_ids]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle(f"Dataset Loading Test - Sample: {sample_id}")

        # Plot Mel
        img = librosa.display.specshow(mel.T, ax=axes[0], sr=config.preprocessing.sample_rate,
                                       hop_length=config.preprocessing.hop_length,
                                       fmin=config.preprocessing.fmin, fmax=config.preprocessing.fmax,
                                       x_axis='time', y_axis='mel')
        axes[0].set_title("Mel Spectrogram (Target)")
        #fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

        # Plot F0
        times = librosa.times_like(f0, sr=config.preprocessing.sample_rate, hop_length=config.preprocessing.hop_length)
        axes[1].plot(times, f0, label='F0', color='cyan')
        axes[1].set_title("F0 Contour")
        axes[1].set_ylabel("Frequency (Hz)")
        axes[1].legend()
        axes[1].grid(True, axis='y', linestyle=':')

        # Plot Phonemes, MIDI, Durations (textual representation on timeline)
        axes[2].set_title("Phonemes, MIDI Notes, Durations")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_yticks([]) # No y-axis ticks needed
        axes[2].set_ylim(0, 1) # Arbitrary y-limits for text placement
        current_time = 0.0
        time_scale = config.preprocessing.hop_length / config.preprocessing.sample_rate
        for i in range(phone_len):
            phone_dur_sec = durations[i] * time_scale
            center_time = current_time + phone_dur_sec / 2.0
            axes[2].text(center_time, 0.6, f"{phoneme_syms[i]}", ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black", lw=0.5))
            axes[2].text(center_time, 0.3, f"M:{midi[i]}", ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="black", lw=0.5))
            # Draw vertical lines for boundaries
            if i > 0:
                axes[2].axvline(current_time, color='gray', linestyle=':', linewidth=0.8)
            current_time += phone_dur_sec
        axes[2].set_xlim(0, times[-1] if len(times) > 0 else 1.0) # Match x-axis to others

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        log_figure_to_tb("Dataset_Loading_Sample", fig)

    except Exception as e:
        print(f"ERROR in test_dataset_loading: {e}")
        import traceback
        traceback.print_exc()

def test_alignment(config: DictConfig, num_samples_to_check=20):
    """Validates data alignment within the HDF5 file for a few random samples."""
    print("\n--- Testing Data Alignment in HDF5 ---")
    hdf5_path = config.data.hdf5_path
    issues_found = 0
    samples_checked = 0

    if not os.path.exists(hdf5_path):
        print(f"Error: HDF5 file not found at {hdf5_path}")
        return

    try:
        with h5py.File(hdf5_path, 'r') as f:
            if 'metadata' not in f or 'file_list' not in f['metadata']:
                 print("Error: HDF5 file missing 'metadata/file_list'.")
                 return
            all_sample_ids = [s.decode('utf-8') for s in f['metadata']['file_list'][:]]

            if not all_sample_ids:
                print("Error: No samples found in metadata.")
                return

            print(f"Found {len(all_sample_ids)} total samples. Checking {min(num_samples_to_check, len(all_sample_ids))} random samples.")

            # Select random samples
            ids_to_check = random.sample(all_sample_ids, min(num_samples_to_check, len(all_sample_ids)))

            for sample_id in ids_to_check:
                 samples_checked += 1
                 print(f"  Checking {sample_id}...")
                 sample_ok = True
                 try:
                    sample_group = f[sample_id]

                    # Check required datasets exist
                    req = {
                        'features': ['mel_spectrogram', 'f0_values'],
                        'phonemes': ['phones', 'durations', 'start_frames', 'end_frames'], # Assuming durations are stored
                        'midi': ['notes']
                    }
                    for group, datasets in req.items():
                        if group not in sample_group:
                             print(f"    ERROR: Missing group '{group}'")
                             sample_ok = False; break
                        for ds in datasets:
                             if ds not in sample_group[group]:
                                 print(f"    ERROR: Missing dataset '{group}/{ds}'")
                                 sample_ok = False; break
                        if not sample_ok: break
                    if not sample_ok: issues_found += 1; continue

                    # Get lengths
                    mel = sample_group['features/mel_spectrogram']
                    f0 = sample_group['features/f0_values']
                    phones = sample_group['phonemes/phones']
                    durs = sample_group['phonemes/durations']
                    midi = sample_group['midi/notes']

                    T_mel = mel.shape[0]
                    T_f0 = f0.shape[0]
                    N_phones = phones.shape[0]
                    N_durs = durs.shape[0]
                    N_midi = midi.shape[0]
                    T_dur_sum = np.sum(durs[:]) # Sum durations

                    # Perform checks
                    if T_mel != T_f0:
                        print(f"    ERROR: Mel/F0 length mismatch! Mel={T_mel}, F0={T_f0}")
                        sample_ok = False
                    if not (N_phones == N_durs == N_midi):
                        print(f"    ERROR: Input sequence length mismatch! Phones={N_phones}, Durs={N_durs}, MIDI={N_midi}")
                        sample_ok = False
                    if T_dur_sum != T_mel:
                        print(f"    ERROR: Sum of durations != Mel length! Sum(Dur)={T_dur_sum}, Mel={T_mel}")
                        print(f"           Durations: {durs[:5]}...") # Print first few durations
                        sample_ok = False

                    if not sample_ok:
                         issues_found += 1
                    else:
                         print("    OK.")


                 except KeyError as e:
                     print(f"    ERROR: Missing data key {e} in sample {sample_id}.")
                     issues_found += 1
                 except Exception as e:
                     print(f"    ERROR: Unexpected error checking {sample_id}: {e}")
                     issues_found += 1

    except Exception as e:
        print(f"Failed to open or process HDF5 file {hdf5_path}: {e}")
        return

    print("--- Alignment Test Summary ---")
    print(f"Checked {samples_checked} samples.")
    if issues_found == 0:
        print("All checked samples seem correctly aligned. Preprocessing likely OK.")
        print("NOTE: This does not check the *quality* of alignment, only internal consistency.")
    else:
        print(f"Found issues in {issues_found} out of {samples_checked} checked samples.")
        print("ACTION: Review your preprocess.py script, specifically the duration calculation and alignment checks before saving!")


def test_model_component(config: DictConfig, component_name: str):
    """Tests forward pass of a single model component with dummy data."""
    print(f"\n--- Testing Model Component: {component_name} ---")

    # Dummy data parameters
    B = 4  # Batch size
    N = 25 # Input sequence length (phonemes)
    T = 200 # Output sequence length (frames) - derived for regulator/decoder
    hidden_dim = config.model.hidden_dim
    n_mels = config.model.n_mels
    vocab_size = 50 # Dummy vocab size, use actual if map is loaded
    midi_vocab_size = config.model.midi_vocab_size
    pad_id = config.preprocessing.pad_phoneme_id
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load phoneme map to get actual vocab size if needed by component
    if config.model.vocab_size is None: # If not set by datamodule beforehand
        try:
             map_path = config.preprocessing.phoneme_map_path
             if map_path and os.path.exists(map_path):
                 p_map = load_phoneme_map(map_path)
             else:
                  # Need hdf5 path to build map, could be complex here
                  # For component test, let's just use a placeholder size
                  print("Warning: Phoneme map not loaded, using dummy vocab_size=50 for component test.")
                  p_map = {f"P{i}": i for i in range(vocab_size)}
                  p_map['<PAD>'] = pad_id
             config.model.vocab_size = len(p_map)
        except Exception as e:
             print(f"Warning: Failed to load/build phoneme map for vocab size: {e}. Using dummy size 50.")
             config.model.vocab_size = 50


    try:
        if component_name == 'encoder':
            # Prepare dummy input for encoder
            dummy_phonemes = torch.randint(1, config.model.vocab_size, (B, N), dtype=torch.long).to(device) # Avoid PAD ID 0 mostly
            dummy_midi = torch.randint(20, 100, (B, N), dtype=torch.long).to(device) # Realistic MIDI range
            dummy_input_mask = (dummy_phonemes != pad_id) # Example mask

            # Instantiate embeddings + projection + encoder blocks
            phoneme_embedding = nn.Embedding(config.model.vocab_size, config.model.encoder_embed_dim, padding_idx=pad_id).to(device)
            midi_embedding = nn.Embedding(midi_vocab_size, config.model.midi_embed_dim, padding_idx=0).to(device)
            encoder_input_dim = config.model.encoder_embed_dim + config.model.midi_embed_dim
            encoder_input_proj = nn.Linear(encoder_input_dim, hidden_dim).to(device) if encoder_input_dim != hidden_dim else nn.Identity().to(device)
            pos_encoding = PositionalEncoding(hidden_dim, config.model.dropout).to(device)
            encoder_blocks = nn.ModuleList([
                FFTBlock(hidden_dim, config.model.fft_n_heads, config.model.fft_conv_ffn_dim, config.model.fft_conv_kernel_size, config.model.dropout).to(device)
                for _ in range(config.model.fft_n_layers)
            ])

            # Forward pass
            phone_emb = phoneme_embedding(dummy_phonemes)
            midi_emb = midi_embedding(dummy_midi)
            enc_input = torch.cat([phone_emb, midi_emb], dim=-1)
            enc_input = encoder_input_proj(enc_input)
            enc_output = pos_encoding(enc_input)
            for block in encoder_blocks:
                enc_output = block(enc_output, mask=dummy_input_mask)

            print(f"Input shape (Phonemes): {dummy_phonemes.shape}")
            print(f"Output shape (Encoder): {enc_output.shape}")
            assert enc_output.shape == (B, N, hidden_dim)
            print(f"{component_name} test PASSED.")

        elif component_name == 'length_regulator':
            # Prepare dummy input for regulator
            dummy_encoder_output = torch.randn(B, N, hidden_dim).to(device)
            # Create realistic durations that sum up near T, some variation
            base_dur = T // N
            dummy_durations = torch.full((B, N), base_dur, dtype=torch.long).to(device)
            # Add some noise and ensure sum matches roughly T
            for i in range(B):
                 noise = torch.randint(-base_dur//2, base_dur//2 + 1, (N,), device=device)
                 dummy_durations[i] += noise
                 dummy_durations[i] = torch.clamp(dummy_durations[i], min=1) # Ensure positive duration
                 # Adjust sum to be exactly T for simplicity in test (real data handled by preprocess)
                 current_sum = dummy_durations[i].sum()
                 diff = T - current_sum
                 dummy_durations[i, -1] += diff # Add difference to last element
                 dummy_durations[i] = torch.clamp(dummy_durations[i], min=1) # Clamp again just in case

            # Instantiate regulator
            regulator = LengthRegulator().to(device)

            # Forward pass
            expanded_output, output_mask = regulator(dummy_encoder_output, dummy_durations)

            print(f"Input shape (Encoder Out): {dummy_encoder_output.shape}")
            print(f"Input shape (Durations): {dummy_durations.shape}")
            print(f"Output shape (Expanded): {expanded_output.shape}")
            print(f"Output shape (Mask): {output_mask.shape}")
            print(f"Target T: {T}, Max Duration Sum: {dummy_durations.sum(1).max().item()}")
            assert expanded_output.shape == (B, T, hidden_dim)
            assert output_mask.shape == (B, T)
            print(f"{component_name} test PASSED.")


        elif component_name == 'decoder':
             # Prepare dummy input for decoder
             dummy_expanded_output = torch.randn(B, T, hidden_dim).to(device)
             dummy_f0 = torch.randn(B, T, 1).to(device) # Add feature dim
             # Create dummy output mask
             dummy_mel_lens = torch.randint(T//2, T+1, (B,), device=device) # Variable lengths
             dummy_output_mask = (torch.arange(T, device=device).unsqueeze(0) < dummy_mel_lens.unsqueeze(1))

             # Instantiate decoder components
             decoder_input_dim = hidden_dim + 1
             decoder_input_proj = nn.Linear(decoder_input_dim, hidden_dim).to(device) if decoder_input_dim != hidden_dim else nn.Identity().to(device)
             pos_encoding = PositionalEncoding(hidden_dim, config.model.dropout).to(device)
             decoder_blocks = nn.ModuleList([
                 FFTBlock(hidden_dim, config.model.fft_n_heads, config.model.fft_conv_ffn_dim, config.model.fft_conv_kernel_size, config.model.dropout).to(device)
                 for _ in range(config.model.fft_n_layers)
             ])
             mel_proj = nn.Linear(hidden_dim, n_mels).to(device)


             # Forward pass
             dec_input = torch.cat([dummy_expanded_output, dummy_f0], dim=-1)
             dec_input = decoder_input_proj(dec_input)
             dec_output = pos_encoding(dec_input)
             for block in decoder_blocks:
                 dec_output = block(dec_output, mask=dummy_output_mask)
             mel_output = mel_proj(dec_output)


             print(f"Input shape (Expanded + F0): ({B}, {T}, {hidden_dim+1})")
             print(f"Output shape (Mel Pred): {mel_output.shape}")
             assert mel_output.shape == (B, T, n_mels)
             print(f"{component_name} test PASSED.")

        elif component_name == 'full_model':
            # Prepare a dummy batch like collate_fn output
            dummy_phonemes = torch.randint(1, config.model.vocab_size, (B, N), dtype=torch.long).to(device)
            dummy_midi = torch.randint(20, 100, (B, N), dtype=torch.long).to(device)
            # Generate durations summing to T
            base_dur = T // N
            dummy_durations = torch.full((B, N), base_dur, dtype=torch.long).to(device)
            for i in range(B):
                 noise = torch.randint(-base_dur//2, base_dur//2 + 1, (N,), device=device)
                 dummy_durations[i] += noise; dummy_durations[i].clamp_(min=1)
                 diff = T - dummy_durations[i].sum(); dummy_durations[i, -1] += diff; dummy_durations[i].clamp_(min=1)

            dummy_f0 = torch.randn(B, T).to(device) * 50 + 150 # Realistic F0 range-ish

            # Create masks
            dummy_input_mask = (dummy_phonemes != pad_id)
            # For full model test, assume all sequences have length T (no padding in T dim)
            dummy_output_mask = torch.ones(B, T, dtype=torch.bool).to(device)

            # Instantiate model
            model = FFTLightSVS(config).to(device)
            model.eval() # Set to eval mode

            # Forward pass
            with torch.no_grad():
                mel_pred = model(dummy_phonemes, dummy_midi, dummy_durations, dummy_f0, dummy_input_mask, dummy_output_mask)

            print(f"Input shapes: Phonemes({B},{N}), MIDI({B},{N}), Dur({B},{N}), F0({B},{T})")
            print(f"Output shape (Mel Pred): {mel_pred.shape}")
            assert mel_pred.shape == (B, T, n_mels)
            print(f"{component_name} test PASSED.")

        else:
            print(f"Error: Unknown component name '{component_name}'")

    except Exception as e:
        print(f"ERROR testing {component_name}: {e}")
        import traceback
        traceback.print_exc()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation tests for SVS components.")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to the configuration file.')
    parser.add_argument('--test', type=str, required=True,
                        choices=['dataset', 'alignment', 'encoder', 'length_regulator', 'decoder', 'full_model'],
                        help='Which test case to run.')
    parser.add_argument('--num_align_checks', type=int, default=20, help='Number of samples to check in alignment test.')

    args = parser.parse_args()

    # Load YAML configuration
    try:
        cfg = OmegaConf.load(args.config)
        print("Configuration loaded.")
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        exit(1)

    # Run selected test
    if args.test == 'dataset':
        test_dataset_loading(cfg)
    elif args.test == 'alignment':
        test_alignment(cfg, num_samples_to_check=args.num_align_checks)
    elif args.test in ['encoder', 'length_regulator', 'decoder', 'full_model']:
        test_model_component(cfg, args.test)

    print("\nValidation script finished.")