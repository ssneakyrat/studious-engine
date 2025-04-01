import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import random

# Import necessary modules from your project
from model import FFTLightSVS, LengthRegulator # Need LR specifically if testing it
from data import SVSDataModule, HDF5Dataset, collate_fn # Need Dataset and collate

def get_random_samples(dataset, num_samples):
    """Gets data for a specified number of random samples."""
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    # Filter out potential None values if dataset __getitem__ can return them
    samples = [s for s in samples if s is not None]
    if not samples:
        raise ValueError("Could not retrieve valid samples from the dataset.")
    batch = collate_fn(samples) # Use the same collate function as training
    return batch


def test_full_model(model, batch, logger, device, config):
    """Runs full forward pass and visualizes input/output Mel."""
    print("--- Testing Full Model Forward Pass ---")
    model.eval()
    model.to(device)

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    with torch.no_grad():
        mel_pred, _, _ = model(
            batch['phonemes'], batch['midi_notes'], batch['durations'],
            batch['f0'], batch['phoneme_mask'], batch['mel_mask']
        )

    mel_pred_cpu = mel_pred.cpu()
    mel_target_cpu = batch['mel_target'].cpu()
    mel_mask_cpu = batch['mel_mask'].cpu()

    num_to_log = min(config['validation']['num_random_samples_per_test'], mel_pred_cpu.shape[0])

    for i in range(num_to_log):
        length = (~mel_mask_cpu[i]).sum().item()
        pred = mel_pred_cpu[i, :length].numpy().T
        target = mel_target_cpu[i, :length].numpy().T

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(f"Full Model Test (Sample {batch['sample_ids'][i]})")

        img_pred = librosa.display.specshow(pred, ax=axes[0], sr=config['data']['sample_rate'],
                                            hop_length=config['data']['hop_length'], x_axis='time', y_axis='mel')
        axes[0].set_title("Predicted Mel")
        fig.colorbar(img_pred, ax=axes[0], format='%+2.0f dB')

        img_target = librosa.display.specshow(target, ax=axes[1], sr=config['data']['sample_rate'],
                                              hop_length=config['data']['hop_length'], x_axis='time', y_axis='mel')
        axes[1].set_title("Target Mel")
        fig.colorbar(img_target, ax=axes[1], format='%+2.0f dB')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        logger.experiment.add_figure(f"ModuleTest/FullModel/Sample_{batch['sample_ids'][i]}", fig, global_step=0)
        plt.close(fig)
        print(f"Logged full model comparison for sample {batch['sample_ids'][i]}")


def test_length_regulator(model, batch, logger, device, config):
    """Tests the Length Regulator output."""
    print("--- Testing Length Regulator ---")
    model.eval()
    model.to(device)

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Need to run the encoder part first to get input for LR
    with torch.no_grad():
        phon_emb = model.phoneme_embedding(batch['phonemes'])
        midi_emb = model.midi_embedding(batch['midi_notes'])
        x = torch.cat([phon_emb, midi_emb], dim=-1)
        encoder_input = model.input_proj(x)
        encoder_input = model.pos_encoder(encoder_input)
        encoder_output = encoder_input # Assume running through blocks modifies 'encoder_output'
        for block in model.encoder_blocks:
             encoder_output, _ = block(encoder_output, mask=batch['phoneme_mask'])

        # Now test the length regulator
        max_t_target = batch['f0'].shape[1]
        x_expanded, lr_mel_mask = model.length_regulator(encoder_output, batch['durations'], max_len=max_t_target)

    x_expanded_cpu = x_expanded.cpu()
    lr_mel_mask_cpu = lr_mel_mask.cpu()
    encoder_output_cpu = encoder_output.cpu()
    durations_cpu = batch['durations'].cpu()

    num_to_log = min(config['validation']['num_random_samples_per_test'], x_expanded_cpu.shape[0])

    for i in range(num_to_log):
        enc_len = (~batch['phoneme_mask'][i]).sum().item()
        exp_len = (~lr_mel_mask_cpu[i]).sum().item() # Actual expanded length
        dur_sum = durations_cpu[i, :enc_len].sum().item()

        if exp_len != dur_sum:
            print(f"WARNING: Sample {batch['sample_ids'][i]} - Expanded length ({exp_len}) != Duration sum ({dur_sum})")

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False) # Don't share x-axis
        fig.suptitle(f"Length Regulator Test (Sample {batch['sample_ids'][i]})")

        # Plot encoder output (sum across features for simplicity)
        im_enc = axes[0].imshow(encoder_output_cpu[i, :enc_len, :].mean(dim=-1, keepdim=True).T.numpy(),
                                aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Encoder Output (Mean Dim) - Len {enc_len}")
        axes[0].set_xlabel("Phoneme Index")
        axes[0].set_yticks([])
        fig.colorbar(im_enc, ax=axes[0])


        # Plot expanded output
        im_exp = axes[1].imshow(x_expanded_cpu[i, :exp_len, :].mean(dim=-1, keepdim=True).T.numpy(),
                                aspect='auto', origin='lower', cmap='viridis')
        axes[1].set_title(f"Expanded Output (Mean Dim) - Len {exp_len}")
        axes[1].set_xlabel("Frame Index")
        axes[1].set_yticks([])
        fig.colorbar(im_exp, ax=axes[1])


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        logger.experiment.add_figure(f"ModuleTest/LengthRegulator/Sample_{batch['sample_ids'][i]}", fig, global_step=0)
        plt.close(fig)
        print(f"Logged Length Regulator visualization for sample {batch['sample_ids'][i]}")

# Add test_encoder, test_decoder functions similarly if needed
# They would involve running parts of the model and potentially visualizing attention maps

def main(args):
    if not Path(args.checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint_path}")
        return

    # --- Load Model and Config ---
    # We need the config used for training this checkpoint
    # Usually saved in checkpoint, or assume default for simplicity
    config_path = args.config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = FFTLightSVS.load_from_checkpoint(args.checkpoint_path, config_path=config_path)
    print(f"Loaded model from {args.checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Data ---
    # Use the validation set for testing modules
    data_module = SVSDataModule(config_path=config_path)
    data_module.setup('validate') # Prepare validation dataset
    val_dataset = data_module.val_dataset
    if not val_dataset:
         print("Error: Could not load validation dataset.")
         return

    # --- Setup Logger ---
    log_dir = "logs/"
    logger = TensorBoardLogger(save_dir=log_dir, name="module_validation", version=Path(args.checkpoint_path).stem)
    print(f"Logging module validations to {logger.log_dir}")


    # --- Get Sample Batch ---
    try:
        batch = get_random_samples(val_dataset, config['validation']['num_random_samples_per_test'])
        if batch is None: raise ValueError("Collate function returned None.")
    except Exception as e:
        print(f"Error getting samples: {e}")
        return


    # --- Run Selected Test ---
    if args.module_to_test == 'full' or args.module_to_test == 'all':
        test_full_model(model, batch, logger, device, config)

    if args.module_to_test == 'length_regulator' or args.module_to_test == 'all':
        test_length_regulator(model, batch, logger, device, config)

    # Add calls to other test functions (test_encoder, test_decoder) here
    # if args.module_to_test == 'encoder' or args.module_to_test == 'all':
    #     test_encoder(...)
    # if args.module_to_test == 'decoder' or args.module_to_test == 'all':
    #     test_decoder(...)

    print(f"\nModule validation finished. Check TensorBoard logs in: {logger.log_dir}")
    print("Run: tensorboard --logdir logs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate specific modules of the SVS model.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint (.ckpt file).')
    parser.add_argument('--module_to_test', type=str, required=True,
                        choices=['full', 'encoder', 'decoder', 'length_regulator', 'all'],
                        help='Which module or test to run.')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to the configuration file matching the checkpoint.')
    # Add arguments like --sample_id if you want to test specific known samples

    args = parser.parse_args()
    main(args)