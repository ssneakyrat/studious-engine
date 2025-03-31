import yaml
import math
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image

def load_config(config_path="config/default.yaml"):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_target_frames(config):
    """Calculates the target number of frames based on config."""
    return math.ceil(
        config['audio']['target_duration_sec'] *
        config['audio']['sample_rate'] /
        config['audio']['hop_length']
    )

def pad_or_truncate(mel, target_length, padding_value):
    """Pads or truncates a mel spectrogram along the time axis."""
    current_length = mel.shape[-1] # Get time dimension length
    if current_length == target_length:
        return mel
    elif current_length < target_length:
        padding_size = target_length - current_length
        # Pad on the right side
        padded_mel = torch.nn.functional.pad(
            mel, (0, padding_size), mode='constant', value=padding_value
        )
        return padded_mel
    else: # current_length > target_length
        truncated_mel = mel[..., :target_length] # Take the beginning
        return truncated_mel

def plot_mel_comparison_to_buf(target_mel, predicted_mel, title, config):
    """
    Generates a comparison plot of two mel spectrograms and returns a buffer.

    Args:
        target_mel (torch.Tensor): Ground truth mel spectrogram (N_MELS, T).
        predicted_mel (torch.Tensor): Predicted mel spectrogram (N_MELS, T).
        title (str): Plot title.
        config (dict): Configuration dictionary with audio parameters.

    Returns:
        io.BytesIO: Buffer containing the PNG image data.
    """
    sr = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    n_mels = config['audio']['n_mels']
    fmin = config['audio'].get('fmin', 0)
    fmax = config['audio'].get('fmax', sr // 2)

    # Convert tensors to numpy, move to CPU if necessary
    target_np = target_mel.squeeze().cpu().numpy()
    predicted_np = predicted_mel.squeeze().cpu().numpy()

    # Shape comparison
    if target_np.shape != predicted_np.shape:
        print(f"WARN: Target shape {target_np.shape} != Predicted shape {predicted_np.shape} after resizing.")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=100)
    fig.suptitle(title, fontsize=14)

    # Plot Ground Truth
    try:
        img_true = librosa.display.specshow(
            target_np, sr=sr, hop_length=hop_length,
            fmin=fmin, fmax=fmax, x_axis='time', y_axis='mel', ax=axes[0]
        )
        axes[0].set_title("Ground Truth Mel")
    except Exception as e:
        print(f"ERROR: Plotting ground truth mel failed: {e}")
        img_true = None # Handle error case
    axes[0].set_title("Ground Truth Mel")
    axes[0].label_outer() # Hide x axis labels
    fig.colorbar(img_true, ax=axes[0], format='%+2.0f dB')

    # Plot Predicted
    try:
        img_pred = librosa.display.specshow(
            predicted_np, sr=sr, hop_length=hop_length,
            fmin=fmin, fmax=fmax, x_axis='time', y_axis='mel', ax=axes[1]
        )
        axes[1].set_title("Predicted Mel")
    except Exception as e:
        print(f"ERROR: Plotting predicted mel failed: {e}")
        img_pred = None
    axes[1].set_title("Predicted Mel")
    fig.colorbar(img_pred, ax=axes[1], format='%+2.0f dB')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    # Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig) # Close the figure to free memory
    return buf

def buf_to_image(buf):
    """Converts an image buffer (BytesIO) to a PIL Image."""
    return Image.open(buf)