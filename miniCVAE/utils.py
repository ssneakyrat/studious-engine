import yaml
import math
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image
import os
from pathlib import Path

def load_config(config_path="config/default.yaml"):
    """Loads YAML configuration file with improved error handling."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate essential config parameters
        required_sections = ['audio', 'model', 'training']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Ensure default values for optional parameters
        if 'data' not in config:
            config['data'] = {}
        
        if 'h5_path' not in config['data']:
            print("WARNING: No h5_path specified in config.")
        
        # Apply sensible defaults for any missing parameters
        if 'latent_dim' not in config['model']:
            print("WARNING: No latent_dim specified, defaulting to 64")
            config['model']['latent_dim'] = 64
            
        if 'annealing_epochs' not in config['training']:
            print("WARNING: No annealing_epochs specified, defaulting to 100")
            config['training']['annealing_epochs'] = 100
        
        return config
        
    except FileNotFoundError:
        print(f"ERROR: Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in config file: {e}")
        raise
    except Exception as e:
        print(f"ERROR: Problem loading config: {e}")
        raise

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

def normalize_mel(mel, floor_db=-80.0, top_db=0.0):
    """Normalize mel spectrogram to a standard range."""
    # Clamp values to a reasonable dB range
    mel = torch.clamp(mel, min=floor_db, max=top_db)
    
    # Scale to [0, 1]
    normalized = (mel - floor_db) / (top_db - floor_db)
    
    return normalized

def denormalize_mel(normalized_mel, floor_db=-80.0, top_db=0.0):
    """Convert normalized values back to mel spectrogram dB values."""
    # Scale back to dB range
    return normalized_mel * (top_db - floor_db) + floor_db

def compute_spectral_convergence_loss(target, prediction):
    """Compute spectral convergence loss between two spectrograms."""
    return torch.norm(target - prediction, p='fro') / (torch.norm(target, p='fro') + 1e-7)

def compute_log_magnitude_loss(target, prediction, eps=1e-7):
    """Compute L1 loss between log magnitudes of spectrograms."""
    return torch.mean(torch.abs(torch.log(target + eps) - torch.log(prediction + eps)))

def plot_mel_comparison_to_buf(target_mel, predicted_mel, title, config, show_diff=True):
    """
    Generates a comparison plot of two mel spectrograms and returns a buffer.
    Improved version with difference visualization.

    Args:
        target_mel (torch.Tensor): Ground truth mel spectrogram (N_MELS, T).
        predicted_mel (torch.Tensor): Predicted mel spectrogram (N_MELS, T).
        title (str): Plot title.
        config (dict): Configuration dictionary with audio parameters.
        show_diff (bool): Whether to show the difference between target and prediction.

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

    # Decide on number of subplots
    n_plots = 3 if show_diff else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4*n_plots), sharex=True, dpi=120)
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
        axes[0].text(0.5, 0.5, "Error plotting ground truth", 
                   ha='center', va='center', transform=axes[0].transAxes)
    
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
        axes[1].text(0.5, 0.5, "Error plotting prediction", 
                   ha='center', va='center', transform=axes[1].transAxes)
    
    # Only add x-axis label if we're not showing difference
    if not show_diff:
        axes[1].set_xlabel("Time (s)")
    else:
        axes[1].label_outer()  # Hide x axis labels
    
    fig.colorbar(img_pred, ax=axes[1], format='%+2.0f dB')

    # Show difference if requested
    if show_diff and target_np.shape == predicted_np.shape:
        try:
            # Compute absolute difference
            diff = np.abs(target_np - predicted_np)
            img_diff = librosa.display.specshow(
                diff, sr=sr, hop_length=hop_length,
                fmin=fmin, fmax=fmax, x_axis='time', y_axis='mel', ax=axes[2]
            )
            axes[2].set_title("Absolute Difference |Ground Truth - Predicted|")
            axes[2].set_xlabel("Time (s)")
            fig.colorbar(img_diff, ax=axes[2], format='%+2.0f dB')
            
            # Add metrics in the title
            mse = np.mean((target_np - predicted_np) ** 2)
            mae = np.mean(np.abs(target_np - predicted_np))
            axes[2].set_title(f"Absolute Difference (MSE: {mse:.2f}, MAE: {mae:.2f})")
        except Exception as e:
            print(f"ERROR: Plotting difference failed: {e}")
            axes[2].text(0.5, 0.5, "Error computing difference", 
                       ha='center', va='center', transform=axes[2].transAxes)

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

def save_audio_sample(mel_spec, sr, hop_length, output_path, min_level_db=-100, ref_level_db=20):
    """
    Convert a mel spectrogram to audio and save it.
    
    Args:
        mel_spec: Mel spectrogram (numpy array)
        sr: Sample rate
        hop_length: Hop length used in STFT
        output_path: Path to save the audio file
        min_level_db: Minimum dB level
        ref_level_db: Reference dB level
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to magnitude
        mel_spec_db = mel_spec.squeeze().cpu().numpy()
        mel_spec_magnitude = librosa.db_to_power(mel_spec_db)
        
        # Approximate inversion (not perfect but gives an idea)
        S_full = librosa.feature.inverse.mel_to_stft(
            mel_spec_magnitude, 
            sr=sr, 
            n_fft=1024, 
            fmin=0, 
            fmax=8000
        )
        
        # Invert STFT to get audio
        wav = librosa.griffinlim(
            S_full,
            hop_length=hop_length,
            n_iter=32  # More iterations = better quality but slower
        )
        
        # Save audio file
        librosa.output.write_wav(output_path, wav, sr)
        print(f"Audio saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving audio: {e}")
        return False

def export_onnx_model(model, filename, input_shape=(1, 1, 80, 864)):
    """
    Export PyTorch model to ONNX format for deployment.
    
    Args:
        model: PyTorch model
        filename: Output ONNX filename
        input_shape: Shape of input tensor (batch, channels, height, width)
    """
    try:
        # Ensure path exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Prepare dummy inputs for export
        dummy_condition = torch.randn(input_shape, dtype=torch.float32)
        dummy_noise = torch.randn(input_shape, dtype=torch.float32)
        
        # Export the model
        torch.onnx.export(
            model,
            (dummy_condition, dummy_noise),
            filename,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['condition', 'noise'],
            output_names=['output', 'mu', 'logvar'],
            dynamic_axes={
                'condition': {0: 'batch_size', 3: 'time_dim'},
                'noise': {0: 'batch_size', 3: 'time_dim'},
                'output': {0: 'batch_size', 3: 'time_dim'}
            }
        )
        print(f"Model exported to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting model: {e}")
        return False