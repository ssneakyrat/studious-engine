import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from typing import Dict, Optional, Union

def griffin_lim(spectrogram: np.ndarray,
                n_fft: int = 1024,
                hop_length: int = 256,
                win_length: int = 1024,
                sample_rate: int = 22050,
                n_iter: int = 60,
                power: float = 1.5,
                n_mels: int = 80,
                fmin: int = 0,
                fmax: int = 8000) -> np.ndarray:
    """
    Convert a mel spectrogram to audio using Griffin-Lim algorithm.
    
    Args:
        spectrogram: Mel spectrogram [n_mels, time]
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        sample_rate: Audio sample rate
        n_iter: Number of Griffin-Lim iterations
        power: Power to raise the magnitude spectrogram
        n_mels: Number of mel bands
        fmin: Minimum frequency for mel filter
        fmax: Maximum frequency for mel filter
        
    Returns:
        Reconstructed audio signal
    """
    # Convert mel spectrogram to linear spectrogram
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    
    # Pseudo-inverse of mel basis
    mel_inverse = np.linalg.pinv(mel_basis)
    
    # Convert from mel to linear scale
    linear_spec = np.dot(mel_inverse, spectrogram)
    
    # Make sure the values are non-negative
    linear_spec = np.maximum(1e-10, linear_spec)
    
    # Apply power law scaling
    linear_spec = linear_spec ** power
    
    # Perform Griffin-Lim
    angles = np.exp(2j * np.pi * np.random.rand(*linear_spec.shape))
    linear_spec_complex = linear_spec.astype(np.complex128) * angles
    
    # Griffin-Lim algorithm
    for _ in range(n_iter):
        audio = librosa.istft(
            linear_spec_complex,
            hop_length=hop_length,
            win_length=win_length
        )
        
        if np.isnan(audio).any() or np.isinf(audio).any():
            print("Warning: NaN or Inf values detected in audio. Stopping Griffin-Lim early.")
            break
            
        spec_complex = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        
        # Update angles
        angles = np.exp(1j * np.angle(spec_complex))
        linear_spec_complex = linear_spec.astype(np.complex128) * angles
    
    # Final reconstruction
    audio = librosa.istft(
        linear_spec_complex,
        hop_length=hop_length,
        win_length=win_length
    )
    
    return audio

def save_mel_spectrogram(spectrogram: np.ndarray,
                         path: str,
                         title: str = "Mel Spectrogram",
                         sample_rate: int = 22050,
                         hop_length: int = 256,
                         fmin: int = 0,
                         fmax: int = 8000):
    """
    Save a mel spectrogram as an image.
    
    Args:
        spectrogram: Mel spectrogram [n_mels, time]
        path: Path to save the image
        title: Title for the plot
        sample_rate: Audio sample rate
        hop_length: Hop length for STFT
        fmin: Minimum frequency for mel filter
        fmax: Maximum frequency for mel filter
    """
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def save_audio_comparison(original_audio: np.ndarray,
                         generated_audio: np.ndarray,
                         path: str,
                         sample_rate: int = 22050,
                         title: str = "Audio Waveform Comparison"):
    """
    Save a comparison of original and generated audio waveforms.
    
    Args:
        original_audio: Original audio waveform
        generated_audio: Generated audio waveform
        path: Path to save the image
        sample_rate: Audio sample rate
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot original audio
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(original_audio)) / sample_rate, original_audio)
    plt.title("Original Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    # Plot generated audio
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(generated_audio)) / sample_rate, generated_audio)
    plt.title("Generated Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def save_spectrogram_comparison(original_mel: np.ndarray,
                               generated_mel: np.ndarray,
                               path: str,
                               sample_rate: int = 22050,
                               hop_length: int = 256,
                               fmin: int = 0,
                               fmax: int = 8000,
                               title: str = "Mel Spectrogram Comparison"):
    """
    Save a comparison of original and generated mel spectrograms.
    
    Args:
        original_mel: Original mel spectrogram [n_mels, time]
        generated_mel: Generated mel spectrogram [n_mels, time]
        path: Path to save the image
        sample_rate: Audio sample rate
        hop_length: Hop length for STFT
        fmin: Minimum frequency for mel filter
        fmax: Maximum frequency for mel filter
        title: Title for the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot original mel spectrogram
    plt.subplot(2, 1, 1)
    librosa.display.specshow(
        original_mel,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Original Mel Spectrogram")
    
    # Plot generated mel spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(
        generated_mel,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Generated Mel Spectrogram")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def mel_to_audio(mel_spectrogram: np.ndarray,
                 config: Dict,
                 use_griffin_lim: bool = True,
                 path: Optional[str] = None) -> np.ndarray:
    """
    Convert a mel spectrogram to audio.
    
    Args:
        mel_spectrogram: Mel spectrogram [n_mels, time] or [time, n_mels]
        config: Configuration dictionary with audio parameters
        use_griffin_lim: Whether to use Griffin-Lim algorithm
        path: Path to save the audio file (optional)
        
    Returns:
        Audio waveform
    """
    # Ensure mel spectrogram is in the correct shape [n_mels, time]
    if mel_spectrogram.shape[0] != config['audio']['n_mels']:
        mel_spectrogram = mel_spectrogram.T
    
    # Generate audio based on method
    if use_griffin_lim:
        audio = griffin_lim(
            spectrogram=mel_spectrogram,
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            win_length=config['audio']['win_length'],
            sample_rate=config['audio']['sample_rate'],
            n_mels=config['audio']['n_mels'],
            fmin=config['audio']['fmin'],
            fmax=config['audio']['fmax'],
            n_iter=60  # Number of Griffin-Lim iterations
        )
    else:
        # Can implement other vocoders here (e.g., WaveNet, WaveGlow, etc.)
        raise NotImplementedError("Only Griffin-Lim is currently supported")
    
    # Save audio if path is provided
    if path is not None:
        sf.write(path, audio, config['audio']['sample_rate'])
    
    return audio


# Test function to validate audio conversion
def test_audio_conversion():
    """Test the audio conversion functions."""
    # Create a simple config
    config = {
        'audio': {
            'sample_rate': 22050,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'n_mels': 80,
            'fmin': 0,
            'fmax': 8000
        }
    }
    
    # Create a synthetic mel spectrogram (sine wave)
    duration = 2  # seconds
    n_frames = int(duration * config['audio']['sample_rate'] / config['audio']['hop_length'])
    
    # Create a sine pattern in the mel spectrogram
    frequency = 440  # Hz
    t = np.linspace(0, duration, n_frames)
    pattern = np.sin(2 * np.pi * frequency * t)
    
    mel_spectrogram = np.zeros((config['audio']['n_mels'], n_frames))
    
    # Add the pattern to a few mel bands
    for i in range(20, 60):
        mel_spectrogram[i, :] = pattern * (i - 20) / 40.0
    
    # Normalize
    mel_spectrogram = (mel_spectrogram - mel_spectrogram.min()) / (mel_spectrogram.max() - mel_spectrogram.min())
    
    # Convert to log scale (similar to librosa.power_to_db)
    mel_spectrogram = 20 * np.log10(np.maximum(1e-5, mel_spectrogram))
    
    # Convert to audio
    print("Converting mel spectrogram to audio...")
    audio = mel_to_audio(mel_spectrogram, config)
    
    # Save results
    output_dir = "audio_test"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mel spectrogram
    save_mel_spectrogram(
        mel_spectrogram,
        os.path.join(output_dir, "test_mel.png"),
        sample_rate=config['audio']['sample_rate'],
        hop_length=config['audio']['hop_length'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    
    # Save audio
    sf.write(
        os.path.join(output_dir, "test_audio.wav"),
        audio,
        config['audio']['sample_rate']
    )
    
    print(f"Test results saved to {output_dir}")
    
    return mel_spectrogram, audio


if __name__ == "__main__":
    test_audio_conversion()
