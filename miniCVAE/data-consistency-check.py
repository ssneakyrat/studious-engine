import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import torch
import librosa
import librosa.display
import yaml
import os

def load_config(config_path="config/default.yaml"):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_statistics(data, name):
    """Check statistics of data array and print warnings for anomalies."""
    if not isinstance(data, np.ndarray):
        print(f"ERROR: {name} is not a numpy array")
        return False
        
    if data.size == 0:
        print(f"ERROR: {name} is empty")
        return False
        
    stats = {
        "shape": data.shape,
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum())
    }
    
    # Print statistics
    print(f"\n{name} Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check for anomalies
    has_warning = False
    if stats["nan_count"] > 0:
        print(f"  WARNING: {name} contains {stats['nan_count']} NaN values")
        has_warning = True
        
    if stats["inf_count"] > 0:
        print(f"  WARNING: {name} contains {stats['inf_count']} infinite values")
        has_warning = True
        
    if stats["std"] < 1e-6:
        print(f"  WARNING: {name} has very low standard deviation ({stats['std']})")
        has_warning = True
    
    # For mel spectrograms in dB scale, typical range is -80 to 0
    if name.endswith("mel_spectrogram"):
        if stats["min"] > -20:
            print(f"  WARNING: Minimum mel value ({stats['min']} dB) is higher than expected")
            has_warning = True
            
        if stats["max"] > 20:
            print(f"  WARNING: Maximum mel value ({stats['max']} dB) is much higher than expected")
            has_warning = True
    
    if not has_warning:
        print("  No anomalies detected")
        
    return not has_warning

def plot_sample(sample_id, h5_file, output_dir, config):
    """Create visualization of a specific sample."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get sample data
        sample_group = h5_file[sample_id]
        mel_spec = sample_group['features']['mel_spectrogram'][:]
        
        # Check if the sample has F0 data
        has_f0 = ('f0_values' in sample_group['features'] and 
                  len(sample_group['features']['f0_values']) > 0)
                  
        # Check if the sample has phoneme data
        has_phonemes = ('phonemes' in sample_group and 
                        'phones' in sample_group['phonemes'] and
                        len(sample_group['phonemes']['phones']) > 0)
                        
        # Check if the sample has MIDI data
        has_midi = ('midi' in sample_group and 
                    'notes' in sample_group['midi'] and
                    len(sample_group['midi']['notes']) > 0)
        
        # Create the plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"Sample ID: {sample_id}", fontsize=16)
        
        # Plot mel spectrogram
        img = librosa.display.specshow(
            mel_spec,
            x_axis='time',
            y_axis='mel',
            sr=config['audio']['sample_rate'],
            hop_length=config['audio']['hop_length'],
            fmin=config['audio'].get('fmin', 0),
            fmax=config['audio'].get('fmax', 8000),
            ax=axes[0]
        )
        axes[0].set_title('Mel Spectrogram')
        fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
        
        # If available, plot F0
        if has_f0:
            f0_times = sample_group['features']['f0_times'][:]
            f0_values = sample_group['features']['f0_values'][:]
            
            # Filter out NaN values for plotting
            valid_indices = ~np.isnan(f0_values)
            valid_times = f0_times[valid_indices]
            valid_f0 = f0_values[valid_indices]
            
            axes[1].plot(valid_times, valid_f0, 'r-', label='F0 (Hz)')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Frequency (Hz)')
            axes[1].set_title('F0 Contour')
            
            # If we have MIDI notes, overlay them
            if has_midi:
                midi_notes = sample_group['midi']['notes'][:]
                midi_start_times = sample_group['midi']['start_times'][:]
                midi_end_times = sample_group['midi']['end_times'][:]
                
                # Plot horizontal lines for each MIDI note
                for i, (note, start, end) in enumerate(zip(midi_notes, midi_start_times, midi_end_times)):
                    freq = 440.0 * (2 ** ((note - 69) / 12.0))  # Convert MIDI to Hz
                    axes[1].plot([start, end], [freq, freq], 'b-', linewidth=2)
                
                # Add legend if we plotted both
                axes[1].legend(['F0 (Hz)', 'MIDI Notes'])
        
        elif has_phonemes:
            # If no F0 but we have phonemes, plot phoneme timeline
            try:
                phones = [p.decode('utf-8') for p in sample_group['phonemes']['phones'][:]]
                start_times = sample_group['phonemes']['start_times'][:]
                end_times = sample_group['phonemes']['end_times'][:]
                
                # Plot phoneme segments
                for i, (phone, start, end) in enumerate(zip(phones, start_times, end_times)):
                    axes[1].plot([start, end], [1, 1], 'k-', linewidth=10, alpha=0.7)
                    axes[1].text((start + end) / 2, 1, phone, 
                               horizontalalignment='center', verticalalignment='center')
                
                axes[1].set_yticks([])
                axes[1].set_title('Phoneme Timeline')
                axes[1].set_xlabel('Time (s)')
            except Exception as e:
                print(f"Error plotting phonemes: {e}")
                axes[1].set_title('Phoneme data could not be plotted')
        
        else:
            # If no extra data, just leave the second subplot empty
            axes[1].set_visible(False)
        
        plt.tight_layout()
        output_path = output_dir / f"{sample_id}_check.png"
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Error plotting sample {sample_id}: {e}")

def analyze_dataset(h5_path, config_path, output_dir="dataset_check", num_samples=5):
    """
    Analyze the HDF5 dataset, check for issues and create visualizations.
    
    Args:
        h5_path: Path to the HDF5 file
        config_path: Path to the config.yaml file
        output_dir: Directory to save visualizations
        num_samples: Number of random samples to visualize
    """
    print(f"Analyzing dataset: {h5_path}")
    
    try:
        config = load_config(config_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(h5_path, 'r') as f:
            # Check if metadata exists
            if 'metadata' not in f:
                print("ERROR: No metadata group found in the HDF5 file")
                return
                
            if 'file_list' not in f['metadata']:
                print("ERROR: No file_list dataset found in metadata")
                return
            
            # Get file list
            try:
                file_list_ds = f['metadata']['file_list']
                if h5py.check_string_dtype(file_list_ds.dtype):
                    sample_ids = [s.decode('utf-8') for s in file_list_ds[:]]
                else:
                    sample_ids = list(file_list_ds[:])
            except Exception as e:
                print(f"ERROR reading file_list: {e}")
                return
            
            # Print dataset summary
            num_samples_total = len(sample_ids)
            print(f"\nDataset Summary:")
            print(f"  Total number of samples: {num_samples_total}")
            print(f"  Audio configuration:")
            for key, value in config['audio'].items():
                print(f"    {key}: {value}")
            
            # Check for empty dataset
            if num_samples_total == 0:
                print("ERROR: Dataset is empty (file_list has 0 entries)")
                return
            
            # Consistency check for random samples
            check_samples = min(num_samples, num_samples_total)
            print(f"\nChecking {check_samples} random samples for consistency...")
            
            # Selected random samples
            selected_indices = np.random.choice(
                range(num_samples_total), 
                size=check_samples, 
                replace=False
            )
            selected_samples = [sample_ids[i] for i in selected_indices]
            
            # Collect statistics for all samples
            all_shapes = []
            all_mins = []
            all_maxes = []
            all_means = []
            all_stds = []
            
            # Check selected samples
            for i, sample_id in enumerate(selected_samples):
                print(f"\nSample {i+1}/{check_samples}: {sample_id}")
                
                if sample_id not in f:
                    print(f"  ERROR: Sample ID not found in HDF5 file")
                    continue
                
                # Get sample group
                sample = f[sample_id]
                
                # Check required groups
                required_groups = ['features', 'phonemes', 'midi']
                for group in required_groups:
                    if group not in sample:
                        print(f"  WARNING: Missing group '{group}'")
                
                # Check mel spectrogram
                if 'features' in sample and 'mel_spectrogram' in sample['features']:
                    mel_spec = sample['features']['mel_spectrogram'][:]
                    check_statistics(mel_spec, f"Sample {sample_id} mel_spectrogram")
                    
                    # Collect statistics
                    all_shapes.append(mel_spec.shape)
                    all_mins.append(np.min(mel_spec))
                    all_maxes.append(np.max(mel_spec))
                    all_means.append(np.mean(mel_spec))
                    all_stds.append(np.std(mel_spec))
                    
                    # Create visualization
                    plot_sample(sample_id, f, output_dir, config)
                else:
                    print(f"  ERROR: Missing mel_spectrogram in features")
            
            # Print overall statistics
            if all_shapes:
                print("\nOverall Statistics:")
                print(f"  Shape: {all_shapes[0]} (checking consistency...)")
                
                # Check if all shapes are the same
                if len(set(str(s) for s in all_shapes)) > 1:
                    print(f"  WARNING: Inconsistent shapes found: {set(str(s) for s in all_shapes)}")
                
                print(f"  Min values range: {min(all_mins):.2f} to {max(all_mins):.2f}")
                print(f"  Max values range: {min(all_maxes):.2f} to {max(all_maxes):.2f}")
                print(f"  Mean values range: {min(all_means):.2f} to {max(all_means):.2f}")
                print(f"  Std dev range: {min(all_stds):.2f} to {max(all_stds):.2f}")
                
                # Plot distribution of statistics
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle("Dataset Statistics Distribution", fontsize=16)
                
                axes[0, 0].hist(all_mins, bins=20)
                axes[0, 0].set_title("Minimum Values")
                axes[0, 0].set_xlabel("Min Value (dB)")
                
                axes[0, 1].hist(all_maxes, bins=20)
                axes[0, 1].set_title("Maximum Values")
                axes[0, 1].set_xlabel("Max Value (dB)")
                
                axes[1, 0].hist(all_means, bins=20)
                axes[1, 0].set_title("Mean Values")
                axes[1, 0].set_xlabel("Mean Value (dB)")
                
                axes[1, 1].hist(all_stds, bins=20)
                axes[1, 1].set_title("Standard Deviations")
                axes[1, 1].set_xlabel("Standard Deviation")
                
                plt.tight_layout()
                stats_path = output_dir / "dataset_statistics.png"
                plt.savefig(stats_path)
                plt.close(fig)
                print(f"\nStatistics visualization saved to {stats_path}")
    
    except FileNotFoundError:
        print(f"ERROR: File not found: {h5_path}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check HDF5 dataset quality")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="dataset_check", help="Output directory for visualizations")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to visualize")
    
    args = parser.parse_args()
    analyze_dataset(args.h5_path, args.config, args.output_dir, args.num_samples)