import os
import numpy as np
import h5py
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from tqdm import tqdm
import logging
import yaml
import json
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('preprocessor')

class SVSPreprocessor:
    """
    Singing Voice Synthesis Preprocessor with phoneme-MIDI alignment.
    
    This preprocessor handles:
    1. Audio processing (loading, resampling, mel extraction)
    2. Phoneme processing from LAB files
    3. F0 extraction and MIDI note conversion
    4. Feature alignment with 1:1 phoneme-to-MIDI mapping
    5. Dataset creation and storage
    """
    
    def __init__(self, config_path="config/default.yaml"):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get required paths
        datasets_dir = self.config['dataset']['datasets_dir']
        self.root_dir = datasets_dir
        self.wav_dir = os.path.join(datasets_dir, "wav")
        self.lab_dir = os.path.join(datasets_dir, "lab")
        self.output_dir = os.path.join(datasets_dir, "processed")
        self.output_file = os.path.join(self.output_dir, "svs_dataset.h5")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Audio parameters
        self.sample_rate = self.config['preprocessing']['sample_rate']
        self.n_fft = self.config['preprocessing']['n_fft']
        self.hop_length = self.config['preprocessing']['hop_length']
        self.n_mels = self.config['preprocessing']['n_mels']
        self.f0_min = self.config['preprocessing']['f0_min']
        self.f0_max = self.config['preprocessing']['f0_max']
        
        # Feature mappings
        self.phone_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_phone = {0: '<PAD>', 1: '<UNK>'}
        self.midi_min = 127
        self.midi_max = 0
        
        # Stats for dataset
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'skipped_few_phonemes': 0,
            'total_frames': 0,
            'total_phonemes': 0,
            'unique_phonemes': 0,
            'alignment_scores': []
        }

    def audio_to_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert audio waveform to mel spectrogram.
        
        Args:
            waveform: Audio waveform (mono, sample_rate)
            
        Returns:
            mel_spectrogram: Mel spectrogram as numpy array
        """
        # Ensure audio is mono
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = np.mean(waveform, axis=0)
        
        # Compute mel spectrogram using librosa for better control
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sample_rate/2
        )
        
        # Convert to dB scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Transpose to get (time, mel_bins) shape
        mel_spectrogram = mel_spectrogram.T
        
        return mel_spectrogram
    
    def load_and_process_audio(self, wav_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Load and process an audio file.
        
        Args:
            wav_path: Path to the WAV file
            
        Returns:
            waveform: Processed audio waveform
            mel_spectrogram: Extracted mel spectrogram
            num_frames: Number of frames in the mel spectrogram
        """
        try:
            # Load audio
            waveform, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)
            
            # Normalize audio
            waveform = librosa.util.normalize(waveform)
            
            # Convert to mel spectrogram
            mel_spectrogram = self.audio_to_mel_spectrogram(waveform)
            
            # Get number of frames
            num_frames = mel_spectrogram.shape[0]
            
            return waveform, mel_spectrogram, num_frames
        
        except Exception as e:
            logger.error(f"Error processing audio file {wav_path}: {e}")
            raise
    
    def extract_f0(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 contour from audio waveform using a direct approach.
        
        Args:
            waveform: Audio waveform
            
        Returns:
            f0: F0 contour in Hz
            midi: MIDI note values
        """
        try:
            # Create Parselmouth Sound object directly from the waveform
            sound = parselmouth.Sound(
                values=waveform,
                sampling_frequency=self.sample_rate
            )
            
            # Extract pitch using Praat's algorithm
            pitch = call(sound, "To Pitch", 0.0, self.f0_min, self.f0_max)
            
            # Get pitch values and timestamps
            pitch_values = pitch.selected_array['frequency']
            pitch_times = np.array([pitch.get_time_from_frame_number(i+1) for i in range(len(pitch_values))])
            
            # Create time points for each frame in the mel spectrogram
            frame_times = librosa.frames_to_time(
                np.arange(int(waveform.shape[0] / self.hop_length) + 1),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
            # Interpolate F0 to align with mel frames
            f0_interp = np.zeros_like(frame_times)
            
            # Handle unvoiced regions
            voiced_indices = pitch_values > 0
            if np.any(voiced_indices):
                # Create interpolation function for voiced regions
                f0_func = interp1d(
                    pitch_times[voiced_indices],
                    pitch_values[voiced_indices],
                    kind='linear',
                    bounds_error=False,
                    fill_value=(
                        pitch_values[voiced_indices][0] if any(voiced_indices) else 0,
                        pitch_values[voiced_indices][-1] if any(voiced_indices) else 0
                    )
                )
                
                # Apply interpolation
                f0_interp = f0_func(frame_times)
                
                # Fill any remaining NaN values
                f0_interp = np.nan_to_num(f0_interp, nan=0.0)
            
            # Convert to MIDI notes (but these will be replaced with phoneme-aligned MIDI later)
            midi = 12 * np.log2(np.maximum(f0_interp, 1e-5) / 440.0) + 69
            midi = np.clip(midi, 0, 127).astype(np.int16)
            
            # Update MIDI range
            voiced_midi = midi[midi > 0]
            if len(voiced_midi) > 0:
                self.midi_min = min(self.midi_min, int(np.min(voiced_midi)))
                self.midi_max = max(self.midi_max, int(np.max(voiced_midi)))
            
            return f0_interp, midi
            
        except Exception as e:
            logger.error(f"Error in F0 extraction: {e}")
            # Return empty arrays with correct shape in case of error
            f0_interp = np.zeros_like(librosa.frames_to_time(
                np.arange(int(waveform.shape[0] / self.hop_length) + 1),
                sr=self.sample_rate,
                hop_length=self.hop_length
            ))
            midi = np.zeros_like(f0_interp, dtype=np.int16)
            return f0_interp, midi
    
    def parse_lab_file(self, lab_path: str) -> Tuple[List[str], List[float], List[float]]:
        """
        Parse a LAB file containing phoneme timing information.
        
        Args:
            lab_path: Path to the LAB file
            
        Returns:
            phonemes: List of phoneme strings
            start_times: List of start times in seconds
            end_times: List of end times in seconds
        """
        phonemes = []
        start_times = []
        end_times = []
        
        with open(lab_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    # LAB format: start_sample end_sample phoneme
                    start_sample = float(parts[0])
                    end_sample = float(parts[1])
                    phoneme = parts[2]
                    
                    # Convert samples to seconds
                    start_time = start_sample / self.sample_rate
                    end_time = end_sample / self.sample_rate
                    
                    # Add to lists
                    phonemes.append(phoneme)
                    start_times.append(start_time)
                    end_times.append(end_time)
                    
                    # Update phoneme dictionary
                    if phoneme not in self.phone_to_id:
                        idx = len(self.phone_to_id)
                        self.phone_to_id[phoneme] = idx
                        self.id_to_phone[idx] = phoneme
        
        return phonemes, start_times, end_times
    
    def align_phonemes_and_midi(
        self, 
        phonemes: List[str], 
        start_times: List[float], 
        end_times: List[float],
        f0: np.ndarray,
        num_frames: int
    ) -> Tuple[List[int], np.ndarray]:
        """
        Align phonemes with MIDI notes, ensuring 1:1 mapping by scaling phoneme durations.
        Each phoneme gets assigned one MIDI note for its entire duration.
        
        Args:
            phonemes: List of phoneme strings
            start_times: List of phoneme start times in seconds
            end_times: List of phoneme end times in seconds
            f0: F0 contour in Hz
            num_frames: Number of frames in the mel spectrogram
            
        Returns:
            frame_phonemes: List of phoneme IDs for each frame
            frame_midi: List of MIDI notes for each frame
        """
        # Calculate total duration of phoneme sequence
        total_phoneme_duration = end_times[-1] - start_times[0]
        
        # Calculate scaling factor to map phoneme time to mel frames
        frames_per_second = num_frames / total_phoneme_duration
        
        # Initialize output arrays
        frame_phonemes = [0] * num_frames
        frame_midi = np.zeros(num_frames, dtype=np.int16)
        
        # Track current frame position
        current_frame = 0
        
        # Process each phoneme
        for i, (phoneme, start, end) in enumerate(zip(phonemes, start_times, end_times)):
            # Calculate frames for this phoneme based on its duration
            phoneme_duration = end - start
            phoneme_frames = round(phoneme_duration * frames_per_second)
            
            # Ensure we don't exceed the mel length (handle last phoneme specially)
            if i == len(phonemes) - 1:
                phoneme_frames = num_frames - current_frame
            elif current_frame + phoneme_frames > num_frames:
                phoneme_frames = num_frames - current_frame
                
            # Skip if no frames (can happen due to rounding)
            if phoneme_frames <= 0:
                continue
                
            # Get frame range for this phoneme
            frame_range = range(current_frame, current_frame + phoneme_frames)
            
            # Assign phoneme ID to all frames in this segment
            phoneme_id = self.phone_to_id.get(phoneme, 1)  # 1 is <UNK>
            for frame_idx in frame_range:
                frame_phonemes[frame_idx] = phoneme_id
            
            # Calculate average F0 for this phoneme's frames in f0 array
            # Only use the frames for this phoneme to get average F0
            phoneme_f0_frames = []
            for frame_idx in frame_range:
                if frame_idx < len(f0):
                    phoneme_f0_frames.append(f0[frame_idx])
            
            phoneme_f0_frames = np.array(phoneme_f0_frames)
            
            if len(phoneme_f0_frames) > 0 and np.any(phoneme_f0_frames > 0):
                # Use only voiced frames for F0 calculation
                voiced_f0 = phoneme_f0_frames[phoneme_f0_frames > 0]
                if len(voiced_f0) > 0:
                    phoneme_f0 = np.mean(voiced_f0)
                    # Convert to MIDI note
                    midi_note = int(round(12 * np.log2(max(phoneme_f0, 1e-5) / 440.0) + 69))
                    midi_note = np.clip(midi_note, 0, 127)
                else:
                    midi_note = 0
            else:
                # Use 0 for unvoiced segments
                midi_note = 0
            
            # Assign this MIDI note to all frames in the phoneme
            for frame_idx in frame_range:
                if frame_idx < len(frame_midi):
                    frame_midi[frame_idx] = midi_note
            
            # Update current frame position
            current_frame += phoneme_frames
        
        # Ensure all frames have a phoneme (just in case)
        if current_frame < num_frames:
            # Fill remaining frames with the last phoneme
            last_phoneme_id = frame_phonemes[current_frame-1] if current_frame > 0 else 0
            last_midi = frame_midi[current_frame-1] if current_frame > 0 else 0
            
            for i in range(current_frame, num_frames):
                frame_phonemes[i] = last_phoneme_id
                frame_midi[i] = last_midi
        
        # Log verification info
        logger.info(f"Alignment scaling: {frames_per_second:.2f} frames per second")
        logger.info(f"Total frames: {num_frames}, Phoneme frames mapped: {current_frame}")
        
        return frame_phonemes, frame_midi
    
    def verify_alignment(
        self, 
        phoneme_frames: List[int], 
        midi: np.ndarray, 
        f0: np.ndarray, 
        mel_spec: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Verify the quality of alignment between features.
        
        Args:
            phoneme_frames: Phoneme IDs for each frame
            midi: MIDI notes for each frame
            f0: F0 values for each frame
            mel_spec: Mel spectrogram
            
        Returns:
            score: Alignment quality score (0-1)
            metrics: Dictionary of alignment metrics
        """
        # Convert to numpy arrays
        phoneme_frames = np.array(phoneme_frames)
        
        # Find phoneme boundaries
        boundaries = np.where(np.diff(phoneme_frames) != 0)[0] + 1
        
        # Count unique phonemes
        unique_phonemes = np.unique(phoneme_frames)
        unique_phoneme_count = len(unique_phonemes[unique_phonemes > 0])
        
        # Check phoneme distribution
        phoneme_counts = {}
        for p in phoneme_frames:
            if p > 0:  # Skip padding
                if p not in phoneme_counts:
                    phoneme_counts[p] = 0
                phoneme_counts[p] += 1
        
        # Calculate metrics
        metrics = {
            'num_frames': len(phoneme_frames),
            'num_boundaries': len(boundaries),
            'unique_phonemes': unique_phoneme_count,
            'longest_stable_segment': 0,
            'boundary_f0_changes': [],
            'boundary_mel_changes': []
        }
        
        # Check for overly long segments (potential alignment issues)
        if len(boundaries) > 0:
            boundaries = np.concatenate(([0], boundaries, [len(phoneme_frames)]))
            segment_lengths = np.diff(boundaries)
            metrics['longest_stable_segment'] = np.max(segment_lengths)
            metrics['avg_segment_length'] = np.mean(segment_lengths)
            
            # Check feature changes at boundaries
            for b in boundaries[1:-1]:
                # F0 change at boundary
                if b > 0 and b < len(f0) - 1:
                    pre_f0 = np.mean(f0[max(0, b-3):b])
                    post_f0 = np.mean(f0[b:min(len(f0), b+3)])
                    f0_change = np.abs(post_f0 - pre_f0)
                    metrics['boundary_f0_changes'].append(f0_change)
                
                # Mel spectrogram change at boundary
                if b > 0 and b < mel_spec.shape[0] - 1:
                    pre_mel = np.mean(mel_spec[max(0, b-3):b], axis=0)
                    post_mel = np.mean(mel_spec[b:min(mel_spec.shape[0], b+3)], axis=0)
                    mel_change = np.mean(np.abs(post_mel - pre_mel))
                    metrics['boundary_mel_changes'].append(mel_change)
        
        # Calculate overall score (0-1)
        score = 0.0
        
        # 1. Penalize if too few unique phonemes
        if unique_phoneme_count <= 1:
            score = 0.0
        else:
            # Start with base score
            score = 0.5
            
            # 2. Reward good boundary changes in F0/mel
            if len(metrics['boundary_f0_changes']) > 0:
                avg_f0_change = np.mean(metrics['boundary_f0_changes'])
                if avg_f0_change > 10:  # Good F0 changes at boundaries
                    score += 0.1
            
            if len(metrics['boundary_mel_changes']) > 0:
                avg_mel_change = np.mean(metrics['boundary_mel_changes'])
                if avg_mel_change > 1.0:  # Good mel changes at boundaries
                    score += 0.1
            
            # 3. Check segment length distribution
            if 'avg_segment_length' in metrics:
                if metrics['avg_segment_length'] > 5 and metrics['longest_stable_segment'] < 100:
                    score += 0.2
                elif metrics['longest_stable_segment'] > 200:  # Too long segments
                    score -= 0.2
            
            # 4. Reward more boundaries
            boundary_density = len(boundaries) / len(phoneme_frames)
            if 0.01 < boundary_density < 0.1:  # Reasonable number of boundaries
                score += 0.1
            
            # Ensure score is between 0 and 1
            score = max(0.0, min(1.0, score))
        
        return score, metrics
    
    def plot_alignment(
        self,
        file_id: str,
        phoneme_frames: List[int],
        midi: np.ndarray, # Phoneme-aligned MIDI
        f0: np.ndarray, # Raw F0
        mel_spec: np.ndarray,
        alignment_score: float,
        metrics: Dict
    ) -> str:
        """
        Create a unified visualization of the alignment between features.
        Mel spectrogram is the background, with pitch and phonemes overlaid.

        Args:
            file_id: File identifier
            phoneme_frames: Phoneme IDs for each frame
            midi: Phoneme-aligned MIDI notes for each frame
            f0: Raw F0 values for each frame
            mel_spec: Mel spectrogram (shape: [num_frames, n_mels])
            alignment_score: Alignment quality score
            metrics: Dictionary of alignment metrics

        Returns:
            plot_path: Path to the saved plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))

        # Define x-axis (frame numbers)
        num_frames = len(phoneme_frames)
        x = np.arange(num_frames)
        n_mels = mel_spec.shape[1] # Get number of mel bins

        # Convert phoneme frames to numpy array
        phoneme_frames = np.array(phoneme_frames)

        # Find phoneme boundaries for vertical lines
        boundaries = np.where(np.diff(phoneme_frames) != 0)[0] + 1

        # --- Plot Mel Spectrogram as Background ---
        # Use actual mel range for extent, transpose mel_spec for imshow
        # Extent: [left, right, bottom, top] in data coordinates
        extent_mel = [0, num_frames, 0, n_mels]
        im = ax.imshow(mel_spec.T, aspect='auto', origin='lower', extent=extent_mel, cmap='viridis', alpha=0.9)
        ax.set_ylabel('Mel Bin')
        ax.set_ylim(0, n_mels) # Set primary Y-axis limits for Mel bins
        # Optional: Add colorbar for Mel spectrogram
        # cbar = fig.colorbar(im, ax=ax, label='Mel Power (dB)')
        # cbar.ax.tick_params(labelsize=8)

        # --- Create Secondary Y-Axis for Pitch ---
        ax_pitch = ax.twinx()

        # --- Prepare Pitch Data (Convert F0 to MIDI scale) ---
        f0_in_midi = np.full_like(f0, np.nan) # Initialize with NaN
        voiced_mask = f0 > 0
        # Use np.maximum to avoid log2(0) or log2(negative) if f0 has noise slightly below 0
        safe_f0 = np.maximum(f0[voiced_mask], 1e-5)
        f0_in_midi[voiced_mask] = 12 * np.log2(safe_f0 / 440.0) + 69
        # Clipping might not be strictly necessary with NaN handling, but good practice
        f0_in_midi = np.clip(f0_in_midi, 0, 127)

        # Determine MIDI range for y-axis (using both F0 and phoneme MIDI)
        all_midi_data = np.concatenate((f0_in_midi[~np.isnan(f0_in_midi)], midi[midi > 0]))
        if len(all_midi_data) > 0:
            min_midi_plot = max(0, np.min(all_midi_data) - 5)
            max_midi_plot = min(127, np.max(all_midi_data) + 5)
        else:
            min_midi_plot = 40 # Default range if no voiced frames
            max_midi_plot = 80

        # --- Plot Pitch Information on Secondary Axis ---
        # Plot detailed F0 contour
        ax_pitch.plot(x, f0_in_midi, color='cyan', linestyle='-', linewidth=1.5, label='F0 Contour (MIDI)')
        # Plot phoneme-aligned MIDI notes (blocky)
        ax_pitch.plot(x, midi, color='red', linestyle='-', linewidth=2.5, drawstyle='steps-post', label='Phoneme MIDI')

        # Set secondary Y-axis limits and label
        ax_pitch.set_ylim(min_midi_plot, max_midi_plot)
        ax_pitch.set_ylabel('MIDI Note')
        ax_pitch.grid(True, axis='y', linestyle=':', alpha=0.6, color='white') # Add horizontal grid for pitch

        # --- Plot Phoneme Boundaries (Full Height) ---
        for boundary in boundaries:
            # Draw line spanning the full plot height
            ax.axvline(x=boundary, ymin=0, ymax=1, color='white', linestyle='--', linewidth=1, alpha=0.7)

        # --- Add Phoneme Labels ---
        prev_boundary = 0
        # Position labels slightly above the bottom of the plot
        label_y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
        for boundary in np.concatenate((boundaries, [num_frames])):
            segment_center = (prev_boundary + boundary) / 2
            segment_width = boundary - prev_boundary

            # Only add label if segment is wide enough
            if segment_width > 3: # Adjust threshold as needed
                # Ensure the center index is valid
                center_idx = int(segment_center)
                if center_idx < num_frames:
                    phoneme_id = phoneme_frames[center_idx] # Get phoneme at center
                    phoneme_text = self.id_to_phone.get(phoneme_id, '?')

                    # Position label in the middle of segment, near the bottom
                    ax.text(segment_center, label_y_pos, phoneme_text,
                            horizontalalignment='center', verticalalignment='bottom',
                            fontsize=9, color='white', weight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.5, edgecolor='none'))

            prev_boundary = boundary

        # --- Set Labels, Title, Legend ---
        quality_label = "Good" if alignment_score > 0.7 else "Moderate" if alignment_score > 0.4 else "Poor"
        ax.set_title(f'Alignment Analysis for {file_id} - Quality: {quality_label} ({alignment_score:.2f})')
        ax.set_xlabel('Frames')
        # Primary Y-label (Mel Bin) already set
        # Secondary Y-label (MIDI Note) already set

        # Combine legends from both axes if needed, or just use pitch legend
        ax_pitch.legend(loc='upper right')

        # --- Add Metrics Annotation ---
        metrics_text = '\n'.join([
            f"Frames: {metrics['num_frames']}",
            f"Boundaries: {metrics['num_boundaries']}",
            f"Unique Phonemes: {metrics['unique_phonemes']}"
        ])
        plt.figtext(0.02, 0.98, metrics_text, fontsize=9, verticalalignment='top', color='black',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # --- Save Plot ---
        plot_dir = os.path.join(self.output_dir, "plots")
        Path(plot_dir).mkdir(exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{file_id}_alignment.png")
        # Adjust layout slightly to prevent title overlap with figtext
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return plot_path
        return plot_path
    
    def normalize_features(
        self, 
        f0: np.ndarray, 
        midi: np.ndarray, 
        mel_spec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features for model training.
        
        Args:
            f0: F0 contour
            midi: MIDI notes
            mel_spec: Mel spectrogram
            
        Returns:
            f0_norm: Normalized F0
            midi_norm: Normalized MIDI notes
            mel_spec_norm: Normalized mel spectrogram
        """
        # Normalize F0 (log scale mapped to [0, 1])
        f0_log = np.log(np.maximum(f0, 1e-5))
        f0_norm = (f0_log - np.log(self.f0_min)) / (np.log(self.f0_max) - np.log(self.f0_min))
        f0_norm = np.clip(f0_norm, 0, 1)
        
        # MIDI is already in a good range (0-127), no need to normalize further
        midi_norm = midi
        
        # Normalize mel spectrogram (dB scale, typically -80 to 0)
        mel_spec_norm = (mel_spec + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        return f0_norm, midi_norm, mel_spec_norm
    
    def process_file(self, base_name: str) -> Optional[Dict]:
        """
        Process a single file with error handling.
        
        Args:
            base_name: Base filename without extension
            
        Returns:
            data: Dictionary of processed data or None if error
        """
        try:
            # Build file paths
            wav_path = os.path.join(self.wav_dir, f"{base_name}.wav")
            lab_path = os.path.join(self.lab_dir, f"{base_name}.lab")
            
            # Check if files exist
            if not os.path.exists(wav_path) or not os.path.exists(lab_path):
                logger.warning(f"Missing files for {base_name}")
                return None
            
            logger.info(f"Processing {base_name}")
            
            # Step 1: Parse phoneme timings first
            phonemes, start_times, end_times = self.parse_lab_file(lab_path)
            logger.info(f"{base_name} - Parsed {len(phonemes)} phonemes from LAB file")
            
            # Skip if fewer than 3 phonemes
            if len(phonemes) < 3:
                logger.warning(f"{base_name} - Skipping file with only {len(phonemes)} phonemes (minimum 3 required)")
                self.stats['skipped_few_phonemes'] += 1
                return None
            
            # Step 2: Load and process audio
            waveform, mel_spec, num_frames = self.load_and_process_audio(wav_path)
            logger.info(f"{base_name} - Extracted mel spectrogram with {num_frames} frames and shape {mel_spec.shape}")
            
            # Step 3: Extract F0 contour
            f0, _ = self.extract_f0(waveform)  # We'll generate MIDI from phoneme alignment instead
            logger.info(f"{base_name} - Extracted F0 contour with shape {f0.shape}")
            
            # Ensure F0 length matches mel spectrogram
            if len(f0) > num_frames:
                f0 = f0[:num_frames]
            elif len(f0) < num_frames:
                # Pad with zeros
                f0 = np.pad(f0, (0, num_frames - len(f0)))
            
            # Step 4: Align phonemes to frames and create 1:1 phoneme-MIDI mapping
            frame_phonemes, frame_midi = self.align_phonemes_and_midi(
                phonemes, start_times, end_times, f0, num_frames
            )
            logger.info(f"{base_name} - Aligned phonemes and MIDI to {len(frame_phonemes)} frames")
            
            # Step 5: Verify alignment quality
            alignment_score, alignment_metrics = self.verify_alignment(
                frame_phonemes, frame_midi, f0, mel_spec
            )
            logger.info(f"{base_name} - Alignment score: {alignment_score:.2f}")
            
            # Log number of unique phonemes after alignment
            unique_phonemes = len(set(frame_phonemes)) - (1 if 0 in frame_phonemes else 0)
            logger.info(f"{base_name} - Unique phonemes after alignment: {unique_phonemes}")
            
            # Skip if alignment is too poor
            if alignment_score < 0.2 or unique_phonemes <= 1:
                logger.warning(f"{base_name} - Poor alignment (score: {alignment_score:.2f}, unique phonemes: {unique_phonemes}). Skipping.")
                self.stats['skipped_files'] += 1
                return None
            
            # Step 6: Normalize features
            f0_norm, midi_norm, mel_spec_norm = self.normalize_features(f0, frame_midi, mel_spec)
            
            # Step 7: Create visualization
            plot_path = self.plot_alignment(
                base_name, frame_phonemes, frame_midi, f0, mel_spec, 
                alignment_score, alignment_metrics
            )
            logger.info(f"{base_name} - Created alignment plot at {plot_path}")
            
            # Step 8: Prepare data for storage
            data = {
                'phonemes': np.array(frame_phonemes, dtype=np.int32),
                'midi': midi_norm,
                'f0': f0_norm,
                'mel_spectrogram': mel_spec_norm,
                'length': num_frames,
                'alignment_score': alignment_score,
                'alignment_metrics': alignment_metrics,
                'plot_path': plot_path
            }
            
            # Update stats
            self.stats['processed_files'] += 1
            self.stats['total_frames'] += num_frames
            self.stats['total_phonemes'] += len(phonemes)
            self.stats['alignment_scores'].append(alignment_score)
            
            return data
            
        except Exception as e:
            logger.error(f"Error processing {base_name}: {e}", exc_info=True)
            self.stats['skipped_files'] += 1
            return None
    
    def convert_numpy_types(self, obj: Any) -> Any:
        """
        Convert NumPy types to standard Python types for JSON serialization.
        
        Args:
            obj: Any object that might contain NumPy types
            
        Returns:
            Object with NumPy types converted to Python standard types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        else:
            return obj
        
    def process_dataset(self):
        """
        Process all files in the dataset.
        """
        logger.info(f"Starting preprocessing with config: sample_rate={self.sample_rate}, n_fft={self.n_fft}, hop_length={self.hop_length}, n_mels={self.n_mels}")
        
        # Create output directories
        Path(self.output_dir).mkdir(exist_ok=True)
        
        # Get list of WAV files with corresponding LAB files
        wav_files = [os.path.splitext(f)[0] for f in os.listdir(self.wav_dir) if f.endswith('.wav')]
        lab_files = [os.path.splitext(f)[0] for f in os.listdir(self.lab_dir) if f.endswith('.lab')]
        
        # Find intersection
        valid_files = sorted(list(set(wav_files) & set(lab_files)))
        self.stats['total_files'] = len(valid_files)
        
        logger.info(f"Found {len(valid_files)} valid files with both WAV and LAB")
        
        # Process files and store data
        with h5py.File(self.output_file, 'w') as h5f:
            # Create dataset groups
            metadata_grp = h5f.create_group('metadata')
            data_grp = h5f.create_group('data')
            
            # Process each file
            for base_name in tqdm(valid_files, desc="Processing files"):
                data = self.process_file(base_name)
                
                if data is not None:
                    # Create dataset for this file
                    file_grp = data_grp.create_group(base_name)
                    file_grp.create_dataset('phonemes', data=data['phonemes'])
                    file_grp.create_dataset('midi', data=data['midi'])
                    file_grp.create_dataset('f0', data=data['f0'])
                    file_grp.create_dataset('mel_spectrogram', data=data['mel_spectrogram'])
                    file_grp.attrs['length'] = data['length']
                    file_grp.attrs['alignment_score'] = data['alignment_score']
                    
                    # Convert NumPy types and store metrics as JSON string
                    file_grp.attrs['alignment_metrics'] = json.dumps(self.convert_numpy_types(data['alignment_metrics']))
            
            # Save metadata - make sure all values are native Python types
            metadata_grp.create_dataset('phone_to_id', data=np.string_([f"{k}:{v}" for k, v in self.phone_to_id.items()]))
            metadata_grp.create_dataset('id_to_phone', data=np.string_([f"{k}:{v}" for k, v in self.id_to_phone.items()]))
            metadata_grp.create_dataset('midi_min', data=int(self.midi_min))
            metadata_grp.create_dataset('midi_max', data=int(self.midi_max))
            metadata_grp.create_dataset('n_phonemes', data=int(len(self.phone_to_id)))
            metadata_grp.create_dataset('sample_rate', data=int(self.sample_rate))
            metadata_grp.create_dataset('hop_length', data=int(self.hop_length))
            metadata_grp.create_dataset('n_mels', data=int(self.n_mels))
            
            # Update stats
            self.stats['unique_phonemes'] = len(self.phone_to_id)
            if self.stats['alignment_scores']:
                self.stats['avg_alignment_score'] = float(np.mean(self.stats['alignment_scores']))
            
             # Store stats - make sure to convert any NumPy types
            for key, value in self.stats.items():
                if key != 'alignment_scores':  # Skip the list of scores
                    if isinstance(value, (np.integer, np.floating, np.ndarray)):
                        metadata_grp.attrs[key] = self.convert_numpy_types(value)
                    else:
                        metadata_grp.attrs[key] = value
        
        # Save phoneme mapping as text file for reference
        phoneme_map_path = os.path.join(self.output_dir, "phoneme_map.txt")
        with open(phoneme_map_path, 'w') as f:
            for phoneme, idx in sorted(self.phone_to_id.items(), key=lambda x: x[1]):
                f.write(f"{idx}: {phoneme}\n")
        
        # Save a summary report
        self._save_summary_report()
        
        logger.info(f"Preprocessing complete. Processed {self.stats['processed_files']} files successfully, skipped {self.stats['skipped_files']} files.")
        logger.info(f"Files skipped due to too few phonemes: {self.stats['skipped_few_phonemes']}")
        logger.info(f"Phoneme vocabulary size: {len(self.phone_to_id)}")
        logger.info(f"MIDI range: {self.midi_min} to {self.midi_max}")
        if self.stats['alignment_scores']:
            logger.info(f"Average alignment score: {self.stats['avg_alignment_score']:.2f}")
        logger.info(f"Output saved to {self.output_file}")
        logger.info(f"Phoneme mapping saved to {phoneme_map_path}")
    
    def _save_summary_report(self):
        """
        Save a summary report of the preprocessing.
        """
        report_path = os.path.join(self.output_dir, "preprocessing_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Singing Voice Synthesis Preprocessing Report\n\n")
            
            f.write("## Dataset Summary\n")
            f.write(f"- Total files: {self.stats['total_files']}\n")
            f.write(f"- Successfully processed files: {self.stats['processed_files']}\n")
            f.write(f"- Skipped files: {self.stats['skipped_files']}\n")
            f.write(f"- Files skipped due to too few phonemes: {self.stats['skipped_few_phonemes']}\n")
            f.write(f"- Total frames: {self.stats['total_frames']}\n")
            f.write(f"- Total phonemes: {self.stats['total_phonemes']}\n")
            f.write(f"- Unique phonemes: {self.stats['unique_phonemes']}\n\n")
            
            if self.stats['alignment_scores']:
                f.write("## Alignment Quality\n")
                f.write(f"- Average alignment score: {self.stats['avg_alignment_score']:.2f}\n")
                f.write(f"- Min alignment score: {min(self.stats['alignment_scores']):.2f}\n")
                f.write(f"- Max alignment score: {max(self.stats['alignment_scores']):.2f}\n\n")
            
            f.write("## Feature Dimensions\n")
            f.write(f"- Sample rate: {self.sample_rate} Hz\n")
            f.write(f"- FFT size: {self.n_fft}\n")
            f.write(f"- Hop length: {self.hop_length}\n")
            f.write(f"- Mel bins: {self.n_mels}\n")
            f.write(f"- MIDI range: {self.midi_min} to {self.midi_max}\n\n")
            
            f.write("## Phoneme Inventory\n")
            for phoneme, idx in sorted(self.phone_to_id.items(), key=lambda x: x[1]):
                f.write(f"- {idx}: `{phoneme}`\n")
        
        logger.info(f"Summary report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess singing voice data")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test mode with single file")
    
    args = parser.parse_args()
    
    preprocessor = SVSPreprocessor(config_path=args.config)
    
    if args.test:
        # Run on a single file for testing
        wav_files = [os.path.splitext(f)[0] for f in os.listdir(preprocessor.wav_dir) if f.endswith('.wav')]
        lab_files = [os.path.splitext(f)[0] for f in os.listdir(preprocessor.lab_dir) if f.endswith('.lab')]
        valid_files = list(set(wav_files) & set(lab_files))
        
        if valid_files:
            test_file = valid_files[0]
            logger.info(f"Running in test mode with file: {test_file}")
            data = preprocessor.process_file(test_file)
            
            if data:
                logger.info(f"Test successful: {test_file} processed with alignment score {data['alignment_score']:.2f}")
                logger.info(f"Plot saved to: {data['plot_path']}")
            else:
                logger.error(f"Test failed: Could not process {test_file}")
        else:
            logger.error("No valid files found for testing")
    else:
        # Process the full dataset
        preprocessor.process_dataset()