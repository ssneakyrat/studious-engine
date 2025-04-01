import os
import numpy as np
import h5py
import torch
import torchaudio
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from tqdm import tqdm
import logging
import random
import yaml
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocessor')

class SVSPreprocessor:
    def __init__(self, root_dir="assets/gin", output_file="svs_dataset.h5", 
                 sr=22050, n_fft=1024, hop_length=256, n_mels=80,
                 lab_time_unit='samples'):
        """
        Initialize the SVS Preprocessor
        
        Parameters:
            root_dir: Root directory for data
            output_file: Output h5 file path
            sr: Sample rate
            n_fft: FFT window size
            hop_length: Hop length for spectrogram
            n_mels: Number of mel bins
            lab_time_unit: Unit of time in LAB files ('samples', 'seconds', or 'auto')
        """
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "wav")
        self.lab_dir = os.path.join(root_dir, "lab")
        self.output_file = os.path.join(root_dir, output_file)
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.lab_time_unit = lab_time_unit
        
        # Initialize mappings
        self.phone_to_id = {'<PAD>': 0}
        self.id_to_phone = {0: '<PAD>'}
        self.midi_min = 127
        self.midi_max = 0
        
        # Create debug directory
        os.makedirs(os.path.join(root_dir, "debug"), exist_ok=True)
        self.debug_dir = os.path.join(root_dir, "debug")
        
    def extract_f0(self, wav_path):
        """
        Extract F0 contour using Parselmouth/Praat with improved parameters for singing voice.
        """
        sound = parselmouth.Sound(wav_path)
        
        # More suitable parameters for singing voice analysis
        time_step = 0.005  # 5ms - finer resolution for singing
        f0_min = 80  # Lower bound for singing voice (including male singers)
        f0_max = 800  # Upper bound for singing voice (including high female notes)
        
        pitch = call(sound, "To Pitch", time_step, f0_min, f0_max)
        pitch_values = pitch.selected_array['frequency']
        
        # Get time points for each F0 value
        times = np.arange(len(pitch_values)) * time_step
        
        # Interpolate through unvoiced regions with improved method
        pitch_values[pitch_values == 0] = np.nan
        indices = np.arange(len(pitch_values))
        valid_indices = ~np.isnan(pitch_values)
        
        if np.any(valid_indices):
            # Use linear interpolation for reliability
            pitch_values = np.interp(
                indices, 
                indices[valid_indices], 
                pitch_values[valid_indices],
                left=pitch_values[valid_indices][0],  # Extend first valid value
                right=pitch_values[valid_indices][-1]  # Extend last valid value
            )
        else:
            # If no voiced regions found, set to a default value
            pitch_values = np.ones_like(pitch_values) * 100.0
            logger.warning(f"No voiced regions found in {wav_path}. Using default F0.")
        
        # Convert to MIDI with smoother handling
        midi_values = 12 * np.log2(np.maximum(pitch_values, 1e-5) / 440.0) + 69
        midi_values = np.clip(midi_values, 0, 127)
        
        # Update MIDI range for dataset statistics
        valid_midi = midi_values[midi_values > 0]
        if len(valid_midi) > 0:
            self.midi_min = min(self.midi_min, int(np.floor(np.min(valid_midi))))
            self.midi_max = max(self.midi_max, int(np.ceil(np.max(valid_midi))))
        
        # Resample to match hop_length timing
        target_length = int(sound.duration * self.sr / self.hop_length) + 1
        times_sec = np.arange(target_length) * self.hop_length / self.sr
        
        # Resample F0 and MIDI to match spectrogram frames
        f0_resampled = np.interp(times_sec, times, pitch_values)
        midi_resampled = np.interp(times_sec, times, midi_values)
        
        return f0_resampled, midi_resampled, sound.duration
    
    def determine_lab_time_unit(self, lab_path, wav_duration):
        """
        Automatically determine the time unit in LAB files by analyzing the values.
        
        Parameters:
            lab_path: Path to the LAB file
            wav_duration: Duration of the WAV file in seconds
            
        Returns:
            String indicating time unit ('samples', 'seconds', or 'milliseconds')
        """
        with open(lab_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return 'samples'  # Default if empty
            
        # Extract the last timestamp (end of the last phoneme)
        try:
            last_line = lines[-1].strip().split()
            if len(last_line) >= 2:
                last_time = float(last_line[1])
            else:
                return 'samples'  # Default if parsing fails
        except:
            return 'samples'  # Default if parsing fails
        
        # Compare with audio duration to determine unit
        if last_time <= wav_duration * 1.1:  # Allow 10% margin
            return 'seconds'
        elif last_time <= wav_duration * 1100:  # 1000 ms/s + 10% margin
            return 'milliseconds'
        else:
            return 'samples'
    
    def read_lab_file(self, lab_path, wav_duration):
        """
        Read LAB file with automatic time unit detection and extract phoneme sequence and durations.
        
        Parameters:
            lab_path: Path to the LAB file
            wav_duration: Duration of the WAV file in seconds
            
        Returns:
            phonemes: List of phoneme IDs
            durations: List of phoneme durations in frames
            boundaries: Original time boundaries for debugging
            time_unit: Detected time unit
        """
        phonemes = []
        durations = []
        boundaries = []  # Store the original timestamps for debugging
        
        # Determine time unit if set to auto
        time_unit = self.lab_time_unit
        if time_unit == 'auto':
            time_unit = self.determine_lab_time_unit(lab_path, wav_duration)
            logger.info(f"Detected time unit for {lab_path}: {time_unit}")
        
        # Debug - print the raw LAB file content
        with open(lab_path, 'r') as f:
            lines = f.readlines()
            logger.debug(f"LAB file {lab_path} contains {len(lines)} lines")
            for line in lines[:5]:  # Print first few lines for debugging
                logger.debug(f"LAB line: {line.strip()}")
        
        with open(lab_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:  # Ensure we have at least start, end, phoneme
                    start_time, end_time, phoneme = parts[0], parts[1], parts[2]
                    start_time = float(start_time)
                    end_time = float(end_time)
                    
                    # Convert to frames based on time unit
                    if time_unit == 'samples':
                        start_frame = start_time / self.hop_length
                        end_frame = end_time / self.hop_length
                    elif time_unit == 'seconds':
                        start_frame = start_time * self.sr / self.hop_length
                        end_frame = end_time * self.sr / self.hop_length
                    elif time_unit == 'milliseconds':
                        start_frame = start_time * self.sr / (1000 * self.hop_length)
                        end_frame = end_time * self.sr / (1000 * self.hop_length)
                    
                    # Calculate duration in frames (keep as float for precision)
                    duration = end_frame - start_frame
                    
                    # Skip if duration is too short (less than 0.5 frame)
                    if duration < 0.5:
                        logger.warning(f"Skipping short phoneme {phoneme} with duration {duration:.2f} frames")
                        continue
                    
                    # Add to phoneme mapping if new
                    if phoneme not in self.phone_to_id:
                        idx = len(self.phone_to_id)
                        self.phone_to_id[phoneme] = idx
                        self.id_to_phone[idx] = phoneme
                    
                    phonemes.append(self.phone_to_id[phoneme])
                    durations.append(duration)
                    boundaries.append((start_time, end_time, phoneme))
        
        # Verify we have valid phonemes
        if not phonemes:
            logger.warning(f"No valid phonemes found in {lab_path}")
        else:
            logger.debug(f"Extracted {len(phonemes)} phonemes from {lab_path}")
            logger.debug(f"First 5 phonemes: {[self.id_to_phone[p] for p in phonemes[:5]]}")
        
        return phonemes, durations, boundaries, time_unit
    
    def extract_mel_spectrogram(self, wav_path):
        """
        Extract mel spectrogram with improved parameters for singing voice.
        """
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sr)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate mel spectrogram with better parameters for singing voice
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=40,  # Lower minimum frequency to capture low notes
            f_max=8000  # Higher maximum to capture overtones
        )(waveform)
        
        # Convert to dB scale
        mel_spec = torchaudio.transforms.AmplitudeToDB()(waveform)
        
        # Return as numpy array
        return mel_spec.squeeze().numpy()
    
    def detect_acoustic_boundaries(self, mel_spec):
        """
        Detect acoustic boundaries based on spectral changes.
        
        Parameters:
            mel_spec: Mel spectrogram (frames, mel_bins)
            
        Returns:
            boundaries: List of frame indices where significant changes occur
            boundary_scores: Array of change scores at each frame
        """
        # Calculate spectral flux (difference between consecutive frames)
        spec_diff = np.zeros(mel_spec.shape[0])
        
        for i in range(1, mel_spec.shape[0]):
            spec_diff[i] = np.mean(np.abs(mel_spec[i] - mel_spec[i-1]))
        
        # Normalize differences
        if np.max(spec_diff) > 0:
            spec_diff = spec_diff / np.max(spec_diff)
        
        # Find peaks (potential boundaries)
        # Adjust min_distance and height parameters based on your data
        peaks, _ = find_peaks(spec_diff, height=0.2, distance=5)
        
        return peaks, spec_diff
    
    def align_features(self, phonemes, durations, f0, midi, mel_spec):
        """
        FIXED: Align phoneme-level features with frame-level features.
        Ensures proper mapping of phonemes to frames.
        
        Parameters:
            phonemes: List of phoneme IDs
            durations: List of phoneme durations (in frames)
            f0: F0 contour
            midi: MIDI notes
            mel_spec: Mel spectrogram
            
        Returns:
            Aligned features
        """
        # Check if we have phonemes
        if not phonemes:
            logger.error("No phonemes to align! Aborting alignment.")
            # Create dummy frame-level features
            dummy_length = min(len(f0), mel_spec.shape[0])
            return ([0] * dummy_length, [0] * dummy_length, f0[:dummy_length], 
                    midi[:dummy_length], mel_spec[:dummy_length], [], [])
        
        # Calculate total expected frames based on durations
        expected_frames = int(np.ceil(sum(durations)))
        actual_frames = mel_spec.shape[0]
        
        logger.info(f"Expected frames: {expected_frames}, Actual frames: {actual_frames}")
        
        # Debug: print phonemes and durations
        logger.debug(f"Phonemes to align: {[self.id_to_phone[p] for p in phonemes]}")
        logger.debug(f"Durations: {[f'{d:.2f}' for d in durations]}")
        
        # Create frame-level phoneme representation
        frame_phonemes = []
        
        # Assign phonemes to frames based on durations
        for i, (phone, duration) in enumerate(zip(phonemes, durations)):
            # Calculate how many frames this phoneme should occupy
            phone_frames = max(1, int(round(duration)))
            
            # Add this phoneme's ID to the frame_phonemes list for each frame
            frame_phonemes.extend([phone] * phone_frames)
            
            logger.debug(f"Assigned phoneme {self.id_to_phone[phone]} to {phone_frames} frames")
        
        # If we have too many frames, truncate
        if len(frame_phonemes) > actual_frames:
            logger.warning(f"Truncating frame_phonemes from {len(frame_phonemes)} to {actual_frames}")
            frame_phonemes = frame_phonemes[:actual_frames]
        
        # If we have too few frames, extend with last phoneme
        while len(frame_phonemes) < actual_frames:
            frame_phonemes.append(phonemes[-1])  # Use last phoneme to pad
        
        # Make sure our length is exactly right
        assert len(frame_phonemes) == actual_frames, "Frame phonemes length mismatch"
        
        # Create frame-level MIDI values with the same approach
        frame_midi = []
        current_frame = 0
        
        for phone, duration in zip(phonemes, durations):
            # Calculate how many frames this phoneme should occupy
            phone_frames = max(1, int(round(duration)))
            
            # Calculate average MIDI note for this phoneme
            start_frame = current_frame
            end_frame = min(current_frame + phone_frames, len(midi))
            
            if start_frame < end_frame:
                # Simply use the average MIDI value for this segment
                avg_midi = int(round(np.mean(midi[start_frame:end_frame])))
                frame_midi.extend([avg_midi] * phone_frames)
            else:
                # If we somehow went past the end, use a safe default
                frame_midi.extend([60] * phone_frames)  # Middle C as safe default
                
            current_frame += phone_frames
        
        # Ensure all features have the same length
        target_length = min(len(frame_phonemes), len(f0), mel_spec.shape[0])
        frame_phonemes = frame_phonemes[:target_length]
        frame_midi = frame_midi[:target_length] if len(frame_midi) > target_length else frame_midi + [60] * (target_length - len(frame_midi))
        f0_aligned = f0[:target_length]
        midi_aligned = midi[:target_length]
        mel_spec_aligned = mel_spec[:target_length]
        
        # Detect acoustic boundaries
        acoustic_boundaries, boundary_scores = self.detect_acoustic_boundaries(mel_spec_aligned)
        
        return frame_phonemes, frame_midi, f0_aligned, midi_aligned, mel_spec_aligned, acoustic_boundaries, boundary_scores
    
    def plot_alignment(self, filename, phonemes, midi, f0, mel_spec, acoustic_boundaries=None, boundary_scores=None):
        """
        Enhanced plot of alignment showing phonemes, MIDI notes, F0 contour, and mel spectrogram
        with improved visualizations and metrics.
        """
        # Check for valid data
        if len(phonemes) == 0:
            logger.error(f"No phonemes to plot for {filename}")
            return f"{filename}_error.png", 0.0
        
        # Create figure with gridspec for better layout control
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 3], hspace=0.3)
        
        # Shared x-axis for all plots
        ax_phonemes = fig.add_subplot(gs[0])  # Phonemes on top
        ax_midi = fig.add_subplot(gs[1], sharex=ax_phonemes)  # MIDI notes
        ax_f0 = fig.add_subplot(gs[2], sharex=ax_phonemes)  # F0 contour
        ax_boundaries = fig.add_subplot(gs[3], sharex=ax_phonemes)  # Boundary detection scores
        ax_mel = fig.add_subplot(gs[4], sharex=ax_phonemes)  # Mel spectrogram at bottom
        
        # Get phoneme text labels
        phone_text = [self.id_to_phone[p] for p in phonemes]
        
        # Find phoneme boundaries
        boundaries = [0]  # Start with the first frame
        for i in range(1, len(phonemes)):
            if phonemes[i] != phonemes[i-1]:
                boundaries.append(i)
        boundaries.append(len(phonemes))  # Add the end frame
        
        # Debug - check how many unique phonemes we have
        unique_phonemes = len(set(phonemes))
        logger.info(f"Plotting {filename} with {unique_phonemes} unique phonemes, {len(boundaries)-1} segments")
        
        # Draw colored regions for each phoneme
        cmap = plt.cm.get_cmap('tab20', 20)  # Use modulo for colors
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            phoneme_idx = phonemes[start]
            
            # Fill region for this phoneme
            ax_phonemes.axvspan(start, end, color=cmap(phoneme_idx % 20), alpha=0.7)
            
            # Add phoneme label
            midpoint = (start + end) // 2
            phone = phone_text[start]
            
            # Debug - check phoneme label
            logger.debug(f"Phoneme at position {start}-{end}: {phone} (ID: {phoneme_idx})")
            
            ax_phonemes.text(midpoint, 0.5, phone, 
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontweight='bold')
        
        # Remove y ticks and spines for phoneme plot
        ax_phonemes.set_yticks([])
        for spine in ax_phonemes.spines.values():
            spine.set_visible(False)
        ax_phonemes.set_ylabel('Phoneme')
        
        # Plot MIDI notes with vertical lines at changes
        ax_midi.plot(midi, 'b-', linewidth=2)
        ax_midi.set_ylabel('MIDI Note')
        
        # Add vertical lines at MIDI note changes
        midi_changes = [0]
        for i in range(1, len(midi)):
            if abs(midi[i] - midi[i-1]) > 0.5:  # Threshold for significant change
                midi_changes.append(i)
                ax_midi.axvline(x=i, color='b', linestyle='--', alpha=0.3)
        
        # Plot F0 contour
        ax_f0.plot(f0, 'r-', linewidth=2)
        ax_f0.set_ylabel('F0 (Hz)')
        
        # Plot boundary detection scores if available
        if boundary_scores is not None:
            ax_boundaries.plot(boundary_scores, 'g-', linewidth=1.5)
            ax_boundaries.set_ylabel('Boundary\nScore')
            # Mark detected acoustic boundaries
            if acoustic_boundaries is not None:
                for boundary in acoustic_boundaries:
                    ax_boundaries.axvline(x=boundary, color='m', linestyle='-', alpha=0.7)
        
        # Plot mel spectrogram correctly oriented
        # NOTE: The mel_spec shape should be (frames, mels)
        # For visualization: x-axis=frames, y-axis=mel bins (frequencies)
        im = ax_mel.imshow(mel_spec.T, aspect='auto', origin='lower', cmap='viridis')
        ax_mel.set_ylabel('Mel Bins')
        ax_mel.set_xlabel('Frames')
        
        # Add vertical lines at phoneme boundaries to all plots
        for boundary in boundaries[1:-1]:  # Skip first and last
            ax_midi.axvline(x=boundary, color='k', linestyle='-', alpha=0.7)
            ax_f0.axvline(x=boundary, color='k', linestyle='-', alpha=0.7)
            ax_boundaries.axvline(x=boundary, color='k', linestyle='-', alpha=0.7)
            ax_mel.axvline(x=boundary, color='w', linestyle='-', alpha=0.7)
        
        # Analyze alignment quality with improved metrics
        alignment_scores = []
        
        for i, boundary in enumerate(boundaries[1:-1]):  # Skip first and last
            window = 5  # frames to check on each side (increased for better context)
            start = max(0, boundary - window)
            end = min(len(midi) - 1, boundary + window)
            
            # Check if there's a significant MIDI change near this boundary
            if end > start:
                midi_change = max(midi[start:end]) - min(midi[start:end])
                midi_gradient = np.mean(np.abs(np.diff(midi[start:end])))
            else:
                midi_change = 0
                midi_gradient = 0
                
            # Check if there's a significant F0 change near this boundary
            if end > start:
                f0_change = max(f0[start:end]) - min(f0[start:end])
                f0_gradient = np.mean(np.abs(np.diff(f0[start:end])))
            else:
                f0_change = 0
                f0_gradient = 0
            
            # Check for spectral change by looking at mel spectrogram difference
            if boundary > 0 and boundary < mel_spec.shape[0] - 1:
                spec_diff = np.mean(np.abs(mel_spec[boundary+1] - mel_spec[boundary-1]))
                
                # Calculate average difference in the region for normalization
                region_diffs = []
                for j in range(max(1, boundary-5), min(mel_spec.shape[0]-1, boundary+5)):
                    if j+1 < mel_spec.shape[0] and j-1 >= 0:
                        region_diffs.append(np.mean(np.abs(mel_spec[j+1] - mel_spec[j-1])))
                
                # Normalize by average difference in the region
                avg_diff = np.mean(region_diffs) if region_diffs else 1.0
                spec_score = spec_diff / avg_diff if avg_diff > 0 else 0
            else:
                spec_score = 0
            
            # Check if this boundary matches an acoustic boundary
            acoustic_match = 0
            if acoustic_boundaries is not None and len(acoustic_boundaries) > 0:
                # Find closest acoustic boundary
                distances = np.abs(np.array(acoustic_boundaries) - boundary)
                min_dist = np.min(distances)
                # Score based on distance (high if close, low if far)
                acoustic_match = max(0, 1 - min_dist / 10)
            
            # Combined score with improved weighting
            boundary_score = (
                0.3 * (1.0 if midi_change > 2 else midi_change / 2) +  # MIDI note change
                0.2 * (1.0 if midi_gradient > 0.5 else midi_gradient / 0.5) +  # MIDI gradient
                0.15 * (1.0 if f0_change > 20 else f0_change / 20) +  # F0 change
                0.15 * (1.0 if f0_gradient > 5 else f0_gradient / 5) +  # F0 gradient
                0.2 * min(1.0, spec_score) +  # Spectral change
                0.2 * acoustic_match  # Match to detected acoustic boundary
            ) / 1.2  # Normalize to 0-1 range
            
            alignment_scores.append(boundary_score)
            
            # Color code the boundary quality
            if boundary_score > 0.7:  # Good alignment
                color = 'g'
            elif boundary_score > 0.4:  # Moderate alignment
                color = 'y'
            else:  # Poor alignment
                color = 'r'
                
            # Highlight boundary quality in all plots
            ax_midi.axvspan(boundary-1, boundary+1, color=color, alpha=0.3)
            ax_f0.axvspan(boundary-1, boundary+1, color=color, alpha=0.3)
            ax_boundaries.axvspan(boundary-1, boundary+1, color=color, alpha=0.3)
            ax_mel.axvspan(boundary-1, boundary+1, color=color, alpha=0.3)
        
        # Calculate overall alignment quality
        alignment_percentage = 100 * sum(alignment_scores) / max(1, len(alignment_scores))
        
        # Add title with alignment quality
        quality_label = "Good" if alignment_percentage > 70 else "Moderate" if alignment_percentage > 40 else "Poor"
        fig.suptitle(f'Alignment Analysis for {filename}\n'
                    f'Alignment Quality: {quality_label} ({alignment_percentage:.1f}%)', 
                    fontsize=14)
        
        # Ensure all x-axes show the same range
        for ax in [ax_phonemes, ax_midi, ax_f0, ax_boundaries, ax_mel]:
            ax.set_xlim(0, len(phonemes))
        
        # Hide x-tick labels for all but the bottom plot
        for ax in [ax_phonemes, ax_midi, ax_f0, ax_boundaries]:
            plt.setp(ax.get_xticklabels(), visible=False)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)  # Make room for suptitle
        
        # Save the figure with improved filename
        plot_path = os.path.join(self.debug_dir, f"{filename}_alignment.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        return plot_path, alignment_percentage
    
    def process(self):
        """Process all WAV and LAB files and save to h5py file with improved alignment."""
        Path("logs").mkdir(exist_ok=True)
        
        # Get list of WAV files with corresponding LAB files
        wav_files = [f for f in os.listdir(self.wav_dir) if f.endswith('.wav')]
        valid_files = []
        
        for wav_file in wav_files:
            base_name = os.path.splitext(wav_file)[0]
            lab_file = f"{base_name}.lab"
            if os.path.exists(os.path.join(self.lab_dir, lab_file)):
                valid_files.append(base_name)
        
        logger.info(f"Found {len(valid_files)} valid files with both WAV and LAB")
        
        # Setup more verbose debugging for the first few files
        for i in range(min(5, len(valid_files))):
            logger.debug(f"Debug file: {valid_files[i]}")
            
            # Open and read the LAB file directly
            lab_path = os.path.join(self.lab_dir, f"{valid_files[i]}.lab")
            with open(lab_path, 'r') as f:
                lab_content = f.read()
                logger.debug(f"LAB content for {valid_files[i]}:\n{lab_content[:500]}...")
        
        # Collect alignment statistics
        alignment_stats = []
        
        # Process files and store data
        with h5py.File(self.output_file, 'w') as h5f:
            # Create dataset groups
            metadata_grp = h5f.create_group('metadata')
            data_grp = h5f.create_group('data')
            debug_grp = h5f.create_group('debug')
            
            processed_count = 0
            random_sample = None
            
            for base_name in tqdm(valid_files, desc="Processing files"):
                wav_path = os.path.join(self.wav_dir, f"{base_name}.wav")
                lab_path = os.path.join(self.lab_dir, f"{base_name}.lab")
                
                try:
                    # Extract features
                    f0, midi, wav_duration = self.extract_f0(wav_path)
                    phonemes, durations, boundaries, time_unit = self.read_lab_file(lab_path, wav_duration)
                    
                    if not phonemes:
                        logger.warning(f"No phonemes found in {lab_path}, skipping")
                        continue
                    
                    # Log file information with more detail
                    unique_phonemes = len(set(phonemes))
                    logger.info(f"Processing {base_name} - {len(phonemes)} phonemes, {unique_phonemes} unique, time unit: {time_unit}")
                    
                    # Print first few phonemes for debugging
                    if len(phonemes) > 0:
                        first_phonemes = [self.id_to_phone[p] for p in phonemes[:min(10, len(phonemes))]]
                        logger.info(f"First phonemes: {first_phonemes}")
                    
                    # Skip files with too few phonemes
                    if len(phonemes) < 3:
                        logger.warning(f"Skipping {base_name} - has fewer than 3 phonemes")
                        continue
                    
                    mel_spec = self.extract_mel_spectrogram(wav_path)
                    
                    # For each file, save original boundaries for debugging
                    boundary_data = '\n'.join([f"{start}\t{end}\t{self.id_to_phone[phonemes[i]]}" 
                                              for i, (start, end, _) in enumerate(boundaries[:len(phonemes)])])
                    debug_grp.create_dataset(f"{base_name}_boundaries", data=np.string_(boundary_data))
                    
                    # DEBUG: Print phoneme/duration data for the first few files
                    if processed_count < 3:
                        logger.info(f"---- Phoneme/Duration data for {base_name} ----")
                        for i, (phone, duration) in enumerate(zip(phonemes, durations)):
                            phoneme_text = self.id_to_phone[phone]
                            logger.info(f"  Phoneme {i}: {phoneme_text}, Duration: {duration:.2f} frames")
                    
                    # Align features with improved algorithm
                    frame_phonemes, frame_midi, f0_aligned, midi_aligned, mel_spec_aligned, acoustic_boundaries, boundary_scores = (
                        self.align_features(phonemes, durations, f0, midi, mel_spec)
                    )
                    
                    # Check number of unique phonemes in frame_phonemes
                    unique_frame_phonemes = len(set(frame_phonemes))
                    logger.info(f"Unique phonemes after alignment: {unique_frame_phonemes}")
                    
                    # Create debug plots
                    plot_path, alignment_score = self.plot_alignment(
                        base_name, frame_phonemes, midi_aligned, f0_aligned, 
                        mel_spec_aligned, acoustic_boundaries, boundary_scores
                    )
                    
                    # Log alignment score
                    alignment_stats.append((base_name, alignment_score))
                    
                    # Create dataset for this file
                    file_grp = data_grp.create_group(base_name)
                    file_grp.create_dataset('phonemes', data=frame_phonemes)
                    file_grp.create_dataset('midi', data=frame_midi)
                    file_grp.create_dataset('f0', data=f0_aligned)
                    file_grp.create_dataset('mel_spectrogram', data=mel_spec_aligned)
                    file_grp.create_dataset('alignment_score', data=alignment_score)
                    
                    processed_count += 1
                    
                    # Select a random sample for visualization
                    if random_sample is None or random.random() < 1.0/processed_count:
                        random_sample = {
                            'filename': base_name,
                            'phonemes': frame_phonemes,
                            'midi': midi_aligned,
                            'f0': f0_aligned,
                            'mel_spectrogram': mel_spec_aligned,
                            'acoustic_boundaries': acoustic_boundaries,
                            'boundary_scores': boundary_scores
                        }
                        
                except Exception as e:
                    logger.error(f"Error processing {base_name}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
            
            # Save metadata
            metadata_grp.create_dataset('phone_to_id', data=np.string_([f"{k}:{v}" for k, v in self.phone_to_id.items()]))
            metadata_grp.create_dataset('id_to_phone', data=np.string_([f"{k}:{v}" for k, v in self.id_to_phone.items()]))
            metadata_grp.create_dataset('midi_min', data=self.midi_min)
            metadata_grp.create_dataset('midi_max', data=self.midi_max)
            metadata_grp.create_dataset('n_phonemes', data=len(self.phone_to_id))
            metadata_grp.create_dataset('sample_rate', data=self.sr)
            metadata_grp.create_dataset('hop_length', data=self.hop_length)
            metadata_grp.create_dataset('n_mels', data=self.n_mels)
            
            # Save alignment statistics
            if alignment_stats:
                alignment_data = '\n'.join([f"{name}\t{score:.1f}" for name, score in alignment_stats])
                metadata_grp.create_dataset('alignment_stats', data=np.string_(alignment_data))
            
                # Calculate overall statistics
                mean_alignment = np.mean([score for _, score in alignment_stats])
                good_alignment = sum(1 for _, score in alignment_stats if score > 70)
                moderate_alignment = sum(1 for _, score in alignment_stats if 40 <= score <= 70)
                poor_alignment = sum(1 for _, score in alignment_stats if score < 40)
                
                logger.info(f"Processed {processed_count} files successfully")
                logger.info(f"Phoneme vocabulary size: {len(self.phone_to_id)}")
                logger.info(f"MIDI range: {self.midi_min} to {self.midi_max}")
                logger.info(f"Alignment quality: {mean_alignment:.1f}% average")
                logger.info(f"Good: {good_alignment}, Moderate: {moderate_alignment}, Poor: {poor_alignment}")
            else:
                logger.info(f"Processed {processed_count} files successfully")
                logger.info(f"Phoneme vocabulary size: {len(self.phone_to_id)}")
                logger.info(f"MIDI range: {self.midi_min} to {self.midi_max}")
                logger.info(f"No files were successfully aligned.")
            
            # Print all phonemes found
            all_phonemes = sorted(self.phone_to_id.items(), key=lambda x: x[1])
            logger.info("All phonemes found in dataset:")
            for phoneme, idx in all_phonemes:
                logger.info(f"  {idx}: {phoneme}")
            
            # Plot random sample for validation if available
            if random_sample:
                alignment_plot, _ = self.plot_alignment(
                    f"{random_sample['filename']}_final",
                    random_sample['phonemes'],
                    random_sample['midi'],
                    random_sample['f0'],
                    random_sample['mel_spectrogram'],
                    random_sample['acoustic_boundaries'],
                    random_sample['boundary_scores']
                )
                logger.info(f"Created final alignment plot: {alignment_plot}")

if __name__ == "__main__":
    # Configure logging to include debug messages in file
    log_file = os.path.join("logs", "preprocessor_debug.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    datasets_dir = config['dataset']['datasets_dir']
    
    # Initialize with auto detection of LAB time units
    preprocessor = SVSPreprocessor(
        root_dir=datasets_dir,
        sr=config['preprocessing']['sample_rate'],
        n_fft=config['preprocessing']['n_fft'],
        hop_length=config['preprocessing']['hop_length'],
        n_mels=config['preprocessing']['n_mels'],
        lab_time_unit='samples'  # Auto-detect time units in LAB files
    )
    
    preprocessor.process()