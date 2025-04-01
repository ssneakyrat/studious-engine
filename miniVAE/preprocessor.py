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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('preprocessor')

class SVSPreprocessor:
    def __init__(self, root_dir="assets/gin", output_file="svs_dataset.h5", 
                 sr=22050, n_fft=1024, hop_length=256, n_mels=80):
        self.root_dir = root_dir
        self.wav_dir = os.path.join(root_dir, "wav")
        self.lab_dir = os.path.join(root_dir, "lab")
        self.output_file = os.path.join(root_dir, output_file) #output_file
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Initialize mappings
        self.phone_to_id = {'<PAD>': 0}
        self.id_to_phone = {0: '<PAD>'}
        self.midi_min = 127
        self.midi_max = 0
        
    def extract_f0(self, wav_path):
        """Extract F0 contour using Parselmouth/Praat."""
        sound = parselmouth.Sound(wav_path)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        pitch_values = pitch.selected_array['frequency']
        
        # Interpolate through unvoiced regions
        pitch_values[pitch_values == 0] = np.nan
        indices = np.arange(len(pitch_values))
        valid_indices = ~np.isnan(pitch_values)
        if np.any(valid_indices):
            pitch_values = np.interp(
                indices, 
                indices[valid_indices], 
                pitch_values[valid_indices]
            )
        else:
            pitch_values = np.zeros_like(pitch_values)
        
        # Convert to MIDI
        midi_values = 12 * np.log2(np.maximum(pitch_values, 1e-5) / 440.0) + 69
        midi_values = np.clip(midi_values, 0, 127)
        
        # Update MIDI range
        self.midi_min = min(self.midi_min, int(np.floor(np.min(midi_values[midi_values > 0]))))
        self.midi_max = max(self.midi_max, int(np.ceil(np.max(midi_values[midi_values > 0]))))
        
        return pitch_values, midi_values
    
    def read_lab_file(self, lab_path):
        """Read LAB file and extract phoneme sequence and durations."""
        phonemes = []
        durations = []
        
        with open(lab_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    start_sample, end_sample, phoneme = parts
                    start_sample = int(start_sample)
                    end_sample = int(end_sample)
                    
                    # Add to phoneme mapping if new
                    if phoneme not in self.phone_to_id:
                        idx = len(self.phone_to_id)
                        self.phone_to_id[phoneme] = idx
                        self.id_to_phone[idx] = phoneme
                    
                    phonemes.append(self.phone_to_id[phoneme])
                    durations.append((end_sample - start_sample) / self.hop_length)
        
        return phonemes, durations
    
    def extract_mel_spectrogram(self, wav_path):
        """Extract mel spectrogram from WAV file."""
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sr)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )(waveform)
        
        # Convert to dB scale
        mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        
        return mel_spec.squeeze().numpy()
    
    def align_features(self, phonemes, durations, f0, midi, mel_spec):
        """Align phoneme-level features with frame-level features."""
        frame_phonemes = []
        frame_midi = []
        
        current_frame = 0
        for phone, duration in zip(phonemes, durations):
            duration_frames = int(round(duration))
            if duration_frames <= 0:
                duration_frames = 1
                
            frame_phonemes.extend([phone] * duration_frames)
            
            # Calculate average MIDI note for this phoneme
            start_frame = current_frame
            end_frame = min(current_frame + duration_frames, len(midi))
            if start_frame < end_frame:
                avg_midi = round(np.mean(midi[start_frame:end_frame]))
                frame_midi.extend([avg_midi] * duration_frames)
            else:
                frame_midi.extend([0] * duration_frames)
                
            current_frame += duration_frames
        
        # Ensure all features have the same length
        target_length = min(len(frame_phonemes), len(f0), mel_spec.shape[0])
        frame_phonemes = frame_phonemes[:target_length]
        frame_midi = frame_midi[:target_length]
        f0_aligned = f0[:target_length]
        midi_aligned = midi[:target_length]
        mel_spec_aligned = mel_spec[:target_length]
        
        return frame_phonemes, frame_midi, f0_aligned, midi_aligned, mel_spec_aligned
    
    def plot_alignment(self, filename, phonemes, midi, f0, mel_spec):
        """
        Plot alignment of phonemes, MIDI notes, F0 contour, and mel spectrogram
        in a more integrated view for easier analysis of alignment.
        """
        # Create figure with gridspec for better layout control
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 3], hspace=0.3)
        
        # Shared x-axis for all plots
        ax_phonemes = fig.add_subplot(gs[0])  # Phonemes on top
        ax_midi = fig.add_subplot(gs[1], sharex=ax_phonemes)  # MIDI notes
        ax_f0 = fig.add_subplot(gs[2], sharex=ax_phonemes)  # F0 contour
        ax_mel = fig.add_subplot(gs[3], sharex=ax_phonemes)  # Mel spectrogram at bottom
        
        # Get phoneme text labels
        phone_text = [self.id_to_phone[p] for p in phonemes]
        
        # Find phoneme boundaries
        boundaries = [0]  # Start with the first frame
        for i in range(1, len(phonemes)):
            if phonemes[i] != phonemes[i-1]:
                boundaries.append(i)
        boundaries.append(len(phonemes))  # Add the end frame
        
        # Draw colored regions for each phoneme
        cmap = plt.cm.get_cmap('tab20', len(boundaries) - 1)
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            # Fill region for this phoneme
            ax_phonemes.axvspan(start, end, color=cmap(i), alpha=0.7)
            # Add phoneme label
            midpoint = (start + end) // 2
            phone = phone_text[start]
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
        
        # Plot mel spectrogram
        im = ax_mel.imshow(mel_spec.T, aspect='auto', origin='lower', cmap='viridis')
        ax_mel.set_ylabel('Mel Bins')
        ax_mel.set_xlabel('Frames')
        
        # Add vertical lines at phoneme boundaries to all plots for alignment analysis
        for boundary in boundaries[1:-1]:  # Skip first and last
            ax_midi.axvline(x=boundary, color='k', linestyle='-', alpha=0.7)
            ax_f0.axvline(x=boundary, color='k', linestyle='-', alpha=0.7)
            ax_mel.axvline(x=boundary, color='w', linestyle='-', alpha=0.7)
        
        # Analyze alignment quality
        alignment_scores = []
        
        for boundary in boundaries[1:-1]:  # Skip first and last
            window = 3  # frames to check on each side
            start = max(0, boundary - window)
            end = min(len(midi) - 1, boundary + window)
            
            # Check if there's a significant MIDI change near this boundary
            midi_change = max(midi[start:end]) - min(midi[start:end])
            # Check if there's a significant F0 change near this boundary
            f0_change = max(f0[start:end]) - min(f0[start:end])
            
            # Score this boundary's alignment (0-1)
            score = 0
            if midi_change > 1:  # Significant MIDI change
                score += 0.5
                # Highlight good alignment
                ax_midi.axvspan(boundary-1, boundary+1, color='g', alpha=0.3)
            
            if f0_change > 10:  # Significant F0 change
                score += 0.5
                # Highlight good alignment
                ax_f0.axvspan(boundary-1, boundary+1, color='g', alpha=0.3)
            
            alignment_scores.append(score)
            
            # Highlight potential misalignments
            if score < 0.5:
                ax_midi.axvspan(boundary-1, boundary+1, color='y', alpha=0.3)
                ax_f0.axvspan(boundary-1, boundary+1, color='y', alpha=0.3)
                ax_mel.axvspan(boundary-1, boundary+1, color='y', alpha=0.3)
        
        # Calculate overall alignment quality
        alignment_percentage = 100 * sum(alignment_scores) / max(1, len(alignment_scores))
        
        # Add title with alignment quality
        quality_label = "Good" if alignment_percentage > 70 else "Moderate" if alignment_percentage > 40 else "Poor"
        fig.suptitle(f'Alignment Analysis for {filename}\n'
                    f'Alignment Quality: {quality_label} ({alignment_percentage:.1f}%)', 
                    fontsize=14)
        
        # Ensure all x-axes show the same range
        for ax in [ax_phonemes, ax_midi, ax_f0, ax_mel]:
            ax.set_xlim(0, len(phonemes))
        
        # Hide x-tick labels for all but the bottom plot
        for ax in [ax_phonemes, ax_midi, ax_f0]:
            plt.setp(ax.get_xticklabels(), visible=False)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)  # Make room for suptitle
        plt.savefig(f"{filename}_alignment.png", dpi=150)
        plt.close()
        
        return f"{filename}_alignment.png"
    
    def process(self):
        """Process all WAV and LAB files and save to h5py file."""
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
        
        # Process files and store data
        with h5py.File(self.output_file, 'w') as h5f:
            # Create dataset groups
            metadata_grp = h5f.create_group('metadata')
            data_grp = h5f.create_group('data')
            
            processed_count = 0
            random_sample = None
            
            for base_name in tqdm(valid_files, desc="Processing files"):
                wav_path = os.path.join(self.wav_dir, f"{base_name}.wav")
                lab_path = os.path.join(self.lab_dir, f"{base_name}.lab")
                
                try:
                    # Extract features
                    phonemes, durations = self.read_lab_file(lab_path)
                    
                    # Skip files with too few phonemes
                    if len(phonemes) < 3:
                        logger.info(f"Skipping {base_name} - has fewer than 3 phonemes")
                        continue
                    
                    f0, midi = self.extract_f0(wav_path)
                    mel_spec = self.extract_mel_spectrogram(wav_path)
                    
                    # Align features
                    frame_phonemes, frame_midi, f0_aligned, midi_aligned, mel_spec_aligned = self.align_features(
                        phonemes, durations, f0, midi, mel_spec
                    )
                    
                    # Create dataset for this file
                    file_grp = data_grp.create_group(base_name)
                    file_grp.create_dataset('phonemes', data=frame_phonemes)
                    file_grp.create_dataset('midi', data=frame_midi)
                    file_grp.create_dataset('f0', data=f0_aligned)
                    file_grp.create_dataset('mel_spectrogram', data=mel_spec_aligned)
                    
                    processed_count += 1
                    
                    # Select a random sample for visualization
                    if random_sample is None or random.random() < 1.0/processed_count:
                        random_sample = {
                            'filename': base_name,
                            'phonemes': frame_phonemes,
                            'midi': midi_aligned,
                            'f0': f0_aligned,
                            'mel_spectrogram': mel_spec_aligned
                        }
                        
                except Exception as e:
                    logger.error(f"Error processing {base_name}: {e}")
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
            
            logger.info(f"Processed {processed_count} files successfully")
            logger.info(f"Phoneme vocabulary size: {len(self.phone_to_id)}")
            logger.info(f"MIDI range: {self.midi_min} to {self.midi_max}")
            
            # Plot random sample for validation
            if random_sample:
                alignment_plot = self.plot_alignment(
                    random_sample['filename'],
                    random_sample['phonemes'],
                    random_sample['midi'],
                    random_sample['f0'],
                    random_sample['mel_spectrogram']
                )
                logger.info(f"Created alignment plot: {alignment_plot}")

if __name__ == "__main__":
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    datasets_dir = config['dataset']['datasets_dir']
    preprocessor = SVSPreprocessor(root_dir=datasets_dir)
    preprocessor.process()