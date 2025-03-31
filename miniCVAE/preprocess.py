import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import h5py
import yaml
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # This must be done before importing pyplot
from matplotlib.patches import Rectangle
import argparse # Added for command-line arguments

def read_config(config_path="config/default.yaml"): # Default path updated slightly for clarity
    """Read the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_lab_file(lab_path):
    """Read a label file and return the phoneme segments."""
    phonemes = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                phonemes.append({'start': start, 'end': end, 'phone': phone})
    return phonemes

def extract_f0(audio, sample_rate, min_f0=70, max_f0=400, frame_length=1024, hop_length=256):
    """
    Extract fundamental frequency (F0) using librosa's pyin algorithm.
    Returns times and f0 values.
    """
    # Use PYIN algorithm for F0 extraction
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=min_f0,
        fmax=max_f0,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length
    )

    # Get corresponding time values
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)

    return times, f0

def extract_mel_spectrogram(audio, config):
    """Extract mel spectrogram based on config parameters."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

def get_phone_color(phone):
    """Get color for phoneme type."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    nasals = ['m', 'n', 'ng', 'em', 'en', 'eng']
    fricatives = ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh']
    stops = ['p', 'b', 't', 'd', 'k', 'g']
    affricates = ['ch', 'jh']
    liquids = ['l', 'r', 'el']
    glides = ['w', 'y']

    if phone in ['pau', 'sil', 'sp']:
        return '#999999'  # Silence/pause
    elif phone in vowels:
        return '#e74c3c'  # Vowels
    elif phone in nasals:
        return '#3498db'  # Nasals
    elif phone in fricatives:
        return '#2ecc71'  # Fricatives
    elif phone in stops:
        return '#f39c12'  # Stops
    elif phone in affricates:
        return '#9b59b6'  # Affricates
    elif phone in liquids:
        return '#1abc9c'  # Liquids
    elif phone in glides:
        return '#d35400'  # Glides
    else:
        return '#34495e'  # Others

def is_vowel(phoneme):
    """Check if a phoneme is a vowel."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    return phoneme in vowels

def hz_to_midi(frequency):
    """Convert frequency in Hz to MIDI note number."""
    if frequency <= 0:
        return 0  # Silence or undefined
    # Use max(1e-5, frequency) to avoid log2(0) errors if frequency is exactly 0
    return 69 + 12 * math.log2(max(1e-5, frequency) / 440.0)


def midi_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    if midi_note == 0:
        return 0.0
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))

def midi_to_note_name(midi_note):
    """Convert MIDI note number to note name (e.g., C4, A#3)."""
    if midi_note == 0:
        return "Rest"

    midi_note = int(round(midi_note)) # Ensure it's an integer
    if not 0 < midi_note <= 127:
        return "Invalid" # Handle out-of-range MIDI notes

    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = notes[midi_note % 12]
    return f"{note}{octave}"

def estimate_midi_from_phoneme(f0_values, start_frame, end_frame, method='median'):
    """
    Estimate a single MIDI note from F0 values within a phoneme.

    Args:
        f0_values: Array of F0 values
        start_frame: Start frame of the phoneme
        end_frame: End frame of the phoneme
        method: Method to estimate the note ('median', 'mean', or 'mode')

    Returns:
        Estimated MIDI note number (rounded integer)
    """
    # Make sure we have valid frame indices
    start_frame = max(0, min(start_frame, len(f0_values)-1))
    # Ensure end_frame is at least start_frame + 1 for slicing
    end_frame = max(start_frame + 1, min(end_frame, len(f0_values)))

    # Extract F0 values for the phoneme duration
    phoneme_f0 = f0_values[start_frame:end_frame]

    # Filter out NaN values (unvoiced frames)
    valid_f0 = phoneme_f0[~np.isnan(phoneme_f0)]

    if len(valid_f0) == 0:
        return 0  # Unvoiced phoneme or segment too short

    # Estimate pitch
    try:
        if method == 'median':
            f0_estimate = np.median(valid_f0)
        elif method == 'mean':
            f0_estimate = np.mean(valid_f0)
        elif method == 'mode':
            # Simple mode implementation using histogram
            if len(valid_f0) < 3: # Need enough points for meaningful mode
                 f0_estimate = np.median(valid_f0)
            else:
                hist, bin_edges = np.histogram(valid_f0, bins=24)  # Bins for each quarter tone
                bin_idx = np.argmax(hist)
                f0_estimate = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
        else:
            raise ValueError(f"Unknown method: {method}")

        if f0_estimate <= 0:
             return 0 # Avoid errors with hz_to_midi

        # Convert to MIDI note
        midi_note = hz_to_midi(f0_estimate)

        # Round to nearest semitone and ensure it's within valid MIDI range
        midi_note_rounded = max(0, min(127, int(round(midi_note))))

        return midi_note_rounded

    except Exception as e:
        print(f"Warning: Error estimating MIDI note: {e}. F0 values: {valid_f0}. Returning 0.")
        return 0


def phonemes_to_frames(phonemes, lab_sample_rate, hop_length, sample_rate, scaling_factor=227.13):
    """Convert phoneme timings to frame indices for the mel spectrogram."""
    phoneme_frames = []
    total_frames_in_audio = 0 # Keep track for validation
    
    for i, p in enumerate(phonemes):
        # Convert from lab time units to seconds
        start_time = p['start'] / lab_sample_rate / scaling_factor
        end_time = p['end'] / lab_sample_rate / scaling_factor

        # Convert from seconds to frame indices using librosa utility for robustness
        start_frame = librosa.time_to_frames(start_time, sr=sample_rate, hop_length=hop_length)
        end_frame = librosa.time_to_frames(end_time, sr=sample_rate, hop_length=hop_length)

        # Ensure end_frame is at least start_frame
        end_frame = max(start_frame, end_frame)

        duration = end_time - start_time

        phoneme_frames.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'phone': p['phone']
        })
        if i == len(phonemes) -1:
            total_frames_in_audio = end_frame

    # Post-processing: Ensure no gaps or overlaps between adjacent phonemes
    for i in range(len(phoneme_frames) - 1):
        current_p = phoneme_frames[i]
        next_p = phoneme_frames[i+1]
        # Adjust end frame/time of current phoneme to match start frame/time of next one
        current_p['end_frame'] = next_p['start_frame']
        current_p['end_time'] = next_p['start_time']
        current_p['duration'] = current_p['end_time'] - current_p['start_time']

    return phoneme_frames, total_frames_in_audio

def extract_midi_notes(f0_values, phoneme_frames, method='median'):
    """
    Extract MIDI notes from F0 values based on phoneme segmentation.

    Args:
        f0_values: Array of F0 values
        phoneme_frames: List of phoneme frame information
        method: Method to estimate notes ('median', 'mean', or 'mode')

    Returns:
        List of dictionaries containing MIDI note information
    """
    midi_notes = []

    for p in phoneme_frames:
        # Check for valid duration and frame range
        if p['end_frame'] <= p['start_frame']:
            # print(f"Skipping phoneme {p['phone']} due to zero/negative duration in frames.")
            continue # Skip phonemes with no duration in frames

        # Estimate MIDI note
        midi_note = estimate_midi_from_phoneme(
            f0_values,
            p['start_frame'],
            p['end_frame'],
            method
        )

        # Only include valid notes (midi_note > 0 means voiced)
        if midi_note > 0:
            note_data = {
                'phone': p['phone'],
                'start_time': p['start_time'],
                'end_time': p['end_time'],
                'duration': p['duration'],
                'start_frame': p['start_frame'],
                'end_frame': p['end_frame'],
                'midi_note': midi_note,
                'is_vowel': is_vowel(p['phone'])
            }
            # Ensure duration is non-negative after potential adjustments
            if note_data['duration'] >= 0:
                midi_notes.append(note_data)

    # Sort by start time to ensure sequential processing (already done by phoneme_frames)
    # Redundant overlap adjustment removed, handled in phonemes_to_frames

    return midi_notes

def create_visualization(sample_id, h5_path, output_path, config):
    """Create visualization with mel spectrogram, F0, and phoneme alignment from the HDF5 file."""
    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if sample_id exists
            if sample_id not in f:
                print(f"Error: Sample ID '{sample_id}' not found in HDF5 file '{h5_path}'. Cannot create visualization.")
                return None

            # Get the sample group
            sample_group = f[sample_id]

            # Extract data
            mel_spec = sample_group['features']['mel_spectrogram'][:]
            f0_times = sample_group['features']['f0_times'][:]
            f0_values = sample_group['features']['f0_values'][:]

            # Get phoneme data
            phones_bytes = sample_group['phonemes']['phones'][:]
            phones = [p.decode('utf-8') for p in phones_bytes]
            start_times = sample_group['phonemes']['start_times'][:]
            end_times = sample_group['phonemes']['end_times'][:]
            durations = sample_group['phonemes']['durations'][:]

            # Get MIDI data if available
            has_midi = 'midi' in sample_group and 'notes' in sample_group['midi'] and sample_group['midi']['notes'].shape[0] > 0
            midi_notes_list = []
            if has_midi:
                midi_notes_data = sample_group['midi']['notes'][:]
                midi_start_times = sample_group['midi']['start_times'][:]
                midi_end_times = sample_group['midi']['end_times'][:]
                midi_durations = sample_group['midi']['durations'][:]
                midi_phones_bytes = sample_group['midi']['phones'][:]
                midi_phones = [p.decode('utf-8') for p in midi_phones_bytes]

                for i in range(len(midi_notes_data)):
                    midi_notes_list.append({
                        'midi_note': midi_notes_data[i],
                        'start_time': midi_start_times[i],
                        'end_time': midi_end_times[i],
                        'duration': midi_durations[i],
                        'phone': midi_phones[i]
                    })

            # Recreate phoneme frames for visualization
            phoneme_frames_vis = []
            total_time = 0
            if len(start_times)>0:
                total_time = end_times[-1] # Get duration from last phoneme end time

            for i, phone in enumerate(phones):
                phoneme_data = {
                    'phone': phone,
                    'start_time': start_times[i],
                    'end_time': end_times[i],
                    'duration': durations[i],
                    'has_midi': False,
                    'midi_note': 0
                }

                # Check if this phoneme has a corresponding MIDI note
                if has_midi:
                    # Match based on overlapping time interval
                    for note in midi_notes_list:
                        # Check if phoneme interval overlaps significantly with note interval
                        overlap_start = max(phoneme_data['start_time'], note['start_time'])
                        overlap_end = min(phoneme_data['end_time'], note['end_time'])
                        if overlap_end > overlap_start: # If there is overlap
                            # Prioritize exact match or first found overlapping note
                            phoneme_data['has_midi'] = True
                            phoneme_data['midi_note'] = note['midi_note']
                            break # Take the first matching note

                phoneme_frames_vis.append(phoneme_data)

    except Exception as e:
        print(f"Error reading data for visualization from {h5_path} for sample {sample_id}: {e}")
        return None

    # Proceed with plotting only if data extraction was successful
    plt.figure(figsize=(14, 10), dpi=100)

    # First subplot: Mel spectrogram
    ax1 = plt.subplot(3, 1, 1)
    img = librosa.display.specshow(
        mel_spec,
        x_axis='time',
        y_axis='mel',
        sr=config['audio']['sample_rate'],
        hop_length=config['audio']['hop_length'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax'],
        ax=ax1 # Specify axis
    )
    ax1.set_title(f'Mel Spectrogram ({sample_id})')
    ax1.set_xlabel('') # Remove x-label as it's shared

    # Second subplot: F0 contour with MIDI notes if available
    ax2 = plt.subplot(3, 1, 2, sharex=ax1) # Share x-axis with ax1
    if len(f0_times) > 0 and len(f0_values) > 0:
         ax2.plot(f0_times, f0_values, 'r-', linewidth=1.5, alpha=0.8, label='F0 Contour')
    else:
         ax2.text(0.5, 0.5, 'No F0 data', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)


    # Overlay MIDI notes if available
    if has_midi:
        plotted_midi = False
        for note in midi_notes_list:
            note_freq = midi_to_hz(note['midi_note'])
            if note['duration'] > 0: # Only plot notes with duration
                 ax2.plot(
                     [note['start_time'], note['end_time']],
                     [note_freq, note_freq],
                     'b-', linewidth=3, alpha=0.7, label='MIDI Notes' if not plotted_midi else ""
                 )
                 plotted_midi = True # Ensure label is added only once

    if has_midi and plotted_midi:
        ax2.legend(loc='upper right')
    elif not has_midi:
        ax2.legend(loc='upper right')


    ax2.set_title('F0 Contour' + (' with MIDI Notes' if has_midi else ''))
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlim(0, total_time if total_time > 0 else (f0_times[-1] if len(f0_times) > 0 else 1.0)) # Use phoneme total time if available
    ax2.set_ylim(config['audio'].get('f0_min', 50), config['audio'].get('f0_max', 500)) # Use get with defaults
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), visible=False) # Hide x-axis labels/ticks

    # Third subplot: Phoneme alignment
    ax3 = plt.subplot(3, 1, 3, sharex=ax1) # Share x-axis with ax1
    ax3.set_title('Phoneme Alignment')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim(0, total_time if total_time > 0 else (f0_times[-1] if len(f0_times) > 0 else 1.0))
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])

    # Add phoneme segments
    for i, p in enumerate(phoneme_frames_vis):
        # Set alpha based on whether it has MIDI note
        alpha = 0.8 if p.get('has_midi', False) else 0.5

        if p['duration'] > 0: # Only draw rectangle if duration is positive
            rect = Rectangle(
                (p['start_time'], 0),
                p['duration'],
                1,
                facecolor=get_phone_color(p['phone']),
                edgecolor='black',
                alpha=alpha,
                linewidth=0.5
            )
            ax3.add_patch(rect)

            # Add phoneme text
            text_x = p['start_time'] + p['duration'] / 2

            # Add MIDI note if available
            if p.get('has_midi', False) and p['midi_note'] > 0:
                note_name = midi_to_note_name(p['midi_note'])
                text = f"{p['phone']}\n{note_name}"
                fontsize = 8
            else:
                text = p['phone']
                fontsize = 9

            ax3.text(
                text_x,
                0.5,
                text,
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='bold',
                fontsize=fontsize,
                clip_on=True # Prevent text spilling outside plot
            )

        # Add vertical alignment lines across all plots (only if duration > 0)
        if i > 0 and p['start_time'] > phoneme_frames_vis[i-1]['start_time']:
             ax1.axvline(x=p['start_time'], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
             ax2.axvline(x=p['start_time'], color='gray', linestyle='--', linewidth=0.5, alpha=0.5)


    # Add phoneme duration table (optional, can make plot cluttered)
    # table_text = "Phoneme durations (seconds):\n"
    # ... (table generation code) ...
    # plt.figtext(0.5, 0.01, table_text, fontsize=7, family='monospace')

    # Add sample ID footer
    plt.figtext(0.02, 0.01, f"Sample ID: {sample_id}" + (" (with MIDI)" if has_midi else ""), fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent footer overlap
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight') # Reduced DPI slightly
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization {output_path}: {e}")

    fig = plt.gcf()
    plt.close(fig) # Close the figure to free memory
    return fig


def process_files(config_path="config/default.yaml",
                  lab_sample_rate=44100, # Consider moving these to config
                  scaling_factor=227.13, # Consider moving these to config
                  midi_method='median',
                  min_phonemes=5): # Added min_phonemes parameter with default
    """Process WAV and LAB files, extract features, and save to a single HDF5 file."""
    # Read config
    config = read_config(config_path)

    # Get paths from config
    data_raw_path = Path(config['data']['data_raw'])
    wav_dir = data_raw_path / "wav"
    lab_dir = data_raw_path / "lab"
    binary_dir = data_raw_path / "binary"

    # Create binary directory if it doesn't exist
    binary_dir.mkdir(parents=True, exist_ok=True)

    # Define the path for the single HDF5 file
    h5_path = binary_dir / "dataset.h5"

    # Get list of WAV files
    wav_files = list(wav_dir.glob('*.wav'))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return None, None

    print(f"Found {len(wav_files)} WAV files. Processing...")

    processed_count = 0
    skipped_count = 0
    sample_id_for_visualization = None

    # Create the HDF5 file
    with h5py.File(h5_path, 'w') as f:
        # Store metadata and configuration
        metadata_group = f.create_group('metadata')
        for key, value in config['audio'].items():
            metadata_group.attrs[key] = value
        # Store preprocessing params used (maybe read from config ideally)
        metadata_group.attrs['lab_sample_rate'] = lab_sample_rate
        metadata_group.attrs['scaling_factor'] = scaling_factor
        metadata_group.attrs['midi_method'] = midi_method
        metadata_group.attrs['min_phonemes_filter'] = min_phonemes

        # Store file list dynamically as samples are processed
        processed_file_list = []

        # Process each WAV file
        for i, wav_path in enumerate(wav_files):
            base_name = wav_path.stem
            lab_file = base_name + '.lab'
            lab_path = lab_dir / lab_file

            # Check if lab file exists
            if not lab_path.exists():
                print(f"Warning: No matching lab file for {wav_path.name}. Skipping.")
                skipped_count += 1
                continue

            # --- Added Phoneme Count Check ---
            try:
                phonemes = read_lab_file(lab_path)
                if not phonemes: # Handle empty lab files
                     print(f"Warning: Lab file {lab_path.name} is empty. Skipping.")
                     skipped_count +=1
                     continue
                if len(phonemes) < min_phonemes:
                    print(f"Skipping {wav_path.name}: Found {len(phonemes)} phonemes, required >= {min_phonemes}")
                    skipped_count += 1
                    continue # Skip this file
            except Exception as e:
                print(f"Error reading lab file {lab_path.name}: {e}. Skipping.")
                skipped_count += 1
                continue # Skip if lab file is unreadable
            # --- End Phoneme Count Check ---

            # Proceed with processing if phoneme count is sufficient
            print(f"Processing {wav_path.name} ({i+1}/{len(wav_files)})")

            try:
                # Load audio
                audio, sample_rate = librosa.load(wav_path, sr=config['audio']['sample_rate'])
                if len(audio) == 0:
                    print(f"Warning: Audio file {wav_path.name} is empty. Skipping.")
                    skipped_count += 1
                    continue

                # Extract features
                mel_spec = extract_mel_spectrogram(audio, config)
                f0_times, f0_values = extract_f0(
                    audio,
                    sample_rate,
                    min_f0=config['audio']['f0_min'],
                    max_f0=config['audio']['f0_max'],
                    frame_length=config['audio']['n_fft'],
                    hop_length=config['audio']['hop_length']
                )

                # Ensure f0 and mel_spec lengths match (pad f0 if necessary)
                target_len = mel_spec.shape[1]
                if len(f0_values) < target_len:
                    pad_width = target_len - len(f0_values)
                    f0_values = np.pad(f0_values, (0, pad_width), constant_values=np.nan)
                    f0_times = np.pad(f0_times, (0, pad_width), constant_values=f0_times[-1] if len(f0_times)>0 else 0) # Pad time too
                elif len(f0_values) > target_len:
                    f0_values = f0_values[:target_len]
                    f0_times = f0_times[:target_len]


                # Convert phoneme timings to frame indices (using the phonemes list read earlier)
                phoneme_frames, total_frames_from_lab = phonemes_to_frames(
                    phonemes,
                    lab_sample_rate,
                    config['audio']['hop_length'],
                    sample_rate,
                    scaling_factor
                )

                # Frame validation: Check if phoneme frames align with spectrogram length
                if total_frames_from_lab > target_len + 1: # Allow for rounding diff of 1 frame
                    print(f"Warning: Phoneme alignment ({total_frames_from_lab} frames) exceeds spectrogram length ({target_len} frames) for {base_name}. Skipping.")
                    skipped_count += 1
                    continue

                # Adjust last phoneme end_frame if necessary
                if phoneme_frames and phoneme_frames[-1]['end_frame'] > target_len:
                    phoneme_frames[-1]['end_frame'] = target_len


                # Extract MIDI notes from phonemes and F0
                midi_notes = extract_midi_notes(f0_values, phoneme_frames, midi_method)

                # --- Data Saving ---
                # Create a group for this sample
                sample_group = f.create_group(base_name)

                # Store the first valid sample ID for visualization
                if sample_id_for_visualization is None:
                    sample_id_for_visualization = base_name

                # Create subgroups
                audio_group = sample_group.create_group('audio')
                feature_group = sample_group.create_group('features')
                phoneme_group = sample_group.create_group('phonemes')
                midi_group = sample_group.create_group('midi')

                # Store audio data (optional, makes file large)
                # audio_group.create_dataset('waveform', data=audio)

                # Store features
                feature_group.create_dataset('mel_spectrogram', data=mel_spec, compression="gzip")
                feature_group.create_dataset('f0_times', data=f0_times, compression="gzip")
                feature_group.create_dataset('f0_values', data=f0_values, compression="gzip")

                # Store phoneme data
                phones = np.array([p['phone'].encode('utf-8') for p in phoneme_frames], dtype='S10')
                start_frames = np.array([p['start_frame'] for p in phoneme_frames], dtype=np.int32)
                end_frames = np.array([p['end_frame'] for p in phoneme_frames], dtype=np.int32)
                start_times = np.array([p['start_time'] for p in phoneme_frames], dtype=np.float32)
                end_times = np.array([p['end_time'] for p in phoneme_frames], dtype=np.float32)
                durations = np.array([p['duration'] for p in phoneme_frames], dtype=np.float32)

                phoneme_group.create_dataset('phones', data=phones)
                phoneme_group.create_dataset('start_frames', data=start_frames)
                phoneme_group.create_dataset('end_frames', data=end_frames)
                phoneme_group.create_dataset('start_times', data=start_times)
                phoneme_group.create_dataset('end_times', data=end_times)
                phoneme_group.create_dataset('durations', data=durations)

                # Store MIDI data
                num_midi_notes = len(midi_notes)
                midi_group.attrs['num_notes'] = num_midi_notes

                if num_midi_notes > 0:
                    midi_notes_data = np.array([note['midi_note'] for note in midi_notes], dtype=np.int32)
                    midi_start_times = np.array([note['start_time'] for note in midi_notes], dtype=np.float32)
                    midi_end_times = np.array([note['end_time'] for note in midi_notes], dtype=np.float32)
                    midi_durations = np.array([note['duration'] for note in midi_notes], dtype=np.float32)
                    midi_start_frames = np.array([note['start_frame'] for note in midi_notes], dtype=np.int32)
                    midi_end_frames = np.array([note['end_frame'] for note in midi_notes], dtype=np.int32)
                    midi_phones = np.array([note['phone'].encode('utf-8') for note in midi_notes], dtype='S10')
                    midi_is_vowel = np.array([note['is_vowel'] for note in midi_notes], dtype=bool)

                    midi_group.create_dataset('notes', data=midi_notes_data)
                    midi_group.create_dataset('start_times', data=midi_start_times)
                    midi_group.create_dataset('end_times', data=midi_end_times)
                    midi_group.create_dataset('durations', data=midi_durations)
                    midi_group.create_dataset('start_frames', data=midi_start_frames)
                    midi_group.create_dataset('end_frames', data=midi_end_frames)
                    midi_group.create_dataset('phones', data=midi_phones)
                    midi_group.create_dataset('is_vowel', data=midi_is_vowel)

                    # Store note names for easy reference
                    note_names = [midi_to_note_name(note) for note in midi_notes_data]
                    note_names_bytes = np.array([name.encode('utf-8') for name in note_names], dtype='S10')
                    midi_group.create_dataset('note_names', data=note_names_bytes)
                else:
                     # Create empty datasets for consistency if no MIDI notes found
                     midi_group.create_dataset('notes', shape=(0,), dtype=np.int32)
                     midi_group.create_dataset('start_times', shape=(0,), dtype=np.float32)
                     midi_group.create_dataset('end_times', shape=(0,), dtype=np.float32)
                     midi_group.create_dataset('durations', shape=(0,), dtype=np.float32)
                     midi_group.create_dataset('start_frames', shape=(0,), dtype=np.int32)
                     midi_group.create_dataset('end_frames', shape=(0,), dtype=np.int32)
                     midi_group.create_dataset('phones', shape=(0,), dtype='S10')
                     midi_group.create_dataset('is_vowel', shape=(0,), dtype=bool)
                     midi_group.create_dataset('note_names', shape=(0,), dtype='S10')


                # Store sample metadata
                sample_group.attrs['filename'] = wav_path.name
                sample_group.attrs['duration_sec'] = len(audio) / sample_rate
                sample_group.attrs['num_frames'] = mel_spec.shape[1]
                sample_group.attrs['num_phonemes'] = len(phones)
                sample_group.attrs['num_midi_notes'] = num_midi_notes

                processed_file_list.append(base_name)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {wav_path.name}: {str(e)}. Skipping.")
                # Clean up potentially partially created group in HDF5
                if base_name in f:
                    del f[base_name]
                skipped_count += 1
                # Optional: break execution on error?
                # raise e # Uncomment to stop execution on first error

        # Store the final list of processed files in metadata
        if processed_file_list:
            file_list_dtype = h5py.string_dtype(encoding='utf-8')
            metadata_group.create_dataset('file_list', data=processed_file_list, dtype=file_list_dtype)
        else:
            # Handle case where no files were processed
            metadata_group.create_dataset('file_list', shape=(0,), dtype=h5py.string_dtype(encoding='utf-8'))


    print("-" * 30)
    print(f"Processing Summary:")
    print(f"  Total WAV files found: {len(wav_files)}")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Skipped (missing .lab, phoneme count < {min_phonemes}, errors): {skipped_count}")
    print(f"Dataset saved to {h5_path}")
    print("-" * 30)

    return h5_path, sample_id_for_visualization

def validate_dataset(h5_path):
    """Validate the entire dataset by checking sample integrity."""
    print(f"Validating dataset: {h5_path}")

    if not Path(h5_path).exists():
        print("Error: HDF5 file not found.")
        return 0, 0

    try:
        with h5py.File(h5_path, 'r') as f:
            # Check if metadata and file_list exist
            if 'metadata' not in f or 'file_list' not in f['metadata']:
                print("Error: Metadata or file_list missing in HDF5 file.")
                return 0, 0

            metadata = f['metadata']
            file_list_bytes = metadata['file_list'][:]
            # Handle potential empty file list
            file_list = [name for name in file_list_bytes] if len(file_list_bytes) > 0 else []


            print(f"\nFound {len(file_list)} samples listed in the dataset metadata.")
            print(f"Audio configuration: {dict(metadata.attrs.items())}")

            # Validate each sample listed in file_list
            valid_samples = 0
            issues = 0

            print("\nSample validation:")
            if not file_list:
                 print("  - No samples listed in file_list to validate.")

            for i, sample_id in enumerate(file_list):
                print(f"  Validating ({i+1}/{len(file_list)}): {sample_id}...", end='\r')
                if sample_id not in f:
                    print(f"\n  - Error: Sample '{sample_id}' in file list but group not found in HDF5.")
                    issues += 1
                    continue

                sample = f[sample_id]
                sample_valid = True

                # Check required groups and datasets
                required_groups = {'features': ['mel_spectrogram', 'f0_values'],
                                   'phonemes': ['phones', 'start_frames', 'end_frames'],
                                   'midi': ['notes']} # MIDI notes is crucial

                for group, datasets in required_groups.items():
                    if group not in sample:
                        print(f"\n  - Warning: Sample '{sample_id}' missing group: {group}")
                        issues += 1
                        sample_valid = False
                        break # Skip dataset checks for this group
                    for dataset in datasets:
                        if dataset not in sample[group]:
                            print(f"\n  - Warning: Sample '{sample_id}' missing dataset: {group}/{dataset}")
                            issues += 1
                            sample_valid = False

                if not sample_valid: continue # Move to next sample if groups/datasets missing

                # Check mel spectrogram and F0 alignment
                try:
                    mel_spec = sample['features']['mel_spectrogram'][:]
                    f0_values = sample['features']['f0_values'][:]
                    n_frames = mel_spec.shape[1]

                    if len(f0_values) != n_frames:
                        print(f"\n  - Warning: Sample '{sample_id}' has misaligned F0 ({len(f0_values)}) and mel spectrogram ({n_frames})")
                        issues += 1
                        sample_valid = False

                    # Check phoneme frame ranges
                    phoneme_end_frames = sample['phonemes']['end_frames'][:]
                    if len(phoneme_end_frames) > 0 and max(phoneme_end_frames) > n_frames:
                        print(f"\n  - Warning: Sample '{sample_id}' has phoneme end frame ({max(phoneme_end_frames)}) exceeding spectrogram length ({n_frames})")
                        issues += 1
                        sample_valid = False

                    # Check MIDI data consistency
                    num_midi_notes_attr = sample['midi'].attrs.get('num_notes', -1)
                    midi_notes = sample['midi']['notes'][:]
                    if num_midi_notes_attr != len(midi_notes):
                        print(f"\n  - Warning: Sample '{sample_id}' MIDI num_notes attribute ({num_midi_notes_attr}) mismatch with notes dataset length ({len(midi_notes)})")
                        issues += 1
                        sample_valid = False


                    if len(midi_notes) > 0:
                        midi_end_frames = sample['midi']['end_frames'][:]
                        # Check for valid MIDI note range (0-127)
                        if any(note < 0 or note > 127 for note in midi_notes):
                            print(f"\n  - Warning: Sample '{sample_id}' has invalid MIDI note values (outside 0-127)")
                            issues += 1
                            sample_valid = False

                        # Check for MIDI frame alignment
                        if len(midi_end_frames) > 0 and max(midi_end_frames) > n_frames:
                            print(f"\n  - Warning: Sample '{sample_id}' has MIDI end frame ({max(midi_end_frames)}) exceeding spectrogram length ({n_frames})")
                            issues += 1
                            sample_valid = False

                except Exception as e:
                    print(f"\n  - Error during validation checks for sample '{sample_id}': {e}")
                    issues += 1
                    sample_valid = False


                if sample_valid:
                    valid_samples += 1

            print("\n" + "="*30) # Newline after progress indicator
            print(f"Validation Summary:")
            if not file_list:
                print("  - No samples found to validate.")
            else:
                print(f"  - Samples listed in metadata: {len(file_list)}")
                print(f"  - Valid samples checked: {valid_samples}")
                print(f"  - Issues found: {issues}")
                if len(file_list)>0:
                     print(f"  - Validity rate: {valid_samples/len(file_list)*100:.1f}%")
            print("="*30)


            # Calculate dataset statistics (only if samples exist)
            if valid_samples > 0 and file_list:
                print("\nCalculating Dataset Statistics (based on listed files):")
                total_duration = 0
                total_phonemes = 0
                total_midi_notes = 0
                phoneme_counts = {}
                note_counts = {}

                for sample_id in file_list:
                    if sample_id in f: # Process only existing samples
                        sample = f[sample_id]
                        total_duration += sample.attrs.get('duration_sec', 0)

                        # Phoneme statistics
                        if 'phonemes' in sample and 'phones' in sample['phonemes']:
                            phones_bytes = sample['phonemes']['phones'][:]
                            total_phonemes += len(phones_bytes)
                            for phone_bytes in phones_bytes:
                                try:
                                    phone = phone_bytes.decode('utf-8')
                                    phoneme_counts[phone] = phoneme_counts.get(phone, 0) + 1
                                except UnicodeDecodeError:
                                    phoneme_counts['<invalid>'] = phoneme_counts.get('<invalid>', 0) + 1


                        # MIDI statistics
                        if 'midi' in sample:
                            num_notes = sample['midi'].attrs.get('num_notes', 0)
                            total_midi_notes += num_notes

                            if num_notes > 0 and 'notes' in sample['midi']:
                                midi_notes = sample['midi']['notes'][:]
                                for note in midi_notes:
                                    note_name = midi_to_note_name(note)
                                    note_counts[note_name] = note_counts.get(note_name, 0) + 1

                print(f"  - Total audio duration: {total_duration:.2f} seconds")
                if total_duration > 0:
                     print(f"  - Total phonemes: {total_phonemes} (Avg: {total_phonemes/total_duration:.2f} per second)")
                     print(f"  - Total MIDI notes: {total_midi_notes} (Avg: {total_midi_notes/total_duration:.2f} per second)")
                     if total_phonemes > 0:
                         print(f"  - Approx. % phonemes with MIDI notes: {total_midi_notes/total_phonemes*100:.1f}%")
                else:
                     print(f"  - Total phonemes: {total_phonemes}")
                     print(f"  - Total MIDI notes: {total_midi_notes}")


                # Print top 10 most common phonemes
                if phoneme_counts:
                    top_phonemes = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    print("\nTop 10 most common phonemes:")
                    for phone, count in top_phonemes:
                        print(f"    - {phone:<6}: {count} occurrences ({count/total_phonemes*100:.1f}%)")

                # Print top 10 most common MIDI notes
                if note_counts:
                    top_notes = sorted(note_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    print("\nTop 10 most common MIDI notes:")
                    for note, count in top_notes:
                        print(f"    - {note:<6}: {count} occurrences ({count/total_midi_notes*100:.1f}%)")

            return valid_samples, issues

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {h5_path}")
        return 0, 0
    except Exception as e:
        print(f"An error occurred during validation: {e}")
        return 0, 0


def main():
    print("Starting audio data processing...")

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Preprocess audio and label files for FutureVox.")
    parser.add_argument('--config', type=str, default="config/default.yaml",
                        help="Path to the configuration file (default: config/default.yaml).")
    parser.add_argument('--min_phonemes', type=int, default=5,
                        help="Minimum number of phonemes required for a file to be processed (default: 5).")
    # Consider adding args for lab_sample_rate, scaling_factor if they shouldn't be hardcoded
    # parser.add_argument('--lab_sr', type=int, default=44100, help="Sample rate of the lab file timings.")
    # parser.add_argument('--scale', type=float, default=227.13, help="Scaling factor for lab timings.")

    args = parser.parse_args()
    # --- End Argument Parsing ---

    try:
        config = read_config(args.config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return
    except Exception as e:
        print(f"Error reading configuration file {args.config}: {e}")
        return

    data_raw_path = Path(config['data']['data_raw'])
    binary_dir = data_raw_path / "binary" # Use Path object
    
    # Use default lab_sample_rate, scaling_factor from process_files signature for now
    # Pass the parsed min_phonemes
    h5_path, sample_id = process_files(
        config_path=args.config,
        min_phonemes=args.min_phonemes
        # lab_sample_rate=args.lab_sr, # Example if args were added
        # scaling_factor=args.scale   # Example if args were added
    )

    if h5_path and Path(h5_path).exists(): # Check if file was actually created
        # Validate the dataset
        validate_dataset(h5_path)

        # Create visualization for the first successfully processed sample (if any)
        if sample_id:
            print(f"\nCreating visualization for sample: {sample_id}")
            vis_output_path = binary_dir / f"{sample_id}_visualization.png"
            create_visualization(sample_id, h5_path, vis_output_path, config)
        else:
            print("\nNo samples were successfully processed, skipping visualization.")

    else:
        print("\nPreprocessing did not generate a dataset file or no samples were processed.")

    print("\nPreprocessing script finished.")

if __name__ == '__main__':
    main()
