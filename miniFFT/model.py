import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import math
import matplotlib.pyplot as plt
import librosa.display # For plotting spectrograms
import numpy as np

# --- Helper Modules ---

class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe) # Not a parameter

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # x.size(1) is the sequence length
        # self.pe shape is (1, max_len, d_model), slice up to seq_len
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FeedForwardNetwork(nn.Module):
    """Position-wise Feed-Forward Network using Conv1D."""
    def __init__(self, d_model, d_ff, kernel_size=9, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1, padding=0) # Use kernel size 1 for the second conv
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() # Or nn.GELU()

    def forward(self, x):
        # Input x shape: (Batch, SeqLen, Dim)
        # Conv1D expects (Batch, Dim, SeqLen)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x.transpose(1, 2) # Back to (Batch, SeqLen, Dim)
        return x


class FFTBlock(nn.Module):
    """A single FFT Block (Transformer Block)."""
    def __init__(self, d_model, n_heads, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(d_model, d_ff, kernel_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): Input tensor (B, SeqLen, Dim)
            mask (Tensor, optional): Attention mask (B, SeqLen) or (B, SeqLen, SeqLen).
                                      For self-attention, typically use padding mask.
        Returns:
            Tensor: Output tensor (B, SeqLen, Dim)
        """
        # Pre-LayerNorm variation (common now)
        x_norm = self.norm1(x)

        # Create attention mask if needed (padding mask -> attention mask)
        # MHA expects mask where True indicates positions NOT allowed to attend
        attn_mask = None
        if mask is not None:
             if mask.dim() == 2: # (B, SeqLen) - Padding mask
                 # Expand to (B, 1, 1, SeqLen) for broadcasting against (B, n_heads, SeqLen, SeqLen) query/key matrix
                 # Then invert, because True in MHA means "masked out"
                 attn_mask = ~mask.unsqueeze(1).unsqueeze(2) # Make sure ~ works as expected or use (1.0 - mask.float()) * -1e9
                 attn_mask = attn_mask.expand(-1, mask.size(1), -1).repeat(self.self_attn.num_heads, 1, 1) # (B*num_heads, T, T) may be needed depending on MHA version
                 # Simpler approach: let MHA handle the broadcast from (B, T) key_padding_mask
                 attn_mask = ~mask # MHA expects True where padded

        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=attn_mask, need_weights=False)
        x = x + self.dropout1(attn_output)

        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)

        return x


class LengthRegulator(nn.Module):
    """ Adjusts sequence length based on durations. """
    def forward(self, encoder_outputs, durations, output_mask=None):
        """
        Args:
            encoder_outputs (Tensor): Encoder hidden states (B, N, Dim).
            durations (Tensor): Phoneme durations (B, N), LongTensor.
            output_mask (Tensor, optional): Pre-computed mask for the target length T (B, T).
                                           If None, max_len T is inferred from durations.

        Returns:
            Tensor: Expanded sequence (B, T, Dim).
            Tensor: Output mask (B, T) indicating non-padded frames.
        """
        B, N, Dim = encoder_outputs.shape
        device = encoder_outputs.device

        if output_mask is None:
            total_durations = torch.sum(durations, dim=1) # (B,)
            T = torch.max(total_durations).item() # Max sequence length in batch after expansion
            # Create output mask based on calculated total durations per sample
            output_mask = torch.arange(T, device=device).unsqueeze(0) < total_durations.unsqueeze(1) # (B, T)
        else:
            T = output_mask.shape[1] # Max length T from provided mask

        expanded_output = torch.zeros(B, T, Dim, device=device, dtype=encoder_outputs.dtype)

        for i in range(B):
            enc_out = encoder_outputs[i] # (N, Dim)
            dur = durations[i]           # (N,)
            t_idx = 0
            for n_idx in range(N):
                phone_dur = dur[n_idx].item()
                if phone_dur > 0 and t_idx < T:
                    # Ensure we don't write past the allocated length T
                    end_t_idx = min(t_idx + phone_dur, T)
                    # Repeat the phoneme's hidden state
                    expanded_output[i, t_idx:end_t_idx] = enc_out[n_idx].unsqueeze(0).expand(end_t_idx - t_idx, -1)
                    t_idx = end_t_idx
                elif t_idx >= T:
                    break # Stop if we've filled the max length

            # Safety check: ensure the mask reflects the actual filled length for this sample
            # This might happen if durations sum to less than T due to truncation/padding issues earlier
            # In theory, sum(durations) should match T based on our preprocessing/collation
            # output_mask[i, t_idx:] = False # Ensure padding mask is correct if something went wrong


        # Apply the output mask to zero out padded regions (might be redundant if init with zeros)
        # expanded_output = expanded_output * output_mask.unsqueeze(-1).float()

        return expanded_output, output_mask


# --- Main Model ---

class FFTLightSVS(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters(config) # Saves config to hparams attribute
        self.config = config

        self.phoneme_embedding = nn.Embedding(config.model.vocab_size, config.model.encoder_embed_dim, padding_idx=config.preprocessing.pad_phoneme_id)
        self.midi_embedding = nn.Embedding(config.model.midi_vocab_size, config.model.midi_embed_dim, padding_idx=0) # Assume MIDI pad=0

        # Calculate encoder input dimension
        encoder_input_dim = config.model.encoder_embed_dim + config.model.midi_embed_dim
        if encoder_input_dim != config.model.hidden_dim:
             print(f"Warning: Encoder input dim ({encoder_input_dim}) != hidden_dim ({config.model.hidden_dim}). Adding projection.")
             self.encoder_input_proj = nn.Linear(encoder_input_dim, config.model.hidden_dim)
        else:
             self.encoder_input_proj = nn.Identity()

        self.encoder_pos_encoding = PositionalEncoding(config.model.hidden_dim, config.model.dropout)

        self.encoder = nn.ModuleList([
            FFTBlock(
                d_model=config.model.hidden_dim,
                n_heads=config.model.fft_n_heads,
                d_ff=config.model.fft_conv_ffn_dim,
                kernel_size=config.model.fft_conv_kernel_size,
                dropout=config.model.dropout
            ) for _ in range(config.model.fft_n_layers)
        ])

        self.length_regulator = LengthRegulator()

        # Decoder input dim: Encoder hidden dim + F0 feature
        decoder_input_dim = config.model.hidden_dim + 1
        if decoder_input_dim != config.model.hidden_dim:
             print(f"Warning: Decoder input dim ({decoder_input_dim}) != hidden_dim ({config.model.hidden_dim}). Adding projection.")
             self.decoder_input_proj = nn.Linear(decoder_input_dim, config.model.hidden_dim)
        else:
             self.decoder_input_proj = nn.Identity()

        self.decoder_pos_encoding = PositionalEncoding(config.model.hidden_dim, config.model.dropout)

        self.decoder = nn.ModuleList([
            FFTBlock(
                d_model=config.model.hidden_dim,
                n_heads=config.model.fft_n_heads,
                d_ff=config.model.fft_conv_ffn_dim,
                kernel_size=config.model.fft_conv_kernel_size,
                dropout=config.model.dropout
            ) for _ in range(config.model.fft_n_layers)
        ])

        self.mel_proj = nn.Linear(config.model.hidden_dim, config.model.n_mels)

        # Loss function
        self.loss_fn = nn.L1Loss(reduction='none') # Use L1 Loss (MAE) for Mels, reduction='none' for masking

    def forward(self, phonemes, midi, durations, f0, input_mask, output_mask):
        """
        Args:
            phonemes (Tensor): (B, N) Phoneme IDs
            midi (Tensor): (B, N) MIDI Note IDs
            durations (Tensor): (B, N) Durations in frames
            f0 (Tensor): (B, T) Ground truth F0 contour
            input_mask (Tensor): (B, N) Mask for phoneme/midi/duration inputs
            output_mask (Tensor): (B, T) Mask for mel/f0 outputs

        Returns:
            Tensor: Predicted Mel Spectrogram (B, T, n_mels)
        """
        # --- Encoder ---
        phone_emb = self.phoneme_embedding(phonemes)  # (B, N, emb_dim)
        midi_emb = self.midi_embedding(midi)      # (B, N, midi_emb_dim)
        encoder_input = torch.cat([phone_emb, midi_emb], dim=-1) # (B, N, emb_dim + midi_emb_dim)
        encoder_input = self.encoder_input_proj(encoder_input) # (B, N, hidden_dim)

        # Apply input mask before positional encoding? Or after? Usually after projection, before FFT blocks.
        # Masking is handled inside FFTBlock via key_padding_mask.
        # Zero out padded inputs explicitly? Embedding pad_idx should handle this.
        # encoder_input = encoder_input * input_mask.unsqueeze(-1).float()

        encoder_output = self.encoder_pos_encoding(encoder_input)

        # Pass through encoder FFT blocks
        for fft_block in self.encoder:
            encoder_output = fft_block(encoder_output, mask=input_mask)

        # Apply mask *after* encoder? Usually not needed if internal masking works.
        # encoder_output = encoder_output * input_mask.unsqueeze(-1).float()

        # --- Length Regulator ---
        # Pass the computed output_mask to ensure consistency if needed, otherwise LR computes it.
        # Using the ground truth output mask ensures the expanded length matches the target Mel length.
        expanded_output, _ = self.length_regulator(encoder_output, durations, output_mask) # (B, T, hidden_dim)

        # --- Decoder ---
        # Add F0 contour
        # Ensure f0 has correct shape (B, T, 1) for concatenation
        if f0.ndim == 2:
            f0 = f0.unsqueeze(-1) # (B, T, 1)

        decoder_input = torch.cat([expanded_output, f0], dim=-1) # (B, T, hidden_dim + 1)
        decoder_input = self.decoder_input_proj(decoder_input) # (B, T, hidden_dim)

        # Apply output mask before pos encoding? Similar to encoder. Let FFT handle mask.
        # decoder_input = decoder_input * output_mask.unsqueeze(-1).float()

        decoder_output = self.decoder_pos_encoding(decoder_input)

        # Pass through decoder FFT blocks
        for fft_block in self.decoder:
            # Use output_mask for decoder self-attention
            decoder_output = fft_block(decoder_output, mask=output_mask)

        # Apply mask after decoder?
        # decoder_output = decoder_output * output_mask.unsqueeze(-1).float()

        # --- Projection to Mel ---
        mel_pred = self.mel_proj(decoder_output) # (B, T, n_mels)

        return mel_pred

    def _calculate_loss(self, mel_pred, mel_target, output_mask):
        """Calculates masked L1 loss."""
        # mel_pred: (B, T, n_mels)
        # mel_target: (B, T, n_mels)
        # output_mask: (B, T)
        loss_unmasked = self.loss_fn(mel_pred, mel_target) # (B, T, n_mels)
        # Expand mask to match loss shape: (B, T) -> (B, T, 1) -> (B, T, n_mels)
        mask_expanded = output_mask.unsqueeze(-1).expand_as(loss_unmasked) # Boolean mask
        loss_masked = torch.where(mask_expanded, loss_unmasked, torch.zeros_like(loss_unmasked))
        # Calculate mean loss only over non-masked elements
        total_elements = mask_expanded.sum()
        if total_elements == 0:
            return torch.tensor(0.0, device=mel_pred.device) # Avoid division by zero
        total_loss = loss_masked.sum() / total_elements
        return total_loss


    def training_step(self, batch, batch_idx):
        mel_pred = self(
            phonemes=batch['phonemes'],
            midi=batch['midi'],
            durations=batch['durations'],
            f0=batch['f0'],
            input_mask=batch['input_mask'],
            output_mask=batch['output_mask']
        )
        loss = self._calculate_loss(mel_pred, batch['mels'], batch['output_mask'])
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch['sample_ids']))
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mel_pred = self(
            phonemes=batch['phonemes'],
            midi=batch['midi'],
            durations=batch['durations'],
            f0=batch['f0'],
            input_mask=batch['input_mask'],
            output_mask=batch['output_mask']
        )
        loss = self._calculate_loss(mel_pred, batch['mels'], batch['output_mask'])
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch['sample_ids']))

        # Store predictions for visualization at epoch end
        # Only log a few examples to avoid excessive memory/log size
        if batch_idx < self.config.validation.num_val_samples_log:
            # Detach and move to CPU *before* storing
            return {
                'val_loss_step': loss.cpu(),
                'mel_pred': mel_pred[0].cpu().numpy(), # Take first sample in batch
                'mel_target': batch['mels'][0].cpu().numpy(),
                'sample_id': batch['sample_ids'][0],
                'mel_len': batch['mel_lens'][0].cpu().item()
            }
        else:
             return {'val_loss_step': loss.cpu()}


    def validation_epoch_end(self, outputs):
        # Optional: Calculate average loss manually if needed
        # avg_loss = torch.stack([x['val_loss_step'] for x in outputs]).mean()
        # self.log('val_loss_epoch', avg_loss)

        # Log Mel spectrogram plots for the first few samples
        if not self.trainer.sanity_checking: # Don't plot during sanity check
            num_to_log = self.config.validation.num_val_samples_log
            logged_count = 0
            for i, output in enumerate(outputs):
                if 'mel_pred' in output and logged_count < num_to_log:
                    sample_id = output['sample_id']
                    mel_len = output['mel_len']
                    mel_pred = output['mel_pred'][:mel_len, :] # Trim padding
                    mel_target = output['mel_target'][:mel_len, :] # Trim padding

                    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    fig.suptitle(f"Sample: {sample_id} (Epoch {self.current_epoch})")

                    img_pred = librosa.display.specshow(mel_pred.T, ax=axes[0], sr=self.config.preprocessing.sample_rate,
                                                        hop_length=self.config.preprocessing.hop_length,
                                                        fmin=self.config.preprocessing.fmin,
                                                        fmax=self.config.preprocessing.fmax, x_axis='time', y_axis='mel')
                    axes[0].set_title("Predicted Mel Spectrogram")
                    fig.colorbar(img_pred, ax=axes[0], format='%+2.0f dB') # Assuming mels are approx dB scale

                    img_target = librosa.display.specshow(mel_target.T, ax=axes[1], sr=self.config.preprocessing.sample_rate,
                                                          hop_length=self.config.preprocessing.hop_length,
                                                          fmin=self.config.preprocessing.fmin,
                                                          fmax=self.config.preprocessing.fmax, x_axis='time', y_axis='mel')
                    axes[1].set_title("Ground Truth Mel Spectrogram")
                    fig.colorbar(img_target, ax=axes[1], format='%+2.0f dB')

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

                    # Log figure to TensorBoard
                    self.logger.experiment.add_figure(f"Validation Mel Specs/{sample_id}", fig, global_step=self.global_step)
                    plt.close(fig) # Close figure to free memory
                    logged_count += 1


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        # Learning rate scheduler (Example: Noam schedule - common for Transformers)
        # Using a simpler warmup + constant schedule for PoC might be easier
        warmup_steps = self.config.training.lr_warmup_steps

        def lr_lambda(current_step):
            # Linear warmup
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Optional: Add decay after warmup (e.g., inverse sqrt or cosine)
            # For PoC, let's keep it constant after warmup
            return 1.0
            # Example Noam decay:
            # return (self.config.model.hidden_dim ** -0.5) * min((current_step + 1) ** -0.5, (current_step + 1) * warmup_steps ** -1.5)


        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Call scheduler every step
                "frequency": 1,
            },
        }