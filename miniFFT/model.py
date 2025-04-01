import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import yaml
import matplotlib.pyplot as plt
import librosa.display # For visualization only

def create_padding_mask(seq_len, max_len):
    """Creates a boolean mask [False=real, True=padded]"""
    return torch.arange(max_len)[None, :] >= seq_len[:, None]

# --- FFT Block Components ---

class MultiHeadAttention(nn.Module):
    """ Standard Multi-Head Attention """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head # Assume d_v = d_k
        self.w_qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x: (B, SeqLen, Dim)
        mask: (B, SeqLen) Boolean mask where True indicates padding
        """
        B, S, D = x.shape
        residual = x

        qkv = self.w_qkv(x).chunk(3, dim=-1) # Tuple of (B, S, D)
        q, k, v = map(lambda t: t.view(B, S, self.n_head, self.d_k).transpose(1, 2), qkv) # (B, n_head, S, d_k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, n_head, S, S)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S) for key mask
            attn_scores = attn_scores.masked_fill(mask, -1e9) # Fill padding keys

        attn_probs = F.softmax(attn_scores, dim=-1) # (B, n_head, S, S)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, v) # (B, n_head, S, d_k)
        output = output.transpose(1, 2).contiguous().view(B, S, D) # (B, S, D)

        output = self.fc(output)
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # Add & Norm

        return output, attn_probs # Return attention probs for visualization

class PositionwiseFeedForward(nn.Module):
    """ Conv1D based FFN from FastSpeech/TransformerTTS """
    def __init__(self, d_model, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = nn.ReLU() # Or GeLU/SiLU

    def forward(self, x):
        """ x: (B, SeqLen, Dim) """
        residual = x
        x = x.transpose(1, 2) # (B, Dim, SeqLen) for Conv1D
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2) # (B, SeqLen, Dim)
        x = self.layer_norm(x + residual) # Add & Norm
        return x

class FFTBlock(nn.Module):
    """ Single FFT Block """
    def __init__(self, d_model, n_head, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, kernel_size, dropout)

    def forward(self, x, mask=None):
        """
        x: (B, SeqLen, Dim)
        mask: (B, SeqLen) Boolean mask where True indicates padding
        """
        attn_output, attn_probs = self.attention(x, mask=mask)
        output = self.feed_forward(attn_output)
        return output, attn_probs

# --- Length Regulator ---
class LengthRegulator(nn.Module):
    """ Expands sequence based on durations """
    def forward(self, x, durations, max_len=None):
        """
        x: (B, N, D) Encoder output
        durations: (B, N) Phoneme durations in frames
        max_len: Optional target maximum frame length for output tensor
        """
        B, N, D = x.shape
        total_lengths = durations.sum(dim=1)
        if max_len is None:
            max_len = total_lengths.max()

        # Create index mapping using repeat_interleave
        # Example: durations = [2, 3, 1] -> indices = [0, 0, 1, 1, 1, 2]
        repeats = durations.flatten()
        base_indices = torch.arange(N, device=x.device).repeat(B)
        expanded_indices = torch.repeat_interleave(base_indices, repeats).view(B, -1) # (B, T_actual)

        # Gather elements based on expanded indices
        # Need to add batch offset to indices for gather
        batch_offsets = torch.arange(B, device=x.device) * N
        expanded_indices_flat = (expanded_indices + batch_offsets.unsqueeze(1)).flatten()

        # Reshape x to (B*N, D)
        x_flat = x.reshape(-1, D)

        # Gather - this gets elements correctly but might be slow for long sequences
        output_flat = torch.index_select(x_flat, 0, expanded_indices_flat)
        output = output_flat.view(B, -1, D) # (B, T_actual, D)

        # Pad to max_len if necessary
        if output.shape[1] < max_len:
            padding_size = max_len - output.shape[1]
            padding = torch.zeros((B, padding_size, D), device=x.device)
            output = torch.cat([output, padding], dim=1)
        elif output.shape[1] > max_len:
             output = output[:, :max_len, :] # Should not happen if durations sum correctly

        # Create mask for the expanded sequence (True where padded)
        mel_mask = torch.arange(max_len, device=x.device)[None, :] >= total_lengths[:, None]

        return output, mel_mask

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # Not a parameter

    def forward(self, x):
        """ x: (B, SeqLen, Dim) """
        # self.pe is (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- Main Lightning Module ---
class FFTLightSVS(pl.LightningModule):
    def __init__(self, config_path="config/default.yaml"):
        super().__init__()
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.save_hyperparameters(self.config) # Makes config available as self.hparams

        # --- Embeddings ---
        self.phoneme_embedding = nn.Embedding(
            self.hparams.data['phoneme_vocab_size'],
            self.hparams.model['phoneme_emb_dim'],
            padding_idx=self.hparams.data['pad_token_id']
        )
        self.midi_embedding = nn.Embedding(
            self.hparams.data['midi_range'], # Assuming MIDI notes are 0-indexed + pad
            self.hparams.model['midi_emb_dim'],
            padding_idx=self.hparams.data['pad_token_id']
        )
        # Simple projection if embedding dims don't match hidden_dim
        input_proj_dim = self.hparams.model['phoneme_emb_dim'] + self.hparams.model['midi_emb_dim']
        self.input_proj = nn.Linear(input_proj_dim, self.hparams.model['hidden_dim'])

        self.pos_encoder = PositionalEncoding(self.hparams.model['hidden_dim'], self.hparams.model['dropout'])
        self.pos_decoder = PositionalEncoding(self.hparams.model['hidden_dim'], self.hparams.model['dropout']) # Separate?

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList([
            FFTBlock(
                d_model=self.hparams.model['hidden_dim'],
                n_head=self.hparams.model['encoder_heads'],
                d_ff=self.hparams.model['ffn_hidden_dim'],
                kernel_size=self.hparams.model['ffn_conv_kernel_size'],
                dropout=self.hparams.model['dropout']
            ) for _ in range(self.hparams.model['encoder_layers'])
        ])

        # --- Length Regulator ---
        self.length_regulator = LengthRegulator()

        # --- F0 Integration ---
        # Project F0 (scalar) to match hidden_dim maybe? Or just concat?
        # Simple concat: Decoder input dim = hidden_dim + 1
        decoder_input_dim = self.hparams.model['hidden_dim'] + 1
        self.decoder_input_proj = nn.Linear(decoder_input_dim, self.hparams.model['hidden_dim'])


        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList([
            FFTBlock(
                d_model=self.hparams.model['hidden_dim'],
                n_head=self.hparams.model['decoder_heads'],
                d_ff=self.hparams.model['ffn_hidden_dim'],
                kernel_size=self.hparams.model['ffn_conv_kernel_size'],
                dropout=self.hparams.model['dropout']
            ) for _ in range(self.hparams.model['decoder_layers'])
        ])

        # --- Output Projection ---
        self.final_proj = nn.Linear(self.hparams.model['hidden_dim'], self.hparams.data['n_mels'])

        # --- Loss ---
        self.criterion = nn.L1Loss(reduction='none') # Use reduction='none' for masking

        self.last_val_batch_for_viz = None # Store for visualization

    def forward(self, phonemes, midi_notes, durations, f0, phoneme_mask=None, mel_mask=None):
        """
        phonemes: (B, N)
        midi_notes: (B, N)
        durations: (B, N)
        f0: (B, T) Frame-level F0 contour
        phoneme_mask: (B, N) Boolean mask (True where padded)
        mel_mask: (B, T) Boolean mask (True where padded, generated by LR or passed in)
        """
        # --- Encoder Path ---
        phon_emb = self.phoneme_embedding(phonemes) # (B, N, ph_emb_dim)
        midi_emb = self.midi_embedding(midi_notes) # (B, N, midi_emb_dim)

        # Combine embeddings
        x = torch.cat([phon_emb, midi_emb], dim=-1) # (B, N, ph_emb+midi_emb)
        x = self.input_proj(x) # (B, N, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x) # (B, N, hidden_dim)

        # Encoder FFT Blocks
        encoder_attns = []
        for block in self.encoder_blocks:
            x, attn = block(x, mask=phoneme_mask) # (B, N, hidden_dim)
            encoder_attns.append(attn)

        # --- Length Regulator ---
        # x is now (B, N, hidden_dim)
        # durations is (B, N)
        # We need the max target length T based on mel_mask if provided, or from durations
        max_t_target = f0.shape[1] # Use provided f0 length as target T
        x_expanded, lr_mel_mask = self.length_regulator(x, durations, max_len=max_t_target)
        # x_expanded: (B, T, hidden_dim)
        # lr_mel_mask: (B, T) mask generated by LR based on actual expanded lengths

        # Combine LR mask with any original padding mask (if sequences were padded *before* LR)
        # If collate_fn pads mel_target and f0, then mel_mask passed in is the one to use.
        # The lr_mel_mask tells us which *content* frames are valid *within* the expanded sequence
        # before padding to max_t_batch in collate_fn. We need the final mask.
        # Let's assume the `mel_mask` from the dataloader is the ultimate truth for padding.
        final_mel_mask = mel_mask if mel_mask is not None else lr_mel_mask

        # --- Decoder Path ---
        # Integrate F0
        f0 = f0.unsqueeze(-1) # (B, T, 1)
        decoder_input = torch.cat([x_expanded, f0], dim=-1) # (B, T, hidden_dim + 1)
        decoder_input = self.decoder_input_proj(decoder_input) # (B, T, hidden_dim)

        # Add positional encoding
        decoder_input = self.pos_decoder(decoder_input) # (B, T, hidden_dim)

        # Decoder FFT Blocks
        decoder_attns = []
        y = decoder_input
        for block in self.decoder_blocks:
            y, attn = block(y, mask=final_mel_mask) # (B, T, hidden_dim)
            decoder_attns.append(attn)

        # --- Final Projection ---
        mel_pred = self.final_proj(y) # (B, T, n_mels)

        # Mask output padding for clarity (optional, loss handles it)
        if final_mel_mask is not None:
             mel_pred = mel_pred.masked_fill(final_mel_mask.unsqueeze(-1), 0.0) # Zero out padded frames

        # Return predictions and potentially attention weights for viz
        return mel_pred, encoder_attns, decoder_attns

    def _common_step(self, batch, batch_idx):
        # Handle case where batch could be None or doesn't have expected keys
        if batch is None:
            print("WARNING: Received None batch in _common_step")
            # Return dummy zero loss and empty tensors
            dummy_mel = torch.zeros((1, 1, self.hparams.data['n_mels']), device=self.device)
            dummy_mask = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
            return torch.tensor(0.0, requires_grad=True, device=self.device), dummy_mel, dummy_mel, dummy_mask

        try:
            phonemes = batch['phonemes']
            midi_notes = batch['midi_notes']
            durations = batch['durations']
            f0 = batch['f0']
            mel_target = batch['mel_target']
            phoneme_mask = batch['phoneme_mask']  # True where padded
            mel_mask = batch['mel_mask']          # True where padded

            # Forward pass
            mel_pred, _, _ = self(phonemes, midi_notes, durations, f0, phoneme_mask, mel_mask)

            # Calculate Loss (L1 Loss)
            loss = self.criterion(mel_pred, mel_target)  # (B, T, n_mels)

            # Apply mask (zero out loss for padded frames)
            loss = loss.masked_fill(mel_mask.unsqueeze(-1), 0.0)

            # Calculate mean loss *only* over non-padded elements
            # Sum over mel bins and time steps, then divide by number of non-padded frames
            # Add small epsilon to avoid division by zero
            non_padded_frames = (~mel_mask).sum() 
            if non_padded_frames > 0:
                loss = loss.sum() / non_padded_frames
            else:
                # Handle extreme case where all frames are padded
                loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                print("WARNING: All frames are padded in batch! Using zero loss.")

            return loss, mel_pred, mel_target, mel_mask
            
        except Exception as e:
            print(f"ERROR in _common_step: {str(e)}")
            # Return dummy values to avoid breaking the training loop
            dummy_mel = torch.zeros((1, 1, self.hparams.data['n_mels']), device=self.device)
            dummy_mask = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
            return torch.tensor(0.0, requires_grad=True, device=self.device), dummy_mel, dummy_mel, dummy_mask

    def training_step(self, batch, batch_idx):
        loss, _, _, _ = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['phonemes'].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mel_pred, mel_target, mel_mask = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch['phonemes'].shape[0])

        # Log visualizations for the first validation batch
        if batch_idx == 0:
             self.last_val_batch_for_viz = (mel_pred.detach().cpu(), mel_target.cpu(), mel_mask.cpu())
        return loss

    def on_validation_epoch_end(self):
        if self.last_val_batch_for_viz is not None and self.logger:
            mel_pred, mel_target, mel_mask = self.last_val_batch_for_viz
            num_to_log = min(self.hparams.training['num_val_samples_to_log'], mel_pred.shape[0])

            for i in range(num_to_log):
                # Find actual length before padding
                length = (~mel_mask[i]).sum().item()
                pred = mel_pred[i, :length].numpy().T # Transpose for librosa display (n_mels, T)
                target = mel_target[i, :length].numpy().T

                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                fig.suptitle(f"Validation Sample {i} (Epoch {self.current_epoch})")

                img_pred = librosa.display.specshow(pred, ax=axes[0], sr=self.hparams.data['sample_rate'],
                                                    hop_length=self.hparams.data['hop_length'], x_axis='time', y_axis='mel')
                axes[0].set_title("Predicted Mel Spectrogram")
                fig.colorbar(img_pred, ax=axes[0], format='%+2.0f dB') # Adjust format if needed

                img_target = librosa.display.specshow(target, ax=axes[1], sr=self.hparams.data['sample_rate'],
                                                      hop_length=self.hparams.data['hop_length'], x_axis='time', y_axis='mel')
                axes[1].set_title("Target Mel Spectrogram")
                fig.colorbar(img_target, ax=axes[1], format='%+2.0f dB')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

                # Log figure to TensorBoard
                self.logger.experiment.add_figure(f"Validation/Sample_{i}", fig, global_step=self.global_step)
                plt.close(fig) # Close figure to free memory

            self.last_val_batch_for_viz = None # Reset after logging


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.training['learning_rate'],
            betas=(self.hparams.training['adam_beta1'], self.hparams.training['adam_beta2']),
            eps=float(self.hparams.training['adam_eps']),
            weight_decay=self.hparams.training['weight_decay']
        )

        # Simple learning rate scheduler: warmup then constant (or decay)
        # Using LambdaLR for warmup - could use others like CosineAnnealingLR later
        def lr_lambda(current_step: int):
            warmup_steps = self.hparams.training['lr_warmup_steps']
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0 # Keep constant after warmup for simplicity in PoC

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Call scheduler every step
                "frequency": 1
            },
        }