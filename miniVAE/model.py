import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pathlib import Path

class Encoder(nn.Module):
    """Encoder for the VAE model."""
    
    def __init__(self, n_phonemes, n_mels, phoneme_dim, pitch_dim, hidden_dim, latent_dim, n_layers=2):
        super().__init__()
        
        # Embeddings
        self.phoneme_embedding = nn.Embedding(n_phonemes, phoneme_dim)
        self.pitch_embedding = nn.Embedding(128, pitch_dim)  # MIDI range
        
        # Feature processing
        self.f0_processing = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Encoder GRU
        self.encoder_rnn = nn.GRU(
            input_size=phoneme_dim + pitch_dim + 32,  # phone + pitch + f0
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # VAE projections
        self.mean = nn.Linear(hidden_dim * 2, latent_dim)  # *2 for bidirectional
        self.logvar = nn.Linear(hidden_dim * 2, latent_dim)
    
    def forward(self, phonemes, midi, f0, lengths):
        # Get embeddings
        phone_embed = self.phoneme_embedding(phonemes)
        pitch_embed = self.pitch_embedding(midi)
        
        # Process F0
        f0_expanded = f0.unsqueeze(-1)
        f0_processed = self.f0_processing(f0_expanded)
        
        # Concatenate features
        encoder_input = torch.cat([phone_embed, pitch_embed, f0_processed], dim=-1)
        
        # Pack for variable length
        packed_input = nn.utils.rnn.pack_padded_sequence(
            encoder_input, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # Run through encoder
        _, hidden = self.encoder_rnn(packed_input)
        
        # Combine bidirectional states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Get last layer forward/backward
        
        # Get latent parameters
        z_mean = self.mean(hidden)
        z_logvar = self.logvar(hidden)
        
        return z_mean, z_logvar
    
class Decoder(nn.Module):
    """Decoder for the VAE model."""
    
    def __init__(self, latent_dim, hidden_dim, n_mels, n_layers=2):
        super().__init__()
        
        # Initial projection from latent to decoder hidden
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        
        # F0 conditioning
        self.f0_conditioning = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1)
        )
        
        # Decoder GRU
        self.decoder_rnn = nn.GRU(
            input_size=hidden_dim + 32,  # hidden + f0 conditioning
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True
        )
        
        # Output projection
        self.mel_projection = nn.Linear(hidden_dim, n_mels)
        
        # Postnet for refining spectrogram
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, n_mels, kernel_size=5, padding=2)
        )
    
    def forward(self, z, f0, lengths, teacher_forcing_ratio=0.5, teacher_targets=None):
        batch_size = z.size(0)
        max_len = f0.size(1)
        
        # Project latent to hidden
        hidden = self.latent_projection(z).unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
        
        # Process F0 conditioning - ensure shapes are correct
        f0_expanded = f0.unsqueeze(-1)  # [B, T, 1]
        f0_expanded = f0_expanded.transpose(1, 2)  # [B, 1, T]
        f0_cond = self.f0_conditioning(f0_expanded)  # [B, 32, T]
        f0_cond = f0_cond.transpose(1, 2)  # [B, T, 32]
        
        # Verify maximum steps to process
        actual_max_steps = min(max_len, lengths.max().item(), f0_cond.size(1))
        
        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, 1, self.decoder_rnn.input_size, device=z.device)
        
        # Use latent size for initialization, not hidden size
        latent_dim = z.size(-1)
        decoder_input[:, :, :latent_dim] = z.unsqueeze(1)
        
        # Pre-allocate outputs
        all_outputs = []
        current_step = 0
        
        # Start autoregressive decoding with proper bounds checking
        while current_step < actual_max_steps:
            # Get F0 conditioning for this step - with bounds check
            f0_step = f0_cond[:, current_step:current_step+1, :]
            
            # Ensure the F0 conditioning size is as expected
            f0_dim = f0_step.size(-1)
            if f0_dim > 0 and f0_dim <= decoder_input.size(-1):
                decoder_input[:, :, -f0_dim:] = f0_step
            
            # Run decoder for one step
            output, hidden = self.decoder_rnn(decoder_input, hidden)
            mel_output = self.mel_projection(output)
            all_outputs.append(mel_output)
            
            # Update input for next step (with bounds check for teacher forcing)
            use_teacher_forcing = (teacher_targets is not None and 
                                current_step < teacher_targets.size(1) and 
                                np.random.random() < teacher_forcing_ratio)
            
            # Prepare next decoder input
            decoder_input = torch.zeros_like(decoder_input)
            decoder_input[:, :, :output.size(-1)] = output
            
            current_step += 1
        
        # Handle case where no outputs were generated (fallback)
        if len(all_outputs) == 0:
            # Create a single frame of zeros as fallback
            mel_output = torch.zeros(batch_size, 1, self.mel_projection.out_features, device=z.device)
            all_outputs.append(mel_output)
        
        # Concatenate outputs
        decoder_outputs = torch.cat(all_outputs, dim=1)
        
        # Apply postnet for refinement
        postnet_outputs = decoder_outputs + self.postnet(decoder_outputs.transpose(1, 2)).transpose(1, 2)
        
        return decoder_outputs, postnet_outputs

class SVSModelLightning(pl.LightningModule):
    """Lightning module for Singing Voice Synthesis."""
    
    def __init__(self, config_path="config/default.yaml"):
        super().__init__()
        self.save_hyperparameters()
        
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Create data placeholders - will be set when we process a batch
        self.n_phonemes = None
        self.n_mels = None
        
        # Model components will be initialized in setup()
        self.encoder = None
        self.decoder = None
        
        # KL annealing
        self.kl_weight = self.config['training']['kl_weight_init']
        self.kl_weight_end = self.config['training']['kl_weight_end']
        self.kl_annealing_epochs = self.config['training']['kl_annealing_epochs']
    
    def setup(self, stage=None):
        # Ensure datamodule is set up
        self.n_phonemes = 50  # Will be updated on first batch
        self.n_mels = self.config['preprocessing']['n_mels']
        
        # Initialize model components
        self.encoder = Encoder(
            n_phonemes=self.n_phonemes,
            n_mels=self.n_mels, 
            phoneme_dim=self.config['model']['phoneme_dim'],
            pitch_dim=self.config['model']['pitch_dim'],
            hidden_dim=self.config['model']['encoder_dim'],
            latent_dim=self.config['model']['latent_dim'],
            n_layers=self.config['model']['encoder_layers']
        )
        
        self.decoder = Decoder(
            latent_dim=self.config['model']['latent_dim'],
            hidden_dim=self.config['model']['decoder_dim'],
            n_mels=self.n_mels,
            n_layers=self.config['model']['decoder_layers']
        )
    
    def update_model_phonemes(self, n_phonemes):
        """Update model if vocabulary size changes."""
        if self.n_phonemes != n_phonemes:
            self.n_phonemes = n_phonemes
            old_phoneme_embedding = self.encoder.phoneme_embedding
            new_phoneme_embedding = nn.Embedding(n_phonemes, self.config['model']['phoneme_dim'])
            
            # Copy weights for existing tokens
            with torch.no_grad():
                common_size = min(old_phoneme_embedding.weight.shape[0], n_phonemes)
                new_phoneme_embedding.weight[:common_size] = old_phoneme_embedding.weight[:common_size]
            
            # Replace embedding
            self.encoder.phoneme_embedding = new_phoneme_embedding
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, phonemes, midi, f0, lengths, teacher_forcing_ratio=0):
        """Full model forward pass."""
        # Update phoneme vocabulary if needed
        self.update_model_phonemes(phonemes.max().item() + 1)
        
        # Encoder forward
        z_mean, z_logvar = self.encoder(phonemes, midi, f0, lengths)
        
        # Sample latent vector
        z = self.reparameterize(z_mean, z_logvar)
        
        # Decoder forward
        decoder_outputs, postnet_outputs = self.decoder(z, f0, lengths)
        
        return decoder_outputs, postnet_outputs, z_mean, z_logvar
    
    def training_step(self, batch, batch_idx):
        # Get batch data
        phonemes = batch['phonemes']
        midi = batch['midi']
        f0 = batch['f0']
        mel_gt = batch['mel_spectrogram']
        lengths = batch['lengths']
        mask = batch['mask']
        
        # Forward pass with teacher forcing during training
        decoder_outputs, postnet_outputs, z_mean, z_logvar = self(
            phonemes, midi, f0, lengths, teacher_forcing_ratio=0.5
        )
        
        # Apply mask for padding
        mask = mask.unsqueeze(-1)  # Add mel dimension
        
        # Calculate losses
        decoder_loss = F.l1_loss(decoder_outputs * mask, mel_gt * mask)
        postnet_loss = F.l1_loss(postnet_outputs * mask, mel_gt * mask)
        
        # Combined reconstruction loss
        mel_loss = decoder_loss + postnet_loss
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        
        # Total loss with KL weight
        loss = mel_loss + self.kl_weight * kl_loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mel_loss', mel_loss)
        self.log('train_kl_loss', kl_loss)
        self.log('train_kl_weight', self.kl_weight)
        
        return loss
    
    def on_train_epoch_end(self):
        # Update KL weight for annealing
        if self.current_epoch < self.kl_annealing_epochs:
            self.kl_weight = self.config['training']['kl_weight_init'] + self.current_epoch * (
                self.kl_weight_end - self.config['training']['kl_weight_init']
            ) / self.kl_annealing_epochs
    
    def validation_step(self, batch, batch_idx):
        # Get batch data
        phonemes = batch['phonemes']
        midi = batch['midi']
        f0 = batch['f0']
        mel_gt = batch['mel_spectrogram']
        lengths = batch['lengths']
        mask = batch['mask']
        
        # Forward pass without teacher forcing
        decoder_outputs, postnet_outputs, z_mean, z_logvar = self(
            phonemes, midi, f0, lengths, teacher_forcing_ratio=0.0
        )
        
        # Apply mask for padding
        mask = mask.unsqueeze(-1)
        
        # Calculate losses
        decoder_loss = F.l1_loss(decoder_outputs * mask, mel_gt * mask)
        postnet_loss = F.l1_loss(postnet_outputs * mask, mel_gt * mask)
        
        # Combined reconstruction loss
        mel_loss = decoder_loss + postnet_loss
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        
        # Total loss with KL weight
        loss = mel_loss + self.kl_weight * kl_loss
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mel_loss', mel_loss)
        self.log('val_kl_loss', kl_loss)
        
        # Log sample spectrograms every few epochs
        if self.current_epoch % self.config['logging']['sample_interval'] == 0 and batch_idx == 0:
            self._log_spectrograms(mel_gt[0], postnet_outputs[0], "val_sample")
        
        return loss
    
    def _log_spectrograms(self, gt_mel, pred_mel, name):
        """Log ground truth and predicted spectrograms to TensorBoard."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Convert tensors to numpy and denormalize
        gt_mel_np = gt_mel.cpu().detach().numpy() * 80 - 80
        pred_mel_np = pred_mel.cpu().detach().numpy() * 80 - 80
        
        # Plot ground truth
        axes[0].imshow(gt_mel_np.T, aspect='auto', origin='lower')
        axes[0].set_title("Ground Truth Mel Spectrogram")
        axes[0].set_ylabel("Mel bins")
        
        # Plot prediction
        axes[1].imshow(pred_mel_np.T, aspect='auto', origin='lower')
        axes[1].set_title("Predicted Mel Spectrogram")
        axes[1].set_ylabel("Mel bins")
        axes[1].set_xlabel("Frames")
        
        plt.tight_layout()
        
        # Log to TensorBoard
        self.logger.experiment.add_figure(name, fig, self.current_epoch)
        plt.close(fig)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }