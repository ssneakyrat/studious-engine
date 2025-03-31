import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import calculate_target_frames, plot_mel_comparison_to_buf, buf_to_image # Import utilities
import numpy as np
import math

class Encoder(nn.Module):
    """Encodes the condition Mel Spectrogram into latent space parameters."""
    def __init__(self, config):
        super().__init__()
        self.n_mels = config['audio']['n_mels']
        self.latent_dim = config['model']['latent_dim']
        num_layers = config['model']['num_layers_encoder']
        base_channels = config['model']['encoder_base_channels']
        kernel_size = config['model']['kernel_size']
        use_bn = config['model']['use_batchnorm']

        # Increase encoder's base channels to match decoder scale
        if base_channels < 32:
            print(f"WARNING: Encoder base_channels increased from {base_channels} to 32 for better balance")
            base_channels = 32

        layers = []
        in_channels = 1
        current_h = self.n_mels # Keep track of spatial dimensions if needed

        for i in range(num_layers):
            out_channels = base_channels * (2**i)
            stride = 2 if i < num_layers - 1 else 1 
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the flattened size dynamically
        target_frames = calculate_target_frames(config)
        dummy_input = torch.randn(1, 1, self.n_mels, target_frames)
        with torch.no_grad():
             conv_output_shape = self.conv_layers(dummy_input).shape
        self.flattened_size = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)

        # Improved weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class SimpleConcatConditioning(nn.Module):
    """Simpler conditioning mechanism that concatenates latent vector with feature maps."""
    def __init__(self, input_dim, condition_dim):
        super().__init__()
        self.condition_proj = nn.Linear(condition_dim, input_dim)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, condition):
        # x: [B, C, H, W]
        # condition: [B, D]
        condition = self.activation(self.condition_proj(condition))  # [B, input_dim]
        condition = condition.unsqueeze(-1).unsqueeze(-1)  # [B, input_dim, 1, 1]
        condition = condition.expand(-1, -1, x.size(2), x.size(3))  # [B, input_dim, H, W]
        return torch.cat([x, condition], dim=1)  # [B, C+input_dim, H, W]

class SimpleConditionalDecoderBlock(nn.Module):
    """A simplified decoder block using concatenation conditioning."""
    def __init__(self, in_channels, out_channels, latent_dim, kernel_size, use_bn):
        super().__init__()
        padding = kernel_size // 2
        self.use_bn = use_bn
        
        # Calculate how many channels to add from condition
        cond_channels = max(16, in_channels // 4)  # Add 25% more channels from condition
        
        # Condition concatenation
        self.cond = SimpleConcatConditioning(cond_channels, latent_dim)
        
        # First convolution after concatenation with condition
        self.conv1 = nn.Conv2d(in_channels + cond_channels, out_channels, kernel_size, padding=padding)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Second conv block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        
        # Residual connection (if needed)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, z):
        # Add condition via concatenation
        conditioned = self.cond(x, z)
        
        # Process the conditioned input
        out = self.conv1(conditioned)
        if self.use_bn:
            out = self.bn1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        
        # Apply residual connection
        residual = self.shortcut(x)
        out = out + residual
        
        out = self.act2(out)
        return out

class Decoder(nn.Module):
    """Decodes noise and latent vector z into a Mel Spectrogram."""
    def __init__(self, config):
        super().__init__()
        self.n_mels = config['audio']['n_mels']
        self.latent_dim = config['model']['latent_dim']
        num_layers = config['model']['num_layers_decoder']
        max_channels = config['model']['decoder_base_channels']
        kernel_size = config['model']['kernel_size']
        use_bn = config['model']['use_batchnorm']
        target_frames = calculate_target_frames(config)

        # Initial convolution to process noise input
        self.initial_conv = nn.Conv2d(1, max_channels, kernel_size, padding=kernel_size//2)

        # Use SimpleConditionalDecoderBlock instead of FiLM for more stability
        layers = []
        in_channels = max_channels
        for i in range(num_layers):
            out_channels = max(max_channels // (2**i), 16)
            layers.append(
                SimpleConditionalDecoderBlock(in_channels, out_channels, self.latent_dim, kernel_size, use_bn)
            )
            in_channels = out_channels

        self.res_blocks = nn.ModuleList(layers)

        # Final convolution
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Improved weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            nn.init.constant_(m.bias, 0)

    def forward(self, noise, z):
        x = self.initial_conv(noise)

        for block in self.res_blocks:
            x = block(x, z)

        output_mel = self.final_conv(x)

        # Ensure output dims match input noise dims
        noise_shape = noise.shape
        output_mel_shape = output_mel.shape
        if output_mel_shape[2] != noise_shape[2] or output_mel_shape[3] != noise_shape[3]:
            print(f"WARN: Decoder output shape {output_mel_shape} does not match noise shape {noise_shape}. Resizing.")
            output_mel = torch.nn.functional.interpolate(
                output_mel,
                size=(noise_shape[2], noise_shape[3]),
                mode='bilinear',
                align_corners=False
            )
        return output_mel

class ConditionalVAE(pl.LightningModule):
    """PyTorch Lightning Module for the Conditional VAE."""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Use smaller latent dimension 
        if self.hparams.model['latent_dim'] > 64:
            print(f"NOTE: Reducing latent dimension from {self.hparams.model['latent_dim']} to 64")
            self.hparams.model['latent_dim'] = 64
            self.config['model']['latent_dim'] = 64

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.target_frames = calculate_target_frames(config)
        self.n_mels = config['audio']['n_mels']
        self.padding_value = config['audio']['padding_value']

        # Mask token
        self.register_buffer(
            "mask_token",
             torch.full((1, 1, self.n_mels, self.target_frames), self.padding_value)
        )
        
        # Increase mask probability for better training
        self.mask_probability = 0.5  # Increased from config value for better training
        print(f"NOTE: Increased mask probability to {self.mask_probability} for more robust training")

        # Validation visualization counter
        self.val_vis_count = 0
        
        # Enable gradient clipping
        self.clip_gradients = True
        self.gradient_clip_val = 1.0

    def forward(self, condition, noise):
        mu, logvar = self.encoder(condition)
        z = self.reparameterize(mu, logvar)
        reconstructed_mel = self.decoder(noise, z)
        return reconstructed_mel, mu, logvar

    def reparameterize(self, mu, logvar):
        # Relaxed logvar clamping for better expressivity
        logvar = torch.clamp(logvar, -10, 10)  # Wider range than before (-5, 5)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _resize_to_match(self, tensor_to_resize, target_tensor):
        resized_tensor = tensor_to_resize
        target_shape = target_tensor.shape
        tensor_shape = tensor_to_resize.shape

        if len(tensor_shape) != 4 or len(target_shape) != 4:
            print(f"WARN: _resize_to_match requires 4D tensors. Got {len(tensor_shape)} and {len(target_shape)} dimensions.")
            return resized_tensor

        if tensor_shape[2] != target_shape[2] or tensor_shape[3] != target_shape[3]:
            print(f"WARN: Resizing tensor from {tensor_shape} to {target_shape}.")
            resized_tensor = torch.nn.functional.interpolate(
                tensor_to_resize,
                size=(target_shape[2], target_shape[3]),
                mode='bilinear',
                align_corners=False
            )
        return resized_tensor

    # Multi-resolution STFT loss functions
    def _stft_loss(self, x, y):
        # Parameters for multi-resolution STFT
        fft_sizes = [1024, 2048, 512]
        hop_sizes = [256, 512, 128]
        win_lengths = [1024, 2048, 512]
        
        # Convert from 2D mel spectrograms to 1D waveforms (simplified approach)
        # This is just for demonstration, as proper mel to waveform would require vocoder
        # Here we just treat the mel as a 1D signal for STFT purposes
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        
        loss = 0.0
        for fft_size, hop_size, win_length in zip(fft_sizes, hop_sizes, win_lengths):
            # Skip if signal is too short
            if x_flat.size(-1) < fft_size:
                continue
                
            # Compute STFT
            x_stft = torch.stft(x_flat, n_fft=fft_size, hop_length=hop_size, 
                               win_length=win_length, return_complex=True)
            y_stft = torch.stft(y_flat, n_fft=fft_size, hop_length=hop_size,
                               win_length=win_length, return_complex=True)
            
            # Get magnitudes
            x_mag = torch.abs(x_stft)
            y_mag = torch.abs(y_stft)
            
            # Spectral convergence loss
            sc_loss = torch.norm(x_mag - y_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-7)
            
            # Log STFT magnitude loss
            lm_loss = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
            
            loss += sc_loss + lm_loss
            
        return loss

    def _vae_loss(self, recon_x, x, mu, logvar):
        # Use the dedicated mel loss from losses.py
        from losses import MultiScaleMelLoss
        mel_loss_fn = MultiScaleMelLoss()
        loss_dict = mel_loss_fn(recon_x, x)
        recon_loss = loss_dict['loss']
        
        # Improved KL divergence calculation
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1)
        kl_loss = torch.mean(kl_loss)

        # Log components individually
        self.log('train/l1_loss', l1_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/mse_loss', mse_loss, on_step=True, on_epoch=True, logger=True)

        return recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        x = batch # Ground truth mel (B, 1, N_MELS, T)
        if batch_idx < 3: # Check first 3 batches
            print(f"Batch {batch_idx} - Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}, Std: {x.std()}")

        # Create noise input
        noise = torch.randn_like(x)

        # Create condition input with masking
        condition = torch.zeros_like(x)
        for i in range(x.size(0)):
            if torch.rand(1).item() < self.mask_probability:
                condition[i] = self.mask_token.to(x.device)
            else:
                condition[i] = x[i]

        # Forward pass
        recon_mel, mu, logvar = self(condition, noise)

        # Calculate loss
        recon_loss, kl_loss = self._vae_loss(recon_mel, x, mu, logvar)
        
        # Improved KL annealing - much slower ramp-up
        annealing_epochs = self.hparams.training.get('annealing_epochs', 100)  # Default to 100 if not set
        if annealing_epochs < 100:
            annealing_epochs = 100
            print(f"NOTE: Increasing annealing_epochs to {annealing_epochs} for better training")
            
        # Start with a much lower beta value and increase slowly
        beta_kl = min(0.05, 0.05 * self.current_epoch / annealing_epochs) * self.hparams.training['beta_kl']
        if self.current_epoch > annealing_epochs:
            beta_kl = 0.05 + min(0.95, (self.current_epoch - annealing_epochs) / annealing_epochs) * self.hparams.training['beta_kl']
        
        # Cap beta_kl at 1.0 for stability
        beta_kl = min(1.0, beta_kl)
        
        total_loss = recon_loss + beta_kl * kl_loss

        # Log latent variables for monitoring
        self.log('train/mu_mean', mu.mean(), on_step=False, on_epoch=True, logger=True)
        self.log('train/mu_std', mu.std(), on_step=False, on_epoch=True, logger=True)
        self.log('train/logvar_mean', logvar.mean(), on_step=False, on_epoch=True, logger=True)
        self.log('train/beta_kl', beta_kl, on_step=False, on_epoch=True, logger=True)

        # Logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True, logger=True)

        # Gradient monitoring (select layers)
        if batch_idx % 100 == 0:  # Only log gradients periodically to save computation
            for name, param in self.named_parameters():
                if param.grad is not None:
                    if 'encoder' in name or 'decoder' in name:
                        if 'weight' in name and not ('bn' in name):  # Focus on main weights, not all params
                            self.logger.experiment.add_histogram(f'gradients/{name}', 
                                                             param.grad, self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x = batch # Ground truth mel (B, 1, N_MELS, T)

        # Create noise input
        noise = torch.randn_like(x)

        # --- Conditional Generation ---
        condition_c = x # Use ground truth as condition
        recon_mel_c, mu_c, logvar_c = self(condition_c, noise)
        recon_loss_c, kl_loss_c = self._vae_loss(recon_mel_c, x, mu_c, logvar_c)
        total_loss_c = recon_loss_c + self.hparams.training['beta_kl'] * kl_loss_c

        # --- Unconditional Generation (using mask token) ---
        mask_cond = self.mask_token.expand(x.size(0), -1, -1, -1).to(x.device)
        recon_mel_u, mu_u, logvar_u = self(mask_cond, noise)
        recon_loss_u, kl_loss_u = self._vae_loss(recon_mel_u, x, mu_u, logvar_u)

        # Logging
        self.log('val/loss', total_loss_c, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/recon_loss', recon_loss_c, on_step=False, on_epoch=True, logger=True)
        self.log('val/kl_loss_cond', kl_loss_c, on_step=False, on_epoch=True, logger=True)
        self.log('val/kl_loss_uncond', kl_loss_u, on_step=False, on_epoch=True, logger=True)

        # Additional metrics for latent space monitoring
        self.log('val/mu_mean', mu_c.mean(), on_step=False, on_epoch=True, logger=True)
        self.log('val/logvar_mean', logvar_c.mean(), on_step=False, on_epoch=True, logger=True)
        self.log('val/mu_std', mu_c.std(), on_step=False, on_epoch=True, logger=True)

        # --- Visualization (same as before) ---
        if self.trainer.is_global_zero and self.global_step > 0 and \
           self.global_step % self.hparams.training['vis_log_every_n_steps'] == 0:

            num_val_batches = self.trainer.num_val_batches[0]
            num_vis_samples = self.hparams.training['num_vis_samples']
            log_probability = float(num_vis_samples) / num_val_batches if num_val_batches > 0 else 0.0

            if torch.rand(1).item() < log_probability:
                target = x[0]
                pred_c = recon_mel_c[0]
                pred_u = recon_mel_u[0]

                log_suffix = f"batch_{batch_idx}"

                target_4d = target.unsqueeze(0) if target.dim() == 3 else target.unsqueeze(0).unsqueeze(0)
                pred_c_4d = pred_c.unsqueeze(0) if pred_c.dim() == 3 else pred_c.unsqueeze(0).unsqueeze(0)
                pred_u_4d = pred_u.unsqueeze(0) if pred_u.dim() == 3 else pred_u.unsqueeze(0).unsqueeze(0)

                pred_c_resized = self._resize_to_match(pred_c_4d, target_4d)
                pred_u_resized = self._resize_to_match(pred_u_4d, target_4d)

                title_c = f"Epoch {self.current_epoch} Step {self.global_step} - Sample (Cond) Batch {batch_idx}"
                try:
                    buf_c = plot_mel_comparison_to_buf(target, pred_c_resized.squeeze(), title_c, self.config)
                    img_c = buf_to_image(buf_c)
                    self.logger.experiment.add_image(f"Validation/Conditional_{log_suffix}", np.array(img_c), self.global_step, dataformats='HWC')
                    buf_c.close()
                except Exception as e:
                    self.print(f"WARN: Failed to log conditional validation image for batch {batch_idx}: {e}")

                title_u = f"Epoch {self.current_epoch} Step {self.global_step} - Sample (Uncond) Batch {batch_idx}"
                try:
                    buf_u = plot_mel_comparison_to_buf(target, pred_u_resized.squeeze(), title_u, self.config)
                    img_u = buf_to_image(buf_u)
                    self.logger.experiment.add_image(f"Validation/Unconditional_{log_suffix}", np.array(img_u), self.global_step, dataformats='HWC')
                    buf_u.close()
                except Exception as e:
                    self.print(f"WARN: Failed to log unconditional validation image for batch {batch_idx}: {e}")

        return total_loss_c

    def configure_optimizers(self):
        # Use RAdam optimizer for better stability with VAEs
        try:
            from torch.optim.radam import RAdam
            optimizer = RAdam(self.parameters(), lr=self.hparams.training['learning_rate'])
            print("Using RAdam optimizer for better stability")
        except ImportError:
            # Fall back to AdamW if RAdam isn't available
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.training['learning_rate'])
        
        # Learning rate scheduler - cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Multiply period by 2 after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }