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

        layers = []
        in_channels = 1
        current_h = self.n_mels # Keep track of spatial dimensions if needed

        for i in range(num_layers):
            out_channels = base_channels * (2**i)
            stride = 2 if i < num_layers -1 else 1 # Don't stride last layer? Or always stride? Let's stride.
            #stride = 1 if current_h <= 4 else 2 # Adaptive stride?
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
            )
            if use_bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            # Calculate output height after conv+stride (assuming padding keeps width roughly proportional)
            #current_h = (current_h + 2*(kernel_size//2) - kernel_size) // stride + 1


        self.conv_layers = nn.Sequential(*layers)

        # Calculate the flattened size dynamically
        # Need target_frames for width calculation
        target_frames = calculate_target_frames(config)
        dummy_input = torch.randn(1, 1, self.n_mels, target_frames)
        with torch.no_grad():
             conv_output_shape = self.conv_layers(dummy_input).shape
        self.flattened_size = conv_output_shape[1] * conv_output_shape[2] * conv_output_shape[3]

        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer."""
    def __init__(self, input_dim, condition_dim):
        super().__init__()
        self.input_dim = input_dim
        # Linear layer to generate scale (gamma) and shift (beta)
        self.cond_proj = nn.Linear(condition_dim, input_dim * 2)

    def forward(self, x, condition):
        # x shape: (B, C, H, W)
        # condition shape: (B, D)
        gamma_beta = self.cond_proj(condition) # (B, C*2)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1) # (B, C), (B, C)
        gamma = torch.tanh(gamma) * 2 + 1  # Range [−1, 3]

        # Reshape gamma and beta for broadcasting: (B, C, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta

class ConditionalDecoderBlock(nn.Module):
    """A single block in the conditional decoder using FiLM."""
    def __init__(self, in_channels, out_channels, latent_dim, kernel_size, use_bn):
        super().__init__()
        padding = kernel_size // 2
        self.use_bn = use_bn

        self.film = FiLM(in_channels, latent_dim)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        # Use ConvTranspose2d for potential upsampling or just Conv2d with padding='same'
        # Let's stick to Conv2d with padding='same' for simplicity as input/output size is fixed
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

        # Optional: Second conv layer within the block
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)

        # Shortcut connection if channels match
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x, z):
        # Modulate
        modulated_x = self.film(x, z)

        # Process
        residual = self.shortcut(modulated_x) # Apply shortcut *after* FiLM? Or before? Let's do after.

        out = modulated_x
        if self.use_bn: out = self.bn1(out)
        out = self.act1(out)
        out = self.conv1(out)

        if self.use_bn: out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        # Add residual
        out += residual

        return out

class Decoder(nn.Module):
    """Decodes noise and latent vector z into a Mel Spectrogram using FiLM."""
    def __init__(self, config):
        super().__init__()
        self.n_mels = config['audio']['n_mels']
        self.latent_dim = config['model']['latent_dim']
        num_layers = config['model']['num_layers_decoder']
        # Decoder often mirrors encoder channel structure
        max_channels = config['model']['decoder_base_channels'] # E.g. 64 or 128
        kernel_size = config['model']['kernel_size']
        use_bn = config['model']['use_batchnorm']
        target_frames = calculate_target_frames(config) # Needed? Maybe not explicitly.

        # Initial convolution to process noise input
        self.initial_conv = nn.Conv2d(1, max_channels, kernel_size, padding=kernel_size//2)

        layers = []
        in_channels = max_channels
        for i in range(num_layers):
            # Decrease channels towards the output
            # E.g., 64 -> 64 -> 32 -> 16
            out_channels = max(max_channels // (2**i), 16) # Ensure minimum channels

            layers.append(
                ConditionalDecoderBlock(in_channels, out_channels, self.latent_dim, kernel_size, use_bn)
            )
            in_channels = out_channels

        self.res_blocks = nn.ModuleList(layers)

        # Final convolution to map to 1 channel (mel spectrogram)
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1, padding=0) # 1x1 conv

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, noise, z):
        # noise shape: (B, 1, N_MELS, T)
        # z shape: (B, LATENT_DIM)

        x = self.initial_conv(noise) # (B, C_max, N_MELS, T)

        for block in self.res_blocks:
            x = block(x, z)

        output_mel = self.final_conv(x) # (B, 1, N_MELS, T)

        # Dimension check and optional resizing
        noise_shape = noise.shape
        output_mel_shape = output_mel.shape
        if output_mel_shape[2] != noise_shape[2] or output_mel_shape[3] != noise_shape[3]:
            print(f"WARN: Decoder output shape {output_mel_shape} does not match noise shape {noise_shape}. Resizing.")
            output_mel = torch.nn.functional.interpolate(
                output_mel,
                size=(noise_shape[2], noise_shape[3]),  # H, W
                mode='bilinear',
                align_corners=False
            )
        return output_mel

class ConditionalVAE(pl.LightningModule):
    """PyTorch Lightning Module for the Conditional VAE."""
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # Saves config to self.hparams and checkpoint
        self.config = config # Keep a direct reference if needed

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.target_frames = calculate_target_frames(config)
        self.n_mels = config['audio']['n_mels']
        self.padding_value = config['audio']['padding_value']

        # Mask token (fixed tensor representing silence or masked input)
        # Use register_buffer for state that should be saved but not trained
        self.register_buffer(
            "mask_token",
             torch.full((1, 1, self.n_mels, self.target_frames), self.padding_value)
        )
        self.mask_probability = config['model']['mask_probability']

        # Validation visualization counter
        self.val_vis_count = 0

    def forward(self, condition, noise):
        mu, logvar = self.encoder(condition)
        z = self.reparameterize(mu, logvar)
        reconstructed_mel = self.decoder(noise, z)
        return reconstructed_mel, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _resize_to_match(self, tensor_to_resize, target_tensor):
        """
        Resizes tensor_to_resize to match the spatial dimensions (H, W) of target_tensor using bilinear interpolation.
        """
        resized_tensor = tensor_to_resize
        target_shape = target_tensor.shape
        tensor_shape = tensor_to_resize.shape

        if len(tensor_shape) != 4 or len(target_shape) != 4:
            print(f"WARN: _resize_to_match requires 4D tensors. Got {len(tensor_shape)} and {len(target_shape)} dimensions.")
            return resized_tensor # Return original if not 4D

        if tensor_shape[2] != target_shape[2] or tensor_shape[3] != target_shape[3]:
            print(f"WARN: Resizing tensor from {tensor_shape} to {target_shape}.")
            resized_tensor = torch.nn.functional.interpolate(
                tensor_to_resize,
                size=(target_shape[2], target_shape[3]),  # H, W
                mode='bilinear',
                align_corners=False
            )
        return resized_tensor

    def _vae_loss(self, recon_x, x, mu, logvar):
        # Reconstruction Loss (MSE)
        # Ensure shapes match: recon_x=(B, 1, M, T), x=(B, 1, M, T)
        recon_loss = F.mse_loss(recon_x, x, reduction='mean') # Average over all elements

        # KL Divergence
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Formula uses logvar = log(sigma^2)
        logvar = torch.clamp(logvar, -5, 5) # Clamp logvar to prevent extreme values
        mu = torch.clamp(mu, -5, 5)  # Also clamp mu values
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1) # Sum over latent dim
        kl_loss = torch.mean(kl_loss) # Average over batch

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
                # Use the registered buffer, ensuring device compatibility
                condition[i] = self.mask_token.to(x.device)
            else:
                condition[i] = x[i]

        # Forward pass
        recon_mel, mu, logvar = self(condition, noise)

        # Calculate loss
        recon_loss, kl_loss = self._vae_loss(recon_mel, x, mu, logvar)
        beta_kl = min(1.0, self.current_epoch / self.hparams.training['annealing_epochs']) * self.hparams.training['beta_kl']
        total_loss = recon_loss + beta_kl * kl_loss

        # Logging
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train/kl_loss', kl_loss, on_step=True, on_epoch=True, logger=True)

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
        # Ensure mask token is expanded to batch size and correct device
        mask_cond = self.mask_token.expand(x.size(0), -1, -1, -1).to(x.device)
        recon_mel_u, mu_u, logvar_u = self(mask_cond, noise)
        # Note: For unconditional, reconstruction loss doesn't make direct sense against 'x'
        # But KL loss against prior is still informative. We log the VAE loss components anyway.
        recon_loss_u, kl_loss_u = self._vae_loss(recon_mel_u, x, mu_u, logvar_u)
        #total_loss_u = recon_loss_u + self.hparams.training['beta_kl'] * kl_loss_u

        # Logging (Focus on conditional loss for main metric, log unconditional KL)
        self.log('val/loss', total_loss_c, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/recon_loss', recon_loss_c, on_step=False, on_epoch=True, logger=True)
        self.log('val/kl_loss_cond', kl_loss_c, on_step=False, on_epoch=True, logger=True)
        self.log('val/kl_loss_uncond', kl_loss_u, on_step=False, on_epoch=True, logger=True) # How close is masked output to prior?

        # --- Visualization (Probabilistic within step) ---
        # Check if running on main process and if visualization is due based on step count
        # Also check global_step > 0 to avoid logging at the very beginning
        if self.trainer.is_global_zero and self.global_step > 0 and \
           self.global_step % self.hparams.training['vis_log_every_n_steps'] == 0:

            # Calculate probability to log this batch to approximate num_vis_samples total for the epoch
            # This requires knowing the total number of validation batches
            num_val_batches = self.trainer.num_val_batches[0] # Assumes single val dataloader
            num_vis_samples = self.hparams.training['num_vis_samples']
            log_probability = float(num_vis_samples) / num_val_batches if num_val_batches > 0 else 0.0

            # Decide probabilistically if we should log this specific batch's first sample
            if torch.rand(1).item() < log_probability:
                # Get first sample from the current batch
                target = x[0] # Assuming batch is just x (ground truth mel)
                # Ensure recon_mel_c and recon_mel_u (conditional/unconditional reconstructions)
                # have been calculated earlier in this validation_step
                pred_c = recon_mel_c[0]
                pred_u = recon_mel_u[0]

                # Use a unique identifier for the log, e.g., based on batch_idx
                log_suffix = f"batch_{batch_idx}"

                # Ensure tensors are 4D (B, C, H, W) before resizing
                target_4d = target.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
                pred_c_4d = pred_c.unsqueeze(0).unsqueeze(0)
                pred_u_4d = pred_u.unsqueeze(0).unsqueeze(0)

                # Resize predicted mels to match target dimensions
                pred_c_resized = self._resize_to_match(pred_c_4d, target_4d)
                pred_u_resized = self._resize_to_match(pred_u_4d, target_4d)

                # Plot Conditional Comparison
                title_c = f"Epoch {self.current_epoch} Step {self.global_step} - Sample (Cond) Batch {batch_idx}"
                try:
                    buf_c = plot_mel_comparison_to_buf(target, pred_c_resized.squeeze(), title_c, self.config)
                    img_c = buf_to_image(buf_c)
                    self.logger.experiment.add_image(f"Validation/Conditional_{log_suffix}", np.array(img_c), self.global_step, dataformats='HWC')
                    buf_c.close()
                except Exception as e:
                    self.print(f"WARN: Failed to log conditional validation image for batch {batch_idx}: {e}")


                # Plot Unconditional Generation (vs Target for reference)
                title_u = f"Epoch {self.current_epoch} Step {self.global_step} - Sample (Uncond) Batch {batch_idx}"
                try:
                    buf_u = plot_mel_comparison_to_buf(target, pred_u_resized.squeeze(), title_u, self.config) # Compare uncond with original target
                    img_u = buf_to_image(buf_u)
                    self.logger.experiment.add_image(f"Validation/Unconditional_{log_suffix}", np.array(img_u), self.global_step, dataformats='HWC')
                    buf_u.close()
                except Exception as e:
                    self.print(f"WARN: Failed to log unconditional validation image for batch {batch_idx}: {e}")

        # --- End Visualization ---


        return total_loss_c # Return the primary validation loss (conditional)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.training['learning_rate'])
        # Optional: Add learning rate scheduler here
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
        return optimizer