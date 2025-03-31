import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiScaleMelLoss(nn.Module):
    """
    Multi-scale mel spectrogram loss with L1, MSE, and spectral components.
    Improved loss function specifically for mel spectrograms.
    """
    def __init__(self, l1_weight=0.5, mse_weight=0.3, sc_weight=0.1, log_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.sc_weight = sc_weight  # spectral convergence weight
        self.log_weight = log_weight  # log magnitude loss weight
        
    def spectral_convergence_loss(self, x_mag, y_mag, eps=1e-8):
        """Spectral convergence loss"""
        return torch.norm(x_mag - y_mag, p='fro') / (torch.norm(y_mag, p='fro') + eps)
    
    def log_magnitude_loss(self, x_mag, y_mag, eps=1e-8):
        """Log magnitude loss"""
        return F.l1_loss(torch.log(x_mag + eps), torch.log(y_mag + eps))
    
    def mel_scale_loss(self, pred, target):
        """
        Compute a frequency-weighted MSE with emphasis on lower mel bins
        which typically contain more perceptually important information.
        """
        # Create frequency weights - higher weight for lower frequency
        # Shape: (1, 1, freq_bins, 1)
        _, _, freq_bins, _ = target.shape
        # Linear weights from 1.5 (low freq) to 0.5 (high freq)
        weights = torch.linspace(1.5, 0.5, freq_bins).reshape(1, 1, -1, 1).to(target.device)
        
        # Compute weighted MSE
        return torch.mean(weights * (pred - target) ** 2)
    
    def forward(self, pred, target):
        """
        Compute multi-component loss
        
        Args:
            pred: Predicted mel spectrogram (B, C, F, T)
            target: Target mel spectrogram (B, C, F, T)
            
        Returns:
            loss: Combined loss value
        """
        # Basic reconstruction losses
        l1_loss = F.l1_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        
        # Mel-specific weighted loss
        mel_loss = self.mel_scale_loss(pred, target)
        
        # Spectral losses
        try:
            # Flatten mels to compute spectral losses (treating them like 1D signals)
            # This is an approximation for demonstration
            b, c, f, t = pred.shape
            pred_flat = pred.reshape(b * c, f * t)
            target_flat = target.reshape(b * c, f * t)
            
            sc_loss = self.spectral_convergence_loss(pred_flat, target_flat)
            log_loss = self.log_magnitude_loss(pred_flat, target_flat)
        except Exception as e:
            # Fall back to zeros if spectral computation fails
            sc_loss = torch.tensor(0.0, device=pred.device)
            log_loss = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = (
            self.l1_weight * l1_loss +
            self.mse_weight * mse_loss +
            self.sc_weight * sc_loss +
            self.log_weight * log_loss +
            0.1 * mel_loss  # Add some mel-specific weighting
        )
        
        # Return total loss and components for logging
        return {
            'loss': total_loss,
            'l1_loss': l1_loss.detach(),
            'mse_loss': mse_loss.detach(),
            'mel_loss': mel_loss.detach(),
            'sc_loss': sc_loss.detach() if not isinstance(sc_loss, float) else sc_loss,
            'log_loss': log_loss.detach() if not isinstance(log_loss, float) else log_loss,
        }

class VAELoss(nn.Module):
    """
    Custom VAE loss combining reconstruction loss with KL divergence.
    Includes advanced KL annealing and weighting schemes.
    """
    def __init__(self, beta=1.0, beta_min=0.0, beta_max=1.0, 
                 kl_weight_scheme='cyclical', free_bits=0.0):
        super().__init__()
        self.beta = beta  # Base KL weight
        self.beta_min = beta_min  # Minimum KL weight for annealing
        self.beta_max = beta_max  # Maximum KL weight for annealing
        self.kl_weight_scheme = kl_weight_scheme  # 'linear', 'cyclical', or 'constant'
        self.free_bits = free_bits  # Free bits for KL
        
        # Initialize mel loss
        self.mel_loss = MultiScaleMelLoss()
        
    def kl_divergence(self, mu, logvar):
        """
        Compute KL divergence between N(mu, var) and N(0, 1)
        with optional free bits
        """
        # KL = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))
        
        # Apply free bits if specified
        if self.free_bits > 0:
            kl_per_dim = torch.max(kl_per_dim, torch.tensor(self.free_bits, device=mu.device))
            
        return kl_per_dim.sum(dim=1).mean()
        
    def get_beta(self, epoch, max_epochs):
        """Get beta value according to annealing scheme"""
        if self.kl_weight_scheme == 'constant':
            return self.beta
        
        elif self.kl_weight_scheme == 'linear':
            # Linear ramp from beta_min to beta_max
            return self.beta_min + (self.beta_max - self.beta_min) * min(1.0, epoch / (max_epochs * 0.75))
        
        elif self.kl_weight_scheme == 'cyclical':
            # Cyclical annealing (from "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing")
            cycle_length = max_epochs // 4  # 4 cycles during training
            cycle_position = (epoch % cycle_length) / cycle_length
            
            if cycle_position < 0.5:
                # Linear increase within first half of cycle
                return self.beta_min + (self.beta_max - self.beta_min) * (2.0 * cycle_position)
            else:
                # Stay at beta_max for second half of cycle
                return self.beta_max
        
        else:
            return self.beta
    
    def forward(self, recon_x, x, mu, logvar, epoch=0, max_epochs=100):
        """
        Compute VAE loss with reconstruction and KL components
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            epoch: Current epoch (for annealing)
            max_epochs: Total training epochs
            
        Returns:
            Dictionary with total loss and components
        """
        # Get reconstruction loss components
        recon_losses = self.mel_loss(recon_x, x)
        recon_loss = recon_losses['loss']
        
        # Compute KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Get beta value according to annealing scheme
        beta = self.get_beta(epoch, max_epochs)
        
        # Combine losses
        total_loss = recon_loss + beta * kl_loss
        
        # Return losses for logging
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'beta': beta,
            **{f'recon_{k}': v for k, v in recon_losses.items() if k != 'loss'}
        }