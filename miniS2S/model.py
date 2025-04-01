import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import librosa
import yaml

from modules import (
    PhonemeEncoder,
    MusicalFeatureEncoder,
    FeatureFusion,
    DecoderRNN,
    Postnet,
    GuidedAttentionLoss
)

class SingingVoiceSynthesisModel(pl.LightningModule):
    """
    PyTorch Lightning module for Singing Voice Synthesis.
    Integrates encoder, attention, decoder, and postnet components.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the model with the given configuration.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model components
        self._init_encoder()
        self._init_decoder()
        self._init_postnet()
        
        # Loss weights
        self.mel_loss_weight = config['training']['loss']['mel_loss_weight']
        self.stop_token_weight = config['training']['loss']['stop_token_weight']
        self.guided_attention_weight = config['training']['loss'].get('guided_attention_weight', 0.0)
        
        # Guided attention loss
        if self.guided_attention_weight > 0:
            self.guided_attention_loss = GuidedAttentionLoss()
    
    def _init_encoder(self):
        """Initialize encoder components."""
        model_config = self.config['model']
        
        # Phoneme encoder
        self.phoneme_encoder = PhonemeEncoder(
            num_phonemes=model_config['num_phonemes'],
            embedding_dim=model_config['embedding_dim'],
            conv_channels=model_config['encoder']['conv_channels'],
            conv_kernel_size=model_config['encoder']['conv_kernel_size'],
            conv_dropout=model_config['encoder']['conv_dropout'],
            lstm_hidden_size=model_config['encoder']['lstm_hidden_size'],
            lstm_layers=model_config['encoder']['lstm_layers'],
            lstm_dropout=model_config['encoder']['lstm_dropout']
        )
        
        # Musical feature encoder
        self.musical_feature_encoder = MusicalFeatureEncoder(
            input_dim=model_config['musical_features']['input_dim'],
            hidden_dims=model_config['musical_features']['hidden_dims'],
            dropout=model_config['musical_features']['dropout']
        )
        
        # Feature fusion
        phoneme_dim = 2 * model_config['encoder']['lstm_hidden_size']  # Bidirectional
        musical_dim = model_config['musical_features']['hidden_dims'][-1]
        
        self.feature_fusion = FeatureFusion(
            phoneme_dim=phoneme_dim,
            musical_dim=musical_dim,
            output_dim=model_config['embedding_dim']  # Use same dim as embedding for output
        )
    
    def _init_decoder(self):
        """Initialize decoder components."""
        model_config = self.config['model']
        
        # Decoder
        self.decoder = DecoderRNN(
            prenet_input_dim=self.config['audio']['n_mels'],
            prenet_dims=model_config['decoder']['prenet_dims'],
            prenet_dropout=model_config['decoder']['prenet_dropout'],
            encoder_dim=model_config['embedding_dim'],  # Output dim from feature fusion
            attention_dim=model_config['attention']['attention_dim'],
            attention_location_features=model_config['attention']['location_features'],
            attention_location_kernel_size=model_config['attention']['location_kernel_size'],
            attention_dropout=model_config['attention']['attention_dropout'],
            decoder_dim=model_config['decoder']['lstm_dim'],
            num_layers=model_config['decoder']['lstm_layers'],
            dropout=model_config['decoder']['lstm_dropout'],
            output_dim=self.config['audio']['n_mels']
        )
    
    def _init_postnet(self):
        """Initialize postnet for mel refinement."""
        model_config = self.config['model']
        
        self.postnet = Postnet(
            mel_dim=self.config['audio']['n_mels'],
            channels=model_config['decoder']['postnet_channels'],
            kernel_size=model_config['decoder']['postnet_kernel'],
            dropout=model_config['decoder']['postnet_dropout']
        )
    
    def forward(self, 
            phoneme_ids: torch.Tensor,
            musical_features: torch.Tensor,
            mel_lengths: torch.Tensor,
            max_decoder_steps: int = 1000,
            teacher_forcing_ratio: float = 1.0,
            target_mel: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            phoneme_ids: Phoneme IDs [batch_size, seq_length]
            musical_features: Musical features [batch_size, seq_length, n_features]
            mel_lengths: Lengths of target mel spectrograms [batch_size]
            max_decoder_steps: Maximum number of decoder steps
            teacher_forcing_ratio: Probability of using teacher forcing
            target_mel: Target mel spectrogram for teacher forcing [batch_size, seq_length, n_mels]
            
        Returns:
            Dictionary with model outputs
        """
        # Prepare mask for padding
        mask = torch.arange(phoneme_ids.size(1)).expand(
            phoneme_ids.size(0), phoneme_ids.size(1)
        ).to(phoneme_ids.device) < mel_lengths.unsqueeze(1)
        
        # Encode phonemes
        phoneme_features = self.phoneme_encoder(phoneme_ids, mel_lengths)
        
        # Encode musical features
        musical_features_encoded = self.musical_feature_encoder(musical_features)
        
        # Fuse features
        encoder_outputs = self.feature_fusion(phoneme_features, musical_features_encoded)
        
        # Set the max_decoder_steps based on the target length when using teacher forcing
        # This ensures we don't generate more frames than necessary
        if target_mel is not None and teacher_forcing_ratio > 0:
            target_max_len = target_mel.size(1)
            # Use the target length as the maximum steps when teacher forcing is used
            effective_max_steps = target_max_len
        else:
            effective_max_steps = max_decoder_steps
        
        # Decode mel spectrogram
        mel_outputs, stop_tokens, alignments = self.decoder(
            memory=encoder_outputs,
            max_decoder_steps=effective_max_steps,
            teacher_forcing_ratio=teacher_forcing_ratio,
            target=target_mel,
            memory_lengths=mel_lengths
        )
        
        # Apply postnet to refine the prediction
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)
        
        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'stop_tokens': stop_tokens,
            'alignments': alignments
        }
    
    def _calculate_loss(self, outputs, batch):
        """
        Calculate the loss for training.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Get target mel and stop tokens
        target_mel = batch['target_mel']  # [batch_size, seq_length, n_mels]
        mel_lengths = batch['mel_lengths']  # [batch_size]
        
        # Create target stop tokens (all zeros except last frame is 1)
        target_stop = torch.zeros_like(outputs['stop_tokens'])
        for i, length in enumerate(mel_lengths):
            if length < target_stop.size(1):
                target_stop[i, length-1:] = 1.0
        
        # Adjust output or target shapes to match if needed
        output_seq_len = outputs['mel_outputs'].size(1)
        target_seq_len = target_mel.size(1)
        
        if output_seq_len > target_seq_len:
            # Truncate outputs to match target length
            mel_outputs = outputs['mel_outputs'][:, :target_seq_len, :]
            mel_outputs_postnet = outputs['mel_outputs_postnet'][:, :target_seq_len, :]
            stop_tokens = outputs['stop_tokens'][:, :target_seq_len, :]
        elif output_seq_len < target_seq_len:
            # This shouldn't typically happen with teacher forcing, but handle it anyway
            # Truncate targets to match output length
            print(f"Warning: Output length ({output_seq_len}) is less than target length ({target_seq_len}). This is unexpected.")
            target_mel = target_mel[:, :output_seq_len, :]
            # Adjust target_stop as well if it was already created based on target_mel
            if target_stop.size(1) > output_seq_len:
                target_stop = target_stop[:, :output_seq_len, :]
            mel_outputs = outputs['mel_outputs']
            mel_outputs_postnet = outputs['mel_outputs_postnet']
            stop_tokens = outputs['stop_tokens']
        else:
            # Lengths already match
            mel_outputs = outputs['mel_outputs']
            mel_outputs_postnet = outputs['mel_outputs_postnet']
            stop_tokens = outputs['stop_tokens']
        
        # Calculate mel loss with correctly sized tensors
        mel_loss = F.l1_loss(mel_outputs, target_mel)
        mel_postnet_loss = F.l1_loss(mel_outputs_postnet, target_mel)
        
        # Calculate stop token loss
        stop_loss = F.binary_cross_entropy_with_logits(
            stop_tokens.squeeze(-1),
            target_stop.squeeze(-1)
        )
        
        # Calculate guided attention loss if enabled
        guided_attn_loss = 0.0
        if self.guided_attention_weight > 0:
            # Adjust alignments as well if needed
            if output_seq_len != target_seq_len and 'alignments' in outputs:
                alignments = outputs['alignments'][:, :min(output_seq_len, target_seq_len), :]
            else:
                alignments = outputs['alignments']
                
            guided_attn_loss = self.guided_attention_loss(
                attention_weights=alignments,
                mel_lengths=mel_lengths,
                memory_lengths=mel_lengths  # Assuming encoder and decoder lengths are the same
            )
        
        # Combine losses
        total_loss = (
            self.mel_loss_weight * (mel_loss + mel_postnet_loss) + 
            self.stop_token_weight * stop_loss +
            self.guided_attention_weight * guided_attn_loss
        )
        
        return {
            'loss': total_loss,
            'mel_loss': mel_loss,
            'mel_postnet_loss': mel_postnet_loss,
            'stop_loss': stop_loss,
            'guided_attn_loss': guided_attn_loss
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """

        # Calculate current epoch fraction (0 to 1 scale over training)
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        progress = min(1.0, current_epoch / (max_epochs * 0.5))  # Anneal over first half
        
        # Reduce guided attention weight over time
        effective_guided_attention_weight = self.guided_attention_weight * (1.0 - progress)

        # Forward pass with teacher forcing
        outputs = self.forward(
            phoneme_ids=batch['phoneme_ids'],
            musical_features=batch['musical_features'],
            mel_lengths=batch['mel_lengths'],
            teacher_forcing_ratio=1.0,  # Always use teacher forcing during training
            target_mel=batch['mel']
        )
        
        # Calculate loss
        loss_dict = self._calculate_loss(outputs, batch)
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f'train_{name}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Forward pass with teacher forcing
        outputs = self.forward(
            phoneme_ids=batch['phoneme_ids'],
            musical_features=batch['musical_features'],
            mel_lengths=batch['mel_lengths'],
            teacher_forcing_ratio=1.0,  # Use teacher forcing for validation too
            target_mel=batch['mel']
        )
        
        # Calculate loss
        loss_dict = self._calculate_loss(outputs, batch)
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f'val_{name}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log additional metrics
        if batch_idx == 0:  # Only log visualizations for the first batch
            self._log_validation_visualizations(batch, outputs)
        
        return loss_dict
    
    def _log_validation_visualizations(self, batch, outputs):
        """
        Log visualizations for validation.
        
        Args:
            batch: Batch data
            outputs: Model outputs
        """
        # Only log if logger is available
        if not self.logger:
            return
        
        # Get sample from batch
        sample_idx = 0  # Use the first sample in the batch
        
        # Get predicted and target mel spectrograms
        pred_mel = outputs['mel_outputs_postnet'][sample_idx].detach().cpu().numpy()
        target_mel = batch['mel'][sample_idx].detach().cpu().numpy()
        attention = outputs['alignments'][sample_idx].detach().cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        # Plot predicted mel
        librosa.display.specshow(
            pred_mel.T,  # Transpose for correct visualization
            x_axis='time',
            y_axis='mel',
            sr=self.config['audio']['sample_rate'],
            hop_length=self.config['audio']['hop_length'],
            fmin=self.config['audio']['fmin'],
            fmax=self.config['audio']['fmax'],
            ax=axes[0]
        )
        axes[0].set_title('Predicted Mel Spectrogram')
        
        # Plot target mel
        librosa.display.specshow(
            target_mel.T,  # Transpose for correct visualization
            x_axis='time',
            y_axis='mel',
            sr=self.config['audio']['sample_rate'],
            hop_length=self.config['audio']['hop_length'],
            fmin=self.config['audio']['fmin'],
            fmax=self.config['audio']['fmax'],
            ax=axes[1]
        )
        axes[1].set_title('Target Mel Spectrogram')
        
        # Plot attention
        axes[2].imshow(attention, aspect='auto', origin='lower')
        axes[2].set_title('Attention')
        axes[2].set_xlabel('Encoder Steps')
        axes[2].set_ylabel('Decoder Steps')
        
        plt.tight_layout()
        
        # Log to TensorBoard
        self.logger.experiment.add_figure(
            'mel_spectrograms', fig, global_step=self.global_step
        )
        
        plt.close(fig)
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Forward pass with reduced teacher forcing
        outputs = self.forward(
            phoneme_ids=batch['phoneme_ids'],
            musical_features=batch['musical_features'],
            mel_lengths=batch['mel_lengths'],
            teacher_forcing_ratio=0.5,  # Use reduced teacher forcing for testing
            target_mel=batch['mel']
        )
        
        # Calculate loss
        loss_dict = self._calculate_loss(outputs, batch)
        
        # Log losses
        for name, value in loss_dict.items():
            self.log(f'test_{name}', value, on_step=False, on_epoch=True)
        
        return loss_dict
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Prediction dictionary
        """
        # Forward pass without teacher forcing
        outputs = self.forward(
            phoneme_ids=batch['phoneme_ids'],
            musical_features=batch['musical_features'],
            mel_lengths=batch['mel_lengths'],
            teacher_forcing_ratio=0.0,  # No teacher forcing during inference
            target_mel=None
        )
        
        return {
            'ids': batch['ids'],
            'mel_outputs': outputs['mel_outputs_postnet'],
            'alignments': outputs['alignments']
        }
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler
        """
        # Get training parameters
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
                        
        # Create scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config['training']['lr_scheduler']['plateau_factor'],
                patience=self.config['training']['lr_scheduler']['plateau_patience'],
                verbose=True,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


def test_model():
    """Test the model implementation."""
    # Load configuration
    with open("config/default.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create small batch for testing
    batch_size = 2
    seq_length = 50
    n_mels = config['audio']['n_mels']
    n_phonemes = config['model']['num_phonemes']
    n_musical_features = 4
    
    batch = {
        'ids': ['sample1', 'sample2'],
        'phoneme_ids': torch.randint(0, n_phonemes, (batch_size, seq_length)),
        'musical_features': torch.randn(batch_size, seq_length, n_musical_features),
        'mel': torch.randn(batch_size, seq_length, n_mels),
        'target_mel': torch.randn(batch_size, seq_length, n_mels),
        'mel_lengths': torch.tensor([seq_length, seq_length - 10]),
        'stop_tokens': torch.zeros(batch_size, seq_length)
    }
    
    # Set stop tokens for last frame
    for i, length in enumerate(batch['mel_lengths']):
        batch['stop_tokens'][i, length-1] = 1.0
    
    # Create model
    model = SingingVoiceSynthesisModel(config)
    
    # Print model summary
    print("Model initialized successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    print("Testing forward pass...")
    outputs = model(
        phoneme_ids=batch['phoneme_ids'],
        musical_features=batch['musical_features'],
        mel_lengths=batch['mel_lengths'],
        teacher_forcing_ratio=1.0,
        target_mel=batch['mel']
    )
    
    print("Output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test loss calculation
    print("Testing loss calculation...")
    loss_dict = model._calculate_loss(outputs, batch)
    
    print("Loss values:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item()}")
    
    print("Model test completed successfully")


if __name__ == "__main__":
    test_model()
