import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import librosa
from torch.utils.tensorboard import SummaryWriter

from dataset import SingingVoiceDataset, SingingVoiceDataModule
from modules import (
    PhonemeEncoder,
    MusicalFeatureEncoder,
    FeatureFusion,
    LocationSensitiveAttention,
    Prenet,
    DecoderRNN,
    Postnet,
    GuidedAttentionLoss
)
from model import SingingVoiceSynthesisModel

class ModuleValidator:
    """
    Validator for individual modules of the singing voice synthesis model.
    """
    
    def __init__(self, config_path: str, log_dir: str = "logs/module_validation"):
        """
        Initialize the validator.
        
        Args:
            config_path: Path to the configuration file
            log_dir: Directory for TensorBoard logs
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Load a small subset of data for testing
        self.data_module = SingingVoiceDataModule(self.config)
        self.data_module.prepare_data()
        self.data_module.setup(stage="test")
        
        # Get a small batch for testing
        self.test_loader = self.data_module.test_dataloader()
        self.test_batch = next(iter(self.test_loader))
        
        # Extract batch data
        self.phoneme_ids = self.test_batch["phoneme_ids"]
        self.musical_features = self.test_batch["musical_features"]
        self.mel = self.test_batch["mel"]
        self.mel_lengths = self.test_batch["mel_lengths"]
        self.attention_mask = self.test_batch["attention_mask"]
        
        # Print batch information
        print(f"Test batch shapes:")
        print(f"  Phoneme IDs: {self.phoneme_ids.shape}")
        print(f"  Musical features: {self.musical_features.shape}")
        print(f"  Mel: {self.mel.shape}")
        print(f"  Mel lengths: {self.mel_lengths}")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def to_device(self, tensors: Union[torch.Tensor, Dict[str, torch.Tensor]]):
        """Move tensors to the device."""
        if isinstance(tensors, torch.Tensor):
            return tensors.to(self.device)
        elif isinstance(tensors, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tensors.items()}
        else:
            return tensors
    
    def test_phoneme_encoder(self):
        """Test the phoneme encoder module."""
        print("\n=== Testing PhonemeEncoder ===")
        
        # Create module
        phoneme_encoder = PhonemeEncoder(
            num_phonemes=self.config["model"]["num_phonemes"],
            embedding_dim=self.config["model"]["embedding_dim"],
            conv_channels=self.config["model"]["encoder"]["conv_channels"],
            conv_kernel_size=self.config["model"]["encoder"]["conv_kernel_size"],
            conv_dropout=self.config["model"]["encoder"]["conv_dropout"],
            lstm_hidden_size=self.config["model"]["encoder"]["lstm_hidden_size"],
            lstm_layers=self.config["model"]["encoder"]["lstm_layers"],
            lstm_dropout=self.config["model"]["encoder"]["lstm_dropout"]
        ).to(self.device)
        
        # Forward pass
        phoneme_ids = self.to_device(self.phoneme_ids)
        mel_lengths = self.to_device(self.mel_lengths)
        
        with torch.no_grad():
            phoneme_features = phoneme_encoder(phoneme_ids, mel_lengths)
        
        # Print output shape
        print(f"Input shape: {phoneme_ids.shape}")
        print(f"Output shape: {phoneme_features.shape}")
        
        # Visualize a random sample of embedding and output
        sample_idx = 0
        
        # Get embeddings for the sample
        with torch.no_grad():
            embeddings = phoneme_encoder.embedding(phoneme_ids[sample_idx])
        
        # Plot before and after
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot embeddings
        im1 = axes[0].imshow(embeddings.cpu().numpy(), aspect="auto")
        axes[0].set_title("Phoneme Embeddings")
        axes[0].set_ylabel("Sequence")
        axes[0].set_xlabel("Embedding Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot encoder outputs
        im2 = axes[1].imshow(phoneme_features[sample_idx].cpu().numpy(), aspect="auto")
        axes[1].set_title("PhonemeEncoder Output")
        axes[1].set_ylabel("Sequence")
        axes[1].set_xlabel("Feature Dim")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "phoneme_encoder.png"))
        self.writer.add_figure("phoneme_encoder", fig)
        plt.close(fig)
        
        # Log parameters and activation histogram
        for name, param in phoneme_encoder.named_parameters():
            self.writer.add_histogram(f"phoneme_encoder/{name}", param.data)
        
        self.writer.add_histogram("phoneme_encoder/output", phoneme_features)
        
        print("PhonemeEncoder test completed successfully")
        return phoneme_features
    
    def test_musical_feature_encoder(self):
        """Test the musical feature encoder module."""
        print("\n=== Testing MusicalFeatureEncoder ===")
        
        # Create module
        musical_feature_encoder = MusicalFeatureEncoder(
            input_dim=self.config["model"]["musical_features"]["input_dim"],
            hidden_dims=self.config["model"]["musical_features"]["hidden_dims"],
            dropout=self.config["model"]["musical_features"]["dropout"]
        ).to(self.device)
        
        # Forward pass
        musical_features = self.to_device(self.musical_features)
        
        with torch.no_grad():
            musical_features_encoded = musical_feature_encoder(musical_features)
        
        # Print output shape
        print(f"Input shape: {musical_features.shape}")
        print(f"Output shape: {musical_features_encoded.shape}")
        
        # Visualize a random sample of input and output
        sample_idx = 0
        
        # Plot before and after
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot input features
        im1 = axes[0].imshow(musical_features[sample_idx].cpu().numpy(), aspect="auto")
        axes[0].set_title("Musical Features Input")
        axes[0].set_ylabel("Sequence")
        axes[0].set_xlabel("Feature Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot encoder outputs
        im2 = axes[1].imshow(musical_features_encoded[sample_idx].cpu().numpy(), aspect="auto")
        axes[1].set_title("MusicalFeatureEncoder Output")
        axes[1].set_ylabel("Sequence")
        axes[1].set_xlabel("Feature Dim")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "musical_feature_encoder.png"))
        self.writer.add_figure("musical_feature_encoder", fig)
        plt.close(fig)
        
        # Log parameters and activation histogram
        for name, param in musical_feature_encoder.named_parameters():
            self.writer.add_histogram(f"musical_feature_encoder/{name}", param.data)
        
        self.writer.add_histogram("musical_feature_encoder/output", musical_features_encoded)
        
        print("MusicalFeatureEncoder test completed successfully")
        return musical_features_encoded
    
    def test_feature_fusion(self, phoneme_features, musical_features_encoded):
        """Test the feature fusion module."""
        print("\n=== Testing FeatureFusion ===")
        
        # Create module
        phoneme_dim = 2 * self.config["model"]["encoder"]["lstm_hidden_size"]  # Bidirectional
        musical_dim = self.config["model"]["musical_features"]["hidden_dims"][-1]
        
        feature_fusion = FeatureFusion(
            phoneme_dim=phoneme_dim,
            musical_dim=musical_dim,
            output_dim=self.config["model"]["embedding_dim"]
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            fused_features = feature_fusion(phoneme_features, musical_features_encoded)
        
        # Print output shape
        print(f"Phoneme features shape: {phoneme_features.shape}")
        print(f"Musical features shape: {musical_features_encoded.shape}")
        print(f"Fused features shape: {fused_features.shape}")
        
        # Visualize a random sample
        sample_idx = 0
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot phoneme features
        im1 = axes[0].imshow(phoneme_features[sample_idx].cpu().numpy(), aspect="auto")
        axes[0].set_title("Phoneme Features")
        axes[0].set_ylabel("Sequence")
        axes[0].set_xlabel("Feature Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot musical features
        im2 = axes[1].imshow(musical_features_encoded[sample_idx].cpu().numpy(), aspect="auto")
        axes[1].set_title("Musical Features")
        axes[1].set_ylabel("Sequence")
        axes[1].set_xlabel("Feature Dim")
        plt.colorbar(im2, ax=axes[1])
        
        # Plot fused features
        im3 = axes[2].imshow(fused_features[sample_idx].cpu().numpy(), aspect="auto")
        axes[2].set_title("Fused Features")
        axes[2].set_ylabel("Sequence")
        axes[2].set_xlabel("Feature Dim")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "feature_fusion.png"))
        self.writer.add_figure("feature_fusion", fig)
        plt.close(fig)
        
        # Log parameters and activation histogram
        for name, param in feature_fusion.named_parameters():
            self.writer.add_histogram(f"feature_fusion/{name}", param.data)
        
        self.writer.add_histogram("feature_fusion/output", fused_features)
        
        print("FeatureFusion test completed successfully")
        return fused_features
    
    def test_attention(self, encoder_outputs):
        """Test the attention mechanism."""
        print("\n=== Testing LocationSensitiveAttention ===")
        
        # Create module
        attention = LocationSensitiveAttention(
            query_dim=self.config["model"]["decoder"]["lstm_dim"],
            encoder_dim=self.config["model"]["embedding_dim"],
            attention_dim=self.config["model"]["attention"]["attention_dim"],
            location_features=self.config["model"]["attention"]["location_features"],
            location_kernel_size=self.config["model"]["attention"]["location_kernel_size"],
            dropout=self.config["model"]["attention"]["attention_dropout"]
        ).to(self.device)
        
        # Create a query (simulating decoder state)
        batch_size = encoder_outputs.size(0)
        query_dim = self.config["model"]["decoder"]["lstm_dim"]
        query = torch.randn(batch_size, query_dim).to(self.device)
        
        # Create previous attention weights (uniform distribution for first step)
        seq_length = encoder_outputs.size(1)
        prev_attn = torch.ones(batch_size, seq_length).to(self.device) / seq_length
        
        # Forward pass
        with torch.no_grad():
            context, attn_weights = attention(
                query=query,
                memory=encoder_outputs,
                attention_weights_prev=prev_attn,
                mask=self.to_device(self.attention_mask)
            )
        
        # Print output shapes
        print(f"Query shape: {query.shape}")
        print(f"Memory shape: {encoder_outputs.shape}")
        print(f"Context shape: {context.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        
        # Visualize attention for a sample
        sample_idx = 0
        
        # Plot attention weights
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_weights[sample_idx].cpu().numpy().reshape(1, -1), aspect="auto")
        ax.set_title("Attention Weights (First Step)")
        ax.set_xlabel("Encoder Steps")
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "attention.png"))
        self.writer.add_figure("attention", fig)
        plt.close(fig)
        
        # Multiple attention steps to see how it evolves
        n_steps = 5
        attention_weights_history = [prev_attn[sample_idx].cpu().numpy()]
        
        curr_attn = prev_attn
        for i in range(n_steps):
            # Update query (in a real scenario, this would be the decoder state)
            query = torch.randn(batch_size, query_dim).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                context, curr_attn = attention(
                    query=query,
                    memory=encoder_outputs,
                    attention_weights_prev=curr_attn,
                    mask=self.to_device(self.attention_mask)
                )
            
            # Store attention weights
            attention_weights_history.append(curr_attn[sample_idx].cpu().numpy())
        
        # Plot attention evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(np.vstack(attention_weights_history), aspect="auto")
        ax.set_title("Attention Weights Evolution")
        ax.set_ylabel("Decoder Steps")
        ax.set_xlabel("Encoder Steps")
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "attention_evolution.png"))
        self.writer.add_figure("attention_evolution", fig)
        plt.close(fig)
        
        # Log parameters
        for name, param in attention.named_parameters():
            self.writer.add_histogram(f"attention/{name}", param.data)
        
        print("LocationSensitiveAttention test completed successfully")
    
    def test_prenet(self):
        """Test the prenet module."""
        print("\n=== Testing Prenet ===")
        
        # Create module
        prenet = Prenet(
            input_dim=self.config["audio"]["n_mels"],
            hidden_dims=self.config["model"]["decoder"]["prenet_dims"],
            dropout=self.config["model"]["decoder"]["prenet_dropout"]
        ).to(self.device)
        
        # Forward pass
        mel_frame = self.to_device(self.mel[:, 0, :])  # First frame
        
        with torch.no_grad():
            prenet_out = prenet(mel_frame)
        
        # Print output shape
        print(f"Input shape: {mel_frame.shape}")
        print(f"Output shape: {prenet_out.shape}")
        
        # Visualize prenet transformations
        sample_idx = 0
        
        # Process multiple frames for visualization
        n_frames = 10
        frames = self.to_device(self.mel[sample_idx, :n_frames, :])
        
        frame_outputs = []
        for i in range(n_frames):
            with torch.no_grad():
                frame_out = prenet(frames[i])
                frame_outputs.append(frame_out.cpu().numpy())
        
        # Plot transformation
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot input mel frames
        im1 = axes[0].imshow(frames.cpu().numpy(), aspect="auto")
        axes[0].set_title("Mel Frames Input")
        axes[0].set_ylabel("Frame")
        axes[0].set_xlabel("Mel Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot prenet outputs
        prenet_outputs = np.array(frame_outputs)
        im2 = axes[1].imshow(prenet_outputs, aspect="auto")
        axes[1].set_title("Prenet Outputs")
        axes[1].set_ylabel("Frame")
        axes[1].set_xlabel("Prenet Output Dim")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "prenet.png"))
        self.writer.add_figure("prenet", fig)
        plt.close(fig)
        
        # Log parameters and activation histogram
        for name, param in prenet.named_parameters():
            self.writer.add_histogram(f"prenet/{name}", param.data)
        
        self.writer.add_histogram("prenet/output", prenet_out)
        
        print("Prenet test completed successfully")
    
    def test_decoder(self, encoder_outputs):
        """Test the decoder module."""
        print("\n=== Testing DecoderRNN ===")
        
        # Create module
        decoder = DecoderRNN(
            prenet_input_dim=self.config["audio"]["n_mels"],
            prenet_dims=self.config["model"]["decoder"]["prenet_dims"],
            prenet_dropout=self.config["model"]["decoder"]["prenet_dropout"],
            encoder_dim=self.config["model"]["embedding_dim"],
            attention_dim=self.config["model"]["attention"]["attention_dim"],
            attention_location_features=self.config["model"]["attention"]["location_features"],
            attention_location_kernel_size=self.config["model"]["attention"]["location_kernel_size"],
            attention_dropout=self.config["model"]["attention"]["attention_dropout"],
            decoder_dim=self.config["model"]["decoder"]["lstm_dim"],
            num_layers=self.config["model"]["decoder"]["lstm_layers"],
            dropout=self.config["model"]["decoder"]["lstm_dropout"],
            output_dim=self.config["audio"]["n_mels"]
        ).to(self.device)
        
        # Forward pass with teacher forcing
        target_mel = self.to_device(self.mel)
        mel_lengths = self.to_device(self.mel_lengths)
        
        with torch.no_grad():
            outputs, stop_tokens, alignments = decoder(
                memory=encoder_outputs,
                max_decoder_steps=100,
                teacher_forcing_ratio=1.0,
                target=target_mel,
                memory_lengths=mel_lengths
            )
        
        # Print output shapes
        print(f"Encoder outputs shape: {encoder_outputs.shape}")
        print(f"Decoder outputs shape: {outputs.shape}")
        print(f"Stop tokens shape: {stop_tokens.shape}")
        print(f"Alignments shape: {alignments.shape}")
        
        # Visualize decoder outputs and attention
        sample_idx = 0
        output_len = min(outputs.size(1), 100)  # Limit for visualization
        
        # Plot decoder outputs and target
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot decoder outputs
        im1 = axes[0].imshow(outputs[sample_idx, :output_len].cpu().numpy(), aspect="auto")
        axes[0].set_title("Decoder Outputs (Mel Spectrogram)")
        axes[0].set_ylabel("Decoder Steps")
        axes[0].set_xlabel("Mel Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot target mel
        target_len = min(target_mel.size(1), output_len)
        im2 = axes[1].imshow(target_mel[sample_idx, :target_len].cpu().numpy(), aspect="auto")
        axes[1].set_title("Target Mel Spectrogram")
        axes[1].set_ylabel("Steps")
        axes[1].set_xlabel("Mel Dim")
        plt.colorbar(im2, ax=axes[1])
        
        # Plot attention alignment
        im3 = axes[2].imshow(
            alignments[sample_idx, :output_len, :encoder_outputs.size(1)].cpu().numpy(),
            aspect="auto",
            origin="lower"
        )
        axes[2].set_title("Attention Alignment")
        axes[2].set_ylabel("Decoder Steps")
        axes[2].set_xlabel("Encoder Steps")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "decoder.png"))
        self.writer.add_figure("decoder", fig)
        plt.close(fig)
        
        # Forward pass without teacher forcing
        with torch.no_grad():
            outputs_no_tf, stop_tokens_no_tf, alignments_no_tf = decoder(
                memory=encoder_outputs,
                max_decoder_steps=100,
                teacher_forcing_ratio=0.0,
                memory_lengths=mel_lengths
            )
        
        # Print output shapes for inference mode
        print(f"Inference decoder outputs shape: {outputs_no_tf.shape}")
        
        # Plot inference outputs
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot decoder outputs (inference)
        output_len_no_tf = min(outputs_no_tf.size(1), 100)
        im1 = axes[0].imshow(outputs_no_tf[sample_idx, :output_len_no_tf].cpu().numpy(), aspect="auto")
        axes[0].set_title("Decoder Outputs (Inference)")
        axes[0].set_ylabel("Decoder Steps")
        axes[0].set_xlabel("Mel Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot attention alignment (inference)
        im2 = axes[1].imshow(
            alignments_no_tf[sample_idx, :output_len_no_tf, :encoder_outputs.size(1)].cpu().numpy(),
            aspect="auto",
            origin="lower"
        )
        axes[1].set_title("Attention Alignment (Inference)")
        axes[1].set_ylabel("Decoder Steps")
        axes[1].set_xlabel("Encoder Steps")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "decoder_inference.png"))
        self.writer.add_figure("decoder_inference", fig)
        plt.close(fig)
        
        # Log parameters
        for name, param in decoder.named_parameters():
            self.writer.add_histogram(f"decoder/{name}", param.data)
        
        print("DecoderRNN test completed successfully")
        return outputs
    
    def test_postnet(self, decoder_outputs):
        """Test the postnet module."""
        print("\n=== Testing Postnet ===")
        
        # Create module
        postnet = Postnet(
            mel_dim=self.config["audio"]["n_mels"],
            channels=self.config["model"]["decoder"]["postnet_channels"],
            kernel_size=self.config["model"]["decoder"]["postnet_kernel"],
            dropout=self.config["model"]["decoder"]["postnet_dropout"]
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            postnet_output = postnet(decoder_outputs)
            final_output = decoder_outputs + postnet_output
        
        # Print output shapes
        print(f"Decoder outputs shape: {decoder_outputs.shape}")
        print(f"Postnet residual shape: {postnet_output.shape}")
        print(f"Final output shape: {final_output.shape}")
        
        # Visualize postnet effect
        sample_idx = 0
        output_len = min(decoder_outputs.size(1), 100)  # Limit for visualization
        
        # Plot before and after postnet
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot decoder outputs
        im1 = axes[0].imshow(decoder_outputs[sample_idx, :output_len].cpu().numpy(), aspect="auto")
        axes[0].set_title("Decoder Outputs (Before Postnet)")
        axes[0].set_ylabel("Steps")
        axes[0].set_xlabel("Mel Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot postnet residual
        im2 = axes[1].imshow(postnet_output[sample_idx, :output_len].cpu().numpy(), aspect="auto")
        axes[1].set_title("Postnet Residual")
        axes[1].set_ylabel("Steps")
        axes[1].set_xlabel("Mel Dim")
        plt.colorbar(im2, ax=axes[1])
        
        # Plot final output
        im3 = axes[2].imshow(final_output[sample_idx, :output_len].cpu().numpy(), aspect="auto")
        axes[2].set_title("Final Output (After Postnet)")
        axes[2].set_ylabel("Steps")
        axes[2].set_xlabel("Mel Dim")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "postnet.png"))
        self.writer.add_figure("postnet", fig)
        plt.close(fig)
        
        # Log parameters
        for name, param in postnet.named_parameters():
            self.writer.add_histogram(f"postnet/{name}", param.data)
        
        print("Postnet test completed successfully")
    
    def test_full_model(self):
        """Test the full model."""
        print("\n=== Testing Full Model ===")
        
        # Create model
        model = SingingVoiceSynthesisModel(self.config).to(self.device)
        
        # Convert batch to device
        batch = self.to_device(self.test_batch)
        
        # Forward pass with teacher forcing
        with torch.no_grad():
            outputs = model(
                phoneme_ids=batch["phoneme_ids"],
                musical_features=batch["musical_features"],
                mel_lengths=batch["mel_lengths"],
                teacher_forcing_ratio=1.0,
                target_mel=batch["mel"]
            )
        
        # Print output shapes
        print(f"Mel outputs shape: {outputs['mel_outputs'].shape}")
        print(f"Mel outputs postnet shape: {outputs['mel_outputs_postnet'].shape}")
        print(f"Stop tokens shape: {outputs['stop_tokens'].shape}")
        print(f"Alignments shape: {outputs['alignments'].shape}")
        
        # Calculate loss
        loss_dict = model._calculate_loss(outputs, batch)
        print("\nLoss values:")
        for name, value in loss_dict.items():
            print(f"  {name}: {value.item()}")
        
        # Visualize model outputs
        sample_idx = 0
        
        # Get predicted and target mel spectrograms
        pred_mel = outputs["mel_outputs_postnet"][sample_idx].cpu().numpy()
        target_mel = batch["mel"][sample_idx].cpu().numpy()
        attention = outputs["alignments"][sample_idx].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot predicted mel
        im1 = axes[0].imshow(pred_mel, aspect="auto")
        axes[0].set_title("Predicted Mel Spectrogram")
        axes[0].set_ylabel("Steps")
        axes[0].set_xlabel("Mel Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot target mel
        im2 = axes[1].imshow(target_mel, aspect="auto")
        axes[1].set_title("Target Mel Spectrogram")
        axes[1].set_ylabel("Steps")
        axes[1].set_xlabel("Mel Dim")
        plt.colorbar(im2, ax=axes[1])
        
        # Plot attention
        im3 = axes[2].imshow(attention, aspect="auto", origin="lower")
        axes[2].set_title("Attention")
        axes[2].set_xlabel("Encoder Steps")
        axes[2].set_ylabel("Decoder Steps")
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "full_model.png"))
        self.writer.add_figure("full_model", fig)
        plt.close(fig)
        
        # Log parameters
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"model/{name}", param.data)
        
        # Log model parameter count
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTotal model parameters: {total_params:,}")
        
        # Try inference mode
        with torch.no_grad():
            inference_outputs = model(
                phoneme_ids=batch["phoneme_ids"],
                musical_features=batch["musical_features"],
                mel_lengths=batch["mel_lengths"],
                teacher_forcing_ratio=0.0,
                target_mel=None
            )
        
        # Visualize inference outputs
        inference_pred_mel = inference_outputs["mel_outputs_postnet"][sample_idx].cpu().numpy()
        inference_attention = inference_outputs["alignments"][sample_idx].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predicted mel (inference)
        im1 = axes[0].imshow(inference_pred_mel, aspect="auto")
        axes[0].set_title("Predicted Mel Spectrogram (Inference)")
        axes[0].set_ylabel("Steps")
        axes[0].set_xlabel("Mel Dim")
        plt.colorbar(im1, ax=axes[0])
        
        # Plot attention (inference)
        im2 = axes[1].imshow(inference_attention, aspect="auto", origin="lower")
        axes[1].set_title("Attention (Inference)")
        axes[1].set_xlabel("Encoder Steps")
        axes[1].set_ylabel("Decoder Steps")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "full_model_inference.png"))
        self.writer.add_figure("full_model_inference", fig)
        plt.close(fig)
        
        print("Full model test completed successfully")
    
    def validate_all(self):
        """Validate all modules."""
        print("\n=== Starting Module Validation ===")
        
        # Test phoneme encoder
        phoneme_features = self.test_phoneme_encoder()
        
        # Test musical feature encoder
        musical_features_encoded = self.test_musical_feature_encoder()
        
        # Test feature fusion
        fused_features = self.test_feature_fusion(phoneme_features, musical_features_encoded)
        
        # Test attention
        self.test_attention(fused_features)
        
        # Test prenet
        self.test_prenet()
        
        # Test decoder
        decoder_outputs = self.test_decoder(fused_features)
        
        # Test postnet
        self.test_postnet(decoder_outputs)
        
        # Test full model
        self.test_full_model()
        
        print("\n=== Module Validation Completed ===")
        print(f"Logs and visualizations saved to: {self.log_dir}")
        
        # Close TensorBoard writer
        self.writer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate model modules")
    
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--log_dir", type=str, default="logs/module_validation",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--module", type=str, default="all",
                        choices=["all", "phoneme_encoder", "musical_encoder", "feature_fusion", 
                                "attention", "prenet", "decoder", "postnet", "full_model"],
                        help="Module to validate (default: all)")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create validator
    validator = ModuleValidator(args.config, args.log_dir)
    
    # Run validation based on selected module
    if args.module == "all":
        validator.validate_all()
    elif args.module == "phoneme_encoder":
        validator.test_phoneme_encoder()
    elif args.module == "musical_encoder":
        validator.test_musical_feature_encoder()
    elif args.module == "feature_fusion":
        phoneme_features = validator.test_phoneme_encoder()
        musical_features_encoded = validator.test_musical_feature_encoder()
        validator.test_feature_fusion(phoneme_features, musical_features_encoded)
    elif args.module == "attention":
        phoneme_features = validator.test_phoneme_encoder()
        musical_features_encoded = validator.test_musical_feature_encoder()
        fused_features = validator.test_feature_fusion(phoneme_features, musical_features_encoded)
        validator.test_attention(fused_features)
    elif args.module == "prenet":
        validator.test_prenet()
    elif args.module == "decoder":
        phoneme_features = validator.test_phoneme_encoder()
        musical_features_encoded = validator.test_musical_feature_encoder()
        fused_features = validator.test_feature_fusion(phoneme_features, musical_features_encoded)
        validator.test_decoder(fused_features)
    elif args.module == "postnet":
        phoneme_features = validator.test_phoneme_encoder()
        musical_features_encoded = validator.test_musical_feature_encoder()
        fused_features = validator.test_feature_fusion(phoneme_features, musical_features_encoded)
        decoder_outputs = validator.test_decoder(fused_features)
        validator.test_postnet(decoder_outputs)
    elif args.module == "full_model":
        validator.test_full_model()


if __name__ == "__main__":
    main()
