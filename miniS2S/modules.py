import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union

class PhonemeEncoder(nn.Module):
    """
    Encodes phoneme sequences using an embedding layer, convolutional layers, and BiLSTM layers.
    """
    def __init__(self, 
                 num_phonemes: int,
                 embedding_dim: int,
                 conv_channels: int,
                 conv_kernel_size: int,
                 conv_dropout: float,
                 lstm_hidden_size: int,
                 lstm_layers: int,
                 lstm_dropout: float):
        """
        Args:
            num_phonemes: Number of phonemes in the vocabulary (including padding)
            embedding_dim: Dimension of phoneme embeddings
            conv_channels: Number of channels in convolutional layers
            conv_kernel_size: Kernel size for convolutional layers
            conv_dropout: Dropout rate for convolutional layers
            lstm_hidden_size: Hidden size of LSTM (per direction)
            lstm_layers: Number of LSTM layers
            lstm_dropout: Dropout rate between LSTM layers
        """
        super().__init__()
        
        # Phoneme embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_phonemes,
            embedding_dim=embedding_dim,
            padding_idx=0  # 0 is reserved for padding
        )
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(3):  # 3 convolutional layers as per design
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=embedding_dim if i == 0 else conv_channels,
                        out_channels=conv_channels,
                        kernel_size=conv_kernel_size,
                        padding=(conv_kernel_size - 1) // 2  # Same padding
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.ReLU(),
                    nn.Dropout(conv_dropout)
                )
            )
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )

    def forward(self, phoneme_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the phoneme encoder.
        
        Args:
            phoneme_ids: Tensor of phoneme IDs [batch_size, seq_length]
            lengths: Lengths of sequences for packing (optional)
            
        Returns:
            Encoded phoneme features [batch_size, seq_length, 2*lstm_hidden_size]
        """
        batch_size, seq_length = phoneme_ids.size()
        
        # Embedding: [batch_size, seq_length] -> [batch_size, seq_length, embedding_dim]
        x = self.embedding(phoneme_ids)
        
        # Prepare for 1D convolution: [batch_size, seq_length, embedding_dim] -> [batch_size, embedding_dim, seq_length]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv in self.convs:
            x = conv(x)
        
        # Prepare for LSTM: [batch_size, conv_channels, seq_length] -> [batch_size, seq_length, conv_channels]
        x = x.transpose(1, 2)
        
        # Apply BiLSTM
        if lengths is not None:
            # Pack sequences for efficient computation
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            lstm_out, _ = self.lstm(x_packed)
            
            # Unpack sequences
            x, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            x, _ = self.lstm(x)
        
        return x  # [batch_size, seq_length, 2*lstm_hidden_size]


class MusicalFeatureEncoder(nn.Module):
    """
    Encodes musical features using an MLP architecture.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout: float):
        """
        Args:
            input_dim: Number of input musical features
            hidden_dims: List of hidden dimensions for MLP layers
            dropout: Dropout rate between layers
        """
        super().__init__()
        
        # Create MLP layers
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, musical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the musical feature encoder.
        
        Args:
            musical_features: Musical features [batch_size, seq_length, input_dim]
            
        Returns:
            Encoded musical features [batch_size, seq_length, output_dim]
        """
        # Process each frame through the MLP
        return self.mlp(musical_features)


class FeatureFusion(nn.Module):
    """
    Fuses phoneme and musical features through concatenation and projection.
    """
    def __init__(self, 
                 phoneme_dim: int,
                 musical_dim: int,
                 output_dim: int):
        """
        Args:
            phoneme_dim: Dimension of phoneme features
            musical_dim: Dimension of musical features
            output_dim: Dimension of fused features
        """
        super().__init__()
        
        # Linear projection for feature fusion
        self.projection = nn.Linear(phoneme_dim + musical_dim, output_dim)

    def forward(self, 
                phoneme_features: torch.Tensor, 
                musical_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature fusion.
        
        Args:
            phoneme_features: Phoneme features [batch_size, seq_length, phoneme_dim]
            musical_features: Musical features [batch_size, seq_length, musical_dim]
            
        Returns:
            Fused features [batch_size, seq_length, output_dim]
        """
        # Concatenate features along the feature dimension
        fused = torch.cat([phoneme_features, musical_features], dim=-1)
        
        # Project to output dimension
        return self.projection(fused)


class LocationSensitiveAttention(nn.Module):
    """
    Location-sensitive attention mechanism based on Tacotron 2 style attention.
    """
    def __init__(self, 
                 query_dim: int,
                 encoder_dim: int,
                 attention_dim: int,
                 location_features: int,
                 location_kernel_size: int,
                 dropout: float):
        """
        Args:
            query_dim: Dimension of query (decoder state)
            encoder_dim: Dimension of encoder outputs (memory)
            attention_dim: Dimension of attention calculations
            location_features: Number of location-aware convolutional filters
            location_kernel_size: Kernel size for location-aware attention
            dropout: Dropout rate
        """
        super().__init__()
        
        # Query projection
        self.query_layer = nn.Linear(query_dim, attention_dim)
        
        # Memory (key) projection
        self.memory_layer = nn.Linear(encoder_dim, attention_dim)
        
        # Location feature extraction
        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=location_features,
            kernel_size=location_kernel_size,
            padding=(location_kernel_size - 1) // 2,
            bias=False
        )
        self.location_layer = nn.Linear(location_features, attention_dim)
        
        # Energy calculation
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters (Glorot/Xavier initialization)
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize the parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.query_layer.weight)
        nn.init.xavier_uniform_(self.memory_layer.weight)
        nn.init.xavier_uniform_(self.location_layer.weight)
        nn.init.xavier_uniform_(self.v.weight)
    
    def forward(self, 
                query: torch.Tensor, 
                memory: torch.Tensor, 
                processed_memory: Optional[torch.Tensor] = None,
                attention_weights_prev: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the location-sensitive attention.
        
        Args:
            query: Decoder state [batch_size, query_dim]
            memory: Encoder outputs [batch_size, seq_length, encoder_dim]
            processed_memory: Pre-processed memory (optional)
            attention_weights_prev: Previous attention weights [batch_size, seq_length]
            mask: Attention mask [batch_size, seq_length]
            
        Returns:
            context: Context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, seq_length]
        """
        batch_size, seq_length, _ = memory.size()
        
        # Process query: [batch_size, query_dim] -> [batch_size, 1, attention_dim]
        processed_query = self.query_layer(query).unsqueeze(1)
        
        # Process memory if not already processed
        if processed_memory is None:
            processed_memory = self.memory_layer(memory)
        
        # Process location features if attention weights are provided
        location_features = torch.zeros(
            batch_size, seq_length, self.location_layer.in_features,
            device=query.device
        )
        
        if attention_weights_prev is not None:
            # Process previous attention weights
            # [batch_size, seq_length] -> [batch_size, 1, seq_length]
            attention_weights_prev = attention_weights_prev.unsqueeze(1)
            
            # Apply convolutional filters to previous weights
            # [batch_size, 1, seq_length] -> [batch_size, location_features, seq_length]
            location_features = self.location_conv(attention_weights_prev)
            
            # [batch_size, location_features, seq_length] -> [batch_size, seq_length, location_features]
            location_features = location_features.transpose(1, 2)
        
        # Project location features
        processed_location = self.location_layer(location_features)
        
        # Calculate energy scores
        # [batch_size, seq_length, attention_dim]
        energy = processed_query + processed_memory + processed_location
        
        # Apply dropout to energy
        energy = self.dropout(torch.tanh(energy))
        
        # [batch_size, seq_length, 1] -> [batch_size, seq_length]
        energy = self.v(energy).squeeze(-1)
        
        # Apply mask if provided
        if mask is not None:
            energy = energy.masked_fill(~mask, -float('inf'))
        
        # Calculate attention weights
        attention_weights = F.softmax(energy, dim=1)
        
        # Calculate context vector
        # [batch_size, 1, seq_length] @ [batch_size, seq_length, encoder_dim] -> [batch_size, 1, encoder_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # [batch_size, 1, encoder_dim] -> [batch_size, encoder_dim]
        context = context.squeeze(1)
        
        return context, attention_weights


class Prenet(nn.Module):
    """
    Prenet for the decoder, processes previous mel frames.
    """
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout: float):
        """
        Args:
            input_dim: Input dimension (mel-spectrogram bands)
            hidden_dims: List of hidden dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        in_dim = input_dim
        for dim in hidden_dims:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            in_dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the prenet.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Processed tensor [batch_size, hidden_dims[-1]]
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderRNN(nn.Module):
    """
    Decoder RNN module for mel-spectrogram generation.
    """
    def __init__(self, 
                 prenet_input_dim: int,
                 prenet_dims: List[int],
                 prenet_dropout: float,
                 encoder_dim: int,
                 attention_dim: int,
                 attention_location_features: int,
                 attention_location_kernel_size: int,
                 attention_dropout: float,
                 decoder_dim: int,
                 num_layers: int,
                 dropout: float,
                 output_dim: int):
        """
        Args:
            prenet_input_dim: Input dimension for prenet (mel bands)
            prenet_dims: Hidden dimensions for prenet
            prenet_dropout: Dropout rate for prenet
            encoder_dim: Dimension of encoder outputs
            attention_dim: Dimension for attention
            attention_location_features: Number of location features
            attention_location_kernel_size: Kernel size for location features
            attention_dropout: Dropout rate for attention
            decoder_dim: Hidden dimension for decoder LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for LSTM
            output_dim: Output dimension (mel bands)
        """
        super().__init__()
        
        # Prenet to process previous mel frame
        self.prenet = Prenet(
            input_dim=prenet_input_dim,
            hidden_dims=prenet_dims,
            dropout=prenet_dropout
        )
        
        # Attention mechanism
        self.attention = LocationSensitiveAttention(
            query_dim=decoder_dim,
            encoder_dim=encoder_dim,
            attention_dim=attention_dim,
            location_features=attention_location_features,
            location_kernel_size=attention_location_kernel_size,
            dropout=attention_dropout
        )
        
        # Dimensions
        self.prenet_output_dim = prenet_dims[-1] if prenet_dims else prenet_input_dim
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.output_dim = output_dim
        
        # First LSTM layer input: prenet output + context vector
        self.lstm_input_dim = self.prenet_output_dim + encoder_dim
        
        # LSTM layers using LSTMCell for autoregressive decoding
        self.lstm_cells = nn.ModuleList()
        
        # First layer has different input dimension
        self.lstm_cells.append(
            nn.LSTMCell(
                input_size=self.lstm_input_dim,
                hidden_size=decoder_dim
            )
        )
        
        # Remaining layers
        for _ in range(1, num_layers):
            self.lstm_cells.append(
                nn.LSTMCell(
                    input_size=decoder_dim,
                    hidden_size=decoder_dim
                )
            )
        
        # Output projection
        self.frame_projection = nn.Linear(decoder_dim + encoder_dim, output_dim)
        
        # Stop token prediction
        self.stop_projection = nn.Linear(decoder_dim + encoder_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_states(self, memory: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple:
        """
        Initialize decoder states.
        
        Args:
            memory: Encoder outputs [batch_size, seq_length, encoder_dim]
            mask: Memory mask [batch_size, seq_length]
            
        Returns:
            hidden_states, cell_states, context, attention_weights, processed_memory
        """
        batch_size = memory.size(0)
        max_length = memory.size(1)
        
        # Initialize hidden states and cell states
        hidden_states = [torch.zeros(batch_size, self.decoder_dim, device=memory.device)
                         for _ in range(len(self.lstm_cells))]
        cell_states = [torch.zeros(batch_size, self.decoder_dim, device=memory.device)
                       for _ in range(len(self.lstm_cells))]
        
        # Initialize context vector
        context = torch.zeros(batch_size, self.encoder_dim, device=memory.device)
        
        # Initialize attention weights
        attention_weights = torch.zeros(batch_size, max_length, device=memory.device)
        if max_length > 0:
            attention_weights[:, 0] = 1.0  # Initial attention on first encoder step
        
        # Pre-compute processed memory for efficiency
        processed_memory = self.attention.memory_layer(memory)
        
        return hidden_states, cell_states, context, attention_weights, processed_memory
    
    def _decode_step(self, 
                    prev_output: torch.Tensor,
                    memory: torch.Tensor,
                    processed_memory: torch.Tensor,
                    attention_weights: torch.Tensor,
                    hidden_states: List[torch.Tensor],
                    cell_states: List[torch.Tensor],
                    mask: Optional[torch.Tensor] = None) -> Tuple:
        """
        Perform one step of decoding.
        
        Args:
            prev_output: Previous output (mel frame) [batch_size, output_dim]
            memory: Encoder outputs [batch_size, seq_length, encoder_dim]
            processed_memory: Pre-processed memory [batch_size, seq_length, attention_dim]
            attention_weights: Previous attention weights [batch_size, seq_length]
            hidden_states: List of hidden states for each LSTM layer
            cell_states: List of cell states for each LSTM layer
            mask: Memory mask [batch_size, seq_length]
            
        Returns:
            output, stop_token, hidden_states, cell_states, context, attention_weights
        """
        # Process previous output through prenet
        prenet_out = self.prenet(prev_output)
        
        # First LSTM layer input: concatenate prenet output and context vector
        lstm_input = torch.cat([prenet_out, context], dim=1)
        
        # Update LSTM states
        next_hidden_states = []
        next_cell_states = []
        
        # First LSTM layer
        h, c = self.lstm_cells[0](lstm_input, (hidden_states[0], cell_states[0]))
        h = self.dropout(h)
        next_hidden_states.append(h)
        next_cell_states.append(c)
        
        # Remaining LSTM layers
        for i in range(1, len(self.lstm_cells)):
            h, c = self.lstm_cells[i](h, (hidden_states[i], cell_states[i]))
            h = self.dropout(h)
            next_hidden_states.append(h)
            next_cell_states.append(c)
        
        # Use the output of the last LSTM layer for attention
        lstm_output = next_hidden_states[-1]
        
        # Calculate attention
        context, attention_weights = self.attention(
            query=lstm_output,
            memory=memory,
            processed_memory=processed_memory,
            attention_weights_prev=attention_weights,
            mask=mask
        )
        
        # Concatenate LSTM output and context vector
        output_input = torch.cat([lstm_output, context], dim=1)
        
        # Project to output dimension
        output = self.frame_projection(output_input)
        
        # Predict stop token
        stop_token = self.stop_projection(output_input)
        stop_token = torch.sigmoid(stop_token)
        
        return output, stop_token, next_hidden_states, next_cell_states, context, attention_weights
    
    def forward(self, 
                memory: torch.Tensor,
                max_decoder_steps: int = 1000,
                teacher_forcing_ratio: float = 1.0,
                target: Optional[torch.Tensor] = None,
                memory_lengths: Optional[torch.Tensor] = None) -> Tuple:
        """
        Forward pass for the decoder.
        
        Args:
            memory: Encoder outputs [batch_size, seq_length, encoder_dim]
            max_decoder_steps: Maximum number of decoding steps
            teacher_forcing_ratio: Probability of using teacher forcing
            target: Target mel spectrogram for teacher forcing [batch_size, target_length, output_dim]
            memory_lengths: Lengths of encoder outputs [batch_size]
            
        Returns:
            outputs: Mel spectrogram frames [batch_size, decoder_steps, output_dim]
            stop_tokens: Stop token predictions [batch_size, decoder_steps, 1]
            alignments: Attention weights [batch_size, decoder_steps, memory_length]
        """
        batch_size = memory.size(0)
        
        # Create mask for attention
        if memory_lengths is not None:
            mask = torch.arange(memory.size(1), device=memory.device).expand(
                batch_size, memory.size(1)
            ) < memory_lengths.unsqueeze(1)
        else:
            mask = None
        
        # Initialize states
        hidden_states, cell_states, context, attention_weights, processed_memory = self._init_states(
            memory=memory, mask=mask
        )
        
        # Determine target length
        if target is not None:
            target_length = target.size(1)
        else:
            target_length = 0
        
        # Maximum number of steps
        max_steps = max(max_decoder_steps, target_length)
        
        # Initialize output tensors
        outputs = []
        stop_tokens = []
        alignments = []
        
        # Initial input: zero vector
        prev_output = torch.zeros(batch_size, self.output_dim, device=memory.device)
        
        # Autoregressive decoding
        for t in range(max_steps):
            # Decide whether to use teacher forcing
            use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)
            
            if use_teacher_forcing and t < target_length:
                prev_output = target[:, t, :]
            
            # Decode one step
            output, stop_token, hidden_states, cell_states, context, attention_weights = self._decode_step(
                prev_output=prev_output,
                memory=memory,
                processed_memory=processed_memory,
                attention_weights=attention_weights,
                hidden_states=hidden_states,
                cell_states=cell_states,
                mask=mask
            )
            
            # Store outputs
            outputs.append(output)
            stop_tokens.append(stop_token)
            alignments.append(attention_weights)
            
            # Update previous output
            prev_output = output
            
            # Stop if stop token is predicted
            if not use_teacher_forcing and stop_token.item() > 0.5:
                break
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # [batch_size, decoder_steps, output_dim]
        stop_tokens = torch.stack(stop_tokens, dim=1)  # [batch_size, decoder_steps, 1]
        alignments = torch.stack(alignments, dim=1)  # [batch_size, decoder_steps, memory_length]
        
        return outputs, stop_tokens, alignments


class Postnet(nn.Module):
    """
    Postnet to refine the predicted mel spectrogram.
    """
    def __init__(self, 
                 mel_dim: int,
                 channels: int,
                 kernel_size: int,
                 n_convs: int = 5,
                 dropout: float = 0.1):
        """
        Args:
            mel_dim: Number of mel bands
            channels: Number of channels in convolutional layers
            kernel_size: Kernel size for convolutions
            n_convs: Number of convolutional layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create convolutional layers
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=mel_dim,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )
        
        # Hidden layers
        for i in range(1, n_convs - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2
                    ),
                    nn.BatchNorm1d(channels),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )
        
        # Final layer
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=mel_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(dropout)
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the postnet.
        
        Args:
            x: Input mel spectrogram [batch_size, seq_length, mel_dim]
            
        Returns:
            Residual to add to the mel spectrogram [batch_size, seq_length, mel_dim]
        """
        # Convert to channel-first format
        x = x.transpose(1, 2)
        
        # Apply convolutions
        for conv in self.convs:
            x = conv(x)
        
        # Convert back to sequence-first format
        return x.transpose(1, 2)


class GuidedAttentionLoss(nn.Module):
    """
    Guided attention loss to encourage monotonic attention during training.
    """
    def __init__(self, sigma: float = 0.2):
        """
        Args:
            sigma: Standard deviation for Gaussian distribution
        """
        super().__init__()
        self.sigma = sigma
    
    def _create_guide_matrix(self, n_rows: int, n_cols: int) -> torch.Tensor:
        """
        Create a guide matrix for monotonic attention.
        
        Args:
            n_rows: Number of decoder steps
            n_cols: Number of encoder steps
            
        Returns:
            Guide matrix [n_rows, n_cols]
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(n_cols), torch.arange(n_rows))
        grid_x = grid_x.T
        grid_y = grid_y.T
        
        # Calculate guide matrix
        W = grid_y.float() / n_rows - grid_x.float() / n_cols
        W = torch.exp(-(W ** 2) / (2 * self.sigma ** 2))
        
        return 1.0 - W
    
    def forward(self, 
                attention_weights: torch.Tensor,
                mel_lengths: torch.Tensor,
                memory_lengths: torch.Tensor) -> torch.Tensor:
        """
        Calculate guided attention loss.
        
        Args:
            attention_weights: Attention weights [batch_size, decoder_steps, encoder_steps]
            mel_lengths: Lengths of mel spectrograms [batch_size]
            memory_lengths: Lengths of encoder outputs [batch_size]
            
        Returns:
            Loss value
        """
        batch_size, decoder_steps, encoder_steps = attention_weights.size()
        
        # Create attention guides for each sample in the batch
        loss = 0.0
        for i in range(batch_size):
            actual_decoder_steps = min(decoder_steps, mel_lengths[i].item())
            actual_encoder_steps = min(encoder_steps, memory_lengths[i].item())
            
            if actual_decoder_steps <= 0 or actual_encoder_steps <= 0:
                continue
            
            # Create guide matrix
            guide = self._create_guide_matrix(
                n_rows=actual_decoder_steps,
                n_cols=actual_encoder_steps
            ).to(attention_weights.device)
            
            # Calculate loss
            A = attention_weights[i, :actual_decoder_steps, :actual_encoder_steps]
            loss += torch.mean(A * guide)
        
        return loss / batch_size
