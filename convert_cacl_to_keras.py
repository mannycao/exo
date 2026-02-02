import torch
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden=64, output=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output))

    def forward(self, x):
        return self.net(x)

class PyTorchTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64, nhead=2, num_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(output_dim) # Added for structural consistency
        self.output_proj = nn.Linear(output_dim, output_dim) # Added for structural consistency

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        out = self.transformer(x)
        out = self.norm(out) # Apply norm
        out = self.output_proj(out) # Apply output projection
        return out.squeeze(1)

class Projector(nn.Module):
    def __init__(self, input_dim=64, proj_dim=32):
        super().__init__()
        self.fc = nn.Linear(input_dim, proj_dim)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=-1)

# --- Keras Model Definitions (copied from models/multimodal_model.py, modified) ---

# --- Keras Model Definitions (copied from models/multimodal_model.py, modified) ---
@tf.keras.utils.register_keras_serializable(name="KerasTransformerEncoder")
class KerasTransformerEncoder(tf.keras.Model):
    def __init__(self, input_dim, head_size, num_heads, ff_dim, num_layers, **kwargs):
        # Handle the **kwargs for proper serialization
        super(KerasTransformerEncoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers

        self.input_proj = tf.keras.layers.Dense(head_size * num_heads)

        self.attention_layers = []
        self.attention_dropout_layers = []
        self.attention_norm_layers = []
        self.ffn_conv1d_1_layers = []
        self.ffn_dropout_layers = []
        self.ffn_conv1d_2_layers = []
        self.ffn_norm_layers = []

        for _ in range(num_layers):
            self.attention_layers.append(tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size))
            self.attention_dropout_layers.append(tf.keras.layers.Dropout(0.1))
            self.attention_norm_layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.ffn_conv1d_1_layers.append(tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu"))
            self.ffn_dropout_layers.append(tf.keras.layers.Dropout(0.1))
            self.ffn_conv1d_2_layers.append(tf.keras.layers.Conv1D(filters=head_size * num_heads, kernel_size=1))
            self.ffn_norm_layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))

    def call(self, inputs):
        x = self.input_proj(inputs)
        x = tf.expand_dims(x, axis=1) # [B, D] -> [B, 1, D]

        for i in range(self.num_layers):
            # MultiHeadAttention
            attention_output = self.attention_layers[i](x, x)
            attention_output = self.attention_dropout_layers[i](attention_output)
            x = self.attention_norm_layers[i](x + attention_output)

            # Feed Forward Network
            ffn_output = self.ffn_conv1d_1_layers[i](x)
            ffn_output = self.ffn_dropout_layers[i](ffn_output)
            ffn_output = self.ffn_conv1d_2_layers[i](ffn_output)
            x = self.ffn_norm_layers[i](x + ffn_output)
            
        return tf.squeeze(x, axis=1) # [B, 1, D] -> [B, D]

    def get_config(self):
        """Enable serialization of the model."""
        config = super(KerasTransformerEncoder, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_layers": self.num_layers,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- Weight Conversion Logic ---
def convert_pytorch_to_keras(
    pytorch_model_path: str,
    keras_model_dir: str,
    input_dim: int,
    output_dim: int = 64,
    nhead: int = 2,
    num_layers: int = 1,
):
    # Parameters for Keras TransformerEncoder
    head_size = output_dim // nhead
    num_heads = nhead
    ff_dim = 2048 # Adjusted to match observed linear1.weight shape from pre-trained model

    # 1. Load PyTorch model state dict
    pytorch_state_dict = torch.load(pytorch_model_path, map_location=torch.device('cpu'))

    # 2. Directly instantiate a PyTorchTransformerEncoder
    pytorch_transformer_encoder = PyTorchTransformerEncoder(input_dim, output_dim, nhead, num_layers)

    # Manually filter the loaded state_dict to get only the keys relevant to the first encoder
    single_encoder_state_dict = {}
    for key, value in pytorch_state_dict.items():
        if key.startswith('encoders.0.'):
            new_key = key.replace('encoders.0.', '')
            single_encoder_state_dict[new_key] = value

    pytorch_transformer_encoder.load_state_dict(single_encoder_state_dict)
    pytorch_transformer_encoder.eval() # Set to eval mode

    # 3. Instantiate Keras TransformerEncoder
    keras_transformer_encoder = KerasTransformerEncoder(
        input_dim=input_dim,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers
    )
    # Build the Keras model by calling it with dummy input
    dummy_keras_input = tf.zeros((1, input_dim))
    _ = keras_transformer_encoder(dummy_keras_input)

    # 4. Perform weight transfer
    print("Transferring weights...")

    # Input Projection Layer
    keras_transformer_encoder.input_proj.set_weights([
        pytorch_transformer_encoder.input_proj.weight.detach().T.cpu().numpy(), # PyTorch Linear.weight is (out, in), Keras Dense.kernel is (in, out)
        pytorch_transformer_encoder.input_proj.bias.detach().cpu().numpy()
    ])

    # Transformer Encoder Layers
    for i in range(num_layers):
        pytorch_layer = pytorch_transformer_encoder.transformer.layers[i]
        print(f"\nPyTorch Layer {i} state_dict keys: {pytorch_layer.state_dict().keys()}")
        
        # MultiHeadAttention
        in_proj_weight = pytorch_layer.self_attn.in_proj_weight.detach().cpu().numpy()
        in_proj_bias = pytorch_layer.self_attn.in_proj_bias.detach().cpu().numpy()
        
        q_weight, k_weight, v_weight = np.split(in_proj_weight, 3, axis=0)
        q_bias, k_bias, v_bias = np.split(in_proj_bias, 3, axis=0)

        out_proj_weight = pytorch_layer.self_attn.out_proj.weight.detach().cpu().numpy()
        out_proj_bias = pytorch_layer.self_attn.out_proj.bias.detach().cpu().numpy()

        # Reshape PyTorch weights to Keras MultiHeadAttention expected format
        # PyTorch: (embed_dim, embed_dim) -> Keras: (embed_dim, num_heads, head_size)
        # For query, key, value kernels
        q_kernel = q_weight.T.reshape(output_dim, num_heads, head_size)
        k_kernel = k_weight.T.reshape(output_dim, num_heads, head_size)
        v_kernel = v_weight.T.reshape(output_dim, num_heads, head_size)

        # For output kernel
        # PyTorch: (embed_dim, embed_dim) -> Keras: (num_heads, head_size, embed_dim)
        out_kernel = out_proj_weight.T.reshape(num_heads, head_size, output_dim)

        # Reshape PyTorch biases to Keras MultiHeadAttention expected format
        # PyTorch: (embed_dim,) -> Keras: (num_heads, head_size)
        q_bias_reshaped = q_bias.reshape(num_heads, head_size)
        k_bias_reshaped = k_bias.reshape(num_heads, head_size)
        v_bias_reshaped = v_bias.reshape(num_heads, head_size)

        keras_transformer_encoder.attention_layers[i].set_weights([
            q_kernel, q_bias_reshaped,
            k_kernel, k_bias_reshaped,
            v_kernel, v_bias_reshaped,
            out_kernel, out_proj_bias
        ])

        # LayerNormalization after attention
        keras_transformer_encoder.attention_norm_layers[i].set_weights([
            pytorch_layer.norm1.weight.detach().cpu().numpy(),
            pytorch_layer.norm1.bias.detach().cpu().numpy()
        ])

        # Feed Forward Network (FFN)
        # FFN Linear 1 (Conv1D in Keras)
        linear1_weight_np = pytorch_layer.linear1.weight.detach().T.cpu().numpy()
        print(f"Shape of linear1_weight_np: {linear1_weight_np.shape}")
        keras_transformer_encoder.ffn_conv1d_1_layers[i].set_weights([
            linear1_weight_np.reshape(1, output_dim, ff_dim), # Keras Conv1D kernel is (kernel_size, input_dim, output_dim)
            pytorch_layer.linear1.bias.detach().cpu().numpy()
        ])

        # FFN Linear 2 (Conv1D in Keras)
        keras_transformer_encoder.ffn_conv1d_2_layers[i].set_weights([
            pytorch_layer.linear2.weight.detach().T.cpu().numpy().reshape(1, ff_dim, output_dim), # Keras Conv1D kernel is (kernel_size, input_dim, output_dim)
            pytorch_layer.linear2.bias.detach().cpu().numpy()
        ])

        # LayerNormalization after FFN
        keras_transformer_encoder.ffn_norm_layers[i].set_weights([
            pytorch_layer.norm2.weight.detach().cpu().numpy(),
            pytorch_layer.norm2.bias.detach().cpu().numpy()
        ])

    print("Weight transfer complete.")

    # 5. Save Keras model
    os.makedirs(keras_model_dir, exist_ok=True)
    keras_model_path = os.path.join(keras_model_dir, "keras_cacl_transformer_encoder.keras") # Use .keras extension
    keras_transformer_encoder.save(keras_model_path)
    print(f"Converted Keras TransformerEncoder model saved to {keras_model_path}")

if __name__ == "__main__":
    # Example usage:
    # Ensure cacl_transformer_feature_extractor.pt exists from pre-training
    pytorch_model_path = "./models/cacl_transformer_feature_extractor.pt"
    keras_model_dir = "./models"

    # These parameters must match those used during PyTorch pre-training
    # and the KerasTransformerEncoder definition
    input_dim = 512 # Corrected: This is the length of a single partition
    output_dim = 64
    nhead = 2
    num_layers = 1

    # 1. Load PyTorch model state dict
    pytorch_state_dict = torch.load(pytorch_model_path, map_location=torch.device('cpu'))

    # 2. Directly instantiate a PyTorchTransformerEncoder
    pytorch_transformer_encoder = PyTorchTransformerEncoder(input_dim, output_dim, nhead, num_layers)

    # Manually filter the loaded state_dict to get only the keys relevant to the first encoder
    single_encoder_state_dict = {}
    for key, value in pytorch_state_dict.items():
        if key.startswith('encoders.0.'):
            new_key = key.replace('encoders.0.', '')
            single_encoder_state_dict[new_key] = value

    pytorch_transformer_encoder.load_state_dict(single_encoder_state_dict)
    pytorch_transformer_encoder.eval() # Set to eval mode

    # Create a dummy PyTorch model and save its state_dict for testing the conversion script
    # In a real scenario, this would be the actual pre-trained model
    if not os.path.exists(pytorch_model_path):
        print("PyTorch pre-trained model not found. Creating a dummy one for testing conversion.")
        dummy_partitions = [[i for i in range(input_dim)]]
        dummy_pytorch_model = PyTorchMultiViewModel(
            partitions=dummy_partitions,
            encoder_type="transformer",
            output_dim=output_dim,
            proj_dim=32,
            nhead=nhead,
            num_layers=num_layers
        )
        # Save only the encoder part
        dummy_encoder_state_dict = dummy_pytorch_model.get_feature_extractor()[0].state_dict()
        torch.save(dummy_encoder_state_dict, pytorch_model_path)
        print(f"Dummy PyTorch model saved to {pytorch_model_path}")

    convert_pytorch_to_keras(
        pytorch_model_path=pytorch_model_path,
        keras_model_dir=keras_model_dir,
        input_dim=input_dim,
        output_dim=output_dim,
        nhead=nhead,
        num_layers=num_layers
    )
