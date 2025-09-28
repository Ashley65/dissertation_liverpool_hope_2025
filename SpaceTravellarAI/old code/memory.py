import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer


# === 1. MULTI-MODAL ENCODER (Embeddings for different inputs) === #
class MultiModalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.sensor_encoder = nn.Linear(10, embed_dim)  # Example: 10 sensor features
        self.log_encoder = nn.Linear(20, embed_dim)  # Example: 20 mission log features
        self.comm_encoder = AutoModel.from_pretrained("bert-base-uncased")  # Text encoding
        self.comm_projection = nn.Linear(768, embed_dim)  # Changed from 384 to 768

    def forward(self, sensor_data, mission_logs, messages):
        sensor_embed = self.sensor_encoder(sensor_data)
        log_embed = self.log_encoder(mission_logs)
        comm_embed = self.comm_encoder(**messages).last_hidden_state[:, 0, :]
        comm_embed = self.comm_projection(comm_embed)
        return sensor_embed, log_embed, comm_embed

# === 2. TRANSFORMER BACKBONE (Fusion of all inputs) === #
class SpaceTravelerTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = MultiModalEmbedding(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )
        # Remove output layer as we want to keep embed_dim dimensions
        self.decision_head = nn.Linear(embed_dim, 3)  # Separate decision head

    def forward(self, sensor_data, mission_logs, messages):
        sensor_embed, log_embed, comm_embed = self.embedding(sensor_data, mission_logs, messages)
        fused_input = torch.stack([sensor_embed, log_embed, comm_embed], dim=0)
        transformed_output = self.transformer(fused_input)
        # Keep both the transformed features and decision
        return transformed_output[0], self.decision_head(transformed_output[1])


# === 3. MEMORY AUGMENTATION (Retrieving past experiences) === #
class MemoryModule(nn.Module):
    def __init__(self, embed_dim, memory_size=100):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(1, memory_size, embed_dim))  # Add batch dim
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=8)

    def forward(self, query):
        # Ensure query is 3D: [sequence_length, batch_size, embed_dim]
        if query.dim() == 2:
            query = query.unsqueeze(0).transpose(0, 1)

        # Memory is already [1, memory_size, embed_dim], expand batch dim to match query
        memory = self.memory.expand(query.size(1), -1, -1).transpose(0, 1)

        attn_output, _ = self.attn(query, memory, memory)
        return attn_output

# === 4. FULL AI SYSTEM (Combining Memory & Decision-making) === #
class SpaceTravelerAI(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, num_layers=6):
        super().__init__()
        self.transformer_model = SpaceTravelerTransformer(embed_dim, num_heads, num_layers)
        self.memory_module = MemoryModule(embed_dim)
        self.final_decision = nn.Linear(embed_dim, 3)

    def forward(self, sensor_data, mission_logs, messages):
        features, decision = self.transformer_model(sensor_data, mission_logs, messages)
        memory_retrieval = self.memory_module(features)
        enhanced_features = features + memory_retrieval.squeeze(0)
        final_decision = self.final_decision(enhanced_features)
        return final_decision


# === 5. EXAMPLE USAGE === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpaceTravelerAI().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Dummy inputs
sensor_data = torch.randn(1, 10).to(device)
mission_logs = torch.randn(1, 20).to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-   -uncased")
messages = tokenizer("Hello, AI. What’s the next maneuver?", return_tensors="pt").to(device)

# Forward pass
output = model(sensor_data, mission_logs, messages)
print("Trajectory Prediction:", output)
