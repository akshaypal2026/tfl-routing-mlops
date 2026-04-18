# %% [markdown]
# # TfL Passenger Density Forecasting: GNN Training
# This notebook is designed to be run on Google Colab (using the free T4 GPU tier).
# It uses PyTorch Geometric to train a Spatio-Temporal Graph Neural Network on the harvested TfL data.

# %%
# !pip install torch-geometric networkx dvc dvc-gdrive

# %%
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    HAS_TORCH = True
except ImportError:
    print("PyTorch not installed locally. Using mock mode.")
    HAS_TORCH = False
import networkx as nx
import json
import os

# %%
# 1. Pull latest data from DVC (assuming DVC is configured with GDrive)
# !dvc pull

# %%
# 2. Load and Prepare Data
data_path = "../data/raw/tfl_density_latest.json"

# Fallback for local testing if run outside colab
if not os.path.exists(data_path):
    print("Data not found. Using dummy data for demonstration.")
    stations = ["Kings Cross", "Euston", "Angel"]
    edges = [("Kings Cross", "Euston"), ("Kings Cross", "Angel")]
    edge_densities = [0.8, 0.4]
else:
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    
    stations = [n["id"] for n in raw_data["nodes"]]
    edges = [(e["source"], e["target"]) for e in raw_data["edges"]]
    edge_densities = [e["density"] for e in raw_data["edges"]]

# Create mapping
node_to_idx = {node: i for i, node in enumerate(stations)}
idx_to_node = {i: node for node, i in node_to_idx.items()}

# Create Edge Index for PyTorch Geometric
edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long).t().contiguous()

# Features (dummy features, e.g., one-hot encoding of station ID or historical density)
num_nodes = len(stations)
x = torch.eye(num_nodes, dtype=torch.float)

# Targets (predicting node density, or edge density. Let's predict node density for simplicity here)
if 'raw_data' in locals():
    y = torch.tensor([n["density"] for n in raw_data["nodes"]], dtype=torch.float).view(-1, 1)
else:
    y = torch.rand((num_nodes, 1))

# %%
if HAS_TORCH:
    # 3. Define the GNN Model
    class DensityGNN(torch.nn.Module):
        def __init__(self, num_node_features, hidden_channels):
            super(DensityGNN, self).__init__()
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 1) # Predict 1 continuous value (density)
    
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return torch.sigmoid(x) # Output between 0 and 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DensityGNN(num_node_features=num_nodes, hidden_channels=16).to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # %%
    # 4. Training Loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.mse_loss(out, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')
    
    # %%
    # 5. Export Model Artifacts
    model.eval()
    with torch.no_grad():
        predictions = model(x, edge_index).cpu().numpy()
    
    # Save predictions for the routing engine
    forecast = {idx_to_node[i]: float(pred[0]) for i, pred in enumerate(predictions)}
else:
    print("Mocking training loop...")
    forecast = {node: 0.5 for node in stations}
os.makedirs("../data/processed", exist_ok=True)
with open("../data/processed/forecast.json", "w") as f:
    json.dump(forecast, f)

if HAS_TORCH:
    torch.save(model.state_dict(), "../data/processed/gnn_model.pth")
else:
    with open("../data/processed/gnn_model.txt", "w") as f:
        f.write("mock_model_weights")
print("Training complete. Artifacts saved to data/processed/")

# %%
# 6. DVC add and push the new artifacts (if running in full pipeline)
# !dvc add ../data/processed/forecast.json ../data/processed/gnn_model.pth
# !dvc push
