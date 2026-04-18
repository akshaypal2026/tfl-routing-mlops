import json
import random
import os
from datetime import datetime

def harvest_mock_tfl_data():
    """
    Harvests mock passenger density data for a simplified TfL network.
    In production, this would call the TfL Unified API (e.g., Crowding endpoints).
    """
    stations = ["Kings Cross", "Euston", "Angel", "Old Street", "Bank", "London Bridge", "Waterloo"]
    edges = [
        ("Kings Cross", "Euston"),
        ("Kings Cross", "Angel"),
        ("Angel", "Old Street"),
        ("Old Street", "Bank"),
        ("Bank", "London Bridge"),
        ("Bank", "Waterloo"),
        ("Euston", "Waterloo")
    ]

    timestamp = datetime.now().isoformat()
    
    # Generate mock density data (0.0 to 1.0)
    density_data = {
        "timestamp": timestamp,
        "nodes": [{"id": s, "density": round(random.uniform(0.1, 0.9), 2)} for s in stations],
        "edges": [{"source": u, "target": v, "density": round(random.uniform(0.1, 0.9), 2)} for u, v in edges]
    }

    # Ensure directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Save to JSON
    filepath = f"data/raw/tfl_density_latest.json"
    with open(filepath, "w") as f:
        json.dump(density_data, f, indent=2)
        
    print(f"Successfully harvested mock data to {filepath}")

if __name__ == "__main__":
    harvest_mock_tfl_data()
