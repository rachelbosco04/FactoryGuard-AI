import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(r"C:\Users\Josy Mol\OneDrive\Desktop\FactoryGuard-AI\data\processed\cleaned_data.csv")

# Generate dummy data
n_samples = 100
n_features = 10
data = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f"feature_{i}" for i in range(n_features)])
data["failure"] = np.random.randint(0, 2, size=n_samples)

# Save CSV
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(DATA_PATH, index=False)
print(f"Dummy cleaned_data.csv created at {DATA_PATH}")