import os

# Define base path in Colab
base_path = "/content/drive/MyDrive/Semester 3/Vision and Language/Project SOTA/ILM-VP"

# Adjust paths for Colab environment
data_path = os.path.join(base_path, "datasets")
results_path = os.path.join(base_path, "projects", "ILM-VP", "results")

# Create the directories if they don't exist
os.makedirs(data_path, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

print(f"Data Path: {data_path}")
print(f"Results Path: {results_path}")
