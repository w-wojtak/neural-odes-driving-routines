import pandas as pd
import hdbscan
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# Path to the dataset directory
data_dir = "Datasets"

# List of dataset filenames
file_names = [f"DL_HMI_{str(i).zfill(2)}.xlsx" for i in range(1, 17)]

# Loop through all datasets
for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)

    # Check if the file exists before loading
    if not os.path.exists(file_path):
        print(f"❌ Dataset {file_name} not found. Skipping...")
        continue

    # Load dataset
    df = pd.read_excel(file_path)

    # Count the number of unique values in the "Local" column
    num_unique_local = df['Local'].nunique()

    # Print the number of unique locations
    print(f"📂 Dataset: {file_name}")
    print(f"📍 Number of unique locations in the 'Local' column: {num_unique_local}")

    # Extract Latitude and Longitude
    coords = df[['Latitude', 'Longitude']].dropna().values  # Drop NaN values if any

    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_samples=4,  # Helps merge close points
        cluster_selection_epsilon=0.0005,  # Allows merging nearby clusters
        metric='euclidean'
    )

    # Check if there are enough points for clustering
    if len(coords) > 4:
        cluster_labels = clusterer.fit_predict(coords)  # Clustering
        df['POI'] = cluster_labels

        # Print the number of unique POIs (excluding noise points labeled as -1)
        num_pois = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        print(f"📊 Number of POIs detected: {num_pois}\n")
    else:
        print("⚠️ Not enough points for clustering. Skipping POI calculation.\n")
