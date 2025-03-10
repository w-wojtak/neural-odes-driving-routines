import pandas as pd
import hdbscan
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os


# Load dataset
data_dir = "Datasets"
file_name = "DL_HMI_01.xlsx"  # Choose a dataset for one vehicle
file_path = os.path.join(data_dir, file_name)

df = pd.read_excel(file_path)

# Count the number of unique values in the "Local" column
num_unique_local = df['Local'].nunique()

# Print the number of unique strings
print(f"Number of unique locations in the 'Local' column: {num_unique_local}")

# # Count unique values and their occurrences in the "Local" column
local_counts = df['Local'].value_counts()

# # Print the unique values along with their counts
print(local_counts)

# Extract Latitude and Longitude
coords = df[['Latitude', 'Longitude']].values

# Apply HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    # min_cluster_size=8,  # Increase to merge small clusters
    min_samples=4,  # Helps merge close points
    cluster_selection_epsilon=0.0005,  # Allows merging nearby clusters
    metric='euclidean'
)

cluster_labels = clusterer.fit_predict(coords)  # Convert to radians for geographic distances

# Assign cluster labels to the dataframe
df['POI'] = cluster_labels

# Print the number of unique POIs (excluding noise points labeled as -1)
num_pois = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
print(f"Number of POIs detected: {num_pois}")

# Save the processed dataframe (so we don't need to cluster again)
processed_file_name = "DL_HMI_01_with_POI.csv"  
processed_file_path = os.path.join(data_dir, processed_file_name)

df.to_csv(processed_file_path, index=False)  # Save as CSV

print(f"Processed data with POIs saved to: {processed_file_path}")

# Create a map centered around the mean coordinates
map_center = [df['Latitude'].mean(), df['Longitude'].mean()]
poi_map = folium.Map(location=map_center, zoom_start=12)

# Create a marker cluster layer
marker_cluster = MarkerCluster().add_to(poi_map)

# Define colors for different POIs
poi_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige',
              'darkblue', 'darkgreen', 'cadetblue', 'darkpurple']

# Add markers for each POI
for _, row in df.iterrows():
    poi_label = row['POI']
    color = poi_colors[poi_label % len(poi_colors)] if poi_label != -1 else 'gray'  # Noise points are gray
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"POI: {poi_label}",
        icon=folium.Icon(color=color)
    ).add_to(poi_map)  # Don't add to MarkerCluster()

# Save the map to an HTML file
poi_map.save("poi_map.html")

# Display the first few rows of the dataframe with POI labels
print(df.head())

