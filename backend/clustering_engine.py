import numpy as np

from sklearn.cluster import DBSCAN

# ---------------------------------------------------
# MOCK HIGH-FDI COORDINATES
# ---------------------------------------------------

# Simulated suspicious debris points
# Format:
# [latitude, longitude]

coordinates = np.array([

    [18.10, 70.20],
    [18.11, 70.19],
    [18.12, 70.21],
    [18.13, 70.20],

    [22.40, 72.50],
    [22.41, 72.49],
    [22.39, 72.51],
    [22.42, 72.50],

    [15.00, 65.00]  # Noise point

])

print("\nInput Coordinate Points:")
print(coordinates)

# ---------------------------------------------------
# RUN DBSCAN CLUSTERING
# ---------------------------------------------------

db = DBSCAN(
    eps=0.02,
    min_samples=3
).fit(coordinates)

# ---------------------------------------------------
# CLUSTER LABELS
# ---------------------------------------------------

labels = db.labels_

print("\nCluster Labels:")
print(labels)

# ---------------------------------------------------
# UNIQUE CLUSTERS
# ---------------------------------------------------

unique_clusters = set(labels)

# Remove noise label (-1)

if -1 in unique_clusters:
    unique_clusters.remove(-1)

print("\nValid Clusters:")
print(unique_clusters)

# ---------------------------------------------------
# COMPUTE CLUSTER CENTERS
# ---------------------------------------------------

cluster_results = []

for cluster_id in unique_clusters:

    cluster_points = coordinates[
        labels == cluster_id
    ]

    center_lat, center_lon = np.mean(
        cluster_points,
        axis=0
    )

    cluster_results.append({

        "cluster_id": int(cluster_id),

        "center_lat": float(center_lat),

        "center_lon": float(center_lon),

        "points_in_cluster": len(cluster_points)

    })

# ---------------------------------------------------
# FINAL HOTSPOT OUTPUT
# ---------------------------------------------------

print("\nDetected Marine Debris Hotspots:\n")

for cluster in cluster_results:

    print(cluster)