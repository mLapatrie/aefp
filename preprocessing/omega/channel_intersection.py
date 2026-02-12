import numpy as np
from pathlib import Path

# List all good_channels.npy
channel_files = sorted(Path(".").rglob("good_channels.npy"))

print(f"Found {len(channel_files)} channel files")

# Load them all
all_channels = []

for ch_file in channel_files:
    channels = np.load(ch_file, allow_pickle=True)
    channels = [ch.split('-')[0] for ch in channels]
    all_channels.append(set(channels))
    print(f"{ch_file}: {len(channels)} channels")

# Compute intersection
common_channels = sorted(set.intersection(*all_channels))

print(f"\nIntersection has {len(common_channels)} channels")

# Save intersection
np.save("common_channels.npy", np.array(common_channels))
print("Saved to common_channels.npy")
