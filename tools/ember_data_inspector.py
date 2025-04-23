import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter


def load_ember_region(data_dir, region):
    """Load data for a specific region."""
    if region == "benign":
        region_dir = os.path.join(data_dir, region)
    else:
        region_dir = os.path.join(data_dir, f"region_{region}")

    # Load features (X)
    x_path = os.path.join(region_dir, "X_train.pkl")
    if os.path.exists(x_path):
        with open(x_path, 'rb') as f:
            X = pickle.load(f)
    else:
        print(f"Error: X_train.pkl not found in {region_dir}")
        return None, None

    # Load labels (y)
    y_path = os.path.join(region_dir, "y_train.pkl")
    if os.path.exists(y_path):
        with open(y_path, 'rb') as f:
            y = pickle.load(f)
    else:
        print(f"Error: y_train.pkl not found in {region_dir}")
        return X, None

    return X, y


def inspect_dataset(data_dir, regions=None, output_dir=None):
    """Inspect and visualize EMBER dataset."""
    if regions is None:
        regions = ["US", "JP", "EU", "benign"]

    if output_dir is None:
        output_dir = "ember_analysis"

    os.makedirs(output_dir, exist_ok=True)

    # Load data from all regions
    data = {}
    for region in regions:
        X, y = load_ember_region(data_dir, region)
        if X is not None:
            data[region] = (X, y)
            print(f"Loaded {region} data: {X.shape[0]} samples, {X.shape[1]} features")
            if y is not None:
                class_counts = Counter(y)
                print(f"  Class distribution: {dict(class_counts)}")

    # Check for feature count consistency
    feature_counts = {region: X.shape[1] for region, (X, _) in data.items()}
    if len(set(feature_counts.values())) > 1:
        print("Warning: Inconsistent feature counts across regions:")
        for region, count in feature_counts.items():
            print(f"  - {region}: {count} features")
    else:
        print(f"Consistent feature count across all regions: {next(iter(feature_counts.values()))}")

    # Visualize class distributions
    plt.figure(figsize=(10, 6))
    for region, (_, y) in data.items():
        if y is not None:
            counts = Counter(y)
            plt.bar([f"{region}-0", f"{region}-1"], [counts[0], counts[1]], alpha=0.7)

    plt.title('Class Distribution by Region')
    plt.xlabel('Region-Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ember_class_distribution.png'))

    # Visualize feature distributions
    for region, (X, _) in data.items():
        plt.figure(figsize=(12, 6))

        # Boxplot of feature distributions (sample 50 features for clarity)
        n_features = min(50, X.shape[1])
        feature_indices = np.random.choice(X.shape[1], n_features, replace=False)

        plt.boxplot([X[:, i] for i in feature_indices])
        plt.title(f'Feature Distribution - {region} (Sample of {n_features} features)')
        plt.xlabel('Feature Index')
        plt.ylabel('Value')
        plt.xticks([])  # Hide x-ticks for clarity
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ember_feature_distribution_{region}.png'))

    print(f"Analysis complete. Visualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inspect EMBER dataset')
    parser.add_argument('--data-dir', type=str, default='./ember_data',
                        help='Directory containing EMBER data')
    parser.add_argument('--regions', type=str, nargs='+',
                        default=["US", "JP", "EU", "benign"],
                        help='Regions to inspect')
    parser.add_argument('--output-dir', type=str, default='ember_analysis',
                        help='Directory to save analysis results')

    args = parser.parse_args()
    inspect_dataset(args.data_dir, args.regions, args.output_dir)