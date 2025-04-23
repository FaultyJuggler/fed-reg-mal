import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from data.feature_extractor import PEFeatureExtractor


class MalwareDataset:
    """
    Dataset handler for malware classification.
    Handles the regional datasets (US, BR, JP) as in the paper.
    """

    def __init__(self, regions=None, max_features=1000, test_size=0.2, random_state=42):
        """
        Initialize the dataset handler.

        Args:
            regions: List of regions to include ('US', 'BR', 'JP') or None for all
            max_features: Maximum number of features to extract
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.regions = regions if regions is not None else ['US', 'BR', 'JP']
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state

        self.feature_extractor = PEFeatureExtractor(max_features=max_features)
        self.feature_names = None

        self.X = {}
        self.y = {}
        self.file_paths = {}

        for split in ['train', 'test']:
            self.X[split] = {}
            self.y[split] = {}
            self.file_paths[split] = {}

    def load_data(self, data_dir, load_cached=True, cache_dir='cached_features'):
        """
        Load malware and goodware samples for all regions.

        Args:
            data_dir: Directory containing the data
            load_cached: Whether to load cached features if available
            cache_dir: Directory to store cached features

        Returns:
            self
        """
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        # Load each region's data
        for region in self.regions:
            cache_path = os.path.join(cache_dir, f"{region}_features.npz")

            if load_cached and os.path.exists(cache_path):
                # Load cached features
                print(f"Loading cached features for {region}...")
                data = np.load(cache_path, allow_pickle=True)
                X = data['X']
                y = data['y']
                file_paths = data['file_paths']
            else:
                # Load and process files
                print(f"Processing files for {region}...")

                # Get file paths
                malware_dir = os.path.join(data_dir, region, 'malware')
                goodware_dir = os.path.join(data_dir, 'goodware')  # Shared goodware

                malware_paths = self._get_file_paths(malware_dir)
                goodware_paths = self._get_file_paths(goodware_dir)

                # Balance dataset by taking same number of goodware as malware
                goodware_paths = goodware_paths[:len(malware_paths)]

                # Combine file paths and create labels
                file_paths = malware_paths + goodware_paths
                y = np.array([1] * len(malware_paths) + [0] * len(goodware_paths))

                # Extract features
                print(f"Extracting features for {region} ({len(file_paths)} files)...")
                X = self.feature_extractor.fit_transform(file_paths)

                # Cache features
                print(f"Caching features for {region}...")
                np.savez_compressed(
                    cache_path,
                    X=X,
                    y=y,
                    file_paths=file_paths
                )

            # Store feature names
            if self.feature_names is None:
                self.feature_names = self.feature_extractor.get_feature_names()

            # Split data into train and test sets
            X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
                X, y, file_paths, test_size=self.test_size, random_state=self.random_state,
                stratify=y
            )

            # Store data
            self.X['train'][region] = X_train
            self.y['train'][region] = y_train
            self.file_paths['train'][region] = paths_train

            self.X['test'][region] = X_test
            self.y['test'][region] = y_test
            self.file_paths['test'][region] = paths_test

            print(f"Loaded {region} dataset: {len(y_train)} training samples, {len(y_test)} test samples")

        return self

    def _get_file_paths(self, directory):
        """Get all PE file paths from a directory."""
        file_paths = []

        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.exe', '.dll', '.cpl')):
                    file_paths.append(os.path.join(root, file))

        return file_paths

    def get_global_dataset(self, split='train'):
        """
        Get combined dataset from all regions.

        Args:
            split: Whether to return 'train' or 'test' data

        Returns:
            X, y for the combined dataset
        """
        X_combined = []
        y_combined = []

        for region in self.regions:
            X_combined.append(self.X[split][region])
            y_combined.append(self.y[split][region])

        X_global = np.vstack(X_combined)
        y_global = np.concatenate(y_combined)

        # Shuffle to mix regions
        X_global, y_global = shuffle(X_global, y_global, random_state=self.random_state)

        return X_global, y_global

    def get_time_ordered_dataset(self, region, timestamp_file=None):
        """
        Get time-ordered dataset for time-series analysis.

        Args:
            region: Region code ('US', 'BR', 'JP')
            timestamp_file: CSV file with 'file_path' and 'timestamp' columns

        Returns:
            X, y ordered by timestamp
        """
        if region not in self.regions:
            raise ValueError(f"Region {region} not loaded")

        if timestamp_file is None or not os.path.exists(timestamp_file):
            print("Warning: No timestamp file provided, using original order")
            return self.X['train'][region], self.y['train'][region]

        # Load timestamps
        timestamps_df = pd.read_csv(timestamp_file)
        timestamps_dict = dict(zip(timestamps_df['file_path'], timestamps_df['timestamp']))

        # Get file paths and data
        file_paths = self.file_paths['train'][region]
        X = self.X['train'][region]
        y = self.y['train'][region]

        # Get timestamps for each file
        file_timestamps = []
        for path in file_paths:
            # Get basename for matching with timestamp file
            basename = os.path.basename(path)
            timestamp = timestamps_dict.get(basename, 0)
            file_timestamps.append(timestamp)

        # Sort by timestamp
        sorted_indices = np.argsort(file_timestamps)

        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]

        return X_sorted, y_sorted

    def load_concept_drift_data(self, region, timestamp_file, n_chunks=6):
        """
        Load data for concept drift evaluation (time series).

        Args:
            region: Region code ('US', 'BR', 'JP')
            timestamp_file: CSV file with 'file_path' and 'timestamp' columns
            n_chunks: Number of time chunks to split the data into

        Returns:
            List of (X, y) tuples for each time chunk
        """
        # Get time-ordered data
        X, y = self.get_time_ordered_dataset(region, timestamp_file)

        # Split into chunks
        chunk_size = len(X) // n_chunks
        chunks = []

        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(X)

            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]

            chunks.append((X_chunk, y_chunk))

        return chunks