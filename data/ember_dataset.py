import os
import pickle
import numpy as np
from data.dataset import MalwareDataset


class EmberDataset(MalwareDataset):
    """
    Dataset loader for the EMBER regional malware data.
    Extends the base MalwareDataset class to maintain compatibility.
    """

    def __init__(self, regions=None):
        """
        Initialize the EMBER dataset.

        Args:
            regions: List of regions to load (default: ["US", "JP", "EU", "benign"])
        """
        # Default regions if none provided
        if regions is None:
            regions = ["US", "JP", "EU"]

        # Add benign as a special case
        if "benign" not in regions:
            self._actual_regions = regions.copy()
            regions = regions + ["benign"]
        else:
            self._actual_regions = [r for r in regions if r != "benign"]

        super().__init__(regions=regions)

        # Initialize data containers
        self.X = {"train": {}, "test": {}}
        self.y = {"train": {}, "test": {}}
        self.feature_names = None
        self.is_loaded = False

    def load_data(self, data_dir, test_size=0.2, random_state=42, max_samples=None):
        """
        Load EMBER data from pickle files for each region and mix benign samples with each region.

        Args:
            data_dir: Directory containing the EMBER data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            max_samples: Maximum number of samples to use per region (limits both malware and benign)

        Returns:
            self
        """
        # Handle relative paths when given with leading slash
        if data_dir.startswith('/') and not os.path.exists(data_dir):
            relative_path = data_dir[1:] if data_dir.startswith('/') else data_dir
            if os.path.exists(relative_path):
                data_dir = relative_path
                print(f"Note: Using relative path instead: {data_dir}")

        print(f"Loading EMBER data from {data_dir}...")
        if max_samples:
            print(f"Sample limit applied: using max {max_samples} samples per region (malware + benign combined)")

        # Track feature counts for information
        feature_counts = {}

        # First, load the benign data
        benign_data = None
        try:
            benign_dir = os.path.join(data_dir, "benign")

            # Load benign features (X)
            x_benign_path = os.path.join(benign_dir, "X_train.pkl")
            if os.path.exists(x_benign_path):
                with open(x_benign_path, 'rb') as f:
                    X_benign_full = pickle.load(f)

                    # Limit benign samples if needed - we'll use half of max_samples
                    # since we want to balance malware and benign
                    if max_samples and len(X_benign_full) > (max_samples // 2):
                        from sklearn.utils import resample
                        benign_limit = max_samples // 2
                        print(f"  Limiting benign samples from {len(X_benign_full)} to {benign_limit}")
                        indices = resample(
                            np.arange(len(X_benign_full)),
                            n_samples=benign_limit,
                            replace=False,
                            random_state=random_state
                        )
                        X_benign = X_benign_full[indices]
                    else:
                        X_benign = X_benign_full
            else:
                print(f"Warning: X_train.pkl not found in {benign_dir}")
                X_benign = None

            # Load benign labels (y)
            y_benign_path = os.path.join(benign_dir, "y_train.pkl")
            if os.path.exists(y_benign_path):
                with open(y_benign_path, 'rb') as f:
                    y_benign_full = pickle.load(f)

                    # Apply the same limit to labels
                    if max_samples and X_benign is not None and len(X_benign) < len(y_benign_full):
                        if hasattr(X_benign, 'indices'):
                            indices = X_benign.indices
                            y_benign = y_benign_full[indices]
                        else:
                            # Just take the first n labels corresponding to our filtered X data
                            y_benign = y_benign_full[:len(X_benign)]
                    else:
                        y_benign = y_benign_full
            else:
                print(f"Warning: y_train.pkl not found in {benign_dir}")
                y_benign = None

            if X_benign is not None and y_benign is not None:
                # Make sure benign labels are 0
                if np.mean(y_benign) > 0.5:  # If most labels are 1, invert them
                    print("  Inverting benign labels to ensure they are 0")
                    y_benign = 1 - y_benign

                benign_data = (X_benign, y_benign)
                print(f"  Loaded {len(X_benign)} benign samples ({X_benign.shape[1]} features)")

        except Exception as e:
            print(f"Error loading benign data: {e}")

        # Now load each malware region and mix with benign data
        malware_regions = [r for r in self.regions if r != "benign"]
        for region in malware_regions:
            try:
                region_dir = os.path.join(data_dir, f"region_{region}")

                # Load features (X)
                x_train_path = os.path.join(region_dir, "X_train.pkl")
                if os.path.exists(x_train_path):
                    with open(x_train_path, 'rb') as f:
                        X_malware_full = pickle.load(f)

                        # Apply limit for malware samples - we'll use half of max_samples
                        if max_samples and len(X_malware_full) > (max_samples // 2):
                            from sklearn.utils import resample
                            malware_limit = max_samples // 2
                            print(f"  Limiting {region} malware samples from {len(X_malware_full)} to {malware_limit}")
                            indices = resample(
                                np.arange(len(X_malware_full)),
                                n_samples=malware_limit,
                                replace=False,
                                random_state=random_state
                            )
                            X_malware = X_malware_full[indices]
                        else:
                            X_malware = X_malware_full
                else:
                    print(f"Warning: X_train.pkl not found in {region_dir}")
                    continue

                # Load labels (y)
                y_train_path = os.path.join(region_dir, "y_train.pkl")
                if os.path.exists(y_train_path):
                    with open(y_train_path, 'rb') as f:
                        y_malware_full = pickle.load(f)

                        # Apply the same limit to labels
                        if max_samples and len(X_malware) < len(y_malware_full):
                            if hasattr(X_malware, 'indices'):
                                indices = X_malware.indices
                                y_malware = y_malware_full[indices]
                            else:
                                # Just take the first n labels corresponding to our filtered X data
                                y_malware = y_malware_full[:len(X_malware)]
                        else:
                            y_malware = y_malware_full
                else:
                    print(f"Warning: y_train.pkl not found in {region_dir}")
                    continue

                # Make sure malware labels are 1
                if np.mean(y_malware) < 0.5:  # If most labels are 0, invert them
                    print(f"  Inverting {region} labels to ensure malware is 1")
                    y_malware = 1 - y_malware

                # If we have benign data, mix it with the malware data
                if benign_data is not None:
                    X_benign, y_benign = benign_data

                    # Determine how many benign samples to use (same as malware count)
                    # but don't exceed what we have available
                    n_malware = len(X_malware)
                    n_benign = min(n_malware, len(X_benign))

                    # Randomly sample the benign data to match malware count
                    if n_benign < len(X_benign):
                        from sklearn.utils import resample
                        indices = resample(
                            np.arange(len(X_benign)),
                            n_samples=n_benign,
                            replace=False,
                            random_state=random_state
                        )
                        X_benign_sampled = X_benign[indices]
                        y_benign_sampled = y_benign[indices]
                    else:
                        X_benign_sampled = X_benign
                        y_benign_sampled = y_benign

                    # Combine malware and benign data
                    X = np.vstack([X_malware, X_benign_sampled])
                    y = np.concatenate([y_malware, y_benign_sampled])

                    print(
                        f"  Combined {region} data: {len(X_malware)} malware + {len(X_benign_sampled)} benign = {len(X)} total samples")
                else:
                    X = X_malware
                    y = y_malware
                    print(f"  No benign data available. Using only {len(X)} malware samples for {region}.")

                # Use a smaller test set when applying sample limits
                test_samples = min(2000, int(len(X) * test_size)) if max_samples else int(len(X) * test_size)

                # Split into train/test sets
                from sklearn.model_selection import train_test_split
                from sklearn.utils import shuffle

                # Shuffle to mix malware and benign
                X, y = shuffle(X, y, random_state=random_state)

                # Use specific test sample counts when max_samples is set
                if max_samples:
                    # Test set size shouldn't be more than 2,000 samples when limiting
                    test_size_actual = test_samples / len(X)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_actual, random_state=random_state, stratify=y
                    )
                    print(f"  Using smaller test set: {len(X_test)} samples (instead of {int(len(X) * test_size)})")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y
                    )

                # Store the data
                self.X["train"][region] = X_train
                self.X["test"][region] = X_test
                self.y["train"][region] = y_train
                self.y["test"][region] = y_test

                # Track feature count
                feature_counts[region] = X.shape[1]

                print(f"  - Training: {len(X_train)} samples, positive rate: {np.mean(y_train):.2f}")
                print(f"  - Testing: {len(X_test)} samples, positive rate: {np.mean(y_test):.2f}")

            except Exception as e:
                print(f"Error loading {region} data: {e}")

        # Also store the original benign data for reference if needed
        if benign_data is not None:
            X_benign, y_benign = benign_data
            from sklearn.model_selection import train_test_split

            # Use smaller test set for benign too when limiting
            if max_samples:
                test_samples = min(2000, int(len(X_benign) * test_size))
                test_size_actual = test_samples / len(X_benign)
                X_train_benign, X_test_benign, y_train_benign, y_test_benign = train_test_split(
                    X_benign, y_benign, test_size=test_size_actual, random_state=random_state, stratify=y_benign
                )
            else:
                X_train_benign, X_test_benign, y_train_benign, y_test_benign = train_test_split(
                    X_benign, y_benign, test_size=test_size, random_state=random_state, stratify=y_benign
                )

            self.X["train"]["benign"] = X_train_benign
            self.X["test"]["benign"] = X_test_benign
            self.y["train"]["benign"] = y_train_benign
            self.y["test"]["benign"] = y_test_benign

            print(
                f"  Also stored original benign data for reference ({len(X_train_benign)} train, {len(X_test_benign)} test)")

        # Check if we loaded any data
        if not self.X["train"]:
            raise ValueError(f"No valid data found in {data_dir}")

        # Set feature names (placeholder)
        # If feature names are available in a metadata file, load them here
        first_region = next(iter(self.X["train"].keys()))
        n_features = self.X["train"][first_region].shape[1]
        self.feature_names = [f"ember_feature_{i}" for i in range(n_features)]

        # Check if all regions have the same number of features
        if len(set(feature_counts.values())) > 1:
            print("Warning: Different regions have different feature counts:")
            for region, count in feature_counts.items():
                print(f"  - {region}: {count} features")

        self.is_loaded = True
        print("EMBER data loaded successfully.")

        return self

    def combine_regions(self, regions=None, subset="train"):
        """
        Combine data from multiple regions.

        Args:
            regions: List of regions to combine (default: all)
            subset: 'train' or 'test'

        Returns:
            tuple: (X, y) combined features and labels
        """
        if regions is None:
            regions = self.regions

        if not all(r in self.regions for r in regions):
            missing = [r for r in regions if r not in self.regions]
            raise ValueError(f"Regions not found: {missing}")

        X_combined = []
        y_combined = []

        for region in regions:
            if region in self.X[subset]:
                X_combined.append(self.X[subset][region])
                y_combined.append(self.y[subset][region])
            else:
                print(f"Warning: No {subset} data for region {region}")

        if not X_combined:
            raise ValueError(f"No {subset} data found for regions {regions}")

        return np.vstack(X_combined), np.concatenate(y_combined)

    def get_feature_matrix(self, region, subset="train"):
        """
        Get feature matrix for a specific region.

        Args:
            region: Region name
            subset: 'train' or 'test'

        Returns:
            Feature matrix
        """
        if region not in self.X[subset]:
            raise ValueError(f"No {subset} data for region {region}")

        return self.X[subset][region]

    def get_target_vector(self, region, subset="train"):
        """
        Get target vector for a specific region.

        Args:
            region: Region name
            subset: 'train' or 'test'

        Returns:
            Target vector
        """
        if region not in self.y[subset]:
            raise ValueError(f"No {subset} data for region {region}")

        return self.y[subset][region]