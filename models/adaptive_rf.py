import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score


class HeterogeneousRandomForest(BaseEstimator, ClassifierMixin):
    """
    Heterogeneous Random Forest with varying feature subset sizes per tree.
    Implementation based on the paper's approach for model distillation.
    """

    def _initialize_model(self):
        """Create appropriate model based on available hardware"""

        try:
            print(f"[MPS_DEBUG] Initializing model with backend: {self.backend}")

            # Debug print for parameters
            print(f"[DEBUG] model_params: {self.model_params}")

            # Ensure n_estimators is explicitly extracted from model_params
            n_estimators = self.model_params.get('n_estimators', 100)
            print(f"[DEBUG] Using n_estimators={n_estimators} for HeterogeneousRandomForest")

            # Import your HeterogeneousRandomForest
            try:
                from models.adaptive_rf import HeterogeneousRandomForest
                print(f"[MPS_DEBUG] Successfully imported HeterogeneousRandomForest")
            except ImportError as e:
                print(f"[MPS_DEBUG] ERROR importing HeterogeneousRandomForest: {str(e)}")
                raise

            if self.backend == "cuda":
                try:
                    # Try to import CUDA libraries
                    import cupy
                    import cuml
                    print(f"[MPS_DEBUG] CUDA libraries available: cupy={cupy.__version__}, cuml available")

                    # We could implement a CUDA optimized version here
                    # For now, just use the CPU version with a warning
                    print("[MPS_DEBUG] CUDA-optimized HeterogeneousRandomForest not yet implemented")
                    print("[MPS_DEBUG] Using CPU implementation with optimal threading")

                    self.model = HeterogeneousRandomForest(
                        n_estimators=n_estimators,  # Use the explicit variable
                        max_features=self.model_params.get("max_features", 'auto'),
                        min_features=self.min_features,
                        max_features_step=self.max_features_step,
                        random_state=self.model_params.get("random_state", None),
                        n_jobs=self.model_params.get("n_jobs", -1),
                        verbose=self.model_params.get("verbose", 0)
                    )
                    # Explicitly verify/set n_estimators after initialization
                    if self.model.n_estimators != n_estimators:
                        print(f"[DEBUG] Correcting n_estimators from {self.model.n_estimators} to {n_estimators}")
                        self.model.n_estimators = n_estimators

                except ImportError as e:
                    print(f"[MPS_DEBUG] CUDA libraries not available: {str(e)}")
                    print("[MPS_DEBUG] Falling back to CPU implementation")
                    # Fall back to CPU implementation
                    self.model = HeterogeneousRandomForest(
                        n_estimators=n_estimators,  # Use the explicit variable
                        max_features=self.model_params.get("max_features", 'auto'),
                        min_features=self.min_features,
                        max_features_step=self.max_features_step,
                        random_state=self.model_params.get("random_state", None),
                        n_jobs=self.model_params.get("n_jobs", -1),
                        verbose=self.model_params.get("verbose", 0)
                    )
                    # Explicitly verify/set n_estimators after initialization
                    if self.model.n_estimators != n_estimators:
                        print(f"[DEBUG] Correcting n_estimators from {self.model.n_estimators} to {n_estimators}")
                        self.model.n_estimators = n_estimators

            elif self.backend == "mps":
                print("[MPS_DEBUG] Using MPS (Apple Silicon) optimizations")
                print("[MPS_DEBUG] Attempting to create HeterogeneousRandomForest with optimal threading")

                # Use CPU implementation with threading optimized for the platform
                self.model = HeterogeneousRandomForest(
                    n_estimators=n_estimators,  # Use the explicit variable
                    max_features=self.model_params.get("max_features", 'auto'),
                    min_features=self.min_features,
                    max_features_step=self.max_features_step,
                    random_state=self.model_params.get("random_state", None),
                    n_jobs=self.model_params.get("n_jobs", -1),
                    verbose=self.model_params.get("verbose", 0)
                )
                # Explicitly verify/set n_estimators after initialization
                if self.model.n_estimators != n_estimators:
                    print(f"[DEBUG] Correcting n_estimators from {self.model.n_estimators} to {n_estimators}")
                    self.model.n_estimators = n_estimators

                print("[MPS_DEBUG] HeterogeneousRandomForest created successfully")
            else:
                print(f"[MPS_DEBUG] Using CPU implementation with {self.model_params.get('n_jobs', -1)} threads")
                # Use CPU implementation with threading optimized for the platform
                self.model = HeterogeneousRandomForest(
                    n_estimators=n_estimators,  # Use the explicit variable
                    max_features=self.model_params.get("max_features", 'auto'),
                    min_features=self.min_features,
                    max_features_step=self.max_features_step,
                    random_state=self.model_params.get("random_state", None),
                    n_jobs=self.model_params.get("n_jobs", -1),
                    verbose=self.model_params.get("verbose", 0)
                )
                # Explicitly verify/set n_estimators after initialization
                if self.model.n_estimators != n_estimators:
                    print(f"[DEBUG] Correcting n_estimators from {self.model.n_estimators} to {n_estimators}")
                    self.model.n_estimators = n_estimators

        except Exception as e:
            print(f"[MPS_DEBUG] CRITICAL ERROR initializing model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def __init__(self, n_estimators=100, max_features='auto', min_features=None,
                 max_features_step=None, random_state=None, n_jobs=-1, verbose=0):
        """
        Initialize the heterogeneous random forest.

        Args:
            n_estimators: Number of trees in the forest
            max_features: Maximum number of features to consider
            min_features: Minimum number of features to consider
            max_features_step: Step size for feature count progression
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all)
            verbose: Controls the verbosity of the tree building process (0=silent)
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_features = min_features
        self.max_features_step = max_features_step
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.estimators_ = []
        self.feature_subset_sizes_ = []

        # print(f"[RF_DEBUG] init with n_estimators={self.n_estimators}")
        # # Add this to see where the second instance is being created
        # if n_estimators == 100:  # Only print stack trace for the default value
        #     import traceback
        #     print("[RF_DEBUG] Default n_estimators=100 used. Stack trace:")
        #     traceback.print_stack()

    def fit(self, X, y, warm_start=False, n_trees_to_add=None):
        """
        Fit the random forest with heterogeneous tree configurations.

        Args:
            X: Feature matrix
            y: Target vector
            warm_start: Whether to build on existing trees
            n_trees_to_add: Number of new trees to add (only used with warm_start)

        Returns:
            self
        """
        from sklearn.utils import check_random_state
        import time

        print(f"[RF_DEBUG] Starting HeterogeneousRandomForest fit with {X.shape[0]} samples, {X.shape[1]} features")
        print(f"[RF_DEBUG] Using n_estimators={self.n_estimators}")

        # Initialize trees if not warm starting or no existing trees
        if not warm_start or not hasattr(self, 'estimators_') or self.estimators_ is None:
            print(f"[RF_DEBUG] Initializing new forest with {self.n_estimators} trees")
            self.estimators_ = []
            self.feature_subset_sizes_ = []
            n_trees_to_build = self.n_estimators
            start_idx = 0
        else:
            # If warm starting, calculate how many trees to add
            if n_trees_to_add is not None:
                n_trees_to_build = n_trees_to_add
            else:
                n_trees_to_build = max(0, self.n_estimators - len(self.estimators_))
            start_idx = len(self.estimators_)
            print(f"[RF_DEBUG] Warm start - adding {n_trees_to_build} trees to existing {start_idx} trees")

        if n_trees_to_build <= 0:
            # No trees to build, return early
            print(f"[RF_DEBUG] No trees to build, returning early")
            return self

        # Generate feature subset sizes if needed
        if not hasattr(self, 'feature_subset_sizes_') or len(self.feature_subset_sizes_) < start_idx + n_trees_to_build:
            print(f"[RF_DEBUG] Generating feature subset sizes for {n_trees_to_build} trees")
            new_subset_sizes = self._generate_feature_subset_sizes(
                n_trees=n_trees_to_build,
                n_features=X.shape[1]
            )
            self.feature_subset_sizes_.extend(new_subset_sizes)
            print(f"[RF_DEBUG] Feature subset sizes: {new_subset_sizes[:5]}... (truncated)")

        # Get random state
        random_state = check_random_state(self.random_state)

        # Build each tree with its specific feature subset size
        from sklearn.tree import DecisionTreeClassifier

        print(f"[RF_DEBUG] Starting tree building process for {n_trees_to_build} trees")
        total_start_time = time.time()
        progress_interval = max(1, n_trees_to_build // 10)  # Report every 10%

        try:
            # For each new tree
            for i in range(start_idx, start_idx + n_trees_to_build):
                tree_start_time = time.time()

                if i % progress_interval == 0 or self.verbose:
                    progress_pct = (i - start_idx + 1) / n_trees_to_build * 100
                    print(f"[RF_DEBUG] Building tree {i + 1}/{start_idx + n_trees_to_build} ({progress_pct:.1f}%)...")

                # Bootstrap sample for this tree
                try:
                    indices = random_state.randint(0, X.shape[0], X.shape[0])
                    X_bootstrap = X[indices]
                    y_bootstrap = y[indices]
                except Exception as e:
                    print(f"[RF_DEBUG] ERROR during bootstrapping: {str(e)}")
                    raise

                # Get max_features for this tree
                tree_max_features = min(self.feature_subset_sizes_[i], X.shape[1])

                # Create and train the tree
                try:
                    tree = DecisionTreeClassifier(
                        max_features=tree_max_features,
                        max_depth=None,
                        random_state=random_state.randint(0, 1000000)
                    )

                    # Fit the tree
                    tree.fit(X_bootstrap, y_bootstrap)

                    # Store the tree
                    self.estimators_.append(tree)

                    tree_time = time.time() - tree_start_time
                    if i % progress_interval == 0 or self.verbose:
                        print(
                            f"[RF_DEBUG] Tree {i + 1} built in {tree_time:.2f} seconds with {tree_max_features} features")

                except Exception as e:
                    print(f"[RF_DEBUG] ERROR building tree {i + 1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise

            total_time = time.time() - total_start_time
            print(f"[RF_DEBUG] Forest building completed in {total_time:.2f} seconds ({n_trees_to_build} trees)")

            return self

        except Exception as e:
            print(f"[RF_DEBUG] CRITICAL ERROR during forest building: {str(e)}")
            import traceback
            traceback.print_exc()
            # Save any progress made
            print(f"[RF_DEBUG] Partial forest has {len(self.estimators_)} trees")
            raise

    def _generate_feature_subset_sizes(self, n_trees, n_features):
        """
        Generate feature subset sizes for each tree to create a heterogeneous forest.

        Args:
            n_trees: Number of trees for which to generate subset sizes
            n_features: Total number of features in the dataset

        Returns:
            List of feature subset sizes for each tree
        """
        # Calculate range of feature subset sizes
        if self.min_features is None:
            self.min_features = max(1, int(n_features * 0.1))  # Default to 10% of features

        max_features = self.max_features
        if max_features == 'auto' or max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(max_features, float) and max_features <= 1.0:
            max_features = int(max_features * n_features)

        # Ensure max_features is within bounds
        max_features = min(max_features, n_features)

        # Calculate range of feature subset sizes
        feature_range = max_features - self.min_features

        if feature_range <= 0:
            # If min_features >= max_features, use uniform feature subset sizes
            return [self.min_features] * n_trees

        # Calculate distribution of feature subset sizes
        subset_sizes = []

        # Linear progression from min to max
        for i in range(n_trees):
            # Calculate proportion along the range (0 to 1)
            prop = i / (n_trees - 1) if n_trees > 1 else 0

            # Calculate feature subset size
            size = self.min_features + int(prop * feature_range)

            # Ensure size is between min and max
            size = max(self.min_features, min(size, max_features))

            subset_sizes.append(size)

        return subset_sizes

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        check_is_fitted(self, ['estimators_', 'feature_subset_sizes_'])
        X = check_array(X)

        # Get predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.estimators_])

        # Majority vote
        y_pred = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=predictions.astype(int)
        )

        return y_pred

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X: Feature matrix

        Returns:
            Predicted class probabilities
        """
        check_is_fitted(self, ['estimators_', 'feature_subset_sizes_'])
        X = check_array(X)

        # Get probability predictions from each tree
        probas = [tree.predict_proba(X) for tree in self.estimators_]

        # Average probabilities
        y_proba = np.mean(probas, axis=0)

        return y_proba

    def distill(self, n_estimators=None, max_features=None):
        """
        Distill the forest to a smaller one with fewer trees/features.

        Args:
            n_estimators: Number of estimators in distilled model
            max_features: Maximum number of features in distilled model

        Returns:
            Distilled heterogeneous random forest
        """
        check_is_fitted(self, ['estimators_', 'feature_subset_sizes_'])

        if n_estimators is None:
            n_estimators = len(self.estimators_) // 2

        if max_features is None:
            max_features = self.max_features

        # Create new forest with specified parameters
        distilled_forest = HeterogeneousRandomForest(
            n_estimators=n_estimators,
            max_features=max_features,
            min_features=self.min_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        # Copy fitted estimators with fewest features
        subset_sizes_with_indices = [(size, i) for i, size in enumerate(self.feature_subset_sizes_)]
        subset_sizes_with_indices.sort()  # Sort by feature subset size

        # Select trees with fewest features
        selected_indices = [idx for _, idx in subset_sizes_with_indices[:n_estimators]]

        distilled_forest.estimators_ = [self.estimators_[i] for i in selected_indices]
        distilled_forest.feature_subset_sizes_ = [self.feature_subset_sizes_[i] for i in selected_indices]

        return distilled_forest

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'min_features': self.min_features,
            'max_features_step': self.max_features_step,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class AdaptiveRandomForest(BaseEstimator, ClassifierMixin):
    """
    Adaptive Random Forest implementation based on the paper's approach.
    This implementation includes:
    1. Concept drift detection
    2. Model updating mechanism
    3. Support for heterogeneous trees
    """

    def __init__(self, n_estimators=100, max_features='auto', min_features=None,
                 warning_delta=0.005, drift_delta=0.01, random_state=None, n_jobs=-1):
        """
        Initialize the adaptive random forest.

        Args:
            n_estimators: Number of trees in the forest
            max_features: Maximum number of features to consider
            min_features: Minimum number of features to consider
            warning_delta: Threshold for drift warning
            drift_delta: Threshold for confirmed drift
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all)
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_features = min_features
        self.warning_delta = warning_delta
        self.drift_delta = drift_delta
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize base forest
        self.forest = HeterogeneousRandomForest(
            n_estimators=n_estimators,
            max_features=max_features,
            min_features=min_features,
            random_state=random_state,
            n_jobs=n_jobs
        )

        # Initialize drift detection variables
        self.drift_detector = None
        self.background_forest = None
        self.warning_zone = False
        self.in_concept_drift = False

        # Metrics tracking
        self.error_rate_history = []
        self.drift_points = []

    def fit(self, X, y):
        """
        Fit the adaptive random forest.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        X, y = check_X_y(X, y)

        # Initialize EDDM drift detector
        self.drift_detector = EDDM(
            warning_level=self.warning_delta,
            drift_level=self.drift_delta
        )

        # Fit base forest
        self.forest.fit(X, y)

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit the model on new data.

        Args:
            X: Feature matrix
            y: Target vector
            classes: Array of class labels

        Returns:
            self
        """
        X, y = check_X_y(X, y)

        # If not fitted, fit the model
        if not hasattr(self, 'forest_') or self.forest_ is None:
            self.fit(X, y)
            return self

        # Make predictions
        y_pred = self.predict(X)

        # Calculate error rate
        error_rate = 1 - accuracy_score(y, y_pred)
        self.error_rate_history.append(error_rate)

        # Update drift detector
        drift_level = self.drift_detector.update(error_rate)

        if drift_level == 2:  # Drift detected
            self.in_concept_drift = True
            self.drift_points.append(len(self.error_rate_history) - 1)

            # Replace forest with background forest if available
            if self.background_forest is not None:
                self.forest = self.background_forest
                self.background_forest = None
            else:
                # Create new forest
                self.forest = HeterogeneousRandomForest(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    min_features=self.min_features,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
                self.forest.fit(X, y)

            # Reset warning zone
            self.warning_zone = False

        elif drift_level == 1:  # Warning zone
            # If not already in warning zone, create background forest
            if not self.warning_zone:
                self.warning_zone = True
                self.background_forest = HeterogeneousRandomForest(
                    n_estimators=self.n_estimators,
                    max_features=self.max_features,
                    min_features=self.min_features,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )

            # Update background forest
            if self.background_forest is not None:
                self.background_forest.fit(X, y)
        else:
            # If in normal zone, incrementally update forest
            # This would be a full implementation of incremental learning
            # For simplicity, we'll just update part of the forest
            n_trees_to_update = max(1, self.n_estimators // 10)
            indices_to_update = np.random.choice(
                len(self.forest.estimators_),
                n_trees_to_update,
                replace=False
            )

            for idx in indices_to_update:
                tree = self.forest.estimators_[idx]
                # Update tree (in a real implementation, this would use proper incremental learning)
                tree.fit(X, y)

        return self

    def predict(self, X):
        """
        Predict class labels for X.

        Args:
            X: Feature matrix

        Returns:
            Predicted class labels
        """
        return self.forest.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Args:
            X: Feature matrix

        Returns:
            Predicted class probabilities
        """
        return self.forest.predict_proba(X)

    def distill(self, n_estimators=None, max_features=None):
        """
        Distill the forest to a smaller one.

        Args:
            n_estimators: Number of estimators in distilled model
            max_features: Maximum number of features in distilled model

        Returns:
            Distilled heterogeneous random forest
        """
        return self.forest.distill(n_estimators, max_features)


class EDDM:
    """
    Early Drift Detection Method (EDDM) implementation.
    Based on the paper: Early Drift Detection Method by Baena-García et al.
    """

    def __init__(self, warning_level=0.95, drift_level=0.90):
        """
        Initialize EDDM.

        Args:
            warning_level: Threshold for warning level
            drift_level: Threshold for drift level
        """
        self.warning_level = warning_level
        self.drift_level = drift_level

        self.m_n = 0  # Number of errors
        self.m_p = 0  # Current probability
        self.m_s = 0  # Current standard deviation
        self.m_pmax = 0  # Maximum probability
        self.m_smax = 0  # Maximum standard deviation

        self.m_lastError = 0  # Distance from last error

    def update(self, error):
        """
        Update the drift detector with a new error rate.

        Args:
            error: Error value (0 for correct, 1 for incorrect prediction)

        Returns:
            0: No change, 1: Warning, 2: Drift
        """
        if error == 0:  # If prediction is correct
            self.m_lastError += 1
            return 0

        # Distance between errors
        distance = self.m_lastError
        self.m_lastError = 0
        self.m_n += 1

        # Initialize metrics
        if self.m_n < 2:
            return 0

        # Calculate p (mean distance between errors)
        if self.m_n == 2:
            self.m_p = distance
            self.m_s = 0
            self.m_pmax = distance
            self.m_smax = 0
            return 0

        # Incremental calculation of mean and standard deviation
        old_p = self.m_p
        old_s = self.m_s

        # Update mean
        self.m_p = old_p + (distance - old_p) / self.m_n

        # Update standard deviation
        self.m_s = np.sqrt(
            old_s * old_s + (distance - old_p) * (distance - self.m_p)
        )

        # Update max values if needed
        if self.m_p + 2 * self.m_s > self.m_pmax + 2 * self.m_smax:
            self.m_pmax = self.m_p
            self.m_smax = self.m_s

        # Calculate drift metrics
        if self.m_smax > 0:
            value = (self.m_p + 2 * self.m_s) / (self.m_pmax + 2 * self.m_smax)
        else:
            value = 1.0

        # Check for drift or warning
        if value < self.drift_level:
            # Drift detected
            self.m_n = 0
            self.m_p = 0
            self.m_s = 0
            self.m_pmax = 0
            self.m_smax = 0
            return 2
        elif value < self.warning_level:
            # Warning zone
            return 1
        else:
            # No change
            return 0