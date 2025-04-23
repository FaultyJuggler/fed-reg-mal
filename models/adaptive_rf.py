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

    def __init__(self, n_estimators=100, max_features='auto', min_features=None,
                 max_features_step=None, random_state=None, n_jobs=-1):
        """
        Initialize the heterogeneous random forest.

        Args:
            n_estimators: Number of trees in the forest
            max_features: Maximum number of features to consider
            min_features: Minimum number of features to consider
            max_features_step: Step size for feature count progression
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all)
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_features = min_features
        self.max_features_step = max_features_step
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.estimators_ = []
        self.feature_subset_sizes_ = []

    def fit(self, X, y):
        """
        Fit the heterogeneous random forest.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            self
        """
        X, y = check_X_y(X, y)

        # Set default min_features if not specified
        if self.min_features is None:
            self.min_features = int(np.sqrt(X.shape[1]))

        # Set default max_features if specified as string
        if isinstance(self.max_features, str):
            if self.max_features == 'auto' or self.max_features == 'sqrt':
                self.max_features = int(np.sqrt(X.shape[1]))
            elif self.max_features == 'log2':
                self.max_features = int(np.log2(X.shape[1]))
            else:
                raise ValueError(f"Invalid max_features: {self.max_features}")
        elif self.max_features is None:
            self.max_features = X.shape[1]

        # Ensure max_features is not greater than the number of features
        self.max_features = min(self.max_features, X.shape[1])

        # Set default max_features_step if not specified
        if self.max_features_step is None:
            self.max_features_step = max(1, (self.max_features - self.min_features) // self.n_estimators)

        # Generate feature subset sizes for each tree
        self.feature_subset_sizes_ = self._generate_feature_subset_sizes(X.shape[1])

        # Train each tree with its specific feature subset size
        for i, n_features in enumerate(self.feature_subset_sizes_):
            # Create tree with specific feature subset size
            tree = DecisionTreeClassifier(
                max_features=n_features,
                random_state=self.random_state + i if self.random_state else None
            )

            # Sample data with replacement (bootstrap)
            bootstrap_indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Fit tree
            tree.fit(X_bootstrap, y_bootstrap)

            # Store tree
            self.estimators_.append(tree)

        return self

    def _generate_feature_subset_sizes(self, n_total_features):
        """
        Generate feature subset sizes for each tree to create a heterogeneous forest.

        Args:
            n_total_features: Total number of features in the dataset

        Returns:
            List of feature subset sizes for each tree
        """
        # Calculate range of feature subset sizes
        feature_range = self.max_features - self.min_features

        if feature_range <= 0:
            # If min_features >= max_features, use uniform feature subset sizes
            return [self.min_features] * self.n_estimators

        # Calculate distribution of feature subset sizes
        subset_sizes = []

        # Linear progression from min to max
        for i in range(self.n_estimators):
            # Calculate proportion along the range (0 to 1)
            prop = i / (self.n_estimators - 1) if self.n_estimators > 1 else 0

            # Calculate feature subset size
            size = self.min_features + int(prop * feature_range)

            # Ensure size is between min and max
            size = max(self.min_features, min(size, self.max_features))

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
            'n_jobs': self.n_jobs
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