import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
import matplotlib.pyplot as plt


class FeatureSelector:
    """
    Feature selection for malware classification, implementing the paper's
    approach to finding optimal feature sets for different regions.
    """

    def __init__(self, selection_method='f_score', n_features=None):
        """
        Initialize the feature selector.

        Args:
            selection_method: Method for feature selection ('f_score', 'chi2', 'mutual_info')
            n_features: Number of features to select (default: None, selects automatically)
        """
        self.selection_method = selection_method
        self.n_features = n_features
        self.selector = None
        self.feature_names = None
        self.feature_importances_ = None

        # Choose scoring function based on method
        if selection_method == 'f_score':
            self.score_func = f_classif
        elif selection_method == 'chi2':
            self.score_func = chi2
        elif selection_method == 'mutual_info':
            self.score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")

    def fit(self, X, y, feature_names=None):
        """
        Fit the feature selector.

        Args:
            X: Feature matrix
            y: Target vector (binary)
            feature_names: Names of features (optional)

        Returns:
            self
        """
        # Initialize selector
        if self.n_features is None:
            # If n_features not specified, use all features initially
            self.selector = SelectKBest(self.score_func, k='all')
            self.selector.fit(X, y)

            # Find optimal number of features (elbow method)
            self.n_features = self._find_optimal_feature_count(self.selector.scores_, feature_names)

            # Re-fit with optimal number of features
            self.selector = SelectKBest(self.score_func, k=self.n_features)
            self.selector.fit(X, y)
        else:
            self.selector = SelectKBest(self.score_func, k=self.n_features)
            self.selector.fit(X, y)

        # Store feature importances and names
        self.feature_importances_ = self.selector.scores_

        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        return self

    def transform(self, X):
        """
        Transform X to use only selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.selector is None:
            raise ValueError("Feature selector must be fitted before transform")

        return self.selector.transform(X)

    def fit_transform(self, X, y, feature_names=None):
        """
        Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: Names of features (optional)

        Returns:
            Transformed feature matrix
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def _find_optimal_feature_count(self, scores, feature_names=None, pareto_threshold=0.8):
        """
        Find the optimal number of features using the elbow method.
        Based on the Pareto principle as described in the paper.

        Args:
            scores: Feature importance scores
            feature_names: Names of features (optional)
            pareto_threshold: Threshold for Pareto principle (default: 0.8)

        Returns:
            Optimal number of features
        """
        # Sort scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[sorted_indices]

        # Calculate cumulative sum
        cumulative_scores = np.cumsum(sorted_scores)
        normalized_scores = cumulative_scores / cumulative_scores[-1]

        # Find index where we reach pareto_threshold of the total importance
        pareto_idx = np.where(normalized_scores >= pareto_threshold)[0][0]

        # Add a margin to ensure we have enough features (as described in the paper)
        optimal_features = int(pareto_idx * 1.2)  # 20% margin

        # Ensure we don't exceed total number of features
        optimal_features = min(optimal_features, len(scores))

        # Visualize if feature names are provided
        if feature_names is not None and len(feature_names) == len(scores):
            self._visualize_feature_selection(
                sorted_indices,
                sorted_scores,
                normalized_scores,
                optimal_features,
                feature_names
            )

        return optimal_features

    def _visualize_feature_selection(self, sorted_indices, sorted_scores,
                                     normalized_scores, optimal_features, feature_names):
        """Visualize feature selection process."""
        plt.figure(figsize=(12, 8))

        # Plot cumulative importance
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(normalized_scores) + 1), normalized_scores, marker='o')
        plt.axvline(x=optimal_features, color='r', linestyle='--',
                    label=f'Optimal features: {optimal_features}')
        plt.axhline(y=normalized_scores[optimal_features - 1], color='g', linestyle='--',
                    label=f'Importance covered: {normalized_scores[optimal_features - 1]:.2f}')
        plt.xlabel('Number of features')
        plt.ylabel('Cumulative importance')
        plt.title('Cumulative Feature Importance')
        plt.legend()
        plt.grid(True)

        # Plot top features
        plt.subplot(2, 1, 2)
        top_n = min(20, len(sorted_indices))
        top_indices = sorted_indices[:top_n]
        top_scores = sorted_scores[:top_n]
        top_names = [feature_names[i] if i < len(feature_names) else f"Feature {i}"
                     for i in top_indices]

        plt.barh(range(top_n), top_scores)
        plt.yticks(range(top_n), top_names)
        plt.xlabel('Importance score')
        plt.title(f'Top {top_n} Features')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"feature_selection_{self.selection_method}.png")

    def get_top_features(self, n=None):
        """
        Get the top N features by importance.

        Args:
            n: Number of top features to return (default: self.n_features)

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature selector must be fitted before getting top features")

        if n is None:
            n = self.n_features

        n = min(n, len(self.feature_importances_))

        # Sort features by importance
        indices = np.argsort(self.feature_importances_)[::-1][:n]

        top_features = []
        for idx in indices:
            if self.feature_names is not None and idx < len(self.feature_names):
                feature_name = self.feature_names[idx]
            else:
                feature_name = f"Feature {idx}"

            top_features.append((feature_name, self.feature_importances_[idx]))

        return top_features

    def get_support(self):
        """Get support mask from the selector."""
        if self.selector is None:
            raise ValueError("Feature selector must be fitted before getting support")

        return self.selector.get_support()

    def get_feature_names_out(self):
        """Get names of selected features."""
        if self.selector is None:
            raise ValueError("Feature selector must be fitted before getting feature names")

        mask = self.selector.get_support()

        if self.feature_names is not None:
            return [self.feature_names[i] for i in range(len(mask)) if mask[i]]
        else:
            return [f"Feature {i}" for i in range(len(mask)) if mask[i]]