import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score

from models.adaptive_rf import HeterogeneousRandomForest


class ModelOptimizer:
    """
    Provides tools for optimizing model size and performance through pruning,
    feature reduction, and hyper-parameter tuning. Used in conjunction with
    the model distillation process to create efficient endpoint models.
    """

    def __init__(self, model=None):
        """
        Initialize the model optimizer.

        Args:
            model: The model to optimize (optional)
        """
        self.model = model
        self.original_size = self._get_model_size(model) if model is not None else None
        self.optimized_size = None

    def set_model(self, model):
        """
        Set the model to optimize.

        Args:
            model: The model to optimize

        Returns:
            self
        """
        self.model = model
        self.original_size = self._get_model_size(model)
        return self

    def _get_model_size(self, model):
        """
        Get the size of a model.

        Args:
            model: The model to check

        Returns:
            Dictionary of size metrics
        """
        if model is None:
            return None

        size_metrics = {}

        # Number of trees
        if hasattr(model, 'estimators_'):
            size_metrics['n_estimators'] = len(model.estimators_)

            # Number of nodes across all trees
            if hasattr(model.estimators_[0], 'tree_'):
                size_metrics['n_nodes'] = sum(
                    tree.tree_.node_count for tree in model.estimators_
                )

        # Feature subset sizes
        if hasattr(model, 'feature_subset_sizes_'):
            size_metrics['max_features'] = max(model.feature_subset_sizes_)
            size_metrics['min_features'] = min(model.feature_subset_sizes_)
            size_metrics['avg_features'] = np.mean(model.feature_subset_sizes_)

        return size_metrics

    def prune_low_importance_features(self, X, y, threshold=0.001):
        """
        Prune features with low importance.

        Args:
            X: Training features
            y: Training labels
            threshold: Importance threshold for keeping features

        Returns:
            Optimized model and feature mask
        """
        if self.model is None:
            raise ValueError("Model has not been set")

        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            # For models without built-in feature importance, use mutual information
            importances = mutual_info_classif(X, y, random_state=42)

        # Identify important features
        important_indices = np.where(importances > threshold)[0]
        feature_mask = np.zeros(X.shape[1], dtype=bool)
        feature_mask[important_indices] = True

        # Select important features
        X_important = X[:, feature_mask]

        # Train new model with selected features
        optimized_model = clone(self.model)
        optimized_model.fit(X_important, y)

        # Update size metrics
        self.optimized_size = self._get_model_size(optimized_model)

        return optimized_model, feature_mask

    def prune_tree_ensemble(self, X, y, accuracy_threshold=0.99):
        """
        Prune trees from the ensemble to reduce model size while maintaining accuracy.

        Args:
            X: Validation features
            y: Validation labels
            accuracy_threshold: Minimum acceptable accuracy (proportion of original)

        Returns:
            Optimized model
        """
        if self.model is None:
            raise ValueError("Model has not been set")

        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Model does not have an ensemble of estimators")

        # Get original accuracy
        original_accuracy = accuracy_score(y, self.model.predict(X))

        # Get feature importances for each tree (if available)
        tree_importances = []

        if hasattr(self.model, 'estimators_'):
            for tree in self.model.estimators_:
                if hasattr(tree, 'feature_importances_'):
                    # Average feature importance for this tree
                    tree_importances.append(np.mean(tree.feature_importances_))
                else:
                    # Default importance (will not be pruned)
                    tree_importances.append(1.0)

        # Sort trees by importance
        sorted_indices = np.argsort(tree_importances)[::-1]  # Descending order

        # Try pruning trees
        best_pruned_model = None
        best_accuracy = 0
        best_size = len(self.model.estimators_)

        for n_trees in range(1, len(sorted_indices) + 1):
            # Select top n_trees
            selected_indices = sorted_indices[:n_trees]
            selected_indices.sort()  # Keep original order

            # Create pruned model
            pruned_model = clone(self.model)
            pruned_model.estimators_ = [self.model.estimators_[i] for i in selected_indices]

            if hasattr(self.model, 'feature_subset_sizes_'):
                pruned_model.feature_subset_sizes_ = [self.model.feature_subset_sizes_[i] for i in selected_indices]

            # Evaluate
            accuracy = accuracy_score(y, pruned_model.predict(X))

            # Check if accuracy is within threshold
            if accuracy >= original_accuracy * accuracy_threshold:
                best_pruned_model = pruned_model
                best_accuracy = accuracy
                best_size = n_trees

                # If accuracy is very close to original, we can stop
                if accuracy >= original_accuracy * 0.995:
                    break

        # If no model met the threshold, return the best one we found
        if best_pruned_model is None:
            print("Warning: Could not prune model while maintaining accuracy threshold")
            return self.model

        # Update size metrics
        self.optimized_size = self._get_model_size(best_pruned_model)

        print(f"Pruned model from {len(self.model.estimators_)} to {best_size} trees")
        print(f"Original accuracy: {original_accuracy:.4f}, Pruned accuracy: {best_accuracy:.4f}")

        return best_pruned_model

    def optimize_feature_subset_sizes(self, X, y, strategy='balanced'):
        """
        Optimize feature subset sizes for trees in the ensemble.

        Args:
            X: Training features
            y: Training labels
            strategy: Optimization strategy ('balanced', 'aggressive', 'conservative')

        Returns:
            Optimized model
        """
        if self.model is None:
            raise ValueError("Model has not been set")

        if not isinstance(self.model, HeterogeneousRandomForest):
            raise ValueError("Model must be a HeterogeneousRandomForest")

        # Get original model parameters
        n_estimators = len(self.model.estimators_)

        # Determine feature subset size distribution
        if strategy == 'balanced':
            # Balanced distribution from min to max
            min_features = max(1, int(np.sqrt(X.shape[1])))
            max_features = min(X.shape[1], int(X.shape[1] * 0.5))
        elif strategy == 'aggressive':
            # More small trees for efficiency
            min_features = max(1, int(np.sqrt(X.shape[1]) * 0.7))
            max_features = min(X.shape[1], int(X.shape[1] * 0.3))
        elif strategy == 'conservative':
            # More large trees for accuracy
            min_features = max(1, int(np.sqrt(X.shape[1]) * 1.0))
            max_features = min(X.shape[1], int(X.shape[1] * 0.7))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Create feature subset sizes
        if n_estimators > 1:
            subset_sizes = np.linspace(min_features, max_features, n_estimators).astype(int)
        else:
            subset_sizes = [min(min_features, max_features)]

        # Create new optimized model
        optimized_model = HeterogeneousRandomForest(
            n_estimators=n_estimators,
            max_features=max_features,
            min_features=min_features,
            random_state=self.model.random_state if hasattr(self.model, 'random_state') else None
        )

        # Train model
        optimized_model.fit(X, y)

        # Update size metrics
        self.optimized_size = self._get_model_size(optimized_model)

        return optimized_model

    def compress_ensemble(self, X, y, target_size=None, target_ratio=0.5):
        """
        Compress the ensemble to a target size using a combination of techniques.

        Args:
            X: Training features
            y: Training labels
            target_size: Target number of trees (if None, use target_ratio)
            target_ratio: Target size ratio compared to original model

        Returns:
            Compressed model
        """
        if self.model is None:
            raise ValueError("Model has not been set")

        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Model does not have an ensemble of estimators")

        # Determine target size
        if target_size is None:
            target_size = max(1, int(len(self.model.estimators_) * target_ratio))

        # Feature selection
        _, feature_mask = self.prune_low_importance_features(X, y, threshold=0.0005)
        X_selected = X[:, feature_mask]

        # Create new model with target size
        if isinstance(self.model, HeterogeneousRandomForest):
            compressed_model = HeterogeneousRandomForest(
                n_estimators=target_size,
                max_features=min(X_selected.shape[1], self.model.max_features),
                min_features=self.model.min_features,
                random_state=self.model.random_state if hasattr(self.model, 'random_state') else None
            )
        else:
            # Generic random forest
            from sklearn.ensemble import RandomForestClassifier
            compressed_model = RandomForestClassifier(
                n_estimators=target_size,
                max_features='sqrt',
                random_state=self.model.random_state if hasattr(self.model, 'random_state') else None
            )

        # Train model
        compressed_model.fit(X_selected, y)

        # Update size metrics
        self.optimized_size = self._get_model_size(compressed_model)

        return compressed_model, feature_mask

    def get_size_comparison(self):
        """
        Get comparison of original and optimized model sizes.

        Returns:
            Dictionary with size comparison metrics
        """
        if self.original_size is None:
            raise ValueError("Original model has not been set")

        if self.optimized_size is None:
            raise ValueError("Model has not been optimized yet")

        comparison = {
            'original': self.original_size,
            'optimized': self.optimized_size
        }

        # Calculate reduction percentages
        for metric in self.original_size:
            if metric in self.optimized_size:
                original_value = self.original_size[metric]
                optimized_value = self.optimized_size[metric]

                if original_value > 0:
                    reduction = 1 - (optimized_value / original_value)
                    comparison[f'{metric}_reduction'] = reduction

        return comparison


class RegionalOptimizer:
    """
    Specialized optimizer for region-specific models, considering the different
    requirements for each region as highlighted in the paper.
    """

    def __init__(self, region_configs=None):
        """
        Initialize the regional optimizer.

        Args:
            region_configs: Dictionary mapping regions to configurations
        """
        self.region_configs = region_configs or {
            'US': {'max_features': 300, 'trees': 100},
            'BR': {'max_features': 400, 'trees': 100},
            'JP': {'max_features': 800, 'trees': 100}
        }

        self.optimizers = {}
        self.models = {}

    def optimize_for_region(self, region, model, X, y):
        """
        Optimize a model for a specific region.

        Args:
            region: Region code ('US', 'BR', 'JP')
            model: Model to optimize
            X: Training features
            y: Training labels

        Returns:
            Optimized model for the region
        """
        if region not in self.region_configs:
            raise ValueError(f"Unknown region: {region}")

        # Get region configuration
        config = self.region_configs[region]

        # Create optimizer if needed
        if region not in self.optimizers:
            self.optimizers[region] = ModelOptimizer(model)
        else:
            self.optimizers[region].set_model(model)

        # Determine optimization strategy based on region
        if region == 'US':
            # US: Favor smaller models with fewer features
            strategy = 'aggressive'
        elif region == 'BR':
            # BR: Balanced approach
            strategy = 'balanced'
        elif region == 'JP':
            # JP: Favor larger models for complex patterns
            strategy = 'conservative'
        else:
            strategy = 'balanced'

        # Apply region-specific optimization
        target_trees = config.get('trees', 100)
        target_features = config.get('max_features', X.shape[1] // 2)

        # Compress model
        optimized_model, feature_mask = self.optimizers[region].compress_ensemble(
            X, y,
            target_size=target_trees
        )

        # Further optimize feature subset sizes
        if isinstance(optimized_model, HeterogeneousRandomForest):
            optimized_model = self.optimizers[region].optimize_feature_subset_sizes(
                X[:, feature_mask], y, strategy=strategy
            )

        # Store optimized model
        self.models[region] = optimized_model

        return optimized_model, feature_mask

    def get_regional_comparisons(self):
        """
        Get size comparisons for all optimized regional models.

        Returns:
            Dictionary mapping regions to size comparison metrics
        """
        comparisons = {}

        for region, optimizer in self.optimizers.items():
            try:
                comparisons[region] = optimizer.get_size_comparison()
            except ValueError:
                # Skip regions without optimized models
                pass

        return comparisons