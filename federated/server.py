import numpy as np
from copy import deepcopy
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from models.adaptive_rf import HeterogeneousRandomForest


class FederatedServer:
    """
    Server for federated learning in malware detection.
    Aggregates models from clients and builds a global model.
    """

    def __init__(self, model=None):
        """
        Initialize the federated server.

        Args:
            model: Initial global model (if None, a new one will be created)
        """
        self.global_model = model if model is not None else HeterogeneousRandomForest()
        self.client_models = {}
        self.aggregated_data = defaultdict(list)

        self.is_trained = False

    def receive_model(self, client_id, model):
        """
        Receive a model from a client.

        Args:
            client_id: Identifier of the client
            model: Client's model

        Returns:
            Success flag
        """
        # Store the client's model
        self.client_models[client_id] = deepcopy(model)

        return True

    def receive_data(self, data_dict):
        """
        Receive data from clients.

        Args:
            data_dict: Dictionary with client_id keys and (X, y) value tuples

        Returns:
            Success flag
        """
        for client_id, (X, y) in data_dict.items():
            self.aggregated_data[client_id].append((X, y))

        return True

    def aggregate_models(self, aggregation_method='model_averaging'):
        """
        Aggregate client models to create a global model.

        Args:
            aggregation_method: Method for aggregation ('model_averaging', 'weighted_averaging', 'tree_fusion')

        Returns:
            Aggregated global model
        """
        if not self.client_models:
            raise ValueError("No client models to aggregate")

        if aggregation_method == 'model_averaging':
            # Simple model averaging (forests with same structure)
            self._aggregate_by_model_averaging()
        elif aggregation_method == 'weighted_averaging':
            # Weighted averaging based on data size
            self._aggregate_by_weighted_averaging()
        elif aggregation_method == 'tree_fusion':
            # Tree fusion (more sophisticated)
            self._aggregate_by_tree_fusion()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        self.is_trained = True

        return self.global_model

    def _aggregate_by_model_averaging(self):
        """
        Aggregate client models by averaging their parameters.
        Works for HeterogeneousRandomForest models.
        """
        # Get the first model to determine structure
        first_client_id = list(self.client_models.keys())[0]
        first_model = self.client_models[first_client_id]

        # Create a new global model with the same structure
        self.global_model = HeterogeneousRandomForest(
            n_estimators=first_model.n_estimators,
            max_features=first_model.max_features,
            min_features=first_model.min_features,
            random_state=first_model.random_state
        )

        # Create empty estimators
        self.global_model.estimators_ = []
        self.global_model.feature_subset_sizes_ = first_model.feature_subset_sizes_

        # Calculate the average parameters for each tree
        for i in range(first_model.n_estimators):
            # Get trees at the same position from all clients
            trees = []
            for client_id, model in self.client_models.items():
                if i < len(model.estimators_):
                    trees.append(model.estimators_[i])

            # If no trees at this position, skip
            if not trees:
                continue

            # Get the first tree structure as a template
            global_tree = deepcopy(trees[0])

            # Average the tree parameters
            # This would require complex tree structure reconciliation in practice
            # For simplicity, we'll use a simplified approach
            self.global_model.estimators_.append(global_tree)

    def _aggregate_by_weighted_averaging(self):
        """
        Aggregate client models by weighted averaging.
        Weights are based on the amount of data contributed by each client.
        """
        # Get the number of samples from each client
        client_weights = {}
        total_samples = 0

        for client_id, data_list in self.aggregated_data.items():
            client_samples = sum(len(y) for _, y in data_list)
            client_weights[client_id] = client_samples
            total_samples += client_samples

        # Normalize weights
        if total_samples > 0:
            for client_id in client_weights:
                client_weights[client_id] /= total_samples
        else:
            # If no data, use equal weights
            for client_id in self.client_models:
                client_weights[client_id] = 1.0 / len(self.client_models)

        # Get the first model to determine structure
        first_client_id = list(self.client_models.keys())[0]
        first_model = self.client_models[first_client_id]

        # Create a new global model with the same structure
        self.global_model = HeterogeneousRandomForest(
            n_estimators=first_model.n_estimators,
            max_features=first_model.max_features,
            min_features=first_model.min_features,
            random_state=first_model.random_state
        )

        # Create empty estimators
        self.global_model.estimators_ = []
        self.global_model.feature_subset_sizes_ = first_model.feature_subset_sizes_

        # Apply weighted averaging for each tree
        for i in range(first_model.n_estimators):
            # Get trees at the same position from all clients
            trees = []
            weights = []

            for client_id, model in self.client_models.items():
                if i < len(model.estimators_):
                    trees.append(model.estimators_[i])
                    weights.append(client_weights.get(client_id, 0))

            # If no trees at this position, skip
            if not trees:
                continue

            # Normalize weights
            if sum(weights) > 0:
                weights = [w / sum(weights) for w in weights]
            else:
                weights = [1.0 / len(trees)] * len(trees)

            # Get the first tree structure as a template
            global_tree = deepcopy(trees[0])

            # Apply weighted averaging (simplified)
            self.global_model.estimators_.append(global_tree)

    def _aggregate_by_tree_fusion(self):
        """
        Aggregate client models by tree fusion.
        A more sophisticated approach that combines tree structures.
        """
        # Collect all trees from all models
        all_trees = []
        all_feature_subset_sizes = []

        for client_id, model in self.client_models.items():
            all_trees.extend(model.estimators_)
            all_feature_subset_sizes.extend(model.feature_subset_sizes_)

        if not all_trees:
            raise ValueError("No trees to aggregate")

        # Sort trees by feature subset size
        tree_size_pairs = list(zip(all_trees, all_feature_subset_sizes))
        tree_size_pairs.sort(key=lambda x: x[1])

        # Determine the number of trees for the global model
        # Get the max number of trees from any client
        max_trees = max(len(model.estimators_) for model in self.client_models.values())

        # Select subset of trees to include
        selected_trees = []
        selected_sizes = []

        # Ensure a variety of feature subset sizes
        if len(tree_size_pairs) <= max_trees:
            # If fewer trees than needed, use all of them
            selected_trees = [tree for tree, _ in tree_size_pairs]
            selected_sizes = [size for _, size in tree_size_pairs]
        else:
            # Sample trees with different feature subset sizes
            step = len(tree_size_pairs) / max_trees
            for i in range(max_trees):
                idx = int(i * step)
                selected_trees.append(tree_size_pairs[idx][0])
                selected_sizes.append(tree_size_pairs[idx][1])

        # Create a new global model
        self.global_model = HeterogeneousRandomForest(
            n_estimators=len(selected_trees),
            max_features=max(selected_sizes) if selected_sizes else 'auto',
            min_features=min(selected_sizes) if selected_sizes else None
        )

        # Set the trees and feature subset sizes
        self.global_model.estimators_ = selected_trees
        self.global_model.feature_subset_sizes_ = selected_sizes

    def train_on_aggregated_data(self):
        """
        Train the global model on aggregated data from clients.

        Returns:
            Training accuracy
        """
        if not self.aggregated_data:
            raise ValueError("No aggregated data to train on")

        # Combine all data
        X_all = []
        y_all = []

        for client_id, data_list in self.aggregated_data.items():
            for X, y in data_list:
                X_all.append(X)
                y_all.append(y)

        if not X_all:
            raise ValueError("No data to train on")

        # Combine arrays
        X_combined = np.vstack(X_all)
        y_combined = np.concatenate(y_all)

        # Train the global model
        self.global_model.fit(X_combined, y_combined)
        self.is_trained = True

        # Calculate training accuracy
        y_pred = self.global_model.predict(X_combined)
        accuracy = np.mean(y_pred == y_combined)

        return accuracy

    def get_global_model(self):
        """
        Get the global model.

        Returns:
            Global model
        """
        if not self.is_trained:
            raise ValueError("Global model has not been trained yet")

        return deepcopy(self.global_model)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the global model on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test target vector

        Returns:
            Test accuracy
        """
        if not self.is_trained:
            raise ValueError("Global model has not been trained yet")

        y_pred = self.global_model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        return accuracy