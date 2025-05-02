import numpy as np
import pickle
from copy import deepcopy
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from models.adaptive_rf import HeterogeneousRandomForest


class FederatedClient:
    """
    Client for federated learning in a regional malware detection scenario.
    Represents a regional AV subsidiary as described in the paper.
    """

    def __init__(self, region_name, model=None, selection_strategy='confidence'):
        """
        Initialize the federated client.

        Args:
            region_name: Name of the region
            model: Initial model (if None, a new one will be created)
            selection_strategy: Strategy for selecting samples to share ('random' or 'confidence')
        """
        self.region_name = region_name
        self.model = model if model is not None else HeterogeneousRandomForest()
        self.selection_strategy = selection_strategy

        # Data
        self.X_train = None
        self.y_train = None

        # State flags
        self.is_trained = False
        self.model_updated = False

    def set_data(self, X, y):
        """
        Set the client's local dataset.

        Args:
            X: Feature matrix
            y: Target vector
        """
        self.X_train = X
        self.y_train = y

    def train(self, reset=False):
        """
        Train the client's model on its local dataset.

        Args:
            reset: Whether to reset the model before training

        Returns:
            Training accuracy
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Client has no data. Call set_data() first.")

        if reset or not self.is_trained:
            # Check if the model is an accelerated model wrapper
            if hasattr(self.model, 'model') and self.model.model is not None:
                # Already have an accelerated model, use it directly
                print(f"Using existing accelerated model for training")
                self.model.fit(self.X_train, self.y_train)
            elif hasattr(self.model, 'estimators_'):
                # Already have a regular HeterogeneousRandomForest model
                print(f"Using existing model for training")
                self.model.fit(self.X_train, self.y_train)
            else:
                # No model exists yet, create a new one
                # Note: This should never happen if clients are created properly with models
                print("WARNING: Creating new HeterogeneousRandomForest model with default parameters")
                print("This is likely not what you want - models should be provided at client initialization")
                from models.adaptive_rf import HeterogeneousRandomForest
                self.model = HeterogeneousRandomForest()
                self.model.fit(self.X_train, self.y_train)

        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True

        # Calculate training accuracy
        y_pred = self.model.predict(self.X_train)
        accuracy = np.mean(y_pred == self.y_train)

        return accuracy

    def evaluate(self, X_test, y_test):
        """
        Evaluate the client's model on test data.

        Args:
            X_test: Test feature matrix
            y_test: Test target vector

        Returns:
            Test accuracy
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        y_pred = self.model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        return accuracy

    def select_samples_to_share(self, X_other, y_other, percentage=10):
        """
        Select samples from another dataset to share with the server.
        Uses confidence-based or random selection according to the paper.

        Args:
            X_other: Feature matrix from another region
            y_other: Target vector from another region
            percentage: Percentage of samples to select

        Returns:
            Selected X and y
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        # Calculate number of samples to select
        n_samples = int(len(X_other) * percentage / 100)
        n_samples = max(1, min(n_samples, len(X_other)))

        if self.selection_strategy == 'confidence':
            # Confidence-based selection (select samples the model is least confident about)
            probas = self.model.predict_proba(X_other)
            confidence = np.max(probas, axis=1)
            indices = np.argsort(confidence)[:n_samples]  # Select samples with lowest confidence
        else:
            # Random selection
            indices = np.random.choice(len(X_other), n_samples, replace=False)

        # Return selected samples
        return X_other[indices], y_other[indices]

    def update_model(self, global_model):
        """
        Update the client's model with the global model.

        Args:
            global_model: Global model from the server

        Returns:
            Success flag
        """
        # Store the global model
        self.model = deepcopy(global_model)
        self.model_updated = True

        # Fine-tune on local data
        if self.X_train is not None and self.y_train is not None:
            self.train(reset=False)

        return True

    def get_model_for_server(self):
        """
        Get the client's model to send to the server.

        Returns:
            Serialized model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        # Return a copy of the model
        return deepcopy(self.model)

    def distill_model(self, n_estimators=None, max_features=None):
        """
        Distill the client's model to create a smaller version.

        Args:
            n_estimators: Number of estimators in distilled model
            max_features: Maximum number of features in distilled model

        Returns:
            Distilled model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")

        if isinstance(self.model, HeterogeneousRandomForest):
            return self.model.distill(n_estimators, max_features)
        else:
            # If model doesn't support distillation, create a smaller model
            # using teacher-student approach (simplified)
            smaller_model = HeterogeneousRandomForest(
                n_estimators=n_estimators if n_estimators is not None else 10,
                max_features=max_features if max_features is not None else 'sqrt'
            )

            # Generate predictions from the original model
            y_pred = self.model.predict(self.X_train)

            # Train the smaller model on the original model's predictions
            smaller_model.fit(self.X_train, y_pred)

            return smaller_model