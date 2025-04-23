import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data.dataset import MalwareDataset
from data.feature_selector import FeatureSelector
from models.adaptive_rf import AdaptiveRandomForest, HeterogeneousRandomForest
from federated.client import FederatedClient
from federated.server import FederatedServer


class TimeSeriesExperiment:
    """
    Experiment to evaluate malware detection over time, with concept drift.
    Replicates the time-series experiment from the paper.
    """

    def __init__(self, data_dir, timestamp_file=None, output_dir='results'):
        """
        Initialize the experiment.

        Args:
            data_dir: Directory containing the data
            timestamp_file: CSV file with timestamp information
            output_dir: Directory to save results
        """
        self.data_dir = data_dir
        self.timestamp_file = timestamp_file
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize dataset handler
        self.dataset = MalwareDataset(regions=['US', 'BR', 'JP'])

        # Initialize results storage
        self.results = {
            'no_update': {},
            'drift_detection': {},
            'federated': {}
        }

    def run(self, n_trees=100, features_per_region=None, n_chunks=6):
        """
        Run the time-series experiment.

        Args:
            n_trees: Number of trees for the models
            features_per_region: Dictionary mapping region to feature count
            n_chunks: Number of time chunks to evaluate

        Returns:
            Dictionary of results
        """
        if features_per_region is None:
            features_per_region = {
                'US': 300,
                'BR': 400,
                'JP': 800
            }

        # Load data
        print("Loading data...")
        self.dataset.load_data(self.data_dir)

        # Run experiments for each region
        for region in self.dataset.regions:
            print(f"Running time-series experiments for {region}...")

            # Get time-ordered data chunks
            time_chunks = self.dataset.load_concept_drift_data(region, self.timestamp_file, n_chunks)

            # Run time-series experiments
            self._run_no_update_experiment(region, time_chunks, n_trees, features_per_region[region])
            self._run_drift_detection_experiment(region, time_chunks, n_trees, features_per_region[region])
            self._run_federated_experiment(region, time_chunks, n_trees, features_per_region)

        # Plot results
        self._plot_results(n_chunks)

        return self.results

    def _run_no_update_experiment(self, region, time_chunks, n_trees, n_features):
        """
        Run the experiment without model updates.

        Args:
            region: Region name
            time_chunks: List of time-ordered data chunks
            n_trees: Number of trees for the model
            n_features: Number of features for the model
        """
        print(f"  Running experiment without updates for {region}...")

        # Initialize results
        self.results['no_update'][region] = []

        # Get initial training data (first chunk)
        X_train_init, y_train_init = time_chunks[0]

        # Select features
        selector = FeatureSelector(selection_method='f_score', n_features=n_features)
        X_train_selected = selector.fit_transform(X_train_init, y_train_init)

        # Train initial model
        model = HeterogeneousRandomForest(n_estimators=n_trees, random_state=42)
        model.fit(X_train_selected, y_train_init)

        # Evaluate on all time chunks
        for i, (X_chunk, y_chunk) in enumerate(time_chunks):
            # Select features for evaluation
            X_chunk_selected = selector.transform(X_chunk)

            # Evaluate model
            y_pred = model.predict(X_chunk_selected)
            accuracy = accuracy_score(y_chunk, y_pred)

            # Store results
            self.results['no_update'][region].append((i + 1, accuracy))

            print(f"    Chunk {i + 1} accuracy: {accuracy:.4f}")

    def _run_drift_detection_experiment(self, region, time_chunks, n_trees, n_features):
        """
        Run the experiment with concept drift detection.

        Args:
            region: Region name
            time_chunks: List of time-ordered data chunks
            n_trees: Number of trees for the model
            n_features: Number of features for the model
        """
        print(f"  Running experiment with drift detection for {region}...")

        # Initialize results
        self.results['drift_detection'][region] = []

        # Get initial training data (first chunk)
        X_train_init, y_train_init = time_chunks[0]

        # Select features
        selector = FeatureSelector(selection_method='f_score', n_features=n_features)
        X_train_selected = selector.fit_transform(X_train_init, y_train_init)

        # Train initial model
        model = AdaptiveRandomForest(
            n_estimators=n_trees,
            random_state=42,
            warning_delta=0.005,
            drift_delta=0.01
        )
        model.fit(X_train_selected, y_train_init)

        # Evaluate on all time chunks
        for i, (X_chunk, y_chunk) in enumerate(time_chunks):
            # Select features for evaluation
            X_chunk_selected = selector.transform(X_chunk)

            # Evaluate model
            y_pred = model.predict(X_chunk_selected)
            accuracy = accuracy_score(y_chunk, y_pred)

            # Store results
            self.results['drift_detection'][region].append((i + 1, accuracy))

            print(f"    Chunk {i + 1} accuracy: {accuracy:.4f}")

            # Check for drift and update model
            if i > 0:  # Skip first chunk (already used for training)
                # Update model with current chunk data
                model.partial_fit(X_chunk_selected, y_chunk)

                # Re-select features if needed (if drift was detected)
                if model.in_concept_drift:
                    print(f"    Drift detected in chunk {i + 1}!")
                    # Re-fit selector
                    selector = FeatureSelector(selection_method='f_score', n_features=n_features)
                    selector.fit(X_chunk_selected, y_chunk)

    def _run_federated_experiment(self, region, time_chunks, n_trees, features_per_region):
        """
        Run the experiment with federated learning.

        Args:
            region: Region name
            time_chunks: List of time-ordered data chunks
            n_trees: Number of trees for the model
            features_per_region: Dictionary mapping region to feature count
        """
        print(f"  Running experiment with federated learning for {region}...")

        # Initialize results
        self.results['federated'][region] = []

        # Initialize federated learning components
        clients = {}
        for r in self.dataset.regions:
            model = AdaptiveRandomForest(
                n_estimators=n_trees,
                random_state=42,
                warning_delta=0.005,
                drift_delta=0.01
            )
            clients[r] = FederatedClient(r, model, 'confidence')

        server = FederatedServer()

        # Get initial training data for all regions
        for r in self.dataset.regions:
            # Get data from dataset
            if r == region:
                # For the target region, use the first time chunk
                X_train, y_train = time_chunks[0]
            else:
                # For other regions, use the training data
                X_train = self.dataset.X['train'][r]
                y_train = self.dataset.y['train'][r]

            # Select features
            n_feat = features_per_region[r]
            selector = FeatureSelector(selection_method='f_score', n_features=n_feat)
            X_train_selected = selector.fit_transform(X_train, y_train)

            # Set client data and train
            clients[r].set_data(X_train_selected, y_train)
            clients[r].train()

            # Send model to server
            server.receive_model(r, clients[r].get_model_for_server())

        # Train global model
        server.aggregate_models()

        # Share knowledge back to clients
        global_model = server.get_global_model()
        for r, client in clients.items():
            client.update_model(global_model)

        # Evaluate on all time chunks
        for i, (X_chunk, y_chunk) in enumerate(time_chunks):
            # Select features for evaluation
            n_feat = features_per_region[region]
            selector = FeatureSelector(selection_method='f_score', n_features=n_feat)

            if i == 0:
                # First chunk was used for training, re-use selector
                X_chunk_selected = selector.fit_transform(X_chunk, y_chunk)
            else:
                # Subsequent chunks need feature selection
                X_chunk_selected = selector.fit_transform(X_chunk, y_chunk)

            # Evaluate model
            client = clients[region]
            y_pred = client.model.predict(X_chunk_selected)
            accuracy = accuracy_score(y_chunk, y_pred)

            # Store results
            self.results['federated'][region].append((i + 1, accuracy))

            print(f"    Chunk {i + 1} accuracy: {accuracy:.4f}")

            # Update model with current chunk
            if i > 0:  # Skip first chunk (already used for training)
                # Update client model
                client.set_data(X_chunk_selected, y_chunk)
                client.train(reset=False)

                # Share updated model with server
                server.receive_model(region, client.get_model_for_server())

                # Aggregate models and share knowledge
                server.aggregate_models()
                global_model = server.get_global_model()

                for r, c in clients.items():
                    c.update_model(global_model)

    def _plot_results(self, n_chunks):
        """
        Plot experiment results.

        Args:
            n_chunks: Number of time chunks
        """
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot time-series results for each region
        for region in self.dataset.regions:
            plt.figure(figsize=(12, 8))

            # Plot no update results
            if region in self.results['no_update']:
                chunks = [c for c, _ in self.results['no_update'][region]]
                accuracies = [acc for _, acc in self.results['no_update'][region]]
                plt.plot(chunks, accuracies, marker='o', label='No Update')

            # Plot drift detection results
            if region in self.results['drift_detection']:
                chunks = [c for c, _ in self.results['drift_detection'][region]]
                accuracies = [acc for _, acc in self.results['drift_detection'][region]]
                plt.plot(chunks, accuracies, marker='s', label='Drift Detection')

            # Plot federated results
            if region in self.results['federated']:
                chunks = [c for c, _ in self.results['federated'][region]]
                accuracies = [acc for _, acc in self.results['federated'][region]]
                plt.plot(chunks, accuracies, marker='^', label='FL Retraining')

            # Add plot details
            plt.xlabel('Time (Month)')
            plt.ylabel('Detection Rate (%)')
            plt.title(f'Detection Rate over Time ({region})')
            plt.xticks(range(1, n_chunks + 1))
            plt.ylim(0.7, 1.0)  # Set y-axis limits for better visualization
            plt.legend()
            plt.grid(True)

            # Save figure
            plt.savefig(os.path.join(fig_dir, f'time_series_{region}.png'))
            plt.close()

        # Plot comparison across all regions
        plt.figure(figsize=(12, 8))

        # Calculate average improvement
        avg_improvement = {}

        for method in ['no_update', 'drift_detection', 'federated']:
            improvements = []

            for region in self.dataset.regions:
                if region in self.results[method]:
                    # Get first and last accuracy
                    first_acc = self.results[method][region][0][1]
                    last_acc = self.results[method][region][-1][1]

                    # Calculate improvement
                    improvement = last_acc - first_acc
                    improvements.append(improvement)

            # Calculate average
            if improvements:
                avg_improvement[method] = np.mean(improvements)
            else:
                avg_improvement[method] = 0

        # Plot average improvement
        methods = list(avg_improvement.keys())
        improvements = [avg_improvement[m] for m in methods]

        plt.bar(methods, improvements)

        # Add plot details
        plt.xlabel('Method')
        plt.ylabel('Average Accuracy Improvement')
        plt.title('Average Accuracy Improvement over Time')
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'time_series_improvement.png'))
        plt.close()