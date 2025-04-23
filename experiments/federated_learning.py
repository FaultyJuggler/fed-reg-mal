import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data.dataset import MalwareDataset
from data.feature_selector import FeatureSelector
from models.adaptive_rf import HeterogeneousRandomForest
from federated.client import FederatedClient
from federated.server import FederatedServer
from distillation.teacher_student import ModelDistiller, RegionalDistiller


class FederatedLearningExperiment:
    """
    Experiment to evaluate federated learning and model distillation.
    Replicates the federated learning experiments from the paper.
    """

    def __init__(self, data_dir, output_dir='results'):
        """
        Initialize the experiment.

        Args:
            data_dir: Directory containing the data
            output_dir: Directory to save results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize dataset handler
        self.dataset = MalwareDataset(regions=['US', 'BR', 'JP'])

        # Initialize results storage
        self.results = {
            'federated': {},
            'distilled': {},
            'incremental': {}
        }

    def run(self, n_trees=100, federated_features=800, distilled_features=None,
            data_portions=None, selection_strategy='confidence'):
        """
        Run the federated learning and model distillation experiment.

        Args:
            n_trees: Number of trees for the models
            federated_features: Number of features for the federated model
            distilled_features: Number of features for distilled models (dict mapping region to feature count)
            data_portions: List of data portions to test (0-100%)
            selection_strategy: Strategy for selecting samples to share ('random' or 'confidence')

        Returns:
            Dictionary of results
        """
        if data_portions is None:
            data_portions = [5, 10, 25, 50, 75, 100]

        if distilled_features is None:
            distilled_features = {
                'US': 300,
                'BR': 400,
                'JP': 900
            }

        # Load data
        print("Loading data...")
        self.dataset.load_data(self.data_dir)

        # Select features for each region
        print("Selecting features...")
        region_data = {}

        for region in self.dataset.regions:
            # Get data
            X_train = self.dataset.X['train'][region]
            y_train = self.dataset.y['train'][region]
            X_test = self.dataset.X['test'][region]
            y_test = self.dataset.y['test'][region]

            # Select features for federated model
            selector = FeatureSelector(selection_method='f_score', n_features=federated_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Store data
            region_data[region] = {
                'X_train': X_train_selected,
                'y_train': y_train,
                'X_test': X_test_selected,
                'y_test': y_test
            }

        # Run incremental federated learning experiment
        self._run_incremental_experiment(region_data, n_trees, data_portions, selection_strategy)

        # Run distillation experiment
        self._run_distillation_experiment(region_data, n_trees, distilled_features)

        # Plot results
        self._plot_results(data_portions)

        return self.results

    def _run_incremental_experiment(self, region_data, n_trees, data_portions, selection_strategy):
        """
        Run the incremental federated learning experiment.

        Args:
            region_data: Dictionary of region data
            n_trees: Number of trees for the models
            data_portions: List of data portions to test
            selection_strategy: Strategy for selecting samples to share
        """
        print("Running incremental federated learning experiment...")

        # Initialize clients and server
        clients = {}
        for region in self.dataset.regions:
            model = HeterogeneousRandomForest(n_estimators=n_trees, random_state=42)
            clients[region] = FederatedClient(region, model, selection_strategy)

        server = FederatedServer()

        # Train initial models on local data
        for region, client in clients.items():
            print(f"Training initial model for {region}...")
            X_train = region_data[region]['X_train']
            y_train = region_data[region]['y_train']

            client.set_data(X_train, y_train)
            client.train()

        # Run incremental federated learning
        for portion in data_portions:
            print(f"Testing with {portion}% of data...")

            # Calculate number of samples to share
            shared_data = {}

            for region, client in clients.items():
                for other_region in self.dataset.regions:
                    if other_region == region:
                        continue

                    # Get data from other region
                    X_other = region_data[other_region]['X_train']
                    y_other = region_data[other_region]['y_train']

                    # Select samples to share
                    X_shared, y_shared = client.select_samples_to_share(X_other, y_other, percentage=portion)

                    # Store shared data
                    if region not in shared_data:
                        shared_data[region] = {}

                    shared_data[region][other_region] = (X_shared, y_shared)

            # Send models to server
            for region, client in clients.items():
                server.receive_model(region, client.get_model_for_server())

            # Send shared data to server
            for region, data_dict in shared_data.items():
                server.receive_data(data_dict)

            # Train global model
            server.train_on_aggregated_data()

            # Get global model and update clients
            global_model = server.get_global_model()

            for region, client in clients.items():
                client.update_model(global_model)

            # Evaluate models
            for region, client in clients.items():
                # Get test data
                X_test = region_data[region]['X_test']
                y_test = region_data[region]['y_test']

                # Evaluate
                accuracy = client.evaluate(X_test, y_test)

                # Store results
                if region not in self.results['federated']:
                    self.results['federated'][region] = []

                self.results['federated'][region].append((portion, accuracy))

                print(f"  {region} accuracy with {portion}% data: {accuracy:.4f}")

    def _run_distillation_experiment(self, region_data, n_trees, distilled_features):
        """
        Run the model distillation experiment.

        Args:
            region_data: Dictionary of region data
            n_trees: Number of trees for the models
            distilled_features: Dictionary mapping region to feature count
        """
        print("Running model distillation experiment...")

        # Train global model on all data
        print("Training global model...")

        # Combine data from all regions
        X_all = np.vstack([data['X_train'] for data in region_data.values()])
        y_all = np.concatenate([data['y_train'] for data in region_data.values()])

        # Train global model
        global_model = HeterogeneousRandomForest(n_estimators=n_trees * 3, random_state=42)
        global_model.fit(X_all, y_all)

        # Initialize regional distiller
        regional_distiller = RegionalDistiller(global_model)

        # Distill models for each region
        for region in self.dataset.regions:
            print(f"Distilling model for {region}...")

            # Get data
            X_train = region_data[region]['X_train']
            y_train = region_data[region]['y_train']
            X_test = region_data[region]['X_test']
            y_test = region_data[region]['y_test']

            # Get feature count for this region
            n_features = distilled_features[region]

            # Distill model
            regional_model = regional_distiller.distill_for_region(
                region, X_train, y_train,
                n_estimators=n_trees,
                max_features=n_features
            )

            # Evaluate model
            y_pred = regional_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Store results
            self.results['distilled'][region] = {
                'accuracy': accuracy,
                'features': n_features,
                'trees': n_trees
            }

            print(f"  {region} distilled model accuracy: {accuracy:.4f}")

    def _plot_results(self, data_portions):
        """
        Plot experiment results.

        Args:
            data_portions: List of data portions tested
        """
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot federated learning results
        plt.figure(figsize=(12, 8))

        for region in self.results['federated']:
            # Get data
            portions = [p for p, _ in self.results['federated'][region]]
            accuracies = [acc for _, acc in self.results['federated'][region]]

            # Plot accuracy vs. data portion
            plt.plot(portions, accuracies, marker='o', label=region)

        # Add plot details
        plt.xlabel('Data Portion (%)')
        plt.ylabel('Accuracy')
        plt.title('Federated Learning: Accuracy vs. Data Portion')
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'federated_learning_accuracy.png'))
        plt.close()

        # Plot distillation results
        plt.figure(figsize=(12, 8))

        # Prepare data
        regions = list(self.results['distilled'].keys())
        accuracies = [self.results['distilled'][region]['accuracy'] for region in regions]
        features = [self.results['distilled'][region]['features'] for region in regions]

        # Create bar plots
        bar_width = 0.35
        index = np.arange(len(regions))

        # Plot accuracy bars
        plt.bar(index, accuracies, bar_width, label='Accuracy')

        # Plot feature bars (scaled)
        feature_scale = max(accuracies) / max(features)
        scaled_features = [f * feature_scale for f in features]
        plt.bar(index + bar_width, scaled_features, bar_width, label='Features (scaled)')

        # Add plot details
        plt.xlabel('Region')
        plt.ylabel('Accuracy / Scaled Features')
        plt.title('Model Distillation Results')
        plt.xticks(index + bar_width / 2, regions)
        plt.legend()
        plt.grid(True)

        # Add feature count annotations
        for i, f in enumerate(features):
            plt.annotate(str(f), xy=(index[i] + bar_width, scaled_features[i] + 0.01),
                         ha='center', va='bottom')

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'distillation_results.png'))
        plt.close()