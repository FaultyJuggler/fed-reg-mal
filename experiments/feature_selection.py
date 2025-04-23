import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data.dataset import MalwareDataset
from data.feature_selector import FeatureSelector
from models.adaptive_rf import HeterogeneousRandomForest


class FeatureSelectionExperiment:
    """
    Experiment to analyze feature selection for different regional datasets.
    Replicates the feature selection experiments from the paper.
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
        self.results = {}

    def run(self, selection_methods=None, n_trees_list=None, max_features=1500):
        """
        Run the feature selection experiment.

        Args:
            selection_methods: List of feature selection methods to test
            n_trees_list: List of numbers of trees to test
            max_features: Maximum number of features to test

        Returns:
            Dictionary of results
        """
        if selection_methods is None:
            selection_methods = ['f_score', 'chi2', 'mutual_info']

        if n_trees_list is None:
            n_trees_list = [2, 10, 100, 500, 1000, 2000]

        # Load data
        print("Loading data...")
        self.dataset.load_data(self.data_dir)

        # Run experiments for each region
        for region in self.dataset.regions:
            print(f"Running experiments for {region}...")

            # Get data
            X_train = self.dataset.X['train'][region]
            y_train = self.dataset.y['train'][region]
            X_test = self.dataset.X['test'][region]
            y_test = self.dataset.y['test'][region]

            # Initialize results for region
            self.results[region] = {}

            # Test each feature selection method
            for method in selection_methods:
                print(f"  Testing feature selection method: {method}")

                # Initialize results for method
                self.results[region][method] = {
                    'accuracy': {},
                    'model_size': {},
                    'optimal_features': {}
                }

                # Test different feature subset sizes
                feature_sizes = list(range(10, max_features + 1, 10))

                for n_trees in n_trees_list:
                    print(f"    Testing with {n_trees} trees...")

                    # Initialize accuracy and model size arrays
                    accuracy_values = []
                    model_sizes = []

                    for n_features in feature_sizes:
                        # Select features
                        selector = FeatureSelector(selection_method=method, n_features=n_features)
                        X_train_selected = selector.fit_transform(X_train, y_train)
                        X_test_selected = selector.transform(X_test)

                        # Train and evaluate model
                        model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
                        model.fit(X_train_selected, y_train)

                        # Calculate accuracy
                        y_pred = model.predict(X_test_selected)
                        accuracy = accuracy_score(y_test, y_pred)

                        # Calculate model size (number of nodes)
                        n_nodes = sum(tree.tree_.node_count for tree in model.estimators_)

                        # Store results
                        accuracy_values.append(accuracy)
                        model_sizes.append(n_nodes)

                        # Early stopping if we reach 99% accuracy
                        if accuracy >= 0.99:
                            # Fill remaining values for plotting
                            accuracy_values.extend([accuracy] * (len(feature_sizes) - len(accuracy_values)))
                            model_sizes.extend([n_nodes] * (len(feature_sizes) - len(model_sizes)))
                            break

                    # Store results for this number of trees
                    self.results[region][method]['accuracy'][n_trees] = accuracy_values
                    self.results[region][method]['model_size'][n_trees] = model_sizes

                    # Find optimal number of features (reaching 99% accuracy)
                    optimal_idx = next((i for i, acc in enumerate(accuracy_values) if acc >= 0.99),
                                       len(accuracy_values) - 1)
                    optimal_features = feature_sizes[optimal_idx]
                    self.results[region][method]['optimal_features'][n_trees] = optimal_features

                    print(f"      Optimal features: {optimal_features}")

        # Plot results
        self._plot_results()

        return self.results

    def _plot_results(self):
        """Plot experiment results."""
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot accuracy vs. features for each region and method
        for region in self.results:
            for method in self.results[region]:
                # Plot accuracy vs. features
                plt.figure(figsize=(12, 8))

                # Get dictionary of accuracy values for different numbers of trees
                accuracy_dict = self.results[region][method]['accuracy']

                # Plot for each number of trees
                for n_trees, accuracy_values in accuracy_dict.items():
                    # Get feature sizes (x-axis values)
                    feature_sizes = list(range(10, 10 * len(accuracy_values) + 1, 10))

                    # Plot accuracy values
                    plt.plot(feature_sizes, accuracy_values, label=f'N={n_trees}')

                # Add plot details
                plt.xlabel('Features (#)')
                plt.ylabel('Accuracy (%)')
                plt.title(f'Classification Accuracy vs. Number of Features ({region})')
                plt.legend()
                plt.grid(True)

                # Save figure
                plt.savefig(os.path.join(fig_dir, f'accuracy_features_{region}_{method}.png'))
                plt.close()

                # Plot model size vs. features
                plt.figure(figsize=(12, 8))

                # Get dictionary of model size values for different numbers of trees
                model_size_dict = self.results[region][method]['model_size']

                # Plot for each number of trees
                for n_trees, model_sizes in model_size_dict.items():
                    # Get feature sizes (x-axis values)
                    feature_sizes = list(range(10, 10 * len(model_sizes) + 1, 10))

                    # Plot model sizes
                    plt.plot(feature_sizes, model_sizes, label=f'N={n_trees}')

                # Add plot details
                plt.xlabel('Features (#)')
                plt.ylabel('Nodes (#)')
                plt.title(f'Model Size vs. Number of Tree Nodes ({region})')
                plt.legend()
                plt.grid(True)

                # Save figure
                plt.savefig(os.path.join(fig_dir, f'model_size_features_{region}_{method}.png'))
                plt.close()

        # Plot optimal features comparison across regions
        plt.figure(figsize=(12, 8))

        # Get feature selection method with best results
        best_method = list(self.results[self.dataset.regions[0]].keys())[0]

        # Plot for each region
        bar_width = 0.2
        index = np.arange(len(self.results))

        for i, n_trees in enumerate([2, 100, 2000]):
            # Get optimal features for each region
            optimal_features = [self.results[region][best_method]['optimal_features'][n_trees]
                                for region in self.results]

            # Plot bar
            plt.bar(index + i * bar_width, optimal_features,
                    bar_width, label=f'N={n_trees}')

        # Add plot details
        plt.xlabel('Region')
        plt.ylabel('Optimal Features (#)')
        plt.title(f'Optimal Number of Features by Region ({best_method})')
        plt.xticks(index + bar_width, self.results.keys())
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, f'optimal_features_comparison.png'))
        plt.close()