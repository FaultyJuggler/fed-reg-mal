import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from data.dataset import MalwareDataset
from data.feature_selector import FeatureSelector
from models.adaptive_rf import HeterogeneousRandomForest


class CrossDatasetExperiment:
    """
    Experiment to analyze cross-regional performance of models.
    Replicates the cross-dataset experiments from the paper.
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

    def run(self, n_trees_list=None, max_features=1500, feature_step=50):
        """
        Run the cross-dataset experiment.

        Args:
            n_trees_list: List of numbers of trees to test
            max_features: Maximum number of features to test
            feature_step: Step size for feature count progression

        Returns:
            Dictionary of results
        """
        if n_trees_list is None:
            n_trees_list = [3, 2000]  # Min and max from the paper

        # Load data
        print("Loading data...")
        self.dataset.load_data(self.data_dir)

        # Feature sizes to test
        feature_sizes = list(range(50, max_features + 1, feature_step))

        # Initialize results
        self.results = {
            'trained_on': {},
            'optimal_features': {}
        }

        # Train models for each region and test on others
        for train_region in self.dataset.regions:
            print(f"Training on {train_region}...")

            # Get training data
            X_train = self.dataset.X['train'][train_region]
            y_train = self.dataset.y['train'][train_region]

            # Initialize results for this training region
            self.results['trained_on'][train_region] = {}
            self.results['optimal_features'][train_region] = {}

            # For each test region
            for test_region in self.dataset.regions:
                print(f"  Testing on {test_region}...")

                # Get test data
                X_test = self.dataset.X['test'][test_region]
                y_test = self.dataset.y['test'][test_region]

                # Initialize results for this test region
                self.results['trained_on'][train_region][test_region] = {}

                # Test with different numbers of trees
                for n_trees in n_trees_list:
                    print(f"    Using {n_trees} trees...")

                    # Initialize accuracy array
                    accuracy_values = []

                    # Test different feature subset sizes
                    for n_features in feature_sizes:
                        # Select features
                        selector = FeatureSelector(selection_method='f_score', n_features=n_features)
                        X_train_selected = selector.fit_transform(X_train, y_train)
                        X_test_selected = selector.transform(X_test)

                        # Train model
                        model = HeterogeneousRandomForest(n_estimators=n_trees, random_state=42)
                        model.fit(X_train_selected, y_train)

                        # Evaluate on test data
                        y_pred = model.predict(X_test_selected)
                        accuracy = accuracy_score(y_test, y_pred)

                        # Store accuracy
                        accuracy_values.append(accuracy)

                    # Store results for this number of trees
                    self.results['trained_on'][train_region][test_region][n_trees] = accuracy_values

                    # Find optimal number of features (reaching max accuracy)
                    max_accuracy = max(accuracy_values)
                    optimal_idx = accuracy_values.index(max_accuracy)
                    optimal_features = feature_sizes[optimal_idx]

                    self.results['optimal_features'][train_region][f"{test_region}_{n_trees}"] = {
                        'optimal_features': optimal_features,
                        'max_accuracy': max_accuracy
                    }

                    print(f"      Optimal features: {optimal_features} (accuracy: {max_accuracy:.4f})")

        # Plot results
        self._plot_results(feature_sizes)

        return self.results

    def _plot_results(self, feature_sizes):
        """
        Plot experiment results.

        Args:
            feature_sizes: List of feature sizes tested
        """
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot accuracy vs. features for each training region
        for train_region in self.results['trained_on']:
            # Plot cross-dataset accuracy for each number of trees
            for n_trees in list(self.results['trained_on'][train_region][self.dataset.regions[0]].keys()):
                plt.figure(figsize=(12, 8))

                # Plot for each test region
                for test_region in self.results['trained_on'][train_region]:
                    # Get accuracy values
                    accuracy_values = self.results['trained_on'][train_region][test_region][n_trees]

                    # Plot accuracy
                    plt.plot(feature_sizes, accuracy_values,
                             label=f'Test on {test_region}',
                             marker='o' if test_region == train_region else None)

                # Add plot details
                plt.xlabel('Features (#)')
                plt.ylabel('Accuracy (%)')
                plt.title(f'Classification Accuracy vs. Number of Features ({train_region} vs. others, N={n_trees})')
                plt.legend()
                plt.grid(True)

                # Save figure
                plt.savefig(os.path.join(fig_dir, f'cross_dataset_{train_region}_N{n_trees}.png'))
                plt.close()

        # Plot optimal features and max accuracy comparison
        self._plot_optimal_features_comparison()

    def _plot_optimal_features_comparison(self):
        """Plot comparison of optimal features and max accuracy across regions."""
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')

        # Plot optimal features
        plt.figure(figsize=(12, 8))

        # Prepare data for plotting
        train_regions = list(self.results['optimal_features'].keys())
        test_regions = self.dataset.regions

        # Set up bar chart
        bar_width = 0.15
        index = np.arange(len(train_regions))

        # Plot for each test region and number of trees
        for i, test_region in enumerate(test_regions):
            for j, n_trees in enumerate([3, 2000]):  # Using min and max from the paper
                # Get optimal features
                optimal_features = []

                for train_region in train_regions:
                    key = f"{test_region}_{n_trees}"
                    if key in self.results['optimal_features'][train_region]:
                        optimal_features.append(
                            self.results['optimal_features'][train_region][key]['optimal_features']
                        )
                    else:
                        optimal_features.append(0)

                # Plot bar
                offset = (i * 2 + j) * bar_width
                plt.bar(index + offset, optimal_features,
                        bar_width, label=f'{test_region} (N={n_trees})')

        # Add plot details
        plt.xlabel('Training Region')
        plt.ylabel('Optimal Features (#)')
        plt.title('Optimal Number of Features for Cross-Region Testing')
        plt.xticks(index + 0.3, train_regions)  # Center the x-ticks
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'optimal_features_cross_region.png'))
        plt.close()

        # Plot max accuracy
        plt.figure(figsize=(12, 8))

        # Plot for each test region and number of trees
        for i, test_region in enumerate(test_regions):
            for j, n_trees in enumerate([3, 2000]):  # Using min and max from the paper
                # Get max accuracy
                max_accuracy = []

                for train_region in train_regions:
                    key = f"{test_region}_{n_trees}"
                    if key in self.results['optimal_features'][train_region]:
                        max_accuracy.append(
                            self.results['optimal_features'][train_region][key]['max_accuracy']
                        )
                    else:
                        max_accuracy.append(0)

                # Plot bar
                offset = (i * 2 + j) * bar_width
                plt.bar(index + offset, max_accuracy,
                        bar_width, label=f'{test_region} (N={n_trees})')

        # Add plot details
        plt.xlabel('Training Region')
        plt.ylabel('Max Accuracy')
        plt.title('Maximum Accuracy for Cross-Region Testing')
        plt.xticks(index + 0.3, train_regions)  # Center the x-ticks
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'max_accuracy_cross_region.png'))
        plt.close()