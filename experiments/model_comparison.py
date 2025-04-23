import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from data.dataset import MalwareDataset
from data.feature_selector import FeatureSelector
from models.adaptive_rf import HeterogeneousRandomForest, AdaptiveRandomForest


class ModelComparisonExperiment:
    """
    Experiment to compare different model architectures for malware detection.
    Evaluates standard classifiers against the custom HeterogeneousRandomForest.
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
            'accuracy': {},
            'precision': {},
            'recall': {},
            'f1': {},
            'auc': {},
            'cross_val': {},
            'model_size': {},
            'training_time': {},
            'prediction_time': {}
        }

    def run(self, feature_counts=None, n_trees=100):
        """
        Run the model comparison experiment.

        Args:
            feature_counts: Dictionary mapping regions to feature counts
            n_trees: Number of trees for ensemble models

        Returns:
            Dictionary of results
        """
        if feature_counts is None:
            feature_counts = {
                'US': 300,
                'BR': 400,
                'JP': 800
            }

        # Load data
        print("Loading data...")
        self.dataset.load_data(self.data_dir)

        # Define models to compare
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=n_trees, random_state=42),
            'AdaBoost': AdaBoostClassifier(n_estimators=n_trees, random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'HeterogeneousRF': HeterogeneousRandomForest(n_estimators=n_trees, random_state=42),
            'AdaptiveRF': AdaptiveRandomForest(n_estimators=n_trees, random_state=42)
        }

        # Run experiments for each region
        for region in self.dataset.regions:
            print(f"Running experiments for {region}...")

            # Get data
            X_train = self.dataset.X['train'][region]
            y_train = self.dataset.y['train'][region]
            X_test = self.dataset.X['test'][region]
            y_test = self.dataset.y['test'][region]

            # Select features
            n_features = feature_counts[region]
            selector = FeatureSelector(selection_method='f_score', n_features=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Initialize results for region
            for metric in self.results:
                if region not in self.results[metric]:
                    self.results[metric][region] = {}

            # Train and evaluate models
            for model_name, model in models.items():
                print(f"  Testing {model_name}...")

                # Skip SVM for large feature sets (too slow)
                if model_name == 'SVM' and n_features > 500:
                    print(f"  Skipping SVM for {region} (too many features)")
                    continue

                # Train model
                import time
                start_time = time.time()
                model.fit(X_train_selected, y_train)
                training_time = time.time() - start_time

                # Predict
                start_time = time.time()
                y_pred = model.predict(X_test_selected)
                prediction_time = time.time() - start_time

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Calculate AUC for models with predict_proba
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test_selected)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                else:
                    auc = None

                # Cross-validation
                cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5)

                # Estimate model size
                model_size = self._estimate_model_size(model)

                # Store results
                self.results['accuracy'][region][model_name] = accuracy
                self.results['precision'][region][model_name] = precision
                self.results['recall'][region][model_name] = recall
                self.results['f1'][region][model_name] = f1
                self.results['auc'][region][model_name] = auc
                self.results['cross_val'][region][model_name] = cv_scores.mean()
                self.results['model_size'][region][model_name] = model_size
                self.results['training_time'][region][model_name] = training_time
                self.results['prediction_time'][region][model_name] = prediction_time

                print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Size: {model_size}")

        # Save results to CSV
        self._save_results_to_csv()

        # Plot results
        self._plot_results()

        return self.results

    def _estimate_model_size(self, model):
        """
        Estimate the size of a model.

        Args:
            model: The model to measure

        Returns:
            Estimated size in KB
        """
        import sys
        import pickle

        # Pickle the model
        pickled = pickle.dumps(model)

        # Get size in KB
        size_kb = sys.getsizeof(pickled) / 1024

        return size_kb

    def _save_results_to_csv(self):
        """Save results to CSV files."""
        # Create directory for CSV files
        csv_dir = os.path.join(self.output_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)

        # Save each metric to a separate CSV
        for metric, region_data in self.results.items():
            # Convert to DataFrame
            df = pd.DataFrame(region_data)

            # Save to CSV
            csv_path = os.path.join(csv_dir, f'model_comparison_{metric}.csv')
            df.to_csv(csv_path)

    def _plot_results(self):
        """Plot experiment results."""
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot metrics comparison
        self._plot_metric_comparison('accuracy', 'Accuracy')
        self._plot_metric_comparison('f1', 'F1 Score')
        self._plot_metric_comparison('model_size', 'Model Size (KB)')
        self._plot_metric_comparison('training_time', 'Training Time (s)')
        self._plot_metric_comparison('prediction_time', 'Prediction Time (s)')

        # Plot accuracy vs model size
        self._plot_accuracy_vs_size()

        # Plot accuracy across regions
        self._plot_accuracy_across_regions()

    def _plot_metric_comparison(self, metric, metric_name):
        """
        Plot comparison of a specific metric across models and regions.

        Args:
            metric: Name of the metric in self.results
            metric_name: Display name for the metric
        """
        plt.figure(figsize=(15, 10))

        # Get data
        regions = list(self.results[metric].keys())
        models = set()
        for region_data in self.results[metric].values():
            models.update(region_data.keys())
        models = sorted(models)

        # Set up bar chart
        n_regions = len(regions)
        n_models = len(models)
        bar_width = 0.8 / n_models
        index = np.arange(n_regions)

        # Plot bars for each model
        for i, model in enumerate(models):
            values = []
            for region in regions:
                region_data = self.results[metric][region]
                value = region_data.get(model, 0)
                values.append(value)

            plt.bar(index + i * bar_width - 0.4 + bar_width / 2, values, bar_width, label=model)

        # Add details
        plt.xlabel('Region')
        plt.ylabel(metric_name)
        plt.title(f'Model Comparison - {metric_name}')
        plt.xticks(index, regions)
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'figures', f'model_comparison_{metric}.png'))
        plt.close()

    def _plot_accuracy_vs_size(self):
        """Plot accuracy vs model size."""
        plt.figure(figsize=(15, 10))

        # Set up scatter colors and markers for regions and models
        regions = list(self.results['accuracy'].keys())
        models = set()
        for region_data in self.results['accuracy'].values():
            models.update(region_data.keys())
        models = sorted(models)

        colors = {'US': 'blue', 'BR': 'green', 'JP': 'red'}
        markers = {
            'RandomForest': 'o',
            'AdaBoost': 's',
            'SVM': '^',
            'HeterogeneousRF': 'D',
            'AdaptiveRF': 'P'
        }

        # Plot points
        for region in regions:
            for model in models:
                if model in self.results['accuracy'][region] and model in self.results['model_size'][region]:
                    accuracy = self.results['accuracy'][region][model]
                    size = self.results['model_size'][region][model]

                    plt.scatter(
                        size, accuracy,
                        color=colors[region],
                        marker=markers.get(model, 'o'),
                        s=100,
                        label=f'{region} - {model}'
                    )

        # Add details
        plt.xlabel('Model Size (KB)')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Model Size')
        plt.grid(True)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = []

        # Region colors
        for region, color in colors.items():
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=region)
            )

        # Model markers
        for model, marker in markers.items():
            legend_elements.append(
                Line2D([0], [0], marker=marker, color='black', markersize=10, label=model)
            )

        plt.legend(handles=legend_elements, loc='best')

        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'figures', 'accuracy_vs_size.png'))
        plt.close()

    def _plot_accuracy_across_regions(self):
        """Plot accuracy of models across all regions."""
        plt.figure(figsize=(15, 10))

        # Get data
        regions = list(self.results['accuracy'].keys())
        models = set()
        for region_data in self.results['accuracy'].values():
            models.update(region_data.keys())
        models = sorted(models)

        # Calculate average accuracy for each model
        avg_accuracy = {}
        for model in models:
            accuracies = []
            for region in regions:
                if model in self.results['accuracy'][region]:
                    accuracies.append(self.results['accuracy'][region][model])

            if accuracies:
                avg_accuracy[model] = np.mean(accuracies)

        # Sort models by average accuracy
        sorted_models = sorted(avg_accuracy.keys(), key=lambda m: avg_accuracy[m], reverse=True)

        # Set up bar chart
        index = np.arange(len(sorted_models))
        bar_width = 0.8 / len(regions)

        # Plot bars for each region
        for i, region in enumerate(regions):
            accuracies = []
            for model in sorted_models:
                if model in self.results['accuracy'][region]:
                    accuracies.append(self.results['accuracy'][region][model])
                else:
                    accuracies.append(0)

            plt.bar(index + i * bar_width - 0.4 + bar_width / 2, accuracies, bar_width, label=region)

        # Add details
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Across Regions')
        plt.xticks(index, sorted_models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'figures', 'accuracy_across_regions.png'))
        plt.close()


class FeatureSelectorComparisonExperiment:
    """
    Experiment to compare different feature selection methods for malware detection.
    Evaluates F-score, Chi-squared, and Mutual Information feature selection.
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
            'optimal_features': {},
            'accuracy': {}
        }

    def run(self, max_features=1000, n_trees=100):
        """
        Run the feature selector comparison experiment.

        Args:
            max_features: Maximum number of features to test
            n_trees: Number of trees for the model

        Returns:
            Dictionary of results
        """
        # Define selection methods
        selection_methods = ['f_score', 'chi2', 'mutual_info']

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
            self.results['optimal_features'][region] = {}
            self.results['accuracy'][region] = {method: [] for method in selection_methods}

            # Test different feature selection methods
            for method in selection_methods:
                print(f"  Testing feature selection method: {method}")

                # Find optimal number of features
                optimal_features, accuracies = self._find_optimal_features(
                    X_train, y_train, X_test, y_test,
                    method, max_features, n_trees
                )

                # Store results
                self.results['optimal_features'][region][method] = optimal_features
                self.results['accuracy'][region][method] = accuracies

                print(f"    Optimal features: {optimal_features}")

        # Plot results
        self._plot_results(max_features)

        return self.results

    def _find_optimal_features(self, X_train, y_train, X_test, y_test, method, max_features, n_trees):
        """
        Find the optimal number of features for a given method.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            method: Feature selection method
            max_features: Maximum number of features to test
            n_trees: Number of trees for the model

        Returns:
            Tuple of (optimal_features, accuracies)
        """
        # Define feature sizes to test
        feature_sizes = list(range(50, max_features + 1, 50))

        # Initialize arrays for storing results
        accuracies = []

        # Test each feature size
        for n_features in feature_sizes:
            # Select features
            selector = FeatureSelector(selection_method=method, n_features=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Train model
            model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
            model.fit(X_train_selected, y_train)

            # Evaluate
            y_pred = model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)

            # Store accuracy
            accuracies.append(accuracy)

            # Early stopping if we reach 99% accuracy
            if accuracy >= 0.99:
                # Fill remaining values for plotting
                accuracies.extend([accuracy] * (len(feature_sizes) - len(accuracies)))
                break

        # Find optimal number of features (reaching 99% accuracy)
        optimal_idx = next((i for i, acc in enumerate(accuracies) if acc >= 0.99), len(accuracies) - 1)
        optimal_features = feature_sizes[optimal_idx]

        return optimal_features, accuracies

    def _plot_results(self, max_features):
        """
        Plot experiment results.

        Args:
            max_features: Maximum number of features tested
        """
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot accuracy vs. features for each region and method
        for region in self.results['accuracy']:
            plt.figure(figsize=(12, 8))

            # Define feature sizes
            feature_sizes = list(range(50, max_features + 1, 50))

            # Plot for each method
            for method, accuracies in self.results['accuracy'][region].items():
                plt.plot(feature_sizes[:len(accuracies)], accuracies, label=method)

            # Add details
            plt.xlabel('Number of Features')
            plt.ylabel('Accuracy')
            plt.title(f'Feature Selection Comparison - {region}')
            plt.legend()
            plt.grid(True)

            # Save figure
            plt.savefig(os.path.join(fig_dir, f'feature_selector_comparison_{region}.png'))
            plt.close()

        # Plot optimal features comparison
        plt.figure(figsize=(12, 8))

        # Prepare data
        regions = list(self.results['optimal_features'].keys())
        methods = list(self.results['optimal_features'][regions[0]].keys())

        # Set up bar chart
        index = np.arange(len(regions))
        bar_width = 0.8 / len(methods)

        # Plot bars for each method
        for i, method in enumerate(methods):
            optimal_features = [self.results['optimal_features'][region][method] for region in regions]
            plt.bar(index + i * bar_width - 0.4 + bar_width / 2, optimal_features, bar_width, label=method)

        # Add details
        plt.xlabel('Region')
        plt.ylabel('Optimal Features')
        plt.title('Optimal Number of Features by Feature Selection Method')
        plt.xticks(index, regions)
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'optimal_features_by_method.png'))
        plt.close()