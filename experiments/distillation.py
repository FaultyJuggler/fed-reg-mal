import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from data.dataset import MalwareDataset
from data.feature_selector import FeatureSelector
from models.adaptive_rf import HeterogeneousRandomForest
from distillation.teacher_student import ModelDistiller, RegionalDistiller
from distillation.optimization import ModelOptimizer, RegionalOptimizer


class DistillationExperiment:
    """
    Experiment to evaluate model distillation for malware detection.
    Implements the distillation approach described in the paper.
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
            'teacher': {},
            'student': {},
            'optimized': {},
            'region_specific': {}
        }

    def run(self, global_features=800, distilled_features=None, n_trees=100, n_distilled_trees=None):
        """
        Run the model distillation experiment.

        Args:
            global_features: Number of features for the global model
            distilled_features: Dict mapping regions to feature counts for distilled models
            n_trees: Number of trees for the global model
            n_distilled_trees: Number of trees for distilled models

        Returns:
            Dictionary of results
        """
        if distilled_features is None:
            distilled_features = {
                'US': 300,
                'BR': 400,
                'JP': 800
            }

        if n_distilled_trees is None:
            n_distilled_trees = max(10, n_trees // 2)

        # Load data
        print("Loading data...")
        self.dataset.load_data(self.data_dir)

        # Run global-to-regional distillation experiment
        self._run_global_to_regional_experiment(
            global_features=global_features,
            distilled_features=distilled_features,
            n_trees=n_trees,
            n_distilled_trees=n_distilled_trees
        )

        # Run optimization experiment
        self._run_optimization_experiment(
            global_features=global_features,
            distilled_features=distilled_features,
            n_trees=n_trees
        )

        # Plot results
        self._plot_results()

        return self.results

    def _run_global_to_regional_experiment(self, global_features, distilled_features, n_trees, n_distilled_trees):
        """
        Run the global-to-regional distillation experiment.

        Args:
            global_features: Number of features for the global model
            distilled_features: Dict mapping regions to feature counts for distilled models
            n_trees: Number of trees for the global model
            n_distilled_trees: Number of trees for distilled models
        """
        print("Running global-to-regional distillation experiment...")

        # Train global model on all data
        print("Training global model...")

        # Combine data from all regions
        global_X_train = []
        global_y_train = []

        for region in self.dataset.regions:
            X_train = self.dataset.X['train'][region]
            y_train = self.dataset.y['train'][region]

            global_X_train.append(X_train)
            global_y_train.append(y_train)

        global_X_train = np.vstack(global_X_train)
        global_y_train = np.concatenate(global_y_train)

        # Select features for global model
        selector = FeatureSelector(selection_method='f_score', n_features=global_features)
        global_X_train_selected = selector.fit_transform(global_X_train, global_y_train)

        # Train global model
        global_model = HeterogeneousRandomForest(n_estimators=n_trees, random_state=42)
        global_model.fit(global_X_train_selected, global_y_train)

        # Store global model results
        self.results['teacher']['global'] = {
            'features': global_features,
            'trees': n_trees,
            'model_size': self._get_model_size(global_model)
        }

        # Initialize regional distiller
        regional_distiller = RegionalDistiller(global_model)

        # Distill models for each region
        for region in self.dataset.regions:
            print(f"Distilling model for {region}...")

            # Get data
            X_train = self.dataset.X['train'][region]
            y_train = self.dataset.y['train'][region]
            X_test = self.dataset.X['test'][region]
            y_test = self.dataset.y['test'][region]

            # Transform features
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            # Get feature count for this region
            n_features = distilled_features[region]

            # Distill model
            regional_model = regional_distiller.distill_for_region(
                region, X_train_selected, y_train,
                n_estimators=n_distilled_trees,
                max_features=n_features
            )

            # Evaluate original global model on this region
            global_y_pred = global_model.predict(X_test_selected)
            global_accuracy = accuracy_score(y_test, global_y_pred)

            # Evaluate distilled model
            regional_y_pred = regional_model.predict(X_test_selected)
            regional_accuracy = accuracy_score(y_test, regional_y_pred)

            # Calculate retention rate
            retention_rate = regional_accuracy / global_accuracy if global_accuracy > 0 else 0

            # Store results
            self.results['teacher'][region] = {
                'accuracy': global_accuracy,
                'features': global_features,
                'trees': n_trees,
                'model_size': self._get_model_size(global_model)
            }

            self.results['student'][region] = {
                'accuracy': regional_accuracy,
                'features': n_features,
                'trees': n_distilled_trees,
                'model_size': self._get_model_size(regional_model),
                'retention_rate': retention_rate
            }

            # Save models
            model_dir = os.path.join(self.output_dir, 'models')
            os.makedirs(model_dir, exist_ok=True)

            joblib.dump(regional_model, os.path.join(model_dir, f'distilled_{region}_model.pkl'))

            print(f"  {region} global accuracy: {global_accuracy:.4f}")
            print(f"  {region} distilled accuracy: {regional_accuracy:.4f}")
            print(f"  {region} retention rate: {retention_rate:.4f}")

    def _run_optimization_experiment(self, global_features, distilled_features, n_trees):
        """
        Run the optimization experiment to further reduce model size.

        Args:
            global_features: Number of features for the global model
            distilled_features: Dict mapping regions to feature counts for distilled models
            n_trees: Number of trees for the global model
        """
        print("Running optimization experiment...")

        # Initialize model optimizer
        regional_optimizer = RegionalOptimizer(
            region_configs={
                region: {'max_features': features, 'trees': max(1, n_trees // 2)}
                for region, features in distilled_features.items()
            }
        )

        # Load global model
        global_model = joblib.load(os.path.join(self.output_dir, 'models', 'distilled_global_model.pkl'))

        # Optimize for each region
        for region in self.dataset.regions:
            print(f"Optimizing model for {region}...")

            # Get data
            X_train = self.dataset.X['train'][region]
            y_train = self.dataset.y['train'][region]
            X_test = self.dataset.X['test'][region]
            y_test = self.dataset.y['test'][region]

            # Load region-specific distilled model
            distilled_model = joblib.load(os.path.join(self.output_dir, 'models', f'distilled_{region}_model.pkl'))

            # Select features for this region
            n_features = distilled_features[region]
            selector = FeatureSelector(selection_method='f_score', n_features=n_features)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Get distilled accuracy
            distilled_y_pred = distilled_model.predict(X_test_selected)
            distilled_accuracy = accuracy_score(y_test, distilled_y_pred)

            # Optimize model
            optimized_model, feature_mask = regional_optimizer.optimize_for_region(
                region, distilled_model, X_train_selected, y_train
            )

            # Evaluate optimized model
            X_test_optimized = X_test_selected[:, feature_mask]
            optimized_y_pred = optimized_model.predict(X_test_optimized)
            optimized_accuracy = accuracy_score(y_test, optimized_y_pred)

            # Calculate retention rate
            retention_rate = optimized_accuracy / distilled_accuracy if distilled_accuracy > 0 else 0

            # Store results
            self.results['optimized'][region] = {
                'accuracy': optimized_accuracy,
                'features': sum(feature_mask),
                'trees': len(optimized_model.estimators_) if hasattr(optimized_model, 'estimators_') else -1,
                'model_size': self._get_model_size(optimized_model),
                'retention_rate': retention_rate
            }

            # Save model
            joblib.dump(optimized_model, os.path.join(self.output_dir, 'models', f'optimized_{region}_model.pkl'))

            print(f"  {region} optimized accuracy: {optimized_accuracy:.4f}")
            print(f"  {region} optimization retention rate: {retention_rate:.4f}")

    def _get_model_size(self, model):
        """
        Get the size of a model in terms of nodes and features.

        Args:
            model: The model to measure

        Returns:
            Dictionary with size metrics
        """
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

    def _plot_results(self):
        """Plot experiment results."""
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        # Plot accuracy comparison
        plt.figure(figsize=(12, 8))

        # Prepare data
        regions = list(self.results['teacher'].keys() - {'global'})
        teacher_acc = [self.results['teacher'][r]['accuracy'] for r in regions]
        student_acc = [self.results['student'][r]['accuracy'] for r in regions]
        optimized_acc = [self.results['optimized'][r]['accuracy'] for r in regions]

        # Set up bar chart
        bar_width = 0.25
        index = np.arange(len(regions))

        # Plot bars
        plt.bar(index - bar_width, teacher_acc, bar_width, label='Global Model')
        plt.bar(index, student_acc, bar_width, label='Distilled Model')
        plt.bar(index + bar_width, optimized_acc, bar_width, label='Optimized Model')

        # Add details
        plt.xlabel('Region')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.xticks(index, regions)
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'distillation_accuracy.png'))
        plt.close()

        # Plot model size comparison
        plt.figure(figsize=(12, 8))

        # Prepare data
        teacher_size = [self.results['teacher'][r]['model_size']['n_nodes'] for r in regions]
        student_size = [self.results['student'][r]['model_size']['n_nodes'] for r in regions]
        optimized_size = [self.results['optimized'][r]['model_size']['n_nodes'] for r in regions]

        # Plot bars
        plt.bar(index - bar_width, teacher_size, bar_width, label='Global Model')
        plt.bar(index, student_size, bar_width, label='Distilled Model')
        plt.bar(index + bar_width, optimized_size, bar_width, label='Optimized Model')

        # Add details
        plt.xlabel('Region')
        plt.ylabel('Model Size (nodes)')
        plt.title('Model Size Comparison')
        plt.xticks(index, regions)
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'distillation_size.png'))
        plt.close()

        # Plot retention rates
        plt.figure(figsize=(12, 8))

        # Prepare data
        student_retention = [self.results['student'][r]['retention_rate'] for r in regions]
        optimized_retention = [self.results['optimized'][r]['retention_rate'] for r in regions]

        # Plot bars
        plt.bar(index - bar_width / 2, student_retention, bar_width, label='Distilled Model')
        plt.bar(index + bar_width / 2, optimized_retention, bar_width, label='Optimized Model')

        # Add details
        plt.xlabel('Region')
        plt.ylabel('Retention Rate')
        plt.title('Accuracy Retention After Distillation')
        plt.xticks(index, regions)
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'distillation_retention.png'))
        plt.close()

        # Plot feature counts
        plt.figure(figsize=(12, 8))

        # Prepare data
        teacher_features = [self.results['teacher'][r]['features'] for r in regions]
        student_features = [self.results['student'][r]['features'] for r in regions]
        optimized_features = [self.results['optimized'][r]['features'] for r in regions]

        # Plot bars
        plt.bar(index - bar_width, teacher_features, bar_width, label='Global Model')
        plt.bar(index, student_features, bar_width, label='Distilled Model')
        plt.bar(index + bar_width, optimized_features, bar_width, label='Optimized Model')

        # Add details
        plt.xlabel('Region')
        plt.ylabel('Number of Features')
        plt.title('Feature Count Comparison')
        plt.xticks(index, regions)
        plt.legend()
        plt.grid(True)

        # Save figure
        plt.savefig(os.path.join(fig_dir, 'distillation_features.png'))
        plt.close()