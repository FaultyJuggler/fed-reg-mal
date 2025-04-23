import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


class VisualizationManager:
    """
    Manages visualization creation for malware detection experiments.
    Provides utilities for creating various plots and charts.
    """

    def __init__(self, output_dir=None, style='default'):
        """
        Initialize the visualization manager.

        Args:
            output_dir: Directory to save visualizations
            style: Plot style ('default', 'paper', 'presentation')
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Set up plot style
        self.set_style(style)

        # Define region-specific colors
        self.region_colors = {
            'US': '#1f77b4',  # Blue
            'BR': '#2ca02c',  # Green
            'JP': '#d62728',  # Red
            'Global': '#9467bd',  # Purple
        }

        # Define model-specific markers
        self.model_markers = {
            'RandomForest': 'o',
            'AdaBoost': 's',
            'SVM': '^',
            'HeterogeneousRF': 'D',
            'AdaptiveRF': 'P',
            'Distilled': 'X',
            'Global': '*'
        }

        # Define strategy-specific line styles
        self.strategy_styles = {
            'no_update': '-',
            'drift_detection': '--',
            'federated': '-.'
        }

    def set_style(self, style='default'):
        """
        Set the plot style.

        Args:
            style: Plot style ('default', 'paper', 'presentation')
        """
        if style == 'paper':
            # Publication-ready style
            plt.style.use('seaborn-paper')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.figsize': (8, 6),
                'savefig.dpi': 300,
                'savefig.format': 'pdf'
            })
        elif style == 'presentation':
            # Presentation-friendly style
            plt.style.use('seaborn-talk')
            plt.rcParams.update({
                'font.size': 14,
                'axes.labelsize': 16,
                'axes.titlesize': 18,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'figure.figsize': (12, 8),
                'savefig.dpi': 150,
                'savefig.format': 'png'
            })
        else:
            # Default style
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'figure.figsize': (10, 8),
                'savefig.dpi': 150
            })

    def _save_figure(self, fig, filename):
        """
        Save a figure to file.

        Args:
            fig: Matplotlib figure
            filename: Filename (without extension)

        Returns:
            Path to saved file
        """
        if self.output_dir is None:
            return None

        # Get format from rcParams, default to png
        fmt = plt.rcParams.get('savefig.format', 'png')
        filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save figure
        fig.savefig(filepath, bbox_inches='tight')

        return filepath

    def plot_accuracy_vs_features(self, feature_sizes, accuracy_values,
                                  region=None, n_trees=None, title=None, filename=None):
        """
        Plot accuracy vs. number of features.

        Args:
            feature_sizes: List of feature counts
            accuracy_values: List of accuracy values
            region: Region name (optional)
            n_trees: Number of trees (optional)
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()

        # Plot accuracy curve
        ax.plot(feature_sizes, accuracy_values, marker='o',
                color=self.region_colors.get(region, 'blue'))

        # Add horizontal line at 99% accuracy
        ax.axhline(y=0.99, color='red', linestyle='--',
                   label='99% Accuracy Threshold')

        # Find index where accuracy reaches 99%
        threshold_idx = next((i for i, acc in enumerate(accuracy_values)
                              if acc >= 0.99), len(accuracy_values) - 1)

        if threshold_idx < len(feature_sizes):
            optimal_features = feature_sizes[threshold_idx]

            # Add vertical line at optimal features
            ax.axvline(x=optimal_features, color='green', linestyle='--',
                       label=f'Optimal Features: {optimal_features}')

            # Add annotation
            ax.annotate(f'Optimal: {optimal_features}',
                        xy=(optimal_features, 0.99),
                        xytext=(optimal_features + 20, 0.97),
                        arrowprops=dict(arrowstyle='->'))

        # Set labels and title
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Accuracy')

        if title:
            ax.set_title(title)
        else:
            title_parts = []
            if region:
                title_parts.append(f"Region: {region}")
            if n_trees:
                title_parts.append(f"Trees: {n_trees}")

            if title_parts:
                ax.set_title("Classification Accuracy vs. Number of Features\n" +
                             ", ".join(title_parts))
            else:
                ax.set_title("Classification Accuracy vs. Number of Features")

        # Add grid and legend
        ax.grid(True)
        ax.legend()

        # Set y-axis limits
        min_acc = min(accuracy_values) * 0.99 if accuracy_values else 0.8
        ax.set_ylim(min_acc, 1.01)

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def plot_model_size_vs_features(self, feature_sizes, model_sizes,
                                    region=None, n_trees_list=None, title=None, filename=None):
        """
        Plot model size vs. number of features.

        Args:
            feature_sizes: List of feature counts
            model_sizes: Dict mapping n_trees to list of model sizes
            region: Region name (optional)
            n_trees_list: List of tree counts (optional)
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()

        # Use provided n_trees_list or extract from model_sizes
        if n_trees_list is None:
            n_trees_list = sorted(model_sizes.keys())

        # Plot model size curves for each n_trees
        for n_trees in n_trees_list:
            if n_trees in model_sizes:
                sizes = model_sizes[n_trees]
                ax.plot(feature_sizes[:len(sizes)], sizes, marker='o',
                        label=f'N={n_trees}')

        # Set labels and title
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Model Size (Nodes)')

        if title:
            ax.set_title(title)
        else:
            if region:
                ax.set_title(f"Model Size vs. Number of Tree Nodes ({region})")
            else:
                ax.set_title("Model Size vs. Number of Tree Nodes")

        # Add grid and legend
        ax.grid(True)
        ax.legend()

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def plot_cross_dataset_accuracy(self, feature_sizes, accuracy_values,
                                    train_region, test_regions, n_trees=None,
                                    title=None, filename=None):
        """
        Plot cross-dataset accuracy.

        Args:
            feature_sizes: List of feature counts
            accuracy_values: Dict mapping test_region to list of accuracy values
            train_region: Training region
            test_regions: List of test regions
            n_trees: Number of trees (optional)
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()

        # Plot accuracy curves for each test region
        for region in test_regions:
            if region in accuracy_values:
                values = accuracy_values[region]
                marker = 'o' if region == train_region else None
                ax.plot(feature_sizes[:len(values)], values,
                        label=f'Test on {region}',
                        color=self.region_colors.get(region),
                        marker=marker)

        # Set labels and title
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Accuracy')

        if title:
            ax.set_title(title)
        else:
            title_parts = [f"Trained on {train_region}"]
            if n_trees:
                title_parts.append(f"Trees: {n_trees}")

            ax.set_title("Cross-Dataset Accuracy vs. Number of Features\n" +
                         ", ".join(title_parts))

        # Add grid and legend
        ax.grid(True)
        ax.legend()

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def plot_federated_learning_accuracy(self, data_portions, accuracies,
                                         regions, title=None, filename=None):
        """
        Plot federated learning accuracy vs. data portion.

        Args:
            data_portions: List of data portions
            accuracies: Dict mapping region to list of accuracy values
            regions: List of regions
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()

        # Plot accuracy curves for each region
        for region in regions:
            if region in accuracies:
                values = accuracies[region]
                ax.plot(data_portions[:len(values)], values,
                        label=region,
                        color=self.region_colors.get(region),
                        marker='o')

        # Set labels and title
        ax.set_xlabel('Data Portion (%)')
        ax.set_ylabel('Accuracy')

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Federated Learning: Accuracy vs. Data Portion")

        # Add grid and legend
        ax.grid(True)
        ax.legend()

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def plot_distillation_results(self, results, regions,
                                  metrics=None, title=None, filename=None):
        """
        Plot model distillation results.

        Args:
            results: Dict with distillation results
            regions: List of regions
            metrics: List of metrics to plot (optional)
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Dict mapping metric to Matplotlib figure
        """
        if metrics is None:
            metrics = ['accuracy', 'model_size', 'features']

        figures = {}

        for metric in metrics:
            fig, ax = plt.subplots()

            # Set up bar positions
            n_regions = len(regions)
            n_models = 3  # Global, Distilled, Optimized
            bar_width = 0.8 / n_models
            index = np.arange(n_regions)

            # Prepare data
            if metric == 'accuracy':
                teacher_values = [results['teacher'].get(r, {}).get('accuracy', 0) for r in regions]
                student_values = [results['student'].get(r, {}).get('accuracy', 0) for r in regions]
                optimized_values = [results['optimized'].get(r, {}).get('accuracy', 0) for r in regions]
                ylabel = 'Accuracy'
            elif metric == 'model_size':
                teacher_values = [results['teacher'].get(r, {}).get('model_size', {}).get('n_nodes', 0) for r in
                                  regions]
                student_values = [results['student'].get(r, {}).get('model_size', {}).get('n_nodes', 0) for r in
                                  regions]
                optimized_values = [results['optimized'].get(r, {}).get('model_size', {}).get('n_nodes', 0) for r in
                                    regions]
                ylabel = 'Model Size (nodes)'
            elif metric == 'features':
                teacher_values = [results['teacher'].get(r, {}).get('features', 0) for r in regions]
                student_values = [results['student'].get(r, {}).get('features', 0) for r in regions]
                optimized_values = [results['optimized'].get(r, {}).get('features', 0) for r in regions]
                ylabel = 'Number of Features'
            else:
                continue

            # Plot bars
            ax.bar(index - bar_width, teacher_values, bar_width, label='Global Model')
            ax.bar(index, student_values, bar_width, label='Distilled Model')
            ax.bar(index + bar_width, optimized_values, bar_width, label='Optimized Model')

            # Set labels and title
            ax.set_xlabel('Region')
            ax.set_ylabel(ylabel)

            if title:
                ax.set_title(f"{title} - {ylabel}")
            else:
                ax.set_title(f"Model Distillation Results - {ylabel}")

            # Set x-ticks
            ax.set_xticks(index)
            ax.set_xticklabels(regions)

            # Add grid and legend
            ax.grid(True, axis='y')
            ax.legend()

            # Save figure if filename provided
            if filename:
                metric_filename = f"{filename}_{metric}"
                self._save_figure(fig, metric_filename)

            figures[metric] = fig

        return figures

    def plot_time_series_results(self, time_indices, accuracy_values,
                                 regions, strategies, time_labels=None,
                                 drift_points=None, title=None, filename=None):
        """
        Plot time-series results.

        Args:
            time_indices: List of time indices
            accuracy_values: Dict mapping (strategy, region) to list of accuracy values
            regions: List of regions
            strategies: List of strategies
            time_labels: List of time labels (optional)
            drift_points: Dict mapping region to list of drift points
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Dict mapping region to Matplotlib figure
        """
        figures = {}

        for region in regions:
            fig, ax = plt.subplots()

            # Plot accuracy curves for each strategy
            for strategy in strategies:
                key = (strategy, region)
                if key in accuracy_values:
                    values = accuracy_values[key]
                    ax.plot(time_indices[:len(values)], values,
                            label=strategy.replace('_', ' ').title(),
                            linestyle=self.strategy_styles.get(strategy, '-'),
                            marker='o')

            # Plot drift points if provided
            if drift_points and region in drift_points:
                for drift_point in drift_points[region]:
                    if drift_point < len(time_indices):
                        ax.axvline(x=time_indices[drift_point], color='red',
                                   linestyle=':', alpha=0.7)
                        ax.annotate('Drift', xy=(time_indices[drift_point], 0.75),
                                    xytext=(time_indices[drift_point], 0.7),
                                    arrowprops=dict(arrowstyle='->'),
                                    ha='center')

            # Set labels and title
            ax.set_xlabel('Time')
            ax.set_ylabel('Detection Rate (%)')

            if title:
                ax.set_title(f"{title} - {region}")
            else:
                ax.set_title(f"Detection Rate over Time ({region})")

            # Set x-ticks
            if time_labels:
                ax.set_xticks(time_indices[:len(time_labels)])
                ax.set_xticklabels(time_labels)
            else:
                ax.set_xticks(time_indices)

            # Set y-axis limits
            ax.set_ylim(0.7, 1.0)

            # Add grid and legend
            ax.grid(True)
            ax.legend()

            # Save figure if filename provided
            if filename:
                region_filename = f"{filename}_{region}"
                self._save_figure(fig, region_filename)

            figures[region] = fig

        return figures

    def plot_roc_curves(self, fpr_dict, tpr_dict, auc_dict,
                        models, regions, title=None, filename=None):
        """
        Plot ROC curves for multiple models and regions.

        Args:
            fpr_dict: Dict mapping (model, region) to false positive rates
            tpr_dict: Dict mapping (model, region) to true positive rates
            auc_dict: Dict mapping (model, region) to AUC values
            models: List of models
            regions: List of regions
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Dict mapping region to Matplotlib figure
        """
        figures = {}

        for region in regions:
            fig, ax = plt.subplots()

            # Plot ROC curves for each model
            for model in models:
                key = (model, region)
                if key in fpr_dict and key in tpr_dict and key in auc_dict:
                    fpr = fpr_dict[key]
                    tpr = tpr_dict[key]
                    auc = auc_dict[key]

                    ax.plot(fpr, tpr,
                            label=f'{model} (AUC = {auc:.4f})',
                            marker=self.model_markers.get(model, None),
                            markevery=10)

            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', label='Random')

            # Set labels and title
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')

            if title:
                ax.set_title(f"{title} - {region}")
            else:
                ax.set_title(f"ROC Curves ({region})")

            # Set axis limits
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            # Add grid and legend
            ax.grid(True)
            ax.legend(loc='lower right')

            # Save figure if filename provided
            if filename:
                region_filename = f"{filename}_{region}"
                self._save_figure(fig, region_filename)

            figures[region] = fig

        return figures

    def plot_confusion_matrices(self, cm_dict, models, regions,
                                normalize=False, title=None, filename=None):
        """
        Plot confusion matrices for multiple models and regions.

        Args:
            cm_dict: Dict mapping (model, region) to confusion matrix
            models: List of models
            regions: List of regions
            normalize: Whether to normalize confusion matrices
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Dict mapping (model, region) to Matplotlib figure
        """
        figures = {}

        # Define class labels
        classes = ['Goodware', 'Malware']

        # Define colormap
        cmap = plt.cm.Blues

        for model in models:
            for region in regions:
                key = (model, region)
                if key in cm_dict:
                    cm = cm_dict[key]

                    # Normalize if requested
                    if normalize:
                        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        cm_type = 'Normalized'
                    else:
                        cm_type = 'Absolute'

                    # Create figure
                    fig, ax = plt.subplots()

                    # Plot confusion matrix
                    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
                    fig.colorbar(im, ax=ax)

                    # Add labels
                    ax.set(xticks=np.arange(cm.shape[1]),
                           yticks=np.arange(cm.shape[0]),
                           xticklabels=classes, yticklabels=classes,
                           xlabel='Predicted label',
                           ylabel='True label')

                    # Set title
                    if title:
                        ax.set_title(f"{title} - {model} ({region})")
                    else:
                        ax.set_title(f"{cm_type} Confusion Matrix - {model} ({region})")

                    # Rotate tick labels
                    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

                    # Loop over data dimensions and create text annotations
                    fmt = '.2f' if normalize else 'd'
                    thresh = cm.max() / 2.
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, format(cm[i, j], fmt),
                                    ha="center", va="center",
                                    color="white" if cm[i, j] > thresh else "black")

                    # Set layout
                    fig.tight_layout()

                    # Save figure if filename provided
                    if filename:
                        model_region_filename = f"{filename}_{model}_{region}"
                        self._save_figure(fig, model_region_filename)

                    figures[key] = fig

        return figures

    def plot_accuracy_vs_model_size(self, accuracy_dict, size_dict,
                                    models, regions, title=None, filename=None):
        """
        Plot accuracy vs. model size scatter plot.

        Args:
            accuracy_dict: Dict mapping (model, region) to accuracy
            size_dict: Dict mapping (model, region) to model size
            models: List of models
            regions: List of regions
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()

        # Plot points
        for model in models:
            for region in regions:
                key = (model, region)
                if key in accuracy_dict and key in size_dict:
                    accuracy = accuracy_dict[key]
                    size = size_dict[key]

                    ax.scatter(size, accuracy,
                               color=self.region_colors.get(region),
                               marker=self.model_markers.get(model, 'o'),
                               s=100,
                               label=f'{region} - {model}')

        # Set labels and title
        ax.set_xlabel('Model Size')
        ax.set_ylabel('Accuracy')

        if title:
            ax.set_title(title)
        else:
            ax.set_title("Accuracy vs. Model Size")

        # Add grid
        ax.grid(True)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = []

        # Region colors
        for region in regions:
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=self.region_colors.get(region),
                       markersize=10, label=region)
            )

        # Model markers
        for model in models:
            if model in self.model_markers:
                legend_elements.append(
                    Line2D([0], [0], marker=self.model_markers[model],
                           color='black', markersize=10, label=model)
                )

        ax.legend(handles=legend_elements, loc='best')

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def plot_model_comparison(self, metrics_df, metric,
                              models, regions, title=None, filename=None):
        """
        Plot model comparison bar chart for a specific metric.

        Args:
            metrics_df: DataFrame with metrics data
            metric: Metric to plot
            models: List of models
            regions: List of regions
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        # Filter DataFrame for the specific metric
        df = metrics_df[metrics_df['metric'] == metric].copy()

        # Create pivot table
        pivot_df = df.pivot(index='model', columns='region', values=metric)

        # Create figure
        fig, ax = plt.subplots()

        # Plot bar chart
        pivot_df.plot(kind='bar', ax=ax)

        # Set labels and title
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.replace('_', ' ').title())

        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Model Comparison - {metric.replace('_', ' ').title()}")

        # Add grid
        ax.grid(True, axis='y')

        # Set legend
        ax.legend(title='Region')

        # Adjust layout
        plt.tight_layout()

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def create_feature_importance_heatmap(self, importance_dict, top_n=20,
                                          regions=None, title=None, filename=None):
        """
        Create heatmap of feature importances across regions.

        Args:
            importance_dict: Dict mapping region to feature importance dict
            top_n: Number of top features to include
            regions: List of regions (optional)
            title: Plot title (optional)
            filename: Filename for saving (optional)

        Returns:
            Matplotlib figure
        """
        if regions is None:
            regions = list(importance_dict.keys())

        # Collect all features and their importances
        all_features = set()
        for region in regions:
            if region in importance_dict:
                all_features.update(importance_dict[region].keys())

        # Find top features across all regions
        feature_scores = {}
        for feature in all_features:
            # Calculate average importance across regions
            score = sum(importance_dict.get(region, {}).get(feature, 0)
                        for region in regions) / len(regions)
            feature_scores[feature] = score

        # Sort features by score and take top_n
        top_features = sorted(feature_scores.keys(),
                              key=lambda f: feature_scores[f],
                              reverse=True)[:top_n]

        # Create data matrix
        data = np.zeros((len(top_features), len(regions)))

        for i, feature in enumerate(top_features):
            for j, region in enumerate(regions):
                data[i, j] = importance_dict.get(region, {}).get(feature, 0)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(data, cmap='viridis')

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Feature Importance', rotation=-90, va="bottom")

        # Set tick labels
        ax.set_xticks(np.arange(len(regions)))
        ax.set_yticks(np.arange(len(top_features)))
        ax.set_xticklabels(regions)
        ax.set_yticklabels(top_features)

        # Rotate x-tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Top {top_n} Feature Importances Across Regions")

        # Add grid lines
        ax.set_xticks(np.arange(len(regions) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(top_features) + 1) - .5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)

        # Adjust layout
        fig.tight_layout()

        # Save figure if filename provided
        if filename:
            self._save_figure(fig, filename)

        return fig

    def create_dashboard(self, results, output_filename='dashboard.html'):
        """
        Create HTML dashboard with all visualization results.

        Args:
            results: Dict with experiment results
            output_filename: Filename for the dashboard HTML

        Returns:
            Path to the dashboard HTML file
        """
        if self.output_dir is None:
            return None

        # Create figures directory
        figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        # Generate and save all figures
        # Feature selection results
        if 'feature_selection' in results:
            for region, region_results in results['feature_selection'].items():
                # Accuracy vs. features
                for n_trees, accuracy_values in region_results['accuracy'].items():
                    feature_sizes = list(range(10, 10 * len(accuracy_values) + 1, 10))
                    self.plot_accuracy_vs_features(
                        feature_sizes, accuracy_values,
                        region=region, n_trees=n_trees,
                        filename=f"feature_selection_{region}_{n_trees}"
                    )

                # Model size vs. features
                model_sizes = region_results.get('model_size', {})
                if model_sizes:
                    feature_sizes = list(range(10, 10 * len(next(iter(model_sizes.values()))) + 1, 10))
                    self.plot_model_size_vs_features(
                        feature_sizes, model_sizes,
                        region=region,
                        filename=f"model_size_{region}"
                    )

        # Cross-dataset results
        if 'cross_dataset' in results:
            for train_region, train_results in results['cross_dataset'].items():
                # Get test regions
                test_regions = [r for r in train_results.keys() if r != train_region]

                for n_trees in train_results.get(test_regions[0], {}):
                    # Collect accuracy values for each test region
                    accuracy_values = {
                        test_region: train_results[test_region][n_trees]
                        for test_region in test_regions
                        if test_region in train_results and n_trees in train_results[test_region]
                    }

                    # Get maximum feature count
                    max_features = max(len(values) for values in accuracy_values.values()) * 10
                    feature_sizes = list(range(10, max_features + 1, 10))

                    self.plot_cross_dataset_accuracy(
                        feature_sizes, accuracy_values,
                        train_region=train_region, test_regions=test_regions,
                        n_trees=n_trees,
                        filename=f"cross_dataset_{train_region}_{n_trees}"
                    )

        # Federated learning results
        if 'federated' in results:
            # Get regions and data portions
            regions = list(results['federated'].keys())
            data_portions = [p for p, _ in results['federated'][regions[0]]]

            # Collect accuracy values for each region
            accuracies = {
                region: [acc for _, acc in results['federated'][region]]
                for region in regions
            }

            self.plot_federated_learning_accuracy(
                data_portions, accuracies, regions,
                filename="federated_learning_accuracy"
            )

        # Distillation results
        if 'distillation' in results:
            # Get regions
            regions = list(set(results['distillation']['teacher'].keys()) - {'global'})

            self.plot_distillation_results(
                results['distillation'], regions,
                filename="distillation_results"
            )

        # Time series results
        if 'time_series' in results:
            # Get regions, strategies, and time indices
            regions = set()
            strategies = set()
            time_indices = []

            for strategy, strategy_results in results['time_series'].items():
                strategies.add(strategy)
                for region, time_results in strategy_results.items():
                    regions.add(region)
                    if not time_indices:
                        time_indices = [r['time_index'] for r in time_results]

            # Collect accuracy values
            accuracy_values = {}
            for strategy in strategies:
                for region in regions:
                    if region in results['time_series'].get(strategy, {}):
                        key = (strategy, region)
                        accuracy_values[key] = [r['accuracy'] for r in results['time_series'][strategy][region]]

            # Get drift points
            drift_points = {}
            if 'drift_detection' in results['time_series']:
                for region, time_results in results['time_series']['drift_detection'].items():
                    drift_points[region] = [i for i, r in enumerate(time_results)
                                            if r.get('drift_detected', False)]

            self.plot_time_series_results(
                time_indices, accuracy_values,
                regions=list(regions), strategies=list(strategies),
                drift_points=drift_points,
                filename="time_series_results"
            )

        # Create HTML dashboard
        html_content = self._generate_dashboard_html()

        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, output_filename)
        with open(dashboard_path, 'w') as f:
            f.write(html_content)

        return dashboard_path

    def _generate_dashboard_html(self):
        """
        Generate HTML for the dashboard.

        Returns:
            HTML content
        """
        # Get all figure files
        figure_files = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith(('.png', '.pdf', '.jpg')):
                    rel_path = os.path.relpath(os.path.join(root, file), self.output_dir)
                    figure_files.append(rel_path)

        # Group figures by experiment
        figure_groups = {}

        for file in figure_files:
            # Extract experiment name from filename
            parts = os.path.basename(file).split('_')
            if len(parts) > 1:
                experiment = parts[0]

                if experiment not in figure_groups:
                    figure_groups[experiment] = []

                figure_groups[experiment].append(file)

        # Generate HTML
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Malware Detection Experiments Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1, h2 {
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .section {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                    margin-bottom: 20px;
                    padding: 20px;
                }
                .figures {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    justify-content: center;
                }
                .figure {
                    max-width: 100%;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .figure img {
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 5px;
                }
                .figure .caption {
                    margin-top: 10px;
                    font-size: 14px;
                    color: #666;
                }
                .tabs {
                    display: flex;
                    margin-bottom: 20px;
                }
                .tab {
                    padding: 10px 20px;
                    background-color: #eee;
                    border: 1px solid #ddd;
                    border-radius: 5px 5px 0 0;
                    cursor: pointer;
                    margin-right: 5px;
                }
                .tab.active {
                    background-color: white;
                    border-bottom: 1px solid white;
                }
                .tab-content {
                    display: none;
                }
                .tab-content.active {
                    display: block;
                }
            </style>
            <script>
                function openTab(evt, tabName) {
                    var i, tabcontent, tablinks;

                    // Hide all tab content
                    tabcontent = document.getElementsByClassName("tab-content");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                    }

                    // Remove active class from tabs
                    tablinks = document.getElementsByClassName("tab");
                    for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }

                    // Show current tab and add active class
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }

                // Open first tab by default when page loads
                window.onload = function() {
                    document.getElementsByClassName("tab")[0].click();
                };
            </script>
        </head>
        <body>
            <div class="container">
                <h1>Cross-Regional Malware Detection Dashboard</h1>

                <div class="tabs">
        """

        # Generate tabs
        for i, experiment in enumerate(figure_groups.keys()):
            active = ' active' if i == 0 else ''
            title = experiment.replace('_', ' ').title()
            html += f'<button class="tab{active}" onclick="openTab(event, \'{experiment}\')">{title}</button>\n'

        html += """
                </div>
        """

        # Generate tab content
        for experiment, files in figure_groups.items():
            title = experiment.replace('_', ' ').title()
            html += f"""
                <div id="{experiment}" class="tab-content">
                    <div class="section">
                        <h2>{title} Results</h2>
                        <div class="figures">
            """

            # Add figures
            for file in files:
                filename = os.path.basename(file)
                caption = ' '.join(filename.split('_')[1:]).split('.')[0].title()

                html += f"""
                            <div class="figure">
                                <img src="{file}" alt="{caption}">
                                <div class="caption">{caption}</div>
                            </div>
                """

            html += """
                        </div>
                    </div>
                </div>
            """

        html += """
            </div>
        </body>
        </html>
        """

        return html