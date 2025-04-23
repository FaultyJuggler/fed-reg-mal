import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import time


class ModelEvaluator:
    """
    Provides utilities for evaluating malware detection models.
    Implements metrics and visualizations from the paper.
    """

    def __init__(self, output_dir=None):
        """
        Initialize the model evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Store evaluation results
        self.results = {}

    def evaluate_model(self, model, X_test, y_test, model_name=None, region=None):
        """
        Evaluate a model using various metrics.

        Args:
            model: The model to evaluate
            X_test: Test feature matrix
            y_test: Test labels
            model_name: Name of the model (optional)
            region: Region name (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        result_key = f"{model_name or 'model'}_{region or 'all'}"
        result = {}

        # Start timing
        start_time = time.time()

        # Make predictions
        y_pred = model.predict(X_test)

        # Basic metrics
        result['accuracy'] = accuracy_score(y_test, y_pred)
        result['precision'] = precision_score(y_test, y_pred)
        result['recall'] = recall_score(y_test, y_pred)
        result['f1'] = f1_score(y_test, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        result['confusion_matrix'] = cm

        # Additional metrics
        result['true_positives'] = cm[1, 1]
        result['false_positives'] = cm[0, 1]
        result['true_negatives'] = cm[0, 0]
        result['false_negatives'] = cm[1, 0]

        # Calculate ROC-AUC if model can predict probabilities
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                result['roc_auc'] = roc_auc_score(y_test, y_proba)
                result['average_precision'] = average_precision_score(y_test, y_proba)

                # Store probabilities for plotting
                result['y_proba'] = y_proba
            except:
                # Handle case where predict_proba fails
                result['roc_auc'] = None
                result['average_precision'] = None
        else:
            result['roc_auc'] = None
            result['average_precision'] = None

        # Prediction time
        result['prediction_time'] = time.time() - start_time

        # Store predictions for later analysis
        result['y_pred'] = y_pred
        result['y_true'] = y_test

        # Store model size if available
        if hasattr(model, 'estimators_'):
            result['n_estimators'] = len(model.estimators_)

            # Check if trees have node count attribute
            if hasattr(model.estimators_[0], 'tree_'):
                result['n_nodes'] = sum(tree.tree_.node_count for tree in model.estimators_)
            else:
                result['n_nodes'] = -1
        else:
            result['n_estimators'] = -1
            result['n_nodes'] = -1

        # Feature subset sizes for heterogeneous models
        if hasattr(model, 'feature_subset_sizes_'):
            result['max_features'] = max(model.feature_subset_sizes_)
            result['min_features'] = min(model.feature_subset_sizes_)
            result['avg_features'] = np.mean(model.feature_subset_sizes_)

        # Store results
        self.results[result_key] = result

        return result

    def plot_roc_curve(self, result_key=None, model_name=None, region=None, ax=None):
        """
        Plot ROC curve for a model.

        Args:
            result_key: Key for stored results
            model_name: Name of the model (used with region if result_key not provided)
            region: Region name (used with model_name if result_key not provided)
            ax: Matplotlib axis (optional)

        Returns:
            Matplotlib axis
        """
        # Determine result key
        if result_key is None:
            result_key = f"{model_name or 'model'}_{region or 'all'}"

        # Get results
        if result_key not in self.results:
            raise ValueError(f"Results not found for {result_key}")

        result = self.results[result_key]

        # Check if we have probability predictions
        if 'y_proba' not in result:
            raise ValueError(f"No probability predictions found for {result_key}")

        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(result['y_true'], result['y_proba'])
        ax.plot(fpr, tpr, lw=2, label=f"{model_name or result_key} (AUC = {result['roc_auc']:.4f})")

        # Add diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', lw=2)

        # Add details
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve for {region or "All Regions"}')
        ax.legend(loc="lower right")
        ax.grid(True)

        # Save figure if output directory is set
        if self.output_dir:
            save_path = os.path.join(self.output_dir, f'roc_curve_{result_key}.png')
            plt.savefig(save_path)

        return ax

    def plot_precision_recall_curve(self, result_key=None, model_name=None, region=None, ax=None):
        """
        Plot precision-recall curve for a model.

        Args:
            result_key: Key for stored results
            model_name: Name of the model (used with region if result_key not provided)
            region: Region name (used with model_name if result_key not provided)
            ax: Matplotlib axis (optional)

        Returns:
            Matplotlib axis
        """
        # Determine result key
        if result_key is None:
            result_key = f"{model_name or 'model'}_{region or 'all'}"

        # Get results
        if result_key not in self.results:
            raise ValueError(f"Results not found for {result_key}")

        result = self.results[result_key]

        # Check if we have probability predictions
        if 'y_proba' not in result:
            raise ValueError(f"No probability predictions found for {result_key}")

        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # Plot precision-recall curve
        precision, recall, _ = precision_recall_curve(result['y_true'], result['y_proba'])
        ax.plot(recall, precision, lw=2,
                label=f"{model_name or result_key} (AP = {result['average_precision']:.4f})")

        # Add details
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve for {region or "All Regions"}')
        ax.legend(loc="lower left")
        ax.grid(True)

        # Save figure if output directory is set
        if self.output_dir:
            save_path = os.path.join(self.output_dir, f'pr_curve_{result_key}.png')
            plt.savefig(save_path)

        return ax

    def plot_confusion_matrix(self, result_key=None, model_name=None, region=None, ax=None):
        """
        Plot confusion matrix for a model.

        Args:
            result_key: Key for stored results
            model_name: Name of the model (used with region if result_key not provided)
            region: Region name (used with model_name if result_key not provided)
            ax: Matplotlib axis (optional)

        Returns:
            Matplotlib axis
        """
        # Determine result key
        if result_key is None:
            result_key = f"{model_name or 'model'}_{region or 'all'}"

        # Get results
        if result_key not in self.results:
            raise ValueError(f"Results not found for {result_key}")

        result = self.results[result_key]

        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        # Get confusion matrix
        cm = result['confusion_matrix']

        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # Add labels and title
        classes = ['Goodware', 'Malware']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               ylabel='True label',
               xlabel='Predicted label',
               title=f'Confusion Matrix for {model_name or result_key}')

        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        # Add accuracy in the title
        if 'accuracy' in result:
            ax.set_title(f'Confusion Matrix for {model_name or result_key}\nAccuracy: {result["accuracy"]:.4f}')

        # Save figure if output directory is set
        if self.output_dir:
            save_path = os.path.join(self.output_dir, f'confusion_matrix_{result_key}.png')
            plt.savefig(save_path)

        return ax

    def compare_models(self, models, regions=None, metrics=None):
        """
        Compare multiple models across regions.

        Args:
            models: List of model names to compare
            regions: List of regions to compare (None for all)
            metrics: List of metrics to compare (None for all)

        Returns:
            DataFrame with comparison results
        """
        if regions is None:
            regions = list(set(key.split('_')[1] for key in self.results.keys()))

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

        # Prepare comparison data
        comparison_data = []

        for model in models:
            for region in regions:
                result_key = f"{model}_{region}"

                if result_key not in self.results:
                    continue

                result = self.results[result_key]
                row = {'model': model, 'region': region}

                for metric in metrics:
                    if metric in result:
                        row[metric] = result[metric]

                comparison_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        # Plot comparison if output directory is set
        if self.output_dir:
            self._plot_model_comparison(df, metrics)

        return df

    def _plot_model_comparison(self, df, metrics):
        """
        Plot model comparison.

        Args:
            df: DataFrame with comparison data
            metrics: List of metrics to plot
        """
        for metric in metrics:
            if metric not in df.columns:
                continue

            # Create figure
            plt.figure(figsize=(12, 8))

            # Get data for plotting
            pivot_df = df.pivot(index='model', columns='region', values=metric)

            # Plot as bar chart
            pivot_df.plot(kind='bar', ax=plt.gca())

            # Add details
            plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Models and Regions')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.grid(True, axis='y')
            plt.tight_layout()

            # Save figure
            save_path = os.path.join(self.output_dir, f'model_comparison_{metric}.png')
            plt.savefig(save_path)
            plt.close()

    def save_results(self, output_dir=None):
        """
        Save evaluation results to CSV files.

        Args:
            output_dir: Directory to save results (uses self.output_dir if None)

        Returns:
            List of saved file paths
        """
        # Use provided output directory or default
        output_dir = output_dir or self.output_dir

        if output_dir is None:
            raise ValueError("No output directory specified")

        os.makedirs(output_dir, exist_ok=True)

        # Create DataFrame with all results
        all_results = []

        for result_key, result in self.results.items():
            # Extract model name and region
            parts = result_key.split('_')
            model_name = parts[0]
            region = '_'.join(parts[1:])

            # Flatten data (exclude arrays)
            row = {'model': model_name, 'region': region}

            for metric, value in result.items():
                if isinstance(value, (np.ndarray, list)) or metric in ('y_pred', 'y_true', 'y_proba'):
                    continue

                row[metric] = value

            all_results.append(row)

        # Create DataFrame
        df = pd.DataFrame(all_results)

        # Save to CSV
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        df.to_csv(csv_path, index=False)

        # Also save individual metric summaries
        saved_files = [csv_path]

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in df.columns:
                metric_path = os.path.join(output_dir, f'{metric}_summary.csv')

                # Create pivot table
                pivot_df = df.pivot(index='model', columns='region', values=metric)
                pivot_df.to_csv(metric_path)

                saved_files.append(metric_path)

        return saved_files


class TimeSeriesEvaluator:
    """
    Specialized evaluator for time-series malware detection.
    Evaluates concept drift and model updating strategies.
    """

    def __init__(self, output_dir=None):
        """
        Initialize the time-series evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Store evaluation results
        self.results = {
            'no_update': {},
            'drift_detection': {},
            'federated': {}
        }

    def evaluate_time_chunk(self, strategy, region, time_index, model, X, y):
        """
        Evaluate a model on a time chunk.

        Args:
            strategy: Update strategy ('no_update', 'drift_detection', 'federated')
            region: Region name
            time_index: Time chunk index (0-based)
            model: Model to evaluate
            X: Feature matrix for this time chunk
            y: True labels for this time chunk

        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        # Calculate ROC-AUC if model can predict probabilities
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)[:, 1]
                roc_auc = roc_auc_score(y, y_proba)
            except:
                roc_auc = None
        else:
            roc_auc = None

        # Store results
        if region not in self.results[strategy]:
            self.results[strategy][region] = []

        self.results[strategy][region].append({
            'time_index': time_index,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'n_samples': len(y)
        })

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def detect_drift(self, y_true, y_pred, warning_threshold=0.05, drift_threshold=0.10):
        """
        Detect concept drift based on error rate changes.
        Simple implementation of the EDDM algorithm.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            warning_threshold: Threshold for warning level
            drift_threshold: Threshold for drift level

        Returns:
            Drift level (0: no drift, 1: warning, 2: drift)
        """
        # Calculate error rate
        error_rate = 1 - accuracy_score(y_true, y_pred)

        # Simple threshold-based detection
        # In a real implementation, this would track error rates over time
        if error_rate > drift_threshold:
            return 2  # Drift detected
        elif error_rate > warning_threshold:
            return 1  # Warning level
        else:
            return 0  # No drift

    def plot_results(self, time_labels=None):
        """
        Plot time-series evaluation results.

        Args:
            time_labels: Labels for time chunks (default: numeric indices)

        Returns:
            Dictionary mapping plot types to file paths
        """
        if self.output_dir is None:
            raise ValueError("No output directory specified")

        saved_plots = {}

        # Plot accuracy over time for each region
        for region in set(region for strategy_results in self.results.values()
                          for region in strategy_results.keys()):
            saved_plots[f'accuracy_over_time_{region}'] = self._plot_accuracy_over_time(region, time_labels)

        # Plot overall comparison
        saved_plots['strategy_comparison'] = self._plot_strategy_comparison(time_labels)

        return saved_plots

    def _plot_accuracy_over_time(self, region, time_labels=None):
        """
        Plot accuracy over time for a specific region.

        Args:
            region: Region to plot
            time_labels: Labels for time chunks

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 8))

        # Plot for each strategy
        for strategy, strategy_results in self.results.items():
            if region not in strategy_results:
                continue

            # Get time indices and accuracy values
            time_indices = [result['time_index'] for result in strategy_results[region]]
            accuracy_values = [result['accuracy'] for result in strategy_results[region]]

            # Use provided time labels or numeric indices
            x_values = time_labels if time_labels is not None else time_indices
            if x_values is not None and len(x_values) > len(time_indices):
                x_values = x_values[:len(time_indices)]

            # Plot line
            label = strategy.replace('_', ' ').title()
            plt.plot(x_values, accuracy_values, marker='o', label=label)

        # Add details
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.title(f'Detection Accuracy Over Time for {region}')
        plt.legend()
        plt.grid(True)

        # Add month numbers if no time labels provided
        if time_labels is None:
            plt.xticks(time_indices)

        # Save figure
        save_path = os.path.join(self.output_dir, f'accuracy_over_time_{region}.png')
        plt.savefig(save_path)
        plt.close()

        return save_path

    def _plot_strategy_comparison(self, time_labels=None):
        """
        Plot comparison of different strategies.

        Args:
            time_labels: Labels for time chunks

        Returns:
            Path to saved plot
        """
        plt.figure(figsize=(12, 8))

        # Calculate average accuracy improvement for each strategy and region
        improvements = {}

        for strategy, strategy_results in self.results.items():
            improvements[strategy] = {}

            for region, results in strategy_results.items():
                if not results:
                    continue

                # Get first and last accuracy
                first_accuracy = results[0]['accuracy']
                last_accuracy = results[-1]['accuracy']

                # Calculate improvement
                improvement = last_accuracy - first_accuracy
                improvements[strategy][region] = improvement

        # Prepare data for plotting
        strategies = list(improvements.keys())
        regions = sorted(set(region for strategy_results in improvements.values()
                             for region in strategy_results.keys()))

        # Set up bar chart
        index = np.arange(len(strategies))
        bar_width = 0.8 / len(regions)

        # Plot bars for each region
        for i, region in enumerate(regions):
            region_improvements = [improvements[strategy].get(region, 0) for strategy in strategies]
            plt.bar(index + i * bar_width - 0.4 + bar_width / 2,
                    region_improvements, bar_width, label=region)

        # Add details
        plt.xlabel('Strategy')
        plt.ylabel('Accuracy Improvement')
        plt.title('Accuracy Improvement by Strategy and Region')
        plt.xticks(index, [s.replace('_', ' ').title() for s in strategies])
        plt.legend()
        plt.grid(True)

        # Save figure
        save_path = os.path.join(self.output_dir, 'strategy_comparison.png')
        plt.savefig(save_path)
        plt.close()

        return save_path

    def save_results(self, output_dir=None):
        """
        Save time-series evaluation results to CSV files.

        Args:
            output_dir: Directory to save results (uses self.output_dir if None)

        Returns:
            List of saved file paths
        """
        # Use provided output directory or default
        output_dir = output_dir or self.output_dir

        if output_dir is None:
            raise ValueError("No output directory specified")

        os.makedirs(output_dir, exist_ok=True)

        saved_files = []

        # Save results for each strategy
        for strategy, strategy_results in self.results.items():
            # Convert to DataFrame
            all_results = []

            for region, results in strategy_results.items():
                for result in results:
                    row = {'region': region}
                    row.update(result)
                    all_results.append(row)

            # Create DataFrame
            if all_results:
                df = pd.DataFrame(all_results)

                # Save to CSV
                csv_path = os.path.join(output_dir, f'{strategy}_results.csv')
                df.to_csv(csv_path, index=False)

                saved_files.append(csv_path)

        # Save summary CSV with accuracies for all strategies and regions
        summary_data = []

        for strategy, strategy_results in self.results.items():
            for region, results in strategy_results.items():
                for result in results:
                    row = {
                        'strategy': strategy,
                        'region': region,
                        'time_index': result['time_index'],
                        'accuracy': result['accuracy']
                    }
                    summary_data.append(row)

        # Create DataFrame
        if summary_data:
            df = pd.DataFrame(summary_data)

            # Save to CSV
            csv_path = os.path.join(output_dir, 'time_series_summary.csv')
            df.to_csv(csv_path, index=False)

            saved_files.append(csv_path)

        return saved_files