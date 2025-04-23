import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    average_precision_score, matthews_corrcoef
)
import time
import logging


class PerformanceMetrics:
    """
    Calculates and tracks various performance metrics for malware detection models.
    Includes both accuracy metrics and efficiency/resource metrics.
    """

    def __init__(self):
        """Initialize the performance metrics tracker."""
        self.metrics = {}
        self.execution_times = {}
        self.resource_usage = {}

    def calculate_classification_metrics(self, y_true, y_pred, y_proba=None, prefix=None):
        """
        Calculate various classification performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            prefix: Prefix for metric names (optional)

        Returns:
            Dictionary of metrics
        """
        if prefix:
            prefix = f"{prefix}_"
        else:
            prefix = ""

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Basic metrics
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            f"{prefix}precision": precision_score(y_true, y_pred),
            f"{prefix}recall": recall_score(y_true, y_pred),
            f"{prefix}f1": f1_score(y_true, y_pred),
            f"{prefix}mcc": matthews_corrcoef(y_true, y_pred),
            f"{prefix}true_positives": tp,
            f"{prefix}false_positives": fp,
            f"{prefix}true_negatives": tn,
            f"{prefix}false_negatives": fn
        }

        # Calculate positive predictive value
        if tp + fp > 0:
            metrics[f"{prefix}ppv"] = tp / (tp + fp)
        else:
            metrics[f"{prefix}ppv"] = 0

        # Calculate negative predictive value
        if tn + fn > 0:
            metrics[f"{prefix}npv"] = tn / (tn + fn)
        else:
            metrics[f"{prefix}npv"] = 0

        # Calculate false positive rate
        if fp + tn > 0:
            metrics[f"{prefix}fpr"] = fp / (fp + tn)
        else:
            metrics[f"{prefix}fpr"] = 0

        # Calculate false negative rate
        if fn + tp > 0:
            metrics[f"{prefix}fnr"] = fn / (fn + tp)
        else:
            metrics[f"{prefix}fnr"] = 0

        # AUC-related metrics if probabilities are provided
        if y_proba is not None:
            try:
                metrics[f"{prefix}roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics[f"{prefix}avg_precision"] = average_precision_score(y_true, y_proba)
            except Exception as e:
                logging.warning(f"Error calculating AUC metrics: {str(e)}")

        # Store metrics
        for key, value in metrics.items():
            self.metrics[key] = value

        return metrics

    def measure_execution_time(self, func, *args, name=None, **kwargs):
        """
        Measure execution time of a function.

        Args:
            func: Function to measure
            *args: Arguments to pass to the function
            name: Name for this measurement (optional)
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Function result and execution time
        """
        # Start time
        start_time = time.time()

        # Execute function
        result = func(*args, **kwargs)

        # End time
        execution_time = time.time() - start_time

        # Store execution time
        if name:
            self.execution_times[name] = execution_time

        return result, execution_time

    def calculate_model_size(self, model, name=None):
        """
        Calculate model size metrics.

        Args:
            model: The model to measure
            name: Name for this measurement (optional)

        Returns:
            Dictionary of size metrics
        """
        size_metrics = {}

        # Number of trees
        if hasattr(model, 'estimators_'):
            size_metrics['n_estimators'] = len(model.estimators_)

            # Number of nodes
            if hasattr(model.estimators_[0], 'tree_'):
                n_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
                size_metrics['n_nodes'] = n_nodes

                # Calculate memory usage (approximate)
                # Each node typically has a feature index, threshold, and pointers
                # Assuming 24 bytes per node (typical for scikit-learn)
                size_metrics['memory_kb'] = n_nodes * 24 / 1024

        # Feature subset sizes
        if hasattr(model, 'feature_subset_sizes_'):
            size_metrics['max_features'] = max(model.feature_subset_sizes_)
            size_metrics['min_features'] = min(model.feature_subset_sizes_)
            size_metrics['avg_features'] = np.mean(model.feature_subset_sizes_)

        # Store size metrics
        if name:
            self.resource_usage[name] = size_metrics

        return size_metrics

    def calculate_prediction_throughput(self, model, X, batch_sizes=None, name=None):
        """
        Calculate prediction throughput (samples per second).

        Args:
            model: The model to measure
            X: Input features
            batch_sizes: List of batch sizes to test (optional)
            name: Name for this measurement (optional)

        Returns:
            Dictionary of throughput metrics
        """
        throughput_metrics = {}

        # Default batch sizes if not provided
        if batch_sizes is None:
            if len(X) > 1000:
                batch_sizes = [1, 10, 100, 1000]
            else:
                batch_sizes = [1, 10, min(100, len(X) // 2)]

        # Measure throughput for each batch size
        for batch_size in batch_sizes:
            if batch_size > len(X):
                continue

            n_batches = min(10, len(X) // batch_size)
            total_time = 0

            # Test with multiple batches
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch = X[start_idx:end_idx]

                # Measure prediction time
                start_time = time.time()
                model.predict(batch)
                total_time += time.time() - start_time

            # Calculate throughput
            avg_time = total_time / n_batches
            throughput = batch_size / avg_time if avg_time > 0 else 0

            throughput_metrics[f'batch_size_{batch_size}'] = {
                'avg_time': avg_time,
                'throughput': throughput
            }

        # Store throughput metrics
        if name:
            self.resource_usage[f'{name}_throughput'] = throughput_metrics

        return throughput_metrics

    def get_all_metrics(self):
        """
        Get all tracked metrics.

        Returns:
            Dictionary of all metrics
        """
        all_metrics = {}
        all_metrics.update(self.metrics)

        # Add execution times
        for name, execution_time in self.execution_times.items():
            all_metrics[f'time_{name}'] = execution_time

        # Add resource usage metrics
        for name, metrics in self.resource_usage.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        all_metrics[f'{name}_{metric_name}'] = value
            else:
                all_metrics[name] = metrics

        return all_metrics

    def to_dataframe(self):
        """
        Convert metrics to DataFrame.

        Returns:
            DataFrame with all metrics
        """
        return pd.DataFrame([self.get_all_metrics()])


class RegionalMetricsTracker:
    """
    Tracks metrics across different regions and models.
    Used to analyze performance differences between regions.
    """

    def __init__(self):
        """Initialize the regional metrics tracker."""
        self.region_metrics = {}
        self.model_metrics = {}

    def track_region(self, region, metrics):
        """
        Track metrics for a region.

        Args:
            region: Region name
            metrics: Metrics dictionary or PerformanceMetrics instance

        Returns:
            self
        """
        if isinstance(metrics, PerformanceMetrics):
            metrics = metrics.get_all_metrics()

        if region not in self.region_metrics:
            self.region_metrics[region] = []

        self.region_metrics[region].append(metrics)
        return self

    def track_model(self, model_name, region, metrics):
        """
        Track metrics for a model and region.

        Args:
            model_name: Name of the model
            region: Region name
            metrics: Metrics dictionary or PerformanceMetrics instance

        Returns:
            self
        """
        if isinstance(metrics, PerformanceMetrics):
            metrics = metrics.get_all_metrics()

        key = f"{model_name}_{region}"
        self.model_metrics[key] = metrics
        return self

    def get_region_metrics(self, region):
        """
        Get metrics for a region.

        Args:
            region: Region name

        Returns:
            List of metrics dictionaries
        """
        return self.region_metrics.get(region, [])

    def get_model_metrics(self, model_name, region=None):
        """
        Get metrics for a model.

        Args:
            model_name: Name of the model
            region: Region name (optional)

        Returns:
            Dictionary of metrics or dict mapping regions to metrics
        """
        if region:
            key = f"{model_name}_{region}"
            return self.model_metrics.get(key, {})
        else:
            # Return metrics for all regions
            return {key.split('_')[1]: metrics
                    for key, metrics in self.model_metrics.items()
                    if key.startswith(f"{model_name}_")}

    def compare_regions(self, metrics_keys=None):
        """
        Compare metrics across regions.

        Args:
            metrics_keys: List of metric keys to compare (optional)

        Returns:
            DataFrame with region comparison
        """
        if not self.region_metrics:
            return pd.DataFrame()

        # Get all regions
        regions = list(self.region_metrics.keys())

        # Get all metrics keys if not provided
        if metrics_keys is None:
            metrics_keys = set()
            for metrics_list in self.region_metrics.values():
                for metrics in metrics_list:
                    metrics_keys.update(metrics.keys())
            metrics_keys = sorted(metrics_keys)

        # Create comparison data
        comparison_data = []

        for metric_key in metrics_keys:
            row = {'metric': metric_key}

            for region in regions:
                # Calculate average value for this metric in this region
                values = [metrics.get(metric_key)
                          for metrics in self.region_metrics[region]
                          if metric_key in metrics]

                if values:
                    row[region] = sum(v for v in values if v is not None) / len(values)
                else:
                    row[region] = None

            comparison_data.append(row)

        # Create DataFrame
        return pd.DataFrame(comparison_data)

    def compare_models(self, models, regions=None, metrics_keys=None):
        """
        Compare metrics across models and regions.

        Args:
            models: List of model names
            regions: List of regions (optional)
            metrics_keys: List of metric keys to compare (optional)

        Returns:
            DataFrame with model comparison
        """
        if not self.model_metrics:
            return pd.DataFrame()

        # Get all regions if not provided
        if regions is None:
            regions = set()
            for key in self.model_metrics:
                if '_' in key:
                    regions.add(key.split('_')[1])
            regions = sorted(regions)

        # Get all metrics keys if not provided
        if metrics_keys is None:
            metrics_keys = set()
            for metrics in self.model_metrics.values():
                metrics_keys.update(metrics.keys())
            metrics_keys = sorted(metrics_keys)

        # Create comparison data
        comparison_data = []

        for model in models:
            for metric_key in metrics_keys:
                row = {'model': model, 'metric': metric_key}

                for region in regions:
                    key = f"{model}_{region}"
                    if key in self.model_metrics and metric_key in self.model_metrics[key]:
                        row[region] = self.model_metrics[key][metric_key]
                    else:
                        row[region] = None

                comparison_data.append(row)

        # Create DataFrame
        return pd.DataFrame(comparison_data)

    def to_dataframe(self):
        """
        Convert all metrics to DataFrame.

        Returns:
            DataFrame with all tracked metrics
        """
        data = []

        # Add region metrics
        for region, metrics_list in self.region_metrics.items():
            for i, metrics in enumerate(metrics_list):
                row = {'type': 'region', 'region': region, 'index': i}
                row.update(metrics)
                data.append(row)

        # Add model metrics
        for key, metrics in self.model_metrics.items():
            if '_' in key:
                model, region = key.split('_', 1)
                row = {'type': 'model', 'model': model, 'region': region}
                row.update(metrics)
                data.append(row)

        # Create DataFrame
        return pd.DataFrame(data)


class TimeSeriesMetricsTracker:
    """
    Tracks metrics over time for drift detection and model updating.
    Used to analyze performance changes and concept drift.
    """

    def __init__(self):
        """Initialize the time series metrics tracker."""
        self.time_metrics = {}
        self.drift_points = {}

    def track_time_point(self, strategy, region, time_index, metrics):
        """
        Track metrics for a time point.

        Args:
            strategy: Strategy name ('no_update', 'drift_detection', 'federated')
            region: Region name
            time_index: Time index or label
            metrics: Metrics dictionary or PerformanceMetrics instance

        Returns:
            self
        """
        if isinstance(metrics, PerformanceMetrics):
            metrics = metrics.get_all_metrics()

        key = f"{strategy}_{region}"

        if key not in self.time_metrics:
            self.time_metrics[key] = {}

        self.time_metrics[key][time_index] = metrics
        return self

    def mark_drift_point(self, region, time_index, drift_level=2):
        """
        Mark a drift point.

        Args:
            region: Region name
            time_index: Time index or label
            drift_level: Drift level (1=warning, 2=drift)

        Returns:
            self
        """
        if region not in self.drift_points:
            self.drift_points[region] = []

        self.drift_points[region].append({
            'time_index': time_index,
            'drift_level': drift_level
        })

        return self

    def get_strategy_metrics(self, strategy, region=None):
        """
        Get metrics for a strategy.

        Args:
            strategy: Strategy name
            region: Region name (optional)

        Returns:
            Dictionary of time metrics
        """
        if region:
            key = f"{strategy}_{region}"
            return self.time_metrics.get(key, {})
        else:
            # Return metrics for all regions
            return {key.split('_')[1]: metrics
                    for key, metrics in self.time_metrics.items()
                    if key.startswith(f"{strategy}_")}

    def get_metric_over_time(self, metric_key, strategy, region):
        """
        Get a specific metric over time.

        Args:
            metric_key: Metric key
            strategy: Strategy name
            region: Region name

        Returns:
            Dictionary mapping time indices to metric values
        """
        key = f"{strategy}_{region}"

        if key not in self.time_metrics:
            return {}

        # Get metric at each time point
        metric_over_time = {}

        for time_index, metrics in self.time_metrics[key].items():
            if metric_key in metrics:
                metric_over_time[time_index] = metrics[metric_key]

        return metric_over_time

    def calculate_improvement(self, metric_key='accuracy'):
        """
        Calculate improvement for each strategy and region.

        Args:
            metric_key: Metric key to calculate improvement for

        Returns:
            DataFrame with improvement calculations
        """
        improvements = []

        for key, time_metrics in self.time_metrics.items():
            if '_' in key:
                strategy, region = key.split('_', 1)

                # Get time indices in order
                time_indices = sorted(time_metrics.keys())

                if not time_indices:
                    continue

                # Get first and last metric values
                first_value = time_metrics[time_indices[0]].get(metric_key)
                last_value = time_metrics[time_indices[-1]].get(metric_key)

                if first_value is not None and last_value is not None:
                    # Calculate improvement
                    improvement = last_value - first_value

                    improvements.append({
                        'strategy': strategy,
                        'region': region,
                        'first_value': first_value,
                        'last_value': last_value,
                        'improvement': improvement,
                        'improvement_percent': improvement / first_value * 100 if first_value > 0 else 0
                    })

        # Create DataFrame
        return pd.DataFrame(improvements)

    def to_dataframe(self):
        """
        Convert time series metrics to DataFrame.

        Returns:
            DataFrame with all time series metrics
        """
        data = []

        # Add time metrics
        for key, time_metrics in self.time_metrics.items():
            if '_' in key:
                strategy, region = key.split('_', 1)

                for time_index, metrics in time_metrics.items():
                    row = {'strategy': strategy, 'region': region, 'time_index': time_index}
                    row.update(metrics)

                    # Check if this is a drift point
                    if region in self.drift_points:
                        for drift_point in self.drift_points[region]:
                            if drift_point['time_index'] == time_index:
                                row['drift_level'] = drift_point['drift_level']
                                break

                    data.append(row)

        # Create DataFrame
        return pd.DataFrame(data)