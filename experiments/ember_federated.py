# Create a new file: experiments/ember_federated.py

import os
import signal
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from data.ember_dataset import EmberDataset
from federated.client import FederatedClient
from federated.server import FederatedServer
from utils.hardware.detection import detect_hardware
from utils.hardware.accelerated_models import AcceleratedHeterogeneousRandomForest



class EmberFederatedExperiment:
    """
    Experiment for federated learning on EMBER regional malware data.
    Implements the paper's federated learning approach for cross-regional detection.
    """

    def __init__(self, data_dir, output_dir, hw_config=None, max_samples=None):
        """
        Initialize the experiment.

        Args:
            data_dir: Directory containing the EMBER data
            output_dir: Directory to save results
            hw_config: Hardware configuration (if None, will be detected)
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_samples = max_samples

        # Detect hardware if not provided
        self.hw_config = hw_config if hw_config is not None else detect_hardware()
        print(f"Using {self.hw_config['recommended_backend']} acceleration for training")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # For clean shutdown
        self.interrupt_received = False
        self.current_stage = "initialization"
        self.clients = {}
        self.server = None
        self.global_model = None
        self.results = {
            "global_accuracies": [],
            "local_accuracies": {},
            "final_global_accuracy": 0.0,
            "global_model_path": None,
            "hardware_config": self.hw_config,
            "completed": False,
            "interrupted": False
        }

        # Setup signal handler
        self._setup_signal_handler()

    def _setup_signal_handler(self):
        """Set up signal handler for interruptions."""
        self.original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, sig, frame):
        """Handle interruption signals."""
        # First time: try to clean up
        if not self.interrupt_received:
            print("\n\nInterrupt received. Attempting clean shutdown...")
            print(f"Current stage: {self.current_stage}")
            self.interrupt_received = True
            self.results["interrupted"] = True
            self.results["interrupt_stage"] = self.current_stage

            # Save any progress if possible
            self._save_progress()

            print("Press Ctrl+C again to force exit immediately.")
        else:
            # Second time: force exit
            print("\nForced exit.")
            signal.signal(signal.SIGINT, self.original_sigint)
            sys.exit(1)

    def _save_progress(self):
        """Save any progress made so far."""
        try:
            # Save the current results
            import joblib
            results_path = os.path.join(self.output_dir, "partial_results.joblib")
            joblib.dump(self.results, results_path)
            print(f"Partial results saved to: {results_path}")

            # Save the current model if available
            if self.global_model is not None:
                model_path = os.path.join(self.output_dir, "partial_model.joblib")
                joblib.dump(self.global_model, model_path)
                print(f"Partial model saved to: {model_path}")

            # Plot any available learning curves
            if self.results["global_accuracies"]:
                rounds = len(self.results["global_accuracies"])
                self._plot_learning_curves(
                    self.results["global_accuracies"],
                    self.results["local_accuracies"],
                    rounds
                )
                print(f"Learning curves saved with {rounds} rounds of data.")

        except Exception as e:
            print(f"Error saving progress: {str(e)}")

    def run(self, n_trees=100, federated_features=800, rounds=5):
        """
        Run the experiment with progress tracking to avoid hanging.

        Args:
            n_trees: Number of trees in the random forest
            federated_features: Number of features for federated models
            rounds: Number of federated learning rounds

        Returns:
            Results dictionary
        """
        try:
            print(f"Running EMBER federated learning experiment (rounds={rounds}, trees={n_trees})")

            # Load EMBER dataset
            self.current_stage = "loading_dataset"
            print(f"Loading EMBER dataset from {self.data_dir}...")
            dataset = EmberDataset()
            dataset.load_data(self.data_dir, max_samples=self.max_samples)
            print(f"Dataset loaded successfully with regions: {dataset.regions}")

            # Initialize clients and server
            self.clients = {}
            malware_regions = [r for r in dataset.regions if r != "benign"]
            print(f"Processing malware regions: {malware_regions}")

            # Configure client dictionaries for results
            for region in malware_regions:
                self.results["local_accuracies"][region] = []

            # Data distribution check
            self.current_stage = "checking_data_distribution"
            print("\nChecking data distribution:")
            for region in malware_regions:
                if self.interrupt_received:
                    break

                X_train = dataset.get_feature_matrix(region, "train")
                y_train = dataset.get_target_vector(region, "train")
                X_test = dataset.get_feature_matrix(region, "test")
                y_test = dataset.get_target_vector(region, "test")

                print(f"Region {region}:")
                print(f"  Training: {X_train.shape[0]} samples, {np.mean(y_train)} positive rate")
                print(f"  Testing:  {X_test.shape[0]} samples, {np.mean(y_test)} positive rate")

                # Check if train and test are too similar
                train_mean = np.mean(X_train, axis=0)
                test_mean = np.mean(X_test, axis=0)
                feature_similarity = np.corrcoef(train_mean, test_mean)[0, 1]
                print(f"  Feature similarity between train/test: {feature_similarity:.4f}")

                # Instead of computing correlation for all features at once, just check a few
                print(f"  Checking correlations for top features...")
                top_correlations = []
                for idx in range(min(5, X_train.shape[1])):
                    try:
                        corr = np.corrcoef(X_train[:, idx], y_train)[0, 1]
                        top_correlations.append((idx, corr))
                    except Exception as e:
                        print(f"    Error calculating correlation for feature {idx}: {str(e)}")

                # Sort and display top correlations
                top_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                for idx, corr in top_correlations:
                    print(f"    Feature {idx}: correlation = {corr:.4f}")

            # Create clients with progressive tree building
            self.current_stage = "creating_clients"
            for region in malware_regions:
                if self.interrupt_received:
                    break

                print(f"\nCreating client for region {region}...")
                X_train = dataset.get_feature_matrix(region, "train")
                y_train = dataset.get_target_vector(region, "train")

                # Start with fewer trees for faster initial training
                initial_trees = min(10, n_trees)

                print(f"  Creating model with {initial_trees} trees (out of {n_trees} total)")

                # Use hardware-accelerated model instead of regular HeterogeneousRandomForest
                accelerated_model = AcceleratedHeterogeneousRandomForest(
                    hardware_config=self.hw_config,
                    n_estimators=initial_trees,
                    max_features=min(federated_features, X_train.shape[1]),  # Ensure not larger than data
                    min_features=min(int(federated_features * 0.5), X_train.shape[1] // 2),
                    random_state=42,
                    # Use optimal thread count for the platform
                    n_jobs=self.hw_config["num_cores"],
                    verbose=1  # Add verbosity
                )

                client = FederatedClient(
                    region_name=region,
                    model=accelerated_model
                )

                # Train on local data
                print(f"  Setting data for client {region} ({X_train.shape[0]} samples)")
                client.set_data(X_train, y_train)

                print(f"  Training client {region} with {initial_trees} trees...")
                print(f"  Using {self.hw_config['recommended_backend']} acceleration")

                # Periodic interrupt checking during training
                self.current_stage = f"training_client_{region}"
                client.train()
                print(f"  Initial training of client {region} complete")

                self.clients[region] = client

            # If n_trees > initial_trees, progressively add more trees
            if n_trees > initial_trees and not self.interrupt_received:
                for region, client in self.clients.items():
                    if self.interrupt_received:
                        break

                    print(f"\nProgressively growing more trees for {region}...")
                    total_added = initial_trees

                    # Add trees in batches
                    batch_size = 10
                    while total_added < n_trees and not self.interrupt_received:
                        trees_to_add = min(batch_size, n_trees - total_added)
                        print(
                            f"  Adding {trees_to_add} more trees to client {region} (progress: {total_added}/{n_trees})")

                        # Create a new model with more trees
                        X_train = client.X_train
                        y_train = client.y_train

                        # Create new accelerated model with current trees + new batch
                        new_model = AcceleratedHeterogeneousRandomForest(
                            hardware_config=self.hw_config,
                            n_estimators=total_added + trees_to_add,
                            max_features=min(federated_features, X_train.shape[1]),
                            min_features=min(int(federated_features * 0.5), X_train.shape[1] // 2),
                            random_state=42,
                            n_jobs=self.hw_config["num_cores"],
                            verbose=1
                        )

                        # Copy existing trees (this assumes the internal model structure is compatible)
                        if hasattr(client.model, 'model') and hasattr(client.model.model, 'estimators_'):
                            new_model.model.estimators_ = client.model.model.estimators_.copy()
                            if hasattr(client.model.model, 'feature_subset_sizes_'):
                                new_model.model.feature_subset_sizes_ = client.model.model.feature_subset_sizes_.copy()
                        elif hasattr(client.model, 'estimators_'):
                            new_model.model.estimators_ = client.model.estimators_.copy()
                            if hasattr(client.model, 'feature_subset_sizes_'):
                                new_model.model.feature_subset_sizes_ = client.model.feature_subset_sizes_.copy()

                        # Fit only the new trees
                        self.current_stage = f"adding_trees_{region}_{total_added}"
                        print(f"  Training additional trees using {self.hw_config['recommended_backend']} acceleration")
                        new_model.fit(X_train, y_train, warm_start=True, n_trees_to_add=trees_to_add)

                        # Update client model
                        client.model = new_model
                        total_added += trees_to_add

                        # Evaluate interim progress
                        X_test = dataset.get_feature_matrix(region, "test")
                        y_test = dataset.get_target_vector(region, "test")
                        interim_acc = client.evaluate(X_test, y_test)
                        print(f"  Interim accuracy with {total_added} trees: {interim_acc:.4f}")

            # Create server
            if not self.interrupt_received:
                self.current_stage = "initializing_server"
                print("\nInitializing federated server...")
                self.server = FederatedServer()

                # Federated training process
                global_accuracies = []
                local_accuracies = {region: [] for region in self.clients}

                for round_idx in range(rounds):
                    if self.interrupt_received:
                        break

                    self.current_stage = f"federated_round_{round_idx + 1}"
                    print(f"\n=== Round {round_idx + 1}/{rounds} ===")

                    # Send models to server
                    for client_id, client in self.clients.items():
                        print(f"  Client {client_id} sending model to server...")
                        self.server.receive_model(client_id, client.get_model_for_server())

                    # Aggregate models
                    print("  Server aggregating models...")
                    self.server.aggregate_models(aggregation_method='tree_fusion')
                    print("  Model aggregation complete")
                    self.global_model = self.server.get_global_model()
                    print(f"  Global model has {len(self.global_model.estimators_)} trees")

                    # Wrap global model with accelerated version for predictions if needed
                    if hasattr(self.global_model, 'model'):
                        # Model is already accelerated
                        pass
                    else:
                        # Wrap in accelerated container
                        original_model = self.global_model
                        self.global_model = AcceleratedHeterogeneousRandomForest(
                            hardware_config=self.hw_config,
                            n_estimators=len(original_model.estimators_),
                            random_state=42
                        )
                        # Transfer model data
                        self.global_model.model = original_model

                    # Distribute global model to clients
                    for client_id, client in self.clients.items():
                        print(f"  Client {client_id} updating model...")
                        client.update_model(self.global_model)

                    # Evaluate on test data
                    print("  Evaluating models:")
                    round_accuracies = {}

                    for region in self.clients:
                        if self.interrupt_received:
                            break

                        X_test = dataset.get_feature_matrix(region, "test")
                        y_test = dataset.get_target_vector(region, "test")

                        # Evaluate local model
                        print(f"    Evaluating local model for {region}...")
                        local_acc = self.clients[region].evaluate(X_test, y_test)
                        local_accuracies[region].append(local_acc)
                        self.results["local_accuracies"][region].append(local_acc)

                        # Evaluate global model
                        print(f"    Evaluating global model for {region}...")
                        print(f"    Using {self.hw_config['recommended_backend']} for prediction")
                        y_pred = self.global_model.predict(X_test)
                        global_acc = accuracy_score(y_test, y_pred)

                        print(f"    - {region}: Local={local_acc:.4f}, Global={global_acc:.4f}")
                        round_accuracies[region] = global_acc

                    # Store global accuracy (average across regions)
                    avg_global_acc = np.mean(list(round_accuracies.values()))
                    global_accuracies.append(avg_global_acc)
                    self.results["global_accuracies"].append(avg_global_acc)
                    print(f"  Average global accuracy: {avg_global_acc:.4f}")

                # Save final accuracy
                if global_accuracies:
                    self.results["final_global_accuracy"] = global_accuracies[-1]

                # Final evaluation on benign data
                if "benign" in dataset.regions and not self.interrupt_received:
                    self.current_stage = "evaluating_benign"
                    print("\nEvaluating on benign data:")
                    X_benign = dataset.get_feature_matrix("benign", "test")
                    y_benign = dataset.get_target_vector("benign", "test")

                    print(f"  Making predictions on {len(X_benign)} benign samples...")
                    print(f"  Using {self.hw_config['recommended_backend']} for prediction")
                    y_pred = self.global_model.predict(X_benign)
                    benign_acc = accuracy_score(y_benign, y_pred)

                    print(f"  - Benign accuracy: {benign_acc:.4f}")
                    self.results["benign_accuracy"] = benign_acc

                    # Calculate false positive rate
                    if 0 in y_benign:  # If there are actual negatives
                        true_negatives = ((y_pred == 0) & (y_benign == 0)).sum()
                        false_positives = ((y_pred == 1) & (y_benign == 0)).sum()
                        total_negatives = (y_benign == 0).sum()

                        fpr = false_positives / total_negatives if total_negatives > 0 else 0
                        print(f"  - False positive rate: {fpr:.4f}")
                        self.results["false_positive_rate"] = fpr

                # Plot learning curves
                if not self.interrupt_received and global_accuracies:
                    self.current_stage = "plotting_curves"
                    print("\nGenerating learning curve plots...")
                    self._plot_learning_curves(global_accuracies, local_accuracies, rounds)

                # Evaluate and plot region-specific performance
                if not self.interrupt_received and self.global_model is not None:
                    self.current_stage = "evaluating_regional"
                    print("\nEvaluating regional performance...")
                    self._evaluate_regional_performance(self.global_model, dataset, malware_regions)

                # Save the global model
                if not self.interrupt_received and self.global_model is not None:
                    self.current_stage = "saving_model"
                    print("\nSaving global model...")
                    import joblib
                    model_path = os.path.join(self.output_dir, "ember_global_model.joblib")
                    joblib.dump(self.global_model, model_path)
                    print(f"Global model saved to: {model_path}")
                    self.results["global_model_path"] = model_path

            # Mark as completed if we got this far without interruption
            if not self.interrupt_received:
                self.results["completed"] = True

            print(
                "\nExperiment complete!" if not self.interrupt_received else "\nExperiment interrupted but saved progress.")

            return self.results

        except Exception as e:
            self.current_stage = "error"
            self.results["error"] = str(e)
            print(f"\nError during experiment: {str(e)}")
            import traceback
            traceback.print_exc()

            # Try to save whatever progress we have
            self._save_progress()

            return self.results

        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, self.original_sigint)

    def _plot_learning_curves(self, global_accuracies, local_accuracies, rounds):
        """Plot learning curves for global and local models."""
        plt.figure(figsize=(10, 6))

        # Plot global accuracy
        plt.plot(range(1, rounds + 1), global_accuracies, 'o-',
                 label='Global Model', linewidth=2, markersize=8)

        # Plot local accuracies
        for region, accuracies in local_accuracies.items():
            plt.plot(range(1, rounds + 1), accuracies, 's--',
                     label=f'Local Model ({region})', alpha=0.7)

        plt.xlabel('Federated Learning Round')
        plt.ylabel('Accuracy')
        plt.title('Federated Learning Performance on EMBER Dataset')
        plt.legend()
        plt.grid(True)
        plt.xticks(range(1, rounds + 1))

        # Save the figure
        plt.savefig(os.path.join(self.output_dir, 'ember_federated_learning_curve.png'))
        plt.close()

    def _evaluate_regional_performance(self, global_model, dataset, regions):
        """Evaluate and visualize performance across regions."""
        print("\nEvaluating regional performance of global model:")

        accuracies = []
        region_names = []

        # Evaluate on each region
        for region in regions:
            X_test = dataset.get_feature_matrix(region, "test")
            y_test = dataset.get_target_vector(region, "test")

            y_pred = global_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print(f"  - {region}: Accuracy = {acc:.4f}")
            report = classification_report(y_test, y_pred)
            print(f"    Classification Report:\n{report}")

            # Save confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {region}')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Add text to cells
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'ember_confusion_matrix_{region}.png'))
            plt.close()

            # Store for bar chart
            accuracies.append(acc)
            region_names.append(region)

        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        plt.bar(region_names, accuracies)
        plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                    label=f'Average: {np.mean(accuracies):.4f}')
        plt.xlabel('Region')
        plt.ylabel('Accuracy')
        plt.title('Global Model Performance by Region')
        plt.legend()

        # Add text labels
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.4f}", ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ember_regional_performance.png'))
        plt.close()