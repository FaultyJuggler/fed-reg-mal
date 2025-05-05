import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Import acceleration utilities
from utils.hardware.detection import detect_hardware
from utils.hardware.accelerated_models import AcceleratedHeterogeneousRandomForest


# Define paths
EMBER_DATA_DIR = "/ember_data"  # Update this to your actual path

# Define regions
REGIONS = ["US", "JP", "EU", "benign"]  # Update with your actual regions


def load_region_data(region):
    """
    Load the prepared data for a specific region

    Args:
        region (str): Region name (e.g., "US", "JP", "EU", "benign")

    Returns:
        dict: Dictionary containing X and y data for the region
    """
    if region == "benign":
        region_dir = os.path.join(EMBER_DATA_DIR, region)
    else:
        region_dir = os.path.join(EMBER_DATA_DIR, f"region_{region}")

    # Load features (X)
    with open(os.path.join(region_dir, "X_train.pkl"), 'rb') as f:
        X = pickle.load(f)

    # Load labels (y)
    with open(os.path.join(region_dir, "y_train.pkl"), 'rb') as f:
        y = pickle.load(f)

    return {"X": X, "y": y}


def load_all_regions():
    """
    Load data from all regions

    Returns:
        dict: Dictionary with region data
    """
    data = {}

    for region in REGIONS:
        try:
            print(f"Loading {region} data...")
            data[region] = load_region_data(region)
            print(f"  - Loaded {len(data[region]['X'])} samples")
        except Exception as e:
            print(f"Error loading {region} data: {e}")

    return data


def combine_regions(data, regions_to_combine=None):
    """
    Combine data from multiple regions

    Args:
        data (dict): Dictionary of region data
        regions_to_combine (list): List of regions to combine, or None for all

    Returns:
        tuple: (X, y) combined features and labels
    """
    if regions_to_combine is None:
        regions_to_combine = REGIONS

    X_combined = []
    y_combined = []

    for region in regions_to_combine:
        if region in data:
            X_combined.append(data[region]["X"])
            y_combined.append(data[region]["y"])

    return np.vstack(X_combined), np.concatenate(y_combined)


def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train a classifier on the given data with timeouts and progress tracking

    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Labels
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed

    Returns:
        tuple: (model, accuracy, X_train, X_test, y_train, y_test, y_pred)
    """
    import time
    import signal
    import sys

    # Detect hardware capabilities
    hw_config = detect_hardware()
    print(f"Using {hw_config['recommended_backend']} for training")
    print(f"Input data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {np.bincount(y)}")

    # Setup timeout handler
    def timeout_handler(signum, frame):
        print("WARNING: Training operation timed out after 3600 seconds!")
        # Don't exit, just warn

    # Set the timeout handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3600)  # 1 hour timeout

    try:
        # Split data
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

        # Train accelerated model
        print("Training hardware-accelerated Random Forest model...")
        print(f"Creating model with backend={hw_config['recommended_backend']}, n_estimators=100")

        start_time = time.time()

        model = AcceleratedHeterogeneousRandomForest(
            hardware_config=hw_config,
            n_estimators=100,
            random_state=random_state,
            verbose=1
        )

        print("Model created, starting training...")

        # Create progress tracker in a separate thread
        import threading

        def progress_monitor():
            elapsed = 0
            while elapsed < 3600:  # 1 hour max
                time.sleep(60)  # Update every minute
                elapsed = time.time() - start_time
                print(f"Training in progress... Elapsed time: {elapsed / 60:.1f} minutes")
                # If we have access to tree count
                if hasattr(model, 'model') and hasattr(model.model, 'estimators_'):
                    tree_count = len(model.model.estimators_)
                    print(f"Trees built so far: {tree_count}/100")

        # Start progress monitor thread
        progress_thread = threading.Thread(target=progress_monitor)
        progress_thread.daemon = True
        progress_thread.start()

        # Fit the model
        try:
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")
        except Exception as e:
            print(f"ERROR during model training: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a partial model if possible
            if hasattr(model, 'model') and hasattr(model.model, 'estimators_') and len(model.model.estimators_) > 0:
                print(f"Partial model created with {len(model.model.estimators_)} trees")
            else:
                raise

        # Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        # Add labels
        classes = np.unique(y)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # Add text
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.show()

        return model, accuracy, X_train, X_test, y_train, y_test, y_pred

    except Exception as e:
        print(f"ERROR in train_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore original signal handler and clear alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def analyze_regional_performance(data, model):
    """
    Analyze model performance on each region separately

    Args:
        data (dict): Dictionary of region data
        model: Trained model

    Returns:
        dict: Performance metrics by region
    """
    results = {}

    for region in data:
        X = data[region]["X"]
        y = data[region]["y"]

        # Predict
        y_pred = model.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)

        results[region] = {
            "accuracy": accuracy,
            "report": report
        }

        print(f"Performance on {region}: Accuracy = {accuracy:.4f}")

    # Plot accuracies by region
    regions = list(results.keys())
    accuracies = [results[r]["accuracy"] for r in regions]

    plt.figure(figsize=(10, 6))
    plt.bar(regions, accuracies)
    plt.title('Model Accuracy by Region')
    plt.xlabel('Region')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.show()

    return results


def main():
    """
    Main function to run the complete pipeline
    """
    # Load data from all regions
    print("Loading regional data...")
    regional_data = load_all_regions()

    # Option 1: Combine all regions and train a global model
    print("\n--- Training Global Model ---")
    X_global, y_global = combine_regions(regional_data)
    global_model, global_accuracy, *_ = train_model(X_global, y_global)

    # Option 2: Analyze regional performance
    print("\n--- Regional Performance Analysis ---")
    regional_results = analyze_regional_performance(regional_data, global_model)

    # Option 3: Experiment with different region combinations
    print("\n--- Training Model with Malware-Only Regions ---")
    malware_regions = [r for r in REGIONS if r != "benign"]
    X_malware, y_malware = combine_regions(regional_data, malware_regions)
    malware_model, malware_accuracy, *_ = train_model(X_malware, y_malware)

    # Save the model
    import joblib
    joblib.dump(global_model, 'ember_global_model.joblib')
    print("\nGlobal model saved to 'ember_global_model.joblib'")

    return global_model, regional_data, regional_results


if __name__ == "__main__":
    model, data, results = main()