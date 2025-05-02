import os
import argparse
from datetime import datetime
import signal
import sys

from data.dataset import MalwareDataset
from experiments.feature_selection import FeatureSelectionExperiment
from experiments.cross_dataset import CrossDatasetExperiment
from experiments.federated_learning import FederatedLearningExperiment
from experiments.time_series import TimeSeriesExperiment
from rules.yara_generator import YaraRuleGenerator

from utils.hardware.detection import detect_hardware

# Global flag for tracking interrupt
interrupt_received = False


def signal_handler(sig, frame):
    """Handle Ctrl+C (SIGINT) for clean program termination."""
    global interrupt_received

    if not interrupt_received:
        print("\n\nInterrupt received. Attempting to shut down cleanly...")
        print("This may take a moment to release hardware resources.")
        print("Press Ctrl+C again to force immediate exit.")
        interrupt_received = True
    else:
        print("\nForced exit.")
        sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Cross-Regional Malware Detection via Model Distilling and Federated Learning')

    # for prepped EMBER data split into simulated regions
    parser.add_argument('--ember-data', action='store_true', help='Use EMBER dataset for experiments')
    parser.add_argument('--ember-dir', type=str, default='/ember_data', help='Directory containing EMBER data')

    # Add the max-samples argument
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum total samples to use for training (per region for EMBER data)')

    # Data directories
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing the datasets')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--timestamp-file', type=str, default=None,
                        help='CSV file with timestamp information for time-series analysis')

    # Experiment selection
    parser.add_argument('--feature-selection', action='store_true', help='Run feature selection experiment')
    parser.add_argument('--cross-dataset', action='store_true', help='Run cross-dataset experiment')
    parser.add_argument('--federated-learning', action='store_true', help='Run federated learning experiment')
    parser.add_argument('--time-series', action='store_true', help='Run time-series experiment')
    parser.add_argument('--yara-rules', action='store_true', help='Generate YARA rules from models')
    parser.add_argument('--all', action='store_true', help='Run all experiments')

    # Model parameters
    parser.add_argument('--n-trees', type=int, default=100, help='Number of trees for random forest models')
    parser.add_argument('--max-features', type=int, default=1500, help='Maximum number of features to test')

    return parser.parse_args()


def main():
    """Main entry point."""

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        args = parse_args()

        # Detect hardware capabilities
        hw_config = detect_hardware()
        print(f"\n=== Hardware Configuration ===")
        print(f"Platform: {hw_config['platform']} {hw_config['architecture']}")
        print(f"Acceleration: {hw_config['recommended_backend']}")
        if hw_config['cuda_available']:
            print(f"CUDA devices: {hw_config['cuda_devices']}")
        if hw_config['mps_available']:
            print(f"Apple Silicon MPS acceleration available")
        print(f"CPU cores: {hw_config['num_cores']}")

        # Set up output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f'experiment_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        # Save experiment configuration
        with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
            for arg, value in vars(args).items():
                f.write(f'{arg}: {value}\n')

        # Run selected experiments
        if args.feature_selection or args.all:
            print("\n=== Running Feature Selection Experiment ===")
            experiment = FeatureSelectionExperiment(args.data_dir, output_dir)
            experiment.run(max_features=args.max_features)

        if args.cross_dataset or args.all:
            print("\n=== Running Cross-Dataset Experiment ===")
            experiment = CrossDatasetExperiment(args.data_dir, output_dir)
            experiment.run(max_features=args.max_features)

        if args.federated_learning or args.all:
            print("\n=== Running Federated Learning Experiment ===")
            experiment = FederatedLearningExperiment(args.data_dir, output_dir)
            experiment.run(n_trees=args.n_trees, federated_features=800)

        if args.time_series or args.all:
            print("\n=== Running Time-Series Experiment ===")
            experiment = TimeSeriesExperiment(args.data_dir, args.timestamp_file, output_dir)
            experiment.run(n_trees=args.n_trees)

        if args.yara_rules or args.all:
            print("\n=== Generating YARA Rules ===")
            # Load dataset for rule generation
            dataset = MalwareDataset(regions=['US', 'BR', 'JP'])
            dataset.load_data(args.data_dir)

            # Generate YARA rules for each region
            for region in dataset.regions:
                print(f"Generating YARA rules for {region}...")
                # Train a model for this region
                from models.adaptive_rf import HeterogeneousRandomForest
                from data.feature_selector import FeatureSelector

                # Get optimal feature count for region
                if region == 'US':
                    n_features = 300
                elif region == 'BR':
                    n_features = 400
                else:  # JP
                    n_features = 800

                # Get data
                X_train = dataset.X['train'][region]
                y_train = dataset.y['train'][region]

                # Select features
                selector = FeatureSelector(selection_method='f_score', n_features=n_features)
                X_train_selected = selector.fit_transform(X_train, y_train)

                # Train model
                model = HeterogeneousRandomForest(n_estimators=args.n_trees, random_state=42)
                model.fit(X_train_selected, y_train)

                # Generate YARA rules
                rule_generator = YaraRuleGenerator(model, feature_names=dataset.feature_names)
                rules = rule_generator.generate_rules(rule_prefix=f"rule_{region}")

                # Save rules
                rules_dir = os.path.join(output_dir, f'yara_rules_{region}')
                rule_generator.save_rules(rules_dir)

                print(f"  Generated {len(rules)} rules")

        if args.ember_data:
            print("\n=== Running EMBER Federated Learning Experiment ===")
            from experiments.ember_federated import EmberFederatedExperiment
            experiment = EmberFederatedExperiment(args.ember_dir, output_dir, hw_config=hw_config, max_samples=args.max_samples)
            # experiment will handle its own interrupts
            results = experiment.run(n_trees=args.n_trees, federated_features=800)

            if results.get("interrupted", False):
                print("\nEMBER experiment was interrupted but saved partial progress.")

    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        # Add any final cleanup here
    finally:
        # Restore original handler
        signal.signal(signal.SIGINT, original_sigint)

        # Final cleanup
        print("\nCleaning up resources...")
        try:
            # Clean up any GPU resources
            if 'hw_config' in locals() and hw_config.get('cuda_available', False):
                import torch
                torch.cuda.empty_cache()
                print("CUDA resources cleaned up.")

            if 'hw_config' in locals() and hw_config.get('mps_available', False):
                import torch
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    print("MPS resources cleaned up.")
        except:
            pass

    return 0

if __name__ == '__main__':
    main()