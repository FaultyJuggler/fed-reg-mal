# ember_diagnosis.py

import os
import pickle
import sys


def inspect_directory(directory):
    """Check if directory exists and list its contents"""
    print(f"\nInspecting directory: {directory}")
    if not os.path.exists(directory):
        print(f"  ERROR: Directory does not exist!")
        return False

    print(f"  Directory exists. Contents:")
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            print(f"  - {item}/ (directory)")
        else:
            print(f"  - {item} ({os.path.getsize(item_path)} bytes)")

    return True


def check_pickle_file(file_path):
    """Check if a pickle file exists and can be loaded"""
    print(f"\nChecking pickle file: {file_path}")

    if not os.path.exists(file_path):
        print(f"  ERROR: File does not exist!")
        return False

    print(f"  File exists ({os.path.getsize(file_path)} bytes)")

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Successfully loaded pickle data")

        if isinstance(data, list) or hasattr(data, 'shape'):
            shape_str = f"shape={data.shape}" if hasattr(data, 'shape') else f"length={len(data)}"
            print(f"  Data info: {type(data)} with {shape_str}")

        return True
    except Exception as e:
        print(f"  ERROR: Could not load pickle data: {e}")
        return False


def main():
    """Main diagnostic function"""
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "/ember_data"

    print(f"Running EMBER data diagnostics on: {base_dir}")

    # Check if base directory exists
    if not inspect_directory(base_dir):
        return

    # Check region directories
    regions = ["US", "JP", "EU", "benign"]
    for region in regions:
        if region == "benign":
            region_dir = os.path.join(base_dir, region)
        else:
            region_dir = os.path.join(base_dir, f"region_{region}")

        if inspect_directory(region_dir):
            # Check pickle files
            x_train_path = os.path.join(region_dir, "X_train.pkl")
            y_train_path = os.path.join(region_dir, "y_train.pkl")

            check_pickle_file(x_train_path)
            check_pickle_file(y_train_path)


if __name__ == "__main__":
    main()