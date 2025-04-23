# Cross-Regional Malware Detection

This project implements the system described in the paper "Cross-Regional Malware Detection via Model Distilling and Federated Learning" by Marcus Botacin and Heitor Gomes. It addresses the challenge of detecting malware across different geographical regions using machine learning with federated learning and model distillation.

## Project Overview

The paper introduces a malware detection approach that:

1. Recognizes that different regions (US, Brazil, Japan) have different malware characteristics
2. Uses federated learning to share knowledge between regional models
3. Employs model distillation to create efficient endpoint models
4. Uses heterogeneous random forests to optimize feature usage

This implementation recreates the experiments from the paper to verify and extend the authors' findings.

## Key Features

- **Regional Dataset Handling**: Process datasets from different regions (US, BR, JP)
- **Feature Selection**: Find optimal feature sets for each region
- **Federated Learning**: Share knowledge between regional models without sharing raw data
- **Model Distillation**: Create compact models for deployment on endpoints
- **Heterogeneous Random Forests**: Custom RF implementation that supports trees with varying feature counts
- **YARA Rule Generation**: Convert ML models to YARA rules for deployment in antivirus systems
- **Time-Series Analysis**: Evaluate concept drift detection and model updating strategies

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/malware-federated-distill.git
cd malware-federated-distill

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Prepare your dataset in the following structure:

```
data/
├── US/
│   ├── malware/
│   │   └── [PE files]
├── BR/
│   ├── malware/
│   │   └── [PE files]
├── JP/
│   ├── malware/
│   │   └── [PE files]
└── goodware/
    └── [PE files]
```

For time-series analysis, prepare a CSV file with timestamp information:

```
file_path,timestamp
path/to/file1.exe,2017-01-15
path/to/file2.exe,2017-02-03
...
```

### Running Experiments

Run individual experiments or all of them:

```bash
# Run feature selection experiment
python main.py --data-dir path/to/data --feature-selection

# Run cross-dataset experiment
python main.py --data-dir path/to/data --cross-dataset

# Run federated learning experiment
python main.py --data-dir path/to/data --federated-learning

# Run time-series experiment
python main.py --data-dir path/to/data --time-series --timestamp-file timestamps.csv

# Generate YARA rules
python main.py --data-dir path/to/data --yara-rules

# Run all experiments
python main.py --data-dir path/to/data --all
```

### Experiment Options

```
usage: main.py [-h] --data-dir DATA_DIR [--output-dir OUTPUT_DIR] [--timestamp-file TIMESTAMP_FILE]
               [--feature-selection] [--cross-dataset] [--federated-learning] [--time-series]
               [--yara-rules] [--all] [--n-trees N_TREES] [--max-features MAX_FEATURES]

optional arguments:
  -h, --help            show this help message and exit
  --data-dir DATA_DIR   Directory containing the datasets
  --output-dir OUTPUT_DIR
                        Directory to save results
  --timestamp-file TIMESTAMP_FILE
                        CSV file with timestamp information for time-series analysis
  --feature-selection   Run feature selection experiment
  --cross-dataset       Run cross-dataset experiment
  --federated-learning  Run federated learning experiment
  --time-series         Run time-series experiment
  --yara-rules          Generate YARA rules from models
  --all                 Run all experiments
  --n-trees N_TREES     Number of trees for random forest models
  --max-features MAX_FEATURES
                        Maximum number of features to test
```

## Project Structure

```
federated_malware_detection/
├── data/
│   ├── dataset.py          # Dataset handling and loading
│   ├── feature_extractor.py # PE feature extraction
│   └── feature_selector.py  # Feature selection implementation
├── models/
│   ├── adaptive_rf.py       # Adaptive Random Forest with heterogeneous trees
│   └── evaluation.py        # Model evaluation utilities
├── federated/
│   ├── client.py            # Regional model client
│   └── server.py            # Global model aggregation
├── distillation/
│   ├── teacher_student.py   # Model distillation implementation
│   └── optimization.py      # Model compression techniques
├── rules/
│   └── yara_generator.py    # Generate YARA rules from models
├── experiments/
│   ├── feature_selection.py # Feature selection experiments
│   ├── cross_dataset.py     # Cross-region model evaluation
│   ├── federated_learning.py # FL experiments
│   └── time_series.py       # Time-series evaluation
├── utils/
│   ├── config.py            # Configuration parameters
│   └── visualization.py     # Plotting and visualization
└── main.py                  # Main entry point
```

## Experimental Results

After running the experiments, results will be saved to the specified output directory. Each experiment produces:

1. **Feature Selection**: Analysis of optimal feature set sizes for each region
2. **Cross-Dataset**: Evaluation of model performance across different regions
3. **Federated Learning**: Performance improvement with knowledge sharing
4. **Time-Series**: Comparison of model updating strategies
5. **YARA Rules**: Generated detection rules for deployment

Results are presented as:
- CSV files with detailed metrics
- Figures visualizing key findings
- YARA rule files ready for deployment

## Key Insights from the Paper

1. **Regional Differences**: Different regions require different numbers of features for optimal detection (e.g., 270 for US, 800 for JP).
2. **Model Size Trade-offs**: Adding more trees (e.g., 0.5% gain for US) doesn't significantly improve detection for individual datasets, supporting the benefit of distilling smaller models.
3. **Global Knowledge Benefits**: Larger models improve generalization across regions (e.g., from 60% to 95% for the US model), supporting the benefits of having global models.
4. **Time-series Performance**: While retraining on concept drift detection allows recovering original detection rates, detection rates only increase when global model data is used.

## Citation

If you use this implementation in your research, please cite the original paper:

```
@inproceedings{botacin2024cross,
  title={Cross-Regional Malware Detection via Model Distilling and Federated Learning},
  author={Botacin, Marcus and Gomes, Heitor},
  booktitle={The 27th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2024)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to Marcus Botacin and Heitor Gomes for their research and insights
- Special thanks to the RAID 2024 conference for publishing this important work
