# Getting Started with Cross-Regional Malware Detection

This guide will help you set up your environment and run the experiments to reproduce the results from the "Cross-Regional Malware Detection via Model Distilling and Federated Learning" paper. The implementation allows you to explore how different regional malware characteristics impact detection models and how federated learning and model distillation can improve detection rates.

## Environment Setup

### Prerequisites

- Python 3.8+ installed
- Git installed
- 8GB+ RAM recommended for larger experiments
- GPU optional but helpful for faster training

### Installation Steps

1. **Clone the repository and set up the environment:**

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

2. **Prepare your dataset:**

If you don't have access to the exact datasets used in the paper, you can use any PE malware dataset by organizing it into regional groups:

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

### Using Public Datasets

If you don't have access to regional datasets, you can use publicly available malware datasets and divide them into groups to simulate regional differences:

1. **VirusShare**: A repository of malware samples (requires registration)
2. **Malware Bazaar**: A collection of recent malware samples
3. **VirusTotal Academic**: Academic access to VirusTotal samples

You can divide these randomly or by family/type to create regional groupings.

## Running Your First Experiment

Let's start with the feature selection experiment to understand how many features are optimal for each region:

```bash
# Run feature selection experiment
python main.py --data-dir path/to/data --output-dir results --feature-selection --n-trees 100 --max-features 1000
```

This will:
1. Load the datasets for all regions
2. Run feature selection with different numbers of features
3. Evaluate detection performance
4. Generate plots showing the relationship between feature count and accuracy
5. Save results to the specified output directory

## Understanding the Experiments

### 1. Feature Selection Experiment

This experiment determines the optimal number of features for each region. The key findings from the paper were:
- US dataset: ~270 features optimal
- BR dataset: ~340 features optimal
- JP dataset: ~800 features optimal

Review the generated plots to see if your results match these patterns.

### 2. Cross-Dataset Experiment

This experiment evaluates how models trained on one region perform when tested on samples from other regions:

```bash
python main.py --data-dir path/to/data --output-dir results --cross-dataset
```

In the paper, models trained on one region performed poorly on others (as low as 60% detection for some combinations), highlighting the need for region-specific optimization.

### 3. Federated Learning Experiment

This experiment shows how federated learning can improve detection rates by sharing knowledge between regions:

```bash
python main.py --data-dir path/to/data --output-dir results --federated-learning
```

The key insight is that sharing just 5% of samples between regions can significantly boost detection rates (up to 95% in some cases).

### 4. Time-Series Experiment

This evaluates model performance over time, comparing different update strategies:

```bash
python main.py --data-dir path/to/data --output-dir results --time-series --timestamp-file timestamps.csv
```

The paper showed that while concept drift detection helps maintain performance, only federated learning actually improves detection rates over time.

## Modifying the Code

### Customizing the Feature Extractor

The feature extractor (in `data/feature_extractor.py`) extracts PE file features. You can modify it to extract additional features:

```python
def _extract_header_features(self, pe):
    """Extract features from PE header."""
    features = {}
    
    # Add standard features
    features['machine'] = pe.FILE_HEADER.Machine
    
    # Add your custom features
    features['your_custom_feature'] = calculate_custom_feature(pe)
    
    return features
```

### Testing Different Tree Configurations

The heterogeneous random forest implementation allows trees with different feature set sizes. You can experiment with different configurations:

```python
# In models/adaptive_rf.py
def _generate_feature_subset_sizes(self, n_total_features):
    """Generate custom feature subset sizes for trees."""
    # Change the distribution pattern here
    # E.g., exponential growth instead of linear
```

### Adding New Regions

To add a new region to the experiments, modify the dataset loading code:

```python
# In data/dataset.py
def __init__(self, regions=None):
    # Add your new region
    self.regions = regions if regions is not None else ['US', 'BR', 'JP', 'YOUR_NEW_REGION']
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch sizes or number of features with `--max-features`
2. **Slow Performance**: Use fewer trees with `--n-trees` or process a subset of the data
3. **Feature Extraction Errors**: Some PE files may be malformed; add error handling

### Debugging Tips

1. Add debugging statements in key functions:
```python
print(f"Processing {file_path}, extracted {len(features)} features")
```

2. Run with a smaller dataset first:
```bash
python main.py --data-dir path/to/small_data --feature-selection
```

3. Check intermediate results in the output directory during long experiments

## Next Steps

After running the basic experiments, you can try these advanced explorations:

1. **Compare Different ML Algorithms**: The code uses Random Forests, but you could implement other algorithms
2. **Explore Feature Types**: Analyze which types of features are most discriminative for each region
3. **Optimize Distillation Parameters**: Find the best balance between model size and performance 
4. **Implement Real-time Updates**: Create a system that updates models as new samples are discovered

## Getting Help

If you encounter issues or have questions:
1. Check the paper for implementation details
2. Review the code comments which explain algorithm choices
3. Consult the comprehensive README
4. Modify the debug logging level for more detailed output

Happy malware detection research!
