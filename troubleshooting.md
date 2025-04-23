# Troubleshooting Guide

This guide addresses common issues you might encounter when working with the Cross-Regional Malware Detection project, particularly when trying to reproduce the experiments from the paper.

## Installation Issues

### Dependency Conflicts

**Issue**: Incompatible package versions causing installation errors.

**Solution**:
```bash
# Create a clean environment
python -m venv fresh_venv
source fresh_venv/bin/activate

# Install dependencies one by one with specific versions
pip install numpy==1.22.4
pip install pandas==1.4.2
pip install scikit-learn==1.1.1
pip install matplotlib==3.5.2
pip install pefile==2023.2.7
pip install yara-python==4.2.3
```

### YARA Installation Problems

**Issue**: `yara-python` failing to install due to missing dependencies.

**Solution**: Install system dependencies first:

```bash
# On Ubuntu/Debian
sudo apt-get install automake libtool make gcc pkg-config

# On macOS
brew install automake libtool

# Then install yara-python
pip install yara-python
```

## Dataset Issues

### File Parsing Errors

**Issue**: Errors when parsing PE files with the feature extractor.

**Solution**: Add more robust error handling to the feature extractor:

```python
# Modify data/feature_extractor.py
def extract_features_from_file(self, file_path):
    try:
        pe = pefile.PE(file_path)
        # Extract features...
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        # Return default/empty features
        return {}, ""
```

### Memory Errors with Large Datasets

**Issue**: Out of memory when processing large datasets.

**Solution**: Process files in batches:

```python
# Add to data/dataset.py
def load_data_in_batches(self, batch_size=1000):
    all_files = self._get_file_paths(directory)
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        # Process batch
        # ...
```

### Missing Timestamp Data

**Issue**: Time-series experiment fails due to missing timestamp information.

**Solution**: Generate synthetic timestamps if real ones aren't available:

```python
# Add to data/dataset.py
def generate_synthetic_timestamps(self, file_paths, start_date='2017-01-01', end_date='2017-12-31'):
    import random
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = (end - start).days
    
    timestamps = {}
    for path in file_paths:
        random_days = random.randint(0, delta)
        timestamp = start + timedelta(days=random_days)
        timestamps[path] = timestamp.strftime('%Y-%m-%d')
        
    return timestamps
```

## Experiment Issues

### Slow Feature Selection

**Issue**: Feature selection experiment takes too long.

**Solution**: Reduce the parameter space and use parallel processing:

```python
# Modify experiments/feature_selection.py
from joblib import Parallel, delayed

def run_parallel(self):
    # Use fewer feature size steps
    feature_sizes = list(range(50, self.max_features + 1, 50))  # Increment by 50 instead of 10
    
    # Parallel processing for different tree counts
    results = Parallel(n_jobs=-1)(
        delayed(self._test_tree_count)(n_trees, feature_sizes) 
        for n_trees in self.n_trees_list
    )
```

### Model Size Explosion

**Issue**: Models become too large with many trees or features.

**Solution**: Implement early stopping based on feature importance:

```python
# Add to models/adaptive_rf.py
def prune_features(self, importance_threshold=0.001):
    """Remove trees with low importance features."""
    if not hasattr(self, 'estimators_'):
        return self
        
    # Get feature importances
    importances = self.feature_importances_
    
    # Find important features
    important_indices = np.where(importances > importance_threshold)[0]
    
    # Keep only trees that primarily use important features
    pruned_estimators = []
    pruned_feature_subset_sizes = []
    
    for i, tree in enumerate(self.estimators_):
        # Check if tree uses important features
        # Simplified check: if the tree's feature set size is small enough
        if self.feature_subset_sizes_[i] <= len(important_indices) * 1.5:
            pruned_estimators.append(tree)
            pruned_feature_subset_sizes.append(self.feature_subset_sizes_[i])
            
    self.estimators_ = pruned_estimators
    self.feature_subset_sizes_ = pruned_feature_subset_sizes
    
    return self
```

### Inconsistent Federated Learning Results

**Issue**: Federated learning produces inconsistent results across runs.

**Solution**: Set fixed random seeds and use deterministic operations:

```python
# Add to federated/server.py
def aggregate_models(self, aggregation_method='model_averaging', random_seed=42):
    """Aggregate client models with deterministic behavior."""
    # Set random seed
    np.random.seed(random_seed)
    
    # Rest of the method...
```

## YARA Rule Generation Issues

### Invalid Rules

**Issue**: Generated YARA rules have syntax errors or don't compile.

**Solution**: Add validation and cleanup for rule generation:

```python
# Add to rules/yara_generator.py
def _sanitize_rule_name(self, name):
    """Ensure rule name is valid for YARA."""
    # Replace invalid characters
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # Ensure it doesn't start with a number
    if sanitized[0].isdigit():
        sanitized = 'rule_' + sanitized
        
    return sanitized

def _validate_condition(self, condition):
    """Validate and fix YARA conditions."""
    # Remove empty conditions
    if not condition or condition.strip() == '':
        return "true"
        
    # Fix common syntax issues
    # ...
    
    return condition
```

### Performance Testing Issues

**Issue**: Slow YARA rule matching when evaluating many rules.

**Solution**: Split rules into smaller sets and test in parallel:

```python
# Add to rules/yara_generator.py
def parallel_match_rules(self, file_paths, n_jobs=4):
    """Match rules against files in parallel."""
    from joblib import Parallel, delayed
    
    # Split rules into chunks
    rule_chunks = []
    chunk_size = max(1, len(self.rules) // n_jobs)
    
    for i in range(0, len(self.rules), chunk_size):
        rule_chunks.append(self.rules[i:i+chunk_size])
        
    # Function to match a subset of rules
    def match_chunk(rules_chunk, paths):
        compiled = self.compile_rules(rules_chunk)
        results = []
        
        for path in paths:
            matches = compiled.match(path)
            results.append((path, matches))
            
        return results
        
    # Process in parallel
    all_results = Parallel(n_jobs=n_jobs)(
        delayed(match_chunk)(chunk, file_paths)
        for chunk in rule_chunks
    )
    
    # Combine results
    combined_results = {}
    for results in all_results:
        for path, matches in results:
            if path not in combined_results:
                combined_results[path] = []
            combined_results[path].extend(matches)
            
    return combined_results
```

## Visualization Issues

### Missing or Incomplete Plots

**Issue**: Plots don't show expected data or are incomplete.

**Solution**: Add more robust plotting code with error checking:

```python
# Add to utils/visualization.py
def safe_plot(x_data, y_data, title, xlabel, ylabel, output_path, 
             plot_type='line', check_data=True):
    """Plot with error checking and fallbacks."""
    if check_data:
        # Verify data is valid
        if len(x_data) == 0 or len(y_data) == 0:
            print(f"Warning: Empty data for plot '{title}'")
            return False
            
        if len(x_data) != len(y_data):
            print(f"Warning: Mismatched data lengths for plot '{title}'")
            # Truncate to shorter length
            min_len = min(len(x_data), len(y_data))
            x_data = x_data[:min_len]
            y_data = y_data[:min_len]
    
    try:
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'line':
            plt.plot(x_data, y_data, marker='o')
        elif plot_type == 'bar':
            plt.bar(x_data, y_data)
        elif plot_type == 'scatter':
            plt.scatter(x_data, y_data)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"Error creating plot '{title}': {str(e)}")
        return False
```

## Performance Optimization

### Slow Feature Extraction

**Issue**: Feature extraction takes too long for large datasets.

**Solution**: Implement caching for extracted features:

```python
# Add to data/feature_extractor.py
import hashlib
import pickle
import os

def extract_features_with_cache(self, file_path, cache_dir='feature_cache'):
    """Extract features with caching for faster processing."""
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache key from file path and file modification time
    file_stat = os.stat(file_path)
    cache_key = f"{file_path}_{file_stat.st_mtime}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_hash}.pkl")
    
    # Check if cache exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass  # Fall back to extraction if cache loading fails
    
    # Extract features
    features = self.extract_features_from_file(file_path)
    
    # Save to cache
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(features, f)
    except:
        pass  # Continue even if caching fails
        
    return features
```

### Memory-Efficient Model Storage

**Issue**: Models become too large to store in memory.

**Solution**: Implement partial model loading and processing:

```python
# Add to models/adaptive_rf.py
def save_trees_separately(self, directory):
    """Save each tree separately for memory-efficient loading."""
    os.makedirs(directory, exist_ok=True)
    
    # Save metadata
    metadata = {
        'n_estimators': len(self.estimators_),
        'feature_subset_sizes': self.feature_subset_sizes_,
        # Other model parameters...
    }
    
    with open(os.path.join(directory, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        
    # Save trees
    for i, tree in enumerate(self.estimators_):
        tree_path = os.path.join(directory, f"tree_{i}.pkl")
        joblib.dump(tree, tree_path)
        
def load_trees_on_demand(self, directory):
    """Load trees only when needed to save memory."""
    # Load metadata
    with open(os.path.join(directory, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
        
    # Set model parameters
    self.feature_subset_sizes_ = metadata['feature_subset_sizes']
    n_estimators = metadata['n_estimators']
    
    # Create tree loader functions
    def load_tree(index):
        tree_path = os.path.join(directory, f"tree_{index}.pkl")
        return joblib.load(tree_path)
        
    # Store tree loaders instead of actual trees
    self.tree_loaders = [load_tree for _ in range(n_estimators)]
    
    # Override predict method to load trees on demand
    original_predict = self.predict
    def predict_with_lazy_loading(X):
        # Load trees if not already loaded
        if not hasattr(self, 'estimators_'):
            self.estimators_ = [loader(i) for i, loader in enumerate(self.tree_loaders)]
        return original_predict(X)
        
    self.predict = predict_with_lazy_loading
```

I hope this troubleshooting guide helps you overcome common issues when working with the project!
