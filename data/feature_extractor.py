import os
import pefile
import hashlib
import numpy as np
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer


class EmberFeatureAdapter:
    """
    Adapter for EMBER pre-extracted features to work with your project's pipeline.
    """

    def __init__(self, max_features=1000):
        """
        Initialize the EMBER feature adapter.

        Args:
            max_features: Maximum number of features to use
        """
        self.max_features = max_features
        self.feature_names = None
        self.fitted = False

    def extract_features_from_file(self, file_path):
        """
        Load pre-extracted features from EMBER NPZ file.

        Args:
            file_path: Path to NPZ file with features

        Returns:
            Tuple of (categorical_features, textual_features)
        """
        # Load NPZ file
        data = np.load(file_path)
        features = data['features']

        # Split features into categorical and numerical parts for compatibility
        # with your existing pipeline
        categorical_features = {}

        # Map EMBER features to categorical format expected by your code
        for i in range(min(len(features), self.max_features)):
            feature_name = f"ember_feature_{i}"
            categorical_features[feature_name] = features[i]

        # Return empty string for textual features (not used with EMBER)
        return categorical_features, ""

    def fit_transform(self, file_paths):
        """
        Fit and transform EMBER features.

        Args:
            file_paths: List of paths to NPZ files

        Returns:
            Feature matrix (X)
        """
        # Initialize feature matrix
        X = np.zeros((len(file_paths), self.max_features))

        # Load features from each file
        for i, file_path in enumerate(file_paths):
            categorical_features, _ = self.extract_features_from_file(file_path)

            # Map features to matrix
            for j, (feature_name, value) in enumerate(categorical_features.items()):
                if j < self.max_features:
                    X[i, j] = value

        # Generate feature names
        self.feature_names = [f"ember_feature_{i}" for i in range(self.max_features)]
        self.fitted = True

        return X

    def transform(self, file_paths):
        """
        Transform EMBER features.

        Args:
            file_paths: List of paths to NPZ files

        Returns:
            Feature matrix (X)
        """
        if not self.fitted:
            raise ValueError("Feature adapter must be fitted before transform")

        # Initialize feature matrix
        X = np.zeros((len(file_paths), self.max_features))

        # Load features from each file
        for i, file_path in enumerate(file_paths):
            categorical_features, _ = self.extract_features_from_file(file_path)

            # Map features to matrix
            for j, (feature_name, value) in enumerate(categorical_features.items()):
                if j < self.max_features:
                    X[i, j] = value

        return X

    def get_feature_names(self):
        """Get names of features."""
        if not self.fitted:
            raise ValueError("Feature adapter must be fitted before getting feature names")

        return self.feature_names

class PEFeatureExtractor:
    """
    Extract features from PE files for malware classification.
    Based on the approach in the paper.
    """

    def __init__(self, max_features=1000):
        """
        Initialize the feature extractor.

        Args:
            max_features: Maximum number of features to extract (default: 1000)
        """
        self.max_features = max_features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features // 2,  # Half features for TF-IDF, half for categorical
            analyzer='word',
            token_pattern=r'[A-Za-z0-9_\.]+',
            ngram_range=(1, 2)
        )
        self.categorical_features = []
        self.feature_names = None
        self.fitted = False

    def _extract_header_features(self, pe):
        """Extract features from PE header."""
        features = {}

        # File header features
        try:
            features['machine'] = pe.FILE_HEADER.Machine
            features['num_sections'] = pe.FILE_HEADER.NumberOfSections
            features['timestamp'] = pe.FILE_HEADER.TimeDateStamp
            features['characteristics'] = pe.FILE_HEADER.Characteristics
        except:
            pass

        # Optional header features
        try:
            features['subsystem'] = pe.OPTIONAL_HEADER.Subsystem
            features['dll_characteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
            features['size_code'] = pe.OPTIONAL_HEADER.SizeOfCode
            features['size_data'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
            features['size_uninit_data'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
            features['entry_point'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            features['image_base'] = pe.OPTIONAL_HEADER.ImageBase
        except:
            pass

        # Section features
        try:
            for i, section in enumerate(pe.sections[:5]):  # Consider max 5 sections
                section_name = section.Name.decode('utf-8', 'ignore').strip('\x00')
                features[f'section_{i}_name_hash'] = self._simple_hash(section_name)
                features[f'section_{i}_vsize'] = section.Misc_VirtualSize
                features[f'section_{i}_size'] = section.SizeOfRawData
                features[f'section_{i}_entropy'] = self._calculate_entropy(section.get_data())
                features[f'section_{i}_characteristics'] = section.Characteristics
        except:
            pass

        return features

    def _extract_import_features(self, pe):
        """Extract features from PE imports."""
        imports = []

        try:
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                dll_name = entry.dll.decode('utf-8', 'ignore').lower()

                for imp in entry.imports:
                    if imp.name:
                        func_name = imp.name.decode('utf-8', 'ignore').lower()
                        imports.append(f"{dll_name}.{func_name}")
                    else:
                        imports.append(f"{dll_name}.ord{imp.ordinal}")
        except:
            pass

        return " ".join(imports)

    def _extract_export_features(self, pe):
        """Extract features from PE exports."""
        exports = []

        try:
            for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                if exp.name:
                    func_name = exp.name.decode('utf-8', 'ignore').lower()
                    exports.append(func_name)
        except:
            pass

        return " ".join(exports)

    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        if not data:
            return 0

        byte_counts = defaultdict(int)
        for byte in data:
            byte_counts[byte] += 1

        entropy = 0
        for count in byte_counts.values():
            probability = count / len(data)
            entropy -= probability * np.log2(probability)

        return entropy

    def _simple_hash(self, text):
        """Simple hash function for strings."""
        return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % 10000

    def extract_features_from_file(self, file_path):
        """
        Extract features from a single PE file.

        Args:
            file_path: Path to the PE file

        Returns:
            Tuple of (categorical_features, textual_features)
        """
        categorical_features = {}
        textual_features = ""

        try:
            pe = pefile.PE(file_path)

            # Extract header features
            categorical_features = self._extract_header_features(pe)

            # Extract import features
            import_features = self._extract_import_features(pe)

            # Extract export features
            export_features = self._extract_export_features(pe)

            # Combine textual features
            textual_features = import_features + " " + export_features

        except Exception as e:
            print(f"Error extracting features from {file_path}: {str(e)}")

        return categorical_features, textual_features

    def fit_transform(self, file_paths):
        """
        Fit feature extractor and transform files to feature vectors.

        Args:
            file_paths: List of paths to PE files

        Returns:
            Feature matrix (X)
        """
        categorical_features_list = []
        textual_features_list = []

        for file_path in file_paths:
            cat_features, text_features = self.extract_features_from_file(file_path)
            categorical_features_list.append(cat_features)
            textual_features_list.append(text_features)

        # Fit TF-IDF vectorizer on textual features
        tfidf_features = self.tfidf_vectorizer.fit_transform(textual_features_list)

        # Process categorical features
        self._process_categorical_features(categorical_features_list)

        # Transform categorical features
        cat_features_matrix = self._transform_categorical_features(categorical_features_list)

        # Combine features
        X = np.hstack([cat_features_matrix, tfidf_features.toarray()])

        # Store feature names
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        self.feature_names = self.categorical_features + list(tfidf_feature_names)

        self.fitted = True
        return X

    def transform(self, file_paths):
        """
        Transform files to feature vectors using fitted extractor.

        Args:
            file_paths: List of paths to PE files

        Returns:
            Feature matrix (X)
        """
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted before transform")

        categorical_features_list = []
        textual_features_list = []

        for file_path in file_paths:
            cat_features, text_features = self.extract_features_from_file(file_path)
            categorical_features_list.append(cat_features)
            textual_features_list.append(text_features)

        # Transform textual features
        tfidf_features = self.tfidf_vectorizer.transform(textual_features_list)

        # Transform categorical features
        cat_features_matrix = self._transform_categorical_features(categorical_features_list)

        # Combine features
        X = np.hstack([cat_features_matrix, tfidf_features.toarray()])

        return X

    def _process_categorical_features(self, categorical_features_list):
        """Process categorical features to find common ones."""
        # Count feature occurrences
        feature_counts = defaultdict(int)

        for features in categorical_features_list:
            for feature in features:
                feature_counts[feature] += 1

        # Select top categorical features
        max_cat_features = self.max_features // 2
        self.categorical_features = sorted(
            feature_counts.keys(),
            key=lambda f: feature_counts[f],
            reverse=True
        )[:max_cat_features]

    def _transform_categorical_features(self, categorical_features_list):
        """Transform categorical features to a matrix."""
        X_cat = np.zeros((len(categorical_features_list), len(self.categorical_features)))

        for i, features in enumerate(categorical_features_list):
            for j, feature in enumerate(self.categorical_features):
                if feature in features:
                    X_cat[i, j] = features[feature]

        return X_cat

    def get_feature_names(self):
        """Get names of features."""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted before getting feature names")

        return self.feature_names