import os
import re
import numpy as np
from sklearn.tree import _tree
import yara


class YaraRuleGenerator:
    """
    Generates YARA rules from random forest models.
    Based on the approach described in the paper.
    """

    def __init__(self, model=None, feature_names=None):
        """
        Initialize the YARA rule generator.

        Args:
            model: The trained model
            feature_names: Names of features (required for meaningful rules)
        """
        self.model = model
        self.feature_names = feature_names
        self.rules = []
        self.rule_coverage = {}  # Maps rule index to covered sample indices

    def set_model(self, model):
        """
        Set the model.

        Args:
            model: The new model

        Returns:
            self
        """
        self.model = model
        return self

    def set_feature_names(self, feature_names):
        """
        Set feature names.

        Args:
            feature_names: Names of features

        Returns:
            self
        """
        self.feature_names = feature_names
        return self

    def generate_rules(self, rule_prefix="rule_from_ml"):
        """
        Generate YARA rules from the model.

        Args:
            rule_prefix: Prefix for rule names

        Returns:
            List of YARA rule strings
        """
        if self.model is None:
            raise ValueError("Model has not been set")

        if not hasattr(self.model, 'estimators_'):
            raise ValueError("Model does not have estimators (not a tree-based model)")

        self.rules = []
        self.rule_coverage = {}

        # Generate rules from trees
        for i, tree in enumerate(self.model.estimators_):
            tree_rules = self._generate_rules_from_tree(tree, f"{rule_prefix}_{i}")

            # Store rules and track coverage
            for j, rule in enumerate(tree_rules):
                rule_idx = len(self.rules)
                self.rules.append(rule)
                self.rule_coverage[rule_idx] = self._estimate_rule_coverage(rule)

        return self.rules

    def _generate_rules_from_tree(self, tree, rule_prefix):
        """
        Generate YARA rules from a decision tree.

        Args:
            tree: Decision tree
            rule_prefix: Prefix for rule names

        Returns:
            List of YARA rule strings
        """
        if not hasattr(tree, 'tree_'):
            raise ValueError("Invalid tree (not a decision tree)")

        # Get tree structure
        tree_ = tree.tree_

        # Get feature names
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(tree_.n_features)]

        # Initialize variables
        rules = []

        # Define function to recursively traverse tree and extract paths
        def tree_to_paths(node_id=0, depth=0, path=None):
            if path is None:
                path = []

            # Get node data
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]

            # If leaf node, create a rule
            if left_child == _tree.TREE_LEAF:
                # Get majority class
                value = tree_.value[node_id]
                class_idx = np.argmax(value)

                # Only generate rules for malware class (1)
                if class_idx == 1:
                    rules.append(self._create_yara_rule(path, f"{rule_prefix}_{len(rules)}"))

                return

            # Get feature and threshold
            feature = feature_names[tree_.feature[node_id]]
            threshold = tree_.threshold[node_id]

            # Traverse left child (feature <= threshold)
            path.append((feature, "<=", threshold))
            tree_to_paths(left_child, depth + 1, path)
            path.pop()

            # Traverse right child (feature > threshold)
            path.append((feature, ">", threshold))
            tree_to_paths(right_child, depth + 1, path)
            path.pop()

        # Generate paths from tree
        tree_to_paths()

        return rules

    def _create_yara_rule(self, path, rule_name):
        """
        Create a YARA rule from a decision path.

        Args:
            path: List of (feature, operator, threshold) tuples
            rule_name: Name of the rule

        Returns:
            YARA rule string
        """
        # Initialize rule components
        imports = ["import \"pe\""]
        conditions = []

        # Process path conditions
        for feature, op, threshold in path:
            condition = self._convert_to_yara_condition(feature, op, threshold)
            if condition is not None:
                conditions.append(condition)

        # Generate rule string
        rule_str = f"rule {rule_name} {{\n"

        # Add imports
        if imports:
            for imp in imports:
                rule_str += f"  {imp}\n"

        # Add condition section
        rule_str += "condition:\n"

        # Add conditions
        if conditions:
            rule_str += "  " + "\n  and ".join(conditions) + "\n"
        else:
            rule_str += "  true\n"  # Fallback condition

        rule_str += "}\n"

        return rule_str

    def _convert_to_yara_condition(self, feature, op, threshold):
        """
        Convert a feature condition to a YARA condition.

        Args:
            feature: Feature name
            op: Operator
            threshold: Threshold value

        Returns:
            YARA condition string or None
        """
        # Convert feature name to YARA-compatible format
        feature_lower = feature.lower()

        # Handle different feature types
        if "import" in feature_lower or ".dll" in feature_lower:
            # Import-related features
            match = re.match(r".*\((.*\.dll).*,\s*(.*)\)", feature_lower)
            if match:
                dll_name, function_name = match.groups()
                return f"pe.imports(/{dll_name}/i, /{function_name}/i)"
            else:
                # Generic import check
                return f"pe.imports(/(.)/i, /{feature_lower}/i)"
        elif "export" in feature_lower:
            # Export-related features
            return f"pe.exports(/{feature_lower}/i)"
        elif "section" in feature_lower and "name" in feature_lower:
            # Section name
            return f"pe.sections[0].name contains \"{feature_lower}\""
        elif "characteristic" in feature_lower:
            # PE characteristics
            return f"pe.characteristics & pe.EXECUTABLE_IMAGE"  # Simplified
        elif "subsystem" in feature_lower:
            # PE subsystem
            return f"pe.subsystem == pe.WINDOWS_GUI"  # Simplified
        elif "size" in feature_lower:
            # Size-related features
            if op == "<=" and threshold > 0:
                return f"filesize <= {int(threshold)}"
            elif op == ">" and threshold > 0:
                return f"filesize > {int(threshold)}"
        elif "entropy" in feature_lower:
            # Entropy-related features (not directly supported in YARA)
            return None  # Skip entropy conditions

        # If no specific handling, return a general condition
        return f"pe.number_of_sections > 0"  # Fallback

    def _estimate_rule_coverage(self, rule):
        """
        Estimate the coverage of a rule.

        Args:
            rule: YARA rule string

        Returns:
            Estimated number of samples that would match the rule
        """
        # This is a simplified estimate
        # In reality, you would compile the rule and test against actual samples

        # Count the number of conditions
        conditions = rule.count("and")

        # More conditions typically means fewer matches
        # This is a very rough estimate
        coverage = 100 / (conditions + 1)

        return max(1, int(coverage))

    def save_rules(self, output_dir):
        """
        Save generated rules to files.

        Args:
            output_dir: Directory to save rules

        Returns:
            List of saved file paths
        """
        if not self.rules:
            raise ValueError("No rules have been generated")

        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Save each rule to a file
        file_paths = []

        for i, rule in enumerate(self.rules):
            file_path = os.path.join(output_dir, f"rule_{i:04d}.yar")

            with open(file_path, 'w') as f:
                f.write(rule)

            file_paths.append(file_path)

        return file_paths

    def compile_rules(self, rules=None):
        """
        Compile YARA rules.

        Args:
            rules: List of rule strings (if None, use generated rules)

        Returns:
            Compiled YARA rules
        """
        if rules is None:
            rules = self.rules

        if not rules:
            raise ValueError("No rules to compile")

        # Combine rules
        combined_rules = "\n".join(rules)

        try:
            # Compile rules
            compiled_rules = yara.compile(source=combined_rules)
            return compiled_rules
        except Exception as e:
            print(f"Error compiling YARA rules: {str(e)}")

            # Try compiling rules individually
            valid_rules = []

            for rule in rules:
                try:
                    yara.compile(source=rule)
                    valid_rules.append(rule)
                except:
                    pass

            if valid_rules:
                return yara.compile(source="\n".join(valid_rules))
            else:
                raise ValueError("No valid YARA rules could be compiled")

    def match_rules(self, file_path, rules=None):
        """
        Match YARA rules against a file.

        Args:
            file_path: Path to the file
            rules: Compiled rules (if None, compile generated rules)

        Returns:
            List of matching rules
        """
        if rules is None:
            rules = self.compile_rules()

        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        # Match rules against file
        matches = rules.match(file_path)

        return matches

    def evaluate_rules(self, file_paths, labels, rules=None):
        """
        Evaluate rules against a set of files.

        Args:
            file_paths: List of file paths
            labels: Binary labels (1 for malware, 0 for goodware)
            rules: Compiled rules (if None, compile generated rules)

        Returns:
            Dictionary of evaluation metrics
        """
        if rules is None:
            rules = self.compile_rules()

        # Initialize counters
        tp = 0  # True positives
        fp = 0  # False positives
        tn = 0  # True negatives
        fn = 0  # False negatives

        # Match rules against files
        for file_path, label in zip(file_paths, labels):
            if not os.path.exists(file_path):
                continue

            # Match rules
            matches = rules.match(file_path)

            # Update counters
            if matches and label == 1:
                tp += 1
            elif matches and label == 0:
                fp += 1
            elif not matches and label == 0:
                tn += 1
            else:  # not matches and label == 1
                fn += 1

        # Calculate metrics
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'rule_count': len(self.rules)
        }

    def analyze_rule_performance(self, file_paths, labels):
        """
        Analyze the performance of individual rules.

        Args:
            file_paths: List of file paths
            labels: Binary labels (1 for malware, 0 for goodware)

        Returns:
            Dictionary mapping rule indices to performance metrics
        """
        rule_performance = {}

        # Analyze each rule individually
        for i, rule in enumerate(self.rules):
            # Compile single rule
            try:
                compiled_rule = yara.compile(source=rule)
            except:
                # Skip invalid rules
                continue

            # Initialize counters
            tp = 0  # True positives
            fp = 0  # False positives
            tn = 0  # True negatives
            fn = 0  # False negatives

            # Match rule against files
            for file_path, label in zip(file_paths, labels):
                if not os.path.exists(file_path):
                    continue

                # Match rule
                matches = compiled_rule.match(file_path)

                # Update counters
                if matches and label == 1:
                    tp += 1
                elif matches and label == 0:
                    fp += 1
                elif not matches and label == 0:
                    tn += 1
                else:  # not matches and label == 1
                    fn += 1

            # Calculate metrics
            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            rule_performance[i] = {
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'coverage': self.rule_coverage.get(i, 0)
            }

        return rule_performance