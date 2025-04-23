import numpy as np
import joblib
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from models.adaptive_rf import HeterogeneousRandomForest


class ModelDistiller:
    """
    Implements teacher-student model distillation for malware detection.
    Based on the approach described in the paper.
    """

    def __init__(self, teacher_model=None):
        """
        Initialize the model distiller.

        Args:
            teacher_model: The teacher model (larger model)
        """
        self.teacher_model = teacher_model
        self.student_model = None

    def distill(self, X, student_model=None, n_estimators=None, max_features=None, min_features=None):
        """
        Distill the teacher model into a student model.

        Args:
            X: Feature matrix for distillation
            student_model: Pre-defined student model (if None, a new one will be created)
            n_estimators: Number of estimators for the student model
            max_features: Maximum features for the student model
            min_features: Minimum features for the student model

        Returns:
            Distilled student model
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model has not been set")

        # Generate predictions from the teacher model
        if hasattr(self.teacher_model, 'predict_proba'):
            # Use probability predictions for better knowledge transfer
            teacher_preds = self.teacher_model.predict_proba(X)
            use_proba = True
        else:
            # Use class predictions
            teacher_preds = self.teacher_model.predict(X)
            use_proba = False

        # Create student model
        if student_model is not None:
            # Use provided student model
            self.student_model = clone(student_model)
        elif isinstance(self.teacher_model, HeterogeneousRandomForest):
            # Create a smaller HeterogeneousRandomForest
            teacher_n_estimators = getattr(self.teacher_model, 'n_estimators', 100)
            teacher_max_features = getattr(self.teacher_model, 'max_features', 'auto')

            if n_estimators is None:
                n_estimators = max(10, teacher_n_estimators // 2)

            if max_features is None:
                if isinstance(teacher_max_features, int):
                    max_features = max(10, teacher_max_features // 2)
                else:
                    max_features = 'sqrt'

            if min_features is None:
                min_features = max(1, max_features // 10)

            self.student_model = HeterogeneousRandomForest(
                n_estimators=n_estimators,
                max_features=max_features,
                min_features=min_features,
                random_state=getattr(self.teacher_model, 'random_state', None)
            )
        else:
            # Default student model
            self.student_model = HeterogeneousRandomForest(
                n_estimators=10 if n_estimators is None else n_estimators,
                max_features='sqrt' if max_features is None else max_features,
                min_features=1 if min_features is None else min_features
            )

        # Train student model on teacher's predictions
        if use_proba:
            # Multi-class case: convert to hard labels
            if teacher_preds.shape[1] > 1:
                hard_labels = np.argmax(teacher_preds, axis=1)
                self.student_model.fit(X, hard_labels)
            else:
                # Binary case: use probabilities directly
                self.student_model.fit(X, teacher_preds.ravel())
        else:
            # Use hard labels
            self.student_model.fit(X, teacher_preds)

        return self.student_model

    def evaluate_distillation(self, X_test, y_test):
        """
        Evaluate the distillation quality.

        Args:
            X_test: Test feature matrix
            y_test: True test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model has not been set")

        if self.student_model is None:
            raise ValueError("Student model has not been distilled yet")

        # Teacher predictions
        teacher_preds = self.teacher_model.predict(X_test)
        teacher_acc = accuracy_score(y_test, teacher_preds)

        # Student predictions
        student_preds = self.student_model.predict(X_test)
        student_acc = accuracy_score(y_test, student_preds)

        # Agreement between teacher and student
        agreement = np.mean(teacher_preds == student_preds)

        # Model size comparison
        if hasattr(self.teacher_model, 'estimators_') and hasattr(self.student_model, 'estimators_'):
            teacher_size = len(self.teacher_model.estimators_)
            student_size = len(self.student_model.estimators_)
            size_reduction = 1 - (student_size / teacher_size) if teacher_size > 0 else 0
        else:
            teacher_size = -1
            student_size = -1
            size_reduction = -1

        # Feature subset size comparison
        if hasattr(self.teacher_model, 'feature_subset_sizes_') and hasattr(self.student_model,
                                                                            'feature_subset_sizes_'):
            teacher_features = np.mean(self.teacher_model.feature_subset_sizes_)
            student_features = np.mean(self.student_model.feature_subset_sizes_)
            feature_reduction = 1 - (student_features / teacher_features) if teacher_features > 0 else 0
        else:
            teacher_features = -1
            student_features = -1
            feature_reduction = -1

        return {
            'teacher_accuracy': teacher_acc,
            'student_accuracy': student_acc,
            'accuracy_retention': student_acc / teacher_acc if teacher_acc > 0 else 0,
            'teacher_student_agreement': agreement,
            'teacher_size': teacher_size,
            'student_size': student_size,
            'size_reduction': size_reduction,
            'teacher_features': teacher_features,
            'student_features': student_features,
            'feature_reduction': feature_reduction
        }

    def set_teacher_model(self, teacher_model):
        """
        Set the teacher model.

        Args:
            teacher_model: The new teacher model

        Returns:
            self
        """
        self.teacher_model = teacher_model
        return self

    def get_student_model(self):
        """
        Get the student model.

        Returns:
            Student model
        """
        if self.student_model is None:
            raise ValueError("Student model has not been distilled yet")

        return self.student_model

    def save_models(self, teacher_path, student_path):
        """
        Save both teacher and student models.

        Args:
            teacher_path: Path to save the teacher model
            student_path: Path to save the student model

        Returns:
            Success flag
        """
        if self.teacher_model is None:
            raise ValueError("Teacher model has not been set")

        if self.student_model is None:
            raise ValueError("Student model has not been distilled yet")

        joblib.dump(self.teacher_model, teacher_path)
        joblib.dump(self.student_model, student_path)

        return True

    def load_models(self, teacher_path, student_path=None):
        """
        Load teacher and optionally student models.

        Args:
            teacher_path: Path to the teacher model
            student_path: Path to the student model (optional)

        Returns:
            self
        """
        self.teacher_model = joblib.load(teacher_path)

        if student_path is not None:
            self.student_model = joblib.load(student_path)

        return self


class RegionalDistiller:
    """
    Implements the regional model distillation approach from the paper.
    This creates optimized models for each region from a global model.
    """

    def __init__(self, global_model=None):
        """
        Initialize the regional distiller.

        Args:
            global_model: The global model
        """
        self.global_model = global_model
        self.region_models = {}
        self.feature_requirements = {}

    def set_global_model(self, global_model):
        """
        Set the global model.

        Args:
            global_model: The new global model

        Returns:
            self
        """
        self.global_model = global_model
        return self

    def distill_for_region(self, region_name, X, y, n_estimators=None, max_features=None):
        """
        Distill the global model for a specific region.

        Args:
            region_name: Name of the region
            X: Region-specific feature matrix
            y: Region-specific target vector
            n_estimators: Number of estimators for the regional model
            max_features: Maximum features for the regional model

        Returns:
            Distilled regional model
        """
        if self.global_model is None:
            raise ValueError("Global model has not been set")

        # Create distiller
        distiller = ModelDistiller(self.global_model)

        # Distill model
        regional_model = distiller.distill(X, n_estimators=n_estimators, max_features=max_features)

        # Fine-tune on regional data
        regional_model.fit(X, y)

        # Store model and feature requirements
        self.region_models[region_name] = regional_model

        if hasattr(regional_model, 'feature_subset_sizes_'):
            self.feature_requirements[region_name] = max(regional_model.feature_subset_sizes_)
        else:
            self.feature_requirements[region_name] = max_features

        return regional_model

    def get_region_model(self, region_name):
        """
        Get the model for a specific region.

        Args:
            region_name: Name of the region

        Returns:
            Regional model
        """
        if region_name not in self.region_models:
            raise ValueError(f"No model distilled for region: {region_name}")

        return self.region_models[region_name]

    def get_feature_requirements(self, region_name=None):
        """
        Get feature requirements for regions.

        Args:
            region_name: Name of a specific region (if None, return all)

        Returns:
            Feature requirements
        """
        if region_name is not None:
            if region_name not in self.feature_requirements:
                raise ValueError(f"No feature requirements for region: {region_name}")

            return self.feature_requirements[region_name]
        else:
            return self.feature_requirements

    def evaluate_regional_models(self, region_data):
        """
        Evaluate all regional models.

        Args:
            region_data: Dictionary mapping region names to (X_test, y_test) tuples

        Returns:
            Evaluation results
        """
        results = {}

        for region_name, model in self.region_models.items():
            if region_name not in region_data:
                continue

            X_test, y_test = region_data[region_name]

            # Evaluate model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Store results
            results[region_name] = {
                'accuracy': accuracy,
                'features': self.feature_requirements.get(region_name, -1),
                'estimators': len(model.estimators_) if hasattr(model, 'estimators_') else -1
            }

        return results