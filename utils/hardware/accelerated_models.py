# accelerated_models.py
import numpy as np
import inspect  # Add this import at the top
from sklearn.ensemble import RandomForestClassifier


class HardwareAcceleratedModel:
    """Base class for hardware-accelerated models"""

    def __init__(self, hardware_config, model_type="random_forest", **kwargs):
        self.hardware_config = hardware_config
        self.backend = hardware_config["recommended_backend"]
        self.model_type = model_type
        self.model = None
        self.model_params = kwargs

        # Add debug print to track n_estimators
        if 'n_estimators' in kwargs:
            print(f"[DEBUG] HardwareAcceleratedModel received n_estimators={kwargs['n_estimators']}")

        self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "random_forest":
            if self.backend == "cpu":
                # Use sklearn with optimal threading
                self.model = RandomForestClassifier(
                    n_jobs=self.hardware_config["num_cores"],
                    **self.model_params
                )
            elif self.backend == "cuda":
                try:
                    # Try cuML for GPU-accelerated RF
                    from cuml.ensemble import RandomForestClassifier as cuRF
                    self.model = cuRF(**self.model_params)
                except ImportError:
                    print("cuML not available, falling back to sklearn")
                    self.model = RandomForestClassifier(
                        n_jobs=self.hardware_config["num_cores"],
                        **self.model_params
                    )
            elif self.backend == "mps":
                # For Apple Silicon, use optimal sklearn config or specialized libraries
                try:
                    # Some specialized MPS-compatible libraries might be available
                    # For now, use optimized sklearn
                    self.model = RandomForestClassifier(
                        n_jobs=self.hardware_config["num_cores"],
                        **self.model_params
                    )
                except Exception:
                    self.model = RandomForestClassifier(**self.model_params)

    def fit(self, X, y, warm_start=False, n_trees_to_add=None):
        """Fit the model to the data with better error reporting"""
        try:
            # Debug info before starting fit
            print(f"[MPS_DEBUG] Starting model fit with {X.shape[0]} samples, {X.shape[1]} features")
            print(f"[MPS_DEBUG] Backend: {self.backend}, Model type: {self.model_type}")
            print(f"[MPS_DEBUG] Memory footprint - X: {X.nbytes / (1024 * 1024):.2f} MB")

            # Preprocess data based on backend
            print(f"[MPS_DEBUG] Preprocessing data...")
            X_processed, y_processed = self._preprocess_data(X, y)
            print(f"[MPS_DEBUG] Preprocessing complete")

            # Call underlying model fit with progress timer
            print(f"[MPS_DEBUG] Starting underlying model training...")
            import time
            start_time = time.time()

            if hasattr(self.model, 'fit') and callable(self.model.fit):
                # Check for specialized fit parameters
                if warm_start and hasattr(self.model, 'fit') and 'warm_start' in inspect.signature(
                        self.model.fit).parameters:
                    print(f"[MPS_DEBUG] Using warm start with {n_trees_to_add} additional trees")
                    self.model.fit(X_processed, y_processed, warm_start=warm_start, n_trees_to_add=n_trees_to_add)
                else:
                    self.model.fit(X_processed, y_processed)

                elapsed = time.time() - start_time
                print(f"[MPS_DEBUG] Model training completed in {elapsed:.2f} seconds")
            else:
                print(f"[MPS_DEBUG] ERROR: Model does not have a fit method")
                raise AttributeError("Model does not have a fit method")

            return self
        except Exception as e:
            print(f"[MPS_DEBUG] ERROR in model fitting: {str(e)}")
            print(f"[MPS_DEBUG] Model parameters: {self.model_params}")
            import traceback
            traceback.print_exc()
            # Don't swallow the exception - re-raise to allow proper handling
            raise

    def predict(self, X):
        """Make predictions"""
        X_processed = self._preprocess_input(X)
        predictions = self.model.predict(X_processed)
        return predictions

    def predict_proba(self, X):
        """Predict probabilities"""
        X_processed = self._preprocess_input(X)
        return self.model.predict_proba(X_processed)

    def _preprocess_data(self, X, y=None):
        """Preprocess data for the selected backend with better error handling"""
        try:
            print(f"[MPS_DEBUG] Preprocessing data for backend: {self.backend}")
            print(f"[MPS_DEBUG] Input shapes - X: {X.shape}, y: {y.shape if y is not None else None}")

            if self.backend == "cuda":
                try:
                    import cupy as cp
                    print("[MPS_DEBUG] Converting data to CUDA format")
                    X_processed = cp.array(X)
                    y_processed = cp.array(y) if y is not None else None
                    print("[MPS_DEBUG] CUDA conversion successful")
                    return X_processed, y_processed
                except ImportError as e:
                    print(f"[MPS_DEBUG] CUDA conversion failed, using CPU arrays: {str(e)}")
                except Exception as e:
                    print(f"[MPS_DEBUG] Error during CUDA conversion: {str(e)}")
                    raise
            elif self.backend == "mps":
                try:
                    # For Apple Silicon, check if we can use optimized arrays
                    # This is a placeholder - MPS doesn't directly support scikit-learn yet
                    print("[MPS_DEBUG] No specialized MPS preprocessing applied, using standard arrays")
                except Exception as e:
                    print(f"[MPS_DEBUG] Error during MPS-specific processing: {str(e)}")

            # Default for CPU and fallback
            print("[MPS_DEBUG] Using standard NumPy arrays")
            return X, y

        except Exception as e:
            print(f"[MPS_DEBUG] CRITICAL ERROR in data preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original data as fallback
            return X, y

    def _preprocess_input(self, X):
        """Preprocess input data for prediction"""
        if self.backend == "cuda":
            try:
                import cupy as cp
                return cp.array(X)
            except ImportError:
                pass

        return X


class AcceleratedHeterogeneousRandomForest(HardwareAcceleratedModel):
    """Hardware-accelerated version of HeterogeneousRandomForest"""

    def __init__(self, hardware_config, n_estimators=100, max_features='auto',
                 min_features=None, max_features_step=None, random_state=None,
                 n_jobs=None, verbose=0):

        # Add debug print to confirm n_estimators value
        print(f"[DEBUG] AcceleratedHeterogeneousRandomForest received n_estimators={n_estimators}")

        # If n_jobs not specified, use detected cores
        if n_jobs is None:
            n_jobs = hardware_config["num_cores"]

        # Store these parameters before calling parent constructor
        self.min_features = min_features
        self.max_features_step = max_features_step
        self.feature_subset_sizes_ = []

        # Call parent constructor with the parameters it needs
        super().__init__(
            hardware_config=hardware_config,
            model_type="heterogeneous_rf",
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )

    def _initialize_model(self):
        """Create appropriate model based on available hardware"""

        try:
            print(f"[MPS_DEBUG] Initializing model with backend: {self.backend}")

            # Import your HeterogeneousRandomForest
            try:
                from models.adaptive_rf import HeterogeneousRandomForest
                print(f"[MPS_DEBUG] Successfully imported HeterogeneousRandomForest")
            except ImportError as e:
                print(f"[MPS_DEBUG] ERROR importing HeterogeneousRandomForest: {str(e)}")
                raise

            if self.backend == "cuda":
                try:
                    # Try to import CUDA libraries
                    import cupy
                    import cuml
                    print(f"[MPS_DEBUG] CUDA libraries available: cupy={cupy.__version__}, cuml available")

                    # We could implement a CUDA optimized version here
                    print("[MPS_DEBUG] CUDA-optimized HeterogeneousRandomForest not yet implemented")
                    print("[MPS_DEBUG] Using CPU implementation with optimal threading")

                    self.model = HeterogeneousRandomForest(
                        n_estimators=self.model_params["n_estimators"],
                        max_features=self.model_params["max_features"],
                        min_features=self.min_features,
                        max_features_step=self.max_features_step,
                        random_state=self.model_params["random_state"],
                        n_jobs=self.model_params["n_jobs"],
                        verbose=self.model_params["verbose"]
                    )
                except ImportError as e:
                    print(f"[MPS_DEBUG] CUDA libraries not available: {str(e)}")
                    print("[MPS_DEBUG] Falling back to CPU implementation")
                    # Fall back to CPU implementation
                    self.model = HeterogeneousRandomForest(
                        n_estimators=self.model_params["n_estimators"],
                        max_features=self.model_params["max_features"],
                        min_features=self.min_features,
                        max_features_step=self.max_features_step,
                        random_state=self.model_params["random_state"],
                        n_jobs=self.model_params["n_jobs"],
                        verbose=self.model_params["verbose"]
                    )
            elif self.backend == "mps":
                print("[MPS_DEBUG] Using MPS (Apple Silicon) optimizations")
                print("[MPS_DEBUG] Attempting to create HeterogeneousRandomForest with optimal threading")

                # Use CPU implementation with threading optimized for the platform
                self.model = HeterogeneousRandomForest(
                    n_estimators=self.model_params["n_estimators"],
                    max_features=self.model_params["max_features"],
                    min_features=self.min_features,
                    max_features_step=self.max_features_step,
                    random_state=self.model_params["random_state"],
                    n_jobs=self.model_params["n_jobs"],
                    verbose=self.model_params["verbose"]
                )
                print("[MPS_DEBUG] HeterogeneousRandomForest created successfully")
            else:
                print(f"[MPS_DEBUG] Using CPU implementation with {self.model_params['n_jobs']} threads")
                # Use CPU implementation with threading optimized for the platform
                self.model = HeterogeneousRandomForest(
                    n_estimators=self.model_params["n_estimators"],
                    max_features=self.model_params["max_features"],
                    min_features=self.min_features,
                    max_features_step=self.max_features_step,
                    random_state=self.model_params["random_state"],
                    n_jobs=self.model_params["n_jobs"],
                    verbose=self.model_params["verbose"]
                )

        except Exception as e:
            print(f"[MPS_DEBUG] CRITICAL ERROR initializing model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise