import os
import json
import yaml
import logging
from datetime import datetime


class ConfigManager:
    """
    Manages configuration settings for the malware detection project.
    Handles loading, saving, and accessing configuration parameters.
    """

    def __init__(self, config_file=None, defaults=None):
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to configuration file (JSON or YAML)
            defaults: Default configuration dictionary
        """
        self.config_file = config_file
        self.config = defaults or self._get_default_config()

        # Load configuration if file provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

        # Set up logging
        self._setup_logging()

    def _get_default_config(self):
        """
        Get default configuration.

        Returns:
            Dictionary of default configuration values
        """
        return {
            # Data default settings
            # 'data': {
            #     'regions': ['US', 'BR', 'JP'],
            #     'data_dir': 'data',
            #     'cache_dir': 'cached_features',
            #     'use_cached': True,
            #     'test_size': 0.2,
            #     'random_state': 42
            # },

            # Data Ember settings
            'data': {
                'regions': ['US', 'EU', 'JP'],
                'data_dir': 'ember_data',
                'cache_dir': 'ember_cached_features',
                'use_cached': True,
                'test_size': 0.2,
                'random_state': 42
            },

            # Feature extraction settings
            'features': {
                'max_features': 1500,
                'selection_method': 'f_score',  # 'f_score', 'chi2', 'mutual_info'
                'min_features_per_region': {
                    'US': 300,
                    'BR': 400,
                    'JP': 800
                }
            },

            # Model settings
            'models': {
                'rf': {
                    'n_estimators': 100,
                    'random_state': 42
                },
                'adaptive_rf': {
                    'n_estimators': 100,
                    'warning_delta': 0.005,
                    'drift_delta': 0.01,
                    'random_state': 42
                },
                'heterogeneous_rf': {
                    'n_estimators': 100,
                    'random_state': 42
                }
            },

            # Federated learning settings
            'federated': {
                'simulation_mode': True,
                'message_dir': 'messages',
                'selection_strategy': 'confidence'  # 'confidence' or 'random'
            },

            # Distillation settings
            'distillation': {
                'global_features': 800,
                'n_trees': 100,
                'n_distilled_trees': 50,
                'distilled_features': {
                    'US': 300,
                    'BR': 400,
                    'JP': 800
                }
            },

            # YARA rule generation settings
            'yara': {
                'rule_prefix': 'rule_from_ml',
                'max_rules': 1000
            },

            # Output settings
            'output': {
                'results_dir': 'results',
                'figures_dir': 'figures',
                'models_dir': 'models',
                'rules_dir': 'rules',
                'save_models': True,
                'save_rules': True
            },

            # Logging settings
            'logging': {
                'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'malware_detection.log'
            }
        }

    def load_config(self, config_file):
        """
        Load configuration from file.

        Args:
            config_file: Path to configuration file (JSON or YAML)

        Returns:
            Loaded configuration dictionary
        """
        if not os.path.exists(config_file):
            logging.warning(f"Configuration file not found: {config_file}")
            return self.config

        _, ext = os.path.splitext(config_file)

        try:
            # Load JSON config
            if ext.lower() == '.json':
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)

            # Load YAML config
            elif ext.lower() in ['.yaml', '.yml']:
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)

            # Unknown format
            else:
                logging.warning(f"Unsupported configuration file format: {ext}")
                return self.config

            # Update config with loaded values
            self._update_config(self.config, loaded_config)
            self.config_file = config_file

            logging.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logging.error(f"Error loading configuration file: {str(e)}")

        return self.config

    def _update_config(self, config, updates):
        """
        Recursively update configuration dictionary.

        Args:
            config: Configuration dictionary to update
            updates: Dictionary with updates

        Returns:
            Updated configuration dictionary
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value

        return config

    def save_config(self, config_file=None):
        """
        Save configuration to file.

        Args:
            config_file: Path to configuration file (if None, use self.config_file)

        Returns:
            Success flag
        """
        config_file = config_file or self.config_file

        if not config_file:
            logging.warning("No configuration file specified for saving")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(config_file)), exist_ok=True)

            _, ext = os.path.splitext(config_file)

            # Save as JSON
            if ext.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(self.config, f, indent=4)

            # Save as YAML
            elif ext.lower() in ['.yaml', '.yml']:
                with open(config_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)

            # Unknown format
            else:
                logging.warning(f"Unsupported configuration file format: {ext}")
                return False

            logging.info(f"Saved configuration to {config_file}")
            return True

        except Exception as e:
            logging.error(f"Error saving configuration file: {str(e)}")
            return False

    def get(self, key, default=None):
        """
        Get configuration value by key.

        Args:
            key: Configuration key (can be nested with dots, e.g., 'data.regions')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key, value):
        """
        Set configuration value by key.

        Args:
            key: Configuration key (can be nested with dots, e.g., 'data.regions')
            value: Value to set

        Returns:
            Success flag
        """
        keys = key.split('.')
        config = self.config

        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}

            if not isinstance(config[k], dict):
                config[k] = {}

            config = config[k]

        # Set the value
        config[keys[-1]] = value
        return True

    def _setup_logging(self):
        """Set up logging based on configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file')

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file) if log_file else logging.StreamHandler()
            ]
        )

    def create_experiment_dirs(self):
        """
        Create directories for experiment outputs.

        Returns:
            Dictionary mapping directory types to paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(self.get('output.results_dir', 'results'), f'experiment_{timestamp}')

        dirs = {
            'base': base_dir,
            'figures': os.path.join(base_dir, self.get('output.figures_dir', 'figures')),
            'models': os.path.join(base_dir, self.get('output.models_dir', 'models')),
            'rules': os.path.join(base_dir, self.get('output.rules_dir', 'rules'))
        }

        # Create directories
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        return dirs


# Global configuration instance
config = ConfigManager()


def get_config(config_file=None):
    """
    Get configuration manager instance.

    Args:
        config_file: Path to configuration file (if not already loaded)

    Returns:
        ConfigManager instance
    """
    global config

    if config_file and config_file != config.config_file:
        config.load_config(config_file)

    return config