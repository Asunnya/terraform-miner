import os
import yaml

def get_config():
    """Load configuration from YAML file or environment variables."""
    # Try to load from config file first
    config = {}
    config_path = os.environ.get('TF_CONFIG_PATH', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        pass  # No config file or invalid YAML
    
    # Override/set from environment variables
    config['dataset_dir'] = os.environ.get('TF_DATASET_DIR', config.get('dataset_dir', 'dataset'))
    config['output_dir'] = os.environ.get('TF_OUTPUT_DIR', config.get('output_dir', 'reports'))
    
    return config