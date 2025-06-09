import yaml

def load_config(path: str):
    """Loads and validates the YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Validate batch_size parameter
    if 'batch_size' in config:
        batch_size = config['batch_size']
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got: {batch_size}")
    
    return config