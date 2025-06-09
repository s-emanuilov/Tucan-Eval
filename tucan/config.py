import yaml

def load_config(path: str):
    """Loads and validates the YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # Add any future validation here
    return config