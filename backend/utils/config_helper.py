import yaml

def load_config(config_path="backend/config/config.yaml"):
    """
    Loads project YAML config file.
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return None
    except Exception as e:
        print(f"Error loading config: {e}")
        return None
