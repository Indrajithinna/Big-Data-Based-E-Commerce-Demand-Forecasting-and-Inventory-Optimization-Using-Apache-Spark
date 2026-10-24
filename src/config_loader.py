# =============================================================================
# Utility: Configuration Loader
# =============================================================================
# Loads config/config.yaml and resolves all paths relative to project root.
# Every module imports this to avoid hard-coded paths or magic numbers.
# =============================================================================

import os
import yaml


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    Walks up from this file's location until it finds config/config.yaml.
    """
    current = os.path.dirname(os.path.abspath(__file__))
    # src/ → project root
    root = os.path.dirname(current)
    config_path = os.path.join(root, "config", "config.yaml")
    if os.path.exists(config_path):
        return root
    # Fallback: try one more level up
    root = os.path.dirname(root)
    if os.path.exists(os.path.join(root, "config", "config.yaml")):
        return root
    raise FileNotFoundError(
        "Could not locate config/config.yaml from project root. "
        "Ensure you are running from the project directory."
    )


def load_config(config_path=None):
    """
    Load the YAML configuration file and resolve relative paths.
    
    Args:
        config_path: Optional absolute path to config file.
                     Defaults to config/config.yaml in project root.
    
    Returns:
        dict: Parsed configuration dictionary.
    """
    if config_path is None:
        root = get_project_root()
        config_path = os.path.join(root, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Resolve all relative paths in config['paths'] to absolute paths
    root = get_project_root()
    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config["paths"][key] = os.path.join(root, value)
    
    # Ensure correct Python and Java environment for Spark
    import sys
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    if sys.platform == "win32":
        java_home = os.environ.get("JAVA_HOME")
        if not java_home or not os.path.exists(java_home):
            for candidate in [
                r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot",
                r"C:\Program Files\Eclipse Adoptium\jdk-17",
                r"C:\Program Files\Java\jdk-17",
                r"C:\Program Files\Java\jdk-11",
            ]:
                if os.path.exists(candidate):
                    os.environ["JAVA_HOME"] = candidate
                    break
        
        # Configure HADOOP_HOME for Windows (winutils.exe)
        hadoop_home = os.environ.get("HADOOP_HOME")
        if not hadoop_home or not os.path.exists(hadoop_home):
            for candidate in [r"C:\hadoop", os.path.join(root, "hadoop")]:
                if os.path.exists(candidate):
                    os.environ["HADOOP_HOME"] = candidate
                    bin_dir = os.path.join(candidate, "bin")
                    if os.path.exists(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + ";" + os.environ.get("PATH", "")
                    break
    
    return config


def ensure_directories(config):
    """
    Create all necessary output directories from config.
    
    Args:
        config: Configuration dictionary.
    """
    dirs_to_create = [
        config["paths"]["raw_data_dir"],
        config["paths"]["processed_data_dir"],
        config["paths"]["results_dir"],
        config["paths"].get("models_dir", os.path.join(get_project_root(), "models")),
        config["paths"].get("analytics_output_dir", ""),
        config["paths"].get("forecast_output_dir", ""),
        config["paths"].get("inventory_output_dir", ""),
    ]
    for d in dirs_to_create:
        if d:
            os.makedirs(d, exist_ok=True)


# Quick test
if __name__ == "__main__":
    cfg = load_config()
    print("Project root:", get_project_root())
    print("Raw data dir:", cfg["paths"]["raw_data_dir"])
    print("Spark app name:", cfg["spark"]["app_name"])
    print("Config loaded successfully!")
