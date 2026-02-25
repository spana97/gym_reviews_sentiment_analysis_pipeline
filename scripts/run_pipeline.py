from src.pipeline import run_pipeline
from src.utils.config_loader import load_config

if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)
