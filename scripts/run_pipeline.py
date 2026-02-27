from pipeline import run_pipeline
from utils.config_loader import load_config

if __name__ == "__main__":
    config = load_config()
    run_pipeline(config)
