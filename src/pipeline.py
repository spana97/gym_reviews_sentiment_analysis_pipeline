from topic_modelling.topic_pipeline import run_topic_model_pipeline
from insights.insight_generator_pipeline import run_insight_generator_pipeline
from utils.logger import logger


def run_pipeline(config: dict):
    """Run the full insight analysis pipeline for Google and Trustpilot reviews."""
    logger.info("Starting insight generation pipeline")

    run_topic_model_pipeline(config)
    run_insight_generator_pipeline(config)

    logger.info("Pipeline executed successfully.")
