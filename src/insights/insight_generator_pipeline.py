import os
import json

from dotenv import load_dotenv

from insights.format_insights import parse_insights
from insights.insight_generator import InsightGenerator
from utils.logger import logger

load_dotenv()


def run_insight_generator_pipeline(config: dict):
    """Run the insight generator insight analysis pipeline for Google and Trustpilot reviews."""
    logger.info("Starting insight generation pipeline")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        logger.error("OPENAI_API_KEY environment variable not set.")
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    representative_docs_path = config["topic_model"]["representative_docs_output"]
    if not os.path.exists(representative_docs_path):
        logger.error("No representative docs file found.")
        raise FileNotFoundError(
            f"Representative docs file not found: {representative_docs_path}"
        )

    try:
        with open(representative_docs_path, "r") as f:
            representative_docs = json.dumps(json.load(f), indent=2)
    except Exception as e:
        logger.error(f"Error reading representative docs: {e}")
        raise

    try:
        insight_generator = InsightGenerator(
            config["insights_generator"],
            api_key=api_key,
        )
        insights = insight_generator.generate_insights(representative_docs)
        parsed_insights = parse_insights(insights)
        parsed_insights.to_csv(
            config["insights_generator"]["insights_output"], index=False
        )
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise

    logger.info("Insight generator pipeline executed successfully.")
