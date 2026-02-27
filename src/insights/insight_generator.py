from openai import OpenAI

from utils.logger import logger


class InsightGenerator:
    """Generates insights from clustered data using OpenAI's API."""

    def __init__(self, config: dict, api_key: str, model: str = "gpt-5-mini"):
        """
        Initializes the InsightGenerator with configuration and API details.

        Args:
            config (dict): Configuration dictionary containing
                developer_prompt, user_prompt and max_output_tokens.
            api_key (str): OpenAI API key
            model (str): OpenAI model. Defaults to gpt-5-mini.

        """
        logger.info(f"Initializing InsightGenerator with model: {model}")
        self.config = config
        self.model = model

        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            raise

        try:
            self.developer_prompt = config["developer_prompt"]
            self.user_prompt = config["user_prompt"]
        except KeyError as e:
            logger.error(f"Missing prompt in configuration: {e}")
            raise

        logger.info("Model initialized successfully.")

    def _build_user_prompt(self, formatted_clusters: str) -> str:
        """Inserts the formatted clusters into the user prompt template."""
        return self.user_prompt.format(clusters=formatted_clusters)

    def generate_insights(self, formatted_clusters: str) -> str:
        """
        Generates insights based on the formatted clusters.

        Args:
            formatted_clusters (str): Formatted BERTopic representative topic documentation.

        Returns:
            str: Insights generated from the provided topics.
        """
        logger.info("Generating insights with formatted clusters")
        user_prompt = self._build_user_prompt(formatted_clusters)

        try:
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "developer", "content": self.developer_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=self.config["max_output_tokens"],
            )
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise

        logger.info("Insights generated successfully.")
        return response.output_text
