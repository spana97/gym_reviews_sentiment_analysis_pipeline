import os

import pandas as pd
from dotenv import load_dotenv

from insights.format_insights import parse_insights
from insights.insight_generator import InsightGenerator
from text_preprocessing.text_preprocessor import TextPreprocessor
from topic_modelling.format_json import format_json
from topic_modelling.topic_model import TopicModel

load_dotenv()


def run_pipeline(config: dict):

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    df = pd.read_parquet(config["data"]["processed_output"])

    preprocessor = TextPreprocessor(
        extra_stop_words=config["text_preprocessing"]["extra_stopwords"]
    )
    topic_text = df["review"].apply(preprocessor.preprocess).tolist()

    topic_model = TopicModel(config["topic_model"])
    topics, _ = topic_model.fit(topic_text)
    topic_model.save(config["topic_model"]["model_output_path"])

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(config["topic_model"]["topics_output_path"], index=False)

    formatted_json = format_json(topic_info["Representative_Docs"][:5])

    insight_generator = InsightGenerator(
        config["insights_generator"],
        api_key=api_key,
    )
    insights = insight_generator.generate_insights(formatted_json)
    parsed_insights = parse_insights(insights)
    parsed_insights.to_csv(
        config["insights_generator"]["insights_output_path"], index=False
    )
    print(
        "Pipeline executed successfully. Insights saved to:",
        config["insights_generator"]["insights_output_path"],
    )
