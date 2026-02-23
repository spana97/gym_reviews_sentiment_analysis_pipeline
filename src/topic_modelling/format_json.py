import json


def format_json(topic_info):
    """
    Format the topic modelling information from BERTopic into a JSON structure.
    """
    topics_docs = topic_info.tolist()

    clusters = {f"cluster_{i+1}": docs for i, docs in enumerate(topics_docs)}

    formatted_input = json.dumps(clusters, indent=2)
    return formatted_input
