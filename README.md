# Gym Reviews Insight Analysis Pipeline

Insight analysis of Google and Trustpilot customer reviews for an undisclosed gym.

---

## Table of Contents

- [Description](#description)
- [Project Status](#project-status)
- [Getting Started](#getting-started)
  - [Dependencies](#dependencies)
  - [Installing](#installing)
  - [Executing the Program](#executing-the-program)
- [Design Decisions](#design-decisions)
- [Help](#help)
- [Version History](#version-history)
- [Author](#authors)
- [Acknowledgements](#acknowledgements)

---

## Description

This project analyses customer feedback from Google and Trustpilot reviews for a gym, with the aim of understanding common customer pain points, identifying recurring themes in negative feedback, and extracting actionable insights to help improve customer satisfaction and overall service quality.

The pipeline implements an ETL (Extract, Transform, Load) process to combine, clean, and structure both datasets before applying topic modelling techniques. The output is parsed directly into an LLM to automatically generate key insights, recommendations and justifications on grouped topics.

> **Note:** The dataset used in this project is **synthetic**, generated using Python's Faker library.
> All names, reviews and other data points are **fictitious** and do not represent real individuals, actual gym locations, topics or insights into this gym.

---

## Project Status

Actively in development. The core pipeline is implemented. Planned next steps include:

- Adding visualisations for topic modelling outputs
- REST API deployment via FastAPI
- Cloud integration (S3 and broader AWS/GCP storage)
- Docker containerisation for portable deployment

---

## Getting Started

### Dependencies

The project uses **Python 3.11.3**. Key dependencies are listed below. For the full list see `requirements.txt`.

**Core**

- `pandas`
- `nltk`
- `openai`
- `python-dotenv`
- `PyYAML`
- `certifi`

**Topic Modelling**

- `bertopic`
- `sentence-transformers`
- `transformers` + `tokenizers`
- `torch`
- `umap-learn`
- `hdbscan`
- `numba` _(requires LLVM; can cause install issues on some systems)_

**Dev & Code Quality**

- `pytest`
- `mypy`
- `pre-commit`
- `ruff`
- `detect-secrets`

---

### Installing

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/spana97/gym-reviews-insight-analysis-pipeline.git
cd gym-reviews-insight-analysis-pipeline
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

> **Mac Intel users** — if you encounter issues installing PyTorch, ensure `torch==2.2.2` is being installed as pinned in `requirements.txt`. If you hit SSL errors during NLTK resource downloads, this is handled automatically via `certifi`.

---

### Executing the Program

Run the ETL pipeline first to extract, clean and combine the raw Google and Trustpilot reviews:

```bash
python3 scripts/run_etl_pipeline.py
```

Then run the main pipeline to perform topic modelling and generate insights:

```bash
python3 scripts/run_pipeline.py
```

Outputs will be saved to `data/processed/` and the BERTopic model to `models/`.

#### Configuration

Behaviour can be adjusted via `config.yaml`. Notable parameters include:

- `filters.low_rating_max` — threshold for filtering low-rated reviews (default: `3`)
- `topic_model.nr_topics` — number of topics to extract (default: `auto`)
- `insights_generator.developer_prompt` / `user_prompt` — customise the LLM prompts

---

## Design Decisions

**Dependency management** — This project uses `requirements.txt` rather than `uv` + `pyproject.toml`. This was a deliberate choice to gain experience with the traditional dependency management workflow. Future projects will adopt `uv` + `pyproject.toml` as the default.

**Infrastructure** — The pipeline currently runs locally. Cloud storage (S3), containerisation (Docker), and REST API deployment (FastAPI) are planned for future iterations.

---

## Help

### Known Issues (Mac Intel)

**PyTorch** — Mac Intel users may encounter compatibility issues with recent versions of PyTorch. This project pins `torch==2.2.2` in `requirements.txt` to ensure stability on older Mac hardware.

**NLTK SSL certificates** — Mac users may hit SSL certificate errors when downloading NLTK resources. This is a known macOS issue and is handled explicitly via `certifi` — see `src/text_preprocessing/helpers.py` for the SSL context override.

---

## Version History

- `0.5.0` — Logging, CI/CD actions, pre-commit updates (ruff, detect-secrets)
- `0.4.0` — Full pipeline integrated
- `0.3.0` — Topic modelling and insight generation implemented
- `0.2.0` — Text preprocessing implemented
- `0.1.0` — ETL pipeline and initial repository setup

---

## Authors

- Sean Panacides
- seanpanacides@gmail.com
- GitHub: https://github.com/spana97

---

## Acknowledgements

The original dataset was provided through a university-industry partnership as part of the [Data Science with Machine Learning & AI Career Accelerator](https://onlinecareeraccelerators.pace.cam.ac.uk/cambridge-data-science-career-accelerator) programme, delivered in collaboration with the University of Cambridge and Forthrev. The client has not been disclosed. This project represents an independent productionisation of that work for learning purposes.
